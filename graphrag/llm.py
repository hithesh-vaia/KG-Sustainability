"""Thin OpenRouter (OpenAI-compatible) chat client with retry + token accounting."""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field

from openai import APIStatusError, OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import CONFIG, Config
from .log import get_logger

log = get_logger()


def _err_reason(exc: BaseException) -> str | None:
    try:
        return exc.body["error"]["metadata"]["reason"]  # type: ignore[attr-defined,index]
    except (AttributeError, TypeError, KeyError):
        return None


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient errors only.

    - 429 / 5xx / network: always retry.
    - 402 `in_flight_budget_exhausted`: transient (too many concurrent requests
      reserving credits) -> retry, it clears as in-flight requests settle.
    - 402 anything else (real out-of-credits), 400, 401: fatal, don't retry.
    """
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        if code in (408, 409, 429):
            return True
        if code == 402:
            return _err_reason(exc) == "in_flight_budget_exhausted"
        return code is not None and code >= 500
    return True  # network errors, timeouts, JSON hiccups


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1

    def summary(self) -> str:
        return (
            f"{self.calls} LLM calls | "
            f"{self.prompt_tokens:,} prompt + {self.completion_tokens:,} completion tokens"
        )


class LLM:
    def __init__(self, config: Config = CONFIG):
        config.validate()
        self.config = config
        self.model = config.model
        self.client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/local/graphrag-lite",
                "X-Title": "graphrag-lite",
            },
        )
        self.usage = Usage()
        # a shell-exported OPENROUTER_API_KEY overrides .env (python-dotenv does not
        # override existing env vars) -- log the tail so the active key is unambiguous
        log.info("LLM: model=%s key=...%s max_tokens=%d",
                 self.model, config.openrouter_api_key[-6:], config.llm_max_tokens)

    @retry(
        reraise=True,  # raise the real APIStatusError, not tenacity's RetryError
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(8),
        wait=wait_exponential(min=5, max=120),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def _complete(self, messages: list[dict], temperature: float, json_mode: bool) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.config.llm_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            log.debug(
                "llm call #%d: %d prompt + %d completion tokens",
                self.usage.calls, resp.usage.prompt_tokens, resp.usage.completion_tokens,
            )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            # output hit max_tokens -> JSON will be truncated; surface it plainly
            log.warning(
                "  output TRUNCATED at max_tokens=%d (raise LLM_MAX_TOKENS, or use a "
                "non-reasoning model — reasoning tokens are spent before the answer)",
                self.config.llm_max_tokens,
            )
        return choice.message.content or ""

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(messages, temperature, json_mode=False)

    def chat_json(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        raw = self._complete(messages, temperature, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # some models wrap JSON in prose / code fences; salvage the object
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start : end + 1])
            raise
