"""Thin OpenRouter (OpenAI-compatible) chat client with retry + token accounting."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CONFIG, Config
from .log import get_logger

log = get_logger()


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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def _complete(self, messages: list[dict], temperature: float, json_mode: bool) -> str:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
            log.debug(
                "llm call #%d: %d prompt + %d completion tokens",
                self.usage.calls, resp.usage.prompt_tokens, resp.usage.completion_tokens,
            )
        return resp.choices[0].message.content or ""

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
