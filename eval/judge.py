"""Optional LLM-as-judge scoring."""
from __future__ import annotations

import os

from graphrag.llm import LLM

JUDGE_PROMPT = """You are grading an answer produced by a retrieval system against a
reference answer. Judge only factual correctness relative to the reference and the
question — ignore style, extra detail, and phrasing.

Question:
{question}

Reference answer (ground truth from the source document):
{expected}

System answer:
{answer}

Return JSON: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}
- "correct": the system answer states the reference fact (units / period / value all right; rounding within 1% is fine).
- "partial": right direction but missing or fuzzing a key part (wrong unit, missing period, only half the comparison).
- "incorrect": contradicts the reference, invents a value, or fails to answer.
For a reference that says information is NOT in the report: "correct" only if the
system answer also declines / says it is not available."""


def judge(item: dict, answer: str, llm: LLM) -> dict:
    model = os.getenv("EVAL_JUDGE_MODEL")
    prev = llm.model
    if model:
        llm.model = model
    try:
        data = llm.chat_json(JUDGE_PROMPT.format(
            question=item["question"],
            expected=item.get("expected", ""),
            answer=answer,
        ))
    except Exception as exc:  # pragma: no cover
        return {"verdict": "error", "reason": str(exc)[:200]}
    finally:
        llm.model = prev
    v = str(data.get("verdict", "")).lower().strip()
    if v not in ("correct", "partial", "incorrect"):
        v = "incorrect"
    return {"verdict": v, "reason": data.get("reason", "")}
