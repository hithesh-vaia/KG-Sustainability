"""Deterministic lexical re-ranking helpers for local search.

Pure Python -- no LLM, no DB. Used to re-order vector-search seed entities so the
answer model is handed the correctly-scoped entity first (e.g. TOTAL EMPLOYEES
rather than a bare EMPLOYEES sibling), and to score source chunks by overlap with
the question.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

_STOP = {
    "the", "a", "an", "of", "in", "for", "to", "was", "were", "did", "do", "does",
    "have", "has", "had", "how", "many", "much", "what", "which", "who", "number",
    "and", "or", "as", "at", "is", "are", "its", "it", "on", "by", "with", "per",
    "according", "report", "reported", "bank", "s",
}

# words that make an entity name MORE specific than a bare metric
QUALIFIERS = {
    "permanent", "non-permanent", "nonpermanent", "contract", "contractual",
    "temporary", "temp", "casual", "trainee", "apprentice", "male", "female",
    "women", "men", "differently", "abled", "disabled", "kmp", "kmps", "director",
    "directors", "board", "worker", "workers", "on-roll", "off-roll", "onroll",
    "offroll", "outsourced", "third-party", "new", "turnover", "attrition", "hire",
    "hires", "hired", "exits", "left", "senior", "junior", "middle", "management",
    "executive", "non-executive", "urban", "rural", "other", "others", "gross",
    "net",
}

TOTAL_WORDS = {"total", "overall", "aggregate", "combined", "entire", "all"}


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    return _TOKEN.findall(_norm(s))


_FY_RANGE = re.compile(r"(?:fy\s*)?'?(\d{4})\s*[-/]\s*'?(\d{2,4})", re.I)
_FY_SHORT = re.compile(r"\bfy\s*'?(\d{2})\b(?![-/\d])", re.I)
_FY_LONG = re.compile(r"\bfy\s*'?(\d{4})\b(?![-/\d])", re.I)
_MAR31 = re.compile(r"(?:as (?:of|at|on)\s+)?31(?:st)?\s+march[ ,]+(\d{4})", re.I)


def _canon(start: int, end: int) -> str:
    if end < 100:
        end = (start // 100) * 100 + end
        if end < start:
            end += 100
    return f"{start}-{str(end)[-2:]}"


def fiscal_years(s: str) -> set[str]:
    """Canonical fiscal-year tokens present in ``s`` (e.g. ``{"2024-25"}``).

    Recognises: 'FY 2024-25', '2024-25', 'FY25', 'FY2025', 'financial year
    2024-25', 'as of 31 March 2025'. Bare integers do not match.
    """
    out: set[str] = set()
    txt = _norm(s)
    for m in _FY_RANGE.finditer(txt):
        try:
            out.add(_canon(int(m.group(1)), int(m.group(2))))
        except ValueError:
            pass
    for m in _FY_SHORT.finditer(txt):
        y = 2000 + int(m.group(1))
        out.add(f"{y - 1}-{str(y)[-2:]}")
    for m in _FY_LONG.finditer(txt):
        y = int(m.group(1))
        out.add(f"{y - 1}-{str(y)[-2:]}")
    for m in _MAR31.finditer(txt):
        y = int(m.group(1))
        out.add(f"{y - 1}-{str(y)[-2:]}")
    return out


def score_seed(seed: dict, q_tokens: set[str], q_years: set[str], rank: int) -> float:
    name = seed.get("name") or ""
    desc = seed.get("description") or ""
    n_tok = set(tokens(name))
    d_tok = set(tokens(desc))

    try:
        vec = float(seed.get("score") or 0.0)
    except (TypeError, ValueError):
        vec = 0.0
    base = vec if vec else 1.0 / (1 + rank)
    s = 1.6 * base

    content_q = {t for t in q_tokens if t not in _STOP}
    if content_q:
        s += 0.9 * (len(content_q & n_tok) / len(content_q))
        if content_q <= n_tok:
            s += 0.5
        s += 0.3 * (len(content_q & d_tok) / len(content_q))

    n_years = fiscal_years(name) | fiscal_years(desc)
    if q_years:
        if n_years & q_years:
            s += 0.8
        elif n_years:
            s -= 1.0

    q_has_total = bool(q_tokens & TOTAL_WORDS)
    name_quals = (n_tok & QUALIFIERS) - q_tokens
    s -= 0.7 * len(name_quals)
    if name_quals and q_has_total:
        s -= 0.6
    if q_has_total and (n_tok & TOTAL_WORDS):
        s += 0.7
    elif q_has_total and not name_quals:
        s += 0.15
    return s


def rerank_seeds(seeds: list[dict], question: str, keep: int = 8) -> list[dict]:
    q_tokens = set(tokens(question))
    q_years = fiscal_years(question)
    scored = [
        (score_seed(sd, q_tokens, q_years, i), -i, sd)
        for i, sd in enumerate(seeds)
        if sd and sd.get("name")
    ]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [sd for _, _, sd in scored[:keep]]
