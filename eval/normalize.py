"""Text / number normalization and matching for scoring answers."""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12",
}


def normalize(s: str) -> str:
    """Lowercase, drop thousands separators (incl. Indian grouping), currency and
    unit punctuation, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower()
    s = s.replace("₹", " rupees ").replace("$", " dollars ")
    s = s.replace("₂", "2").replace("²", "2")          # tco₂e / tco²e -> tco2e
    # remove commas that group digits: 3,33,091.94 -> 333091.94 ; 12,333 -> 12333
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[·• ]", " ", s)
    s = re.sub(r"[^\w.%/+\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = " ".join(_WORD_NUM.get(w, w) for w in s.split())
    return s


def _numbers(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def number_close(needle: str, haystack: str, rel_tol: float = 0.01) -> bool:
    """True if `needle` is numeric and a number within rel_tol appears in haystack."""
    ns = _numbers(needle)
    if not ns:
        return False
    target = ns[0]
    for v in _numbers(haystack):
        if target == 0:
            if abs(v) < 1e-9:
                return True
        elif abs(v - target) <= abs(target) * rel_tol:
            return True
    return False


def matches(needle: str, haystack_norm: str) -> bool:
    n = normalize(needle)
    if n and n in haystack_norm:
        return True
    return number_close(needle, haystack_norm)


def contains_all(haystack: str, needles: list[str]) -> tuple[bool, list[str]]:
    hn = normalize(haystack)
    missing = [x for x in (needles or []) if not matches(x, hn)]
    return (not missing, missing)


def contains_any(haystack: str, needles: list[str]) -> list[str]:
    hn = normalize(haystack)
    return [x for x in (needles or []) if matches(x, hn)]
