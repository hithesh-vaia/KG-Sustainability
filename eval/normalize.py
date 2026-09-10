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


_NON_NUMERIC = re.compile(r"[\d.,%/:+-]")


def _is_numeric_needle(needle: str) -> bool:
    """Should `needle` be compared as a number?

    Only when the needle is essentially a quantity ("57,763.14", "4.30 MT") --
    not when it merely happens to contain a digit. Without this, a needle like
    "tCO2e for the entire loan portfolio" yields the number 2 and then matches
    any answer containing a 2; likewise "ISO 45001:2018" or "24x7 SOC".
    """
    n = normalize(needle).lstrip("rupees dollars ").strip()
    if not n or not n[0].isdigit():
        return False                  # "iso 45001:2018", "tco2e ..." are labels
    residue = _NON_NUMERIC.sub("", n).replace(" ", "")
    return len(residue) <= 8          # leaves room for a unit like "tco2e" / "mt"


def matches(needle: str, haystack_norm: str) -> bool:
    n = normalize(needle)
    if n and n in haystack_norm:
        return True
    return _is_numeric_needle(needle) and number_close(needle, haystack_norm)


def _matches_needle(needle, haystack_norm: str) -> bool:
    """A needle is a string, or a list of alternatives meaning "any of these".

    Alternatives matter for questions where several correct phrasings exist --
    "none" / "zero" / "nil" all answer "were there any fatalities?" -- and without
    them such items can only be scored with substrings so short they match anything.
    """
    if isinstance(needle, (list, tuple)):
        return any(matches(alt, haystack_norm) for alt in needle)
    return matches(needle, haystack_norm)


def _label(needle) -> str:
    return " | ".join(str(x) for x in needle) if isinstance(needle, (list, tuple)) else str(needle)


def contains_all(haystack: str, needles: list) -> tuple[bool, list[str]]:
    hn = normalize(haystack)
    missing = [_label(x) for x in (needles or []) if not _matches_needle(x, hn)]
    return (not missing, missing)


def contains_any(haystack: str, needles: list) -> list[str]:
    hn = normalize(haystack)
    return [_label(x) for x in (needles or []) if _matches_needle(x, hn)]
