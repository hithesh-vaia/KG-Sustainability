"""Shared retrieval helpers: token budgeting + context assembly."""
from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def ntokens(text: str) -> int:
    return len(_enc.encode(text))


def pack(sections: list[tuple[str, list[str]]], budget: int) -> str:
    """Greedily pack titled sections of lines into a token budget."""
    out: list[str] = []
    used = 0
    for title, lines in sections:
        if not lines:
            continue
        header = f"\n## {title}\n"
        if used + ntokens(header) > budget:
            break
        out.append(header)
        used += ntokens(header)
        for line in lines:
            cost = ntokens(line) + 1
            if used + cost > budget:
                break
            out.append(line)
            used += cost
    return "\n".join(out).strip()
