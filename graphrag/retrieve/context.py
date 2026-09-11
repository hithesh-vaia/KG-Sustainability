"""Shared retrieval helpers: token budgeting + context assembly."""
from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def ntokens(text: str) -> int:
    return len(_enc.encode(text))


def pack(
    sections: list[tuple[str, list[str]]],
    budget: int,
    first_section_frac: float = 0.7,
) -> str:
    """Greedily pack titled sections of lines into a token budget.

    The first section is capped at ``first_section_frac`` of the budget so the
    remaining sections (knowledge-graph entities / relationships) are not starved
    when the leading section (source evidence) is large.
    """
    out: list[str] = []
    used = 0
    for index, (title, lines) in enumerate(sections):
        if not lines:
            continue
        section_cap = (
            int(budget * first_section_frac)
            if index == 0 and len(sections) > 1
            else budget
        )
        header = f"\n## {title}\n"
        if used + ntokens(header) > budget:
            break
        out.append(header)
        used += ntokens(header)
        for line in lines:
            cost = ntokens(line) + 1
            if used + cost > min(section_cap, budget):
                break
            out.append(line)
            used += cost
    return "\n".join(out).strip()
