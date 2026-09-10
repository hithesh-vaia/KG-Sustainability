"""Global search: map-reduce over community reports."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..graphdb import Neo4jClient
from ..llm import LLM
from .. import prompts as P
from .context import ntokens


@dataclass
class GlobalResult:
    answer: str
    themes: list[dict] = field(default_factory=list)
    communities_used: int = 0


def _batch_reports(
    reports: list[dict],
    batch_tokens: int,
) -> list[str]:
    batches: list[str] = []
    cur: list[str] = []
    used = 0

    for r in reports:
        block = (
            f"### {r['title']} "
            f"(rating {r.get('rating')})\n"
            f"{r['full_content']}\n"
        )

        cost = ntokens(block)

        if cur and used + cost > batch_tokens:
            batches.append("\n".join(cur))
            cur = []
            used = 0

        cur.append(block)
        used += cost

    if cur:
        batches.append("\n".join(cur))

    return batches


def global_search(
    question: str,
    db: Neo4jClient,
    llm: LLM,
    level: int | None = 1,
    batch_tokens: int = 6000,
) -> GlobalResult:

    # 1. Get all community reports
    reports = db.community_reports(level=level)

    if not reports and level is not None:
        reports = db.community_reports(level=None)

    if not reports:
        return GlobalResult(
            answer="No community reports available. Run ingestion first."
        )

    # 2. MAP: extract themes from every batch
    all_themes: list[dict] = []

    batches = _batch_reports(reports, batch_tokens)

    for batch in batches:
        try:
            data = llm.chat_json(
                P.GLOBAL_MAP.format(
                    context=batch,
                    question=question,
                )
            )
        except Exception:
            continue

        for theme in data.get("themes", []) or []:
            name = (theme.get("theme") or "").strip()
            description = (theme.get("description") or "").strip()

            if not name or not description:
                continue

            try:
                importance = int(
                    theme.get("importance", 0) or 0
                )
            except (TypeError, ValueError):
                importance = 0

            all_themes.append(
                {
                    "theme": name,
                    "description": description,
                    "importance": importance,
                }
            )

    # 3. Nothing found
    if not all_themes:
        return GlobalResult(
            answer=(
                "The community reports do not contain enough information "
                "to answer this global question."
            ),
            communities_used=len(reports),
        )

    # 4. Give ALL extracted themes to REDUCE
    context = "\n".join(
        f"- {t['theme']} "
        f"(importance={t['importance']}): "
        f"{t['description']}"
        for t in all_themes
    )

    # 5. REDUCE: synthesize across all communities
    answer = llm.chat(
        P.GLOBAL_REDUCE.format(
            context=context,
            question=question,
        )
    )

    return GlobalResult(
        answer=answer.strip(),
        themes=all_themes,
        communities_used=len(reports),
    )
