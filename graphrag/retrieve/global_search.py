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
    points: list[dict] = field(default_factory=list)
    communities_used: int = 0


def _batch_reports(reports: list[dict], batch_tokens: int) -> list[str]:
    batches: list[str] = []
    cur: list[str] = []
    used = 0
    for r in reports:
        block = f"### {r['title']} (rating {r.get('rating')})\n{r['full_content']}\n"
        cost = ntokens(block)
        if cur and used + cost > batch_tokens:
            batches.append("\n".join(cur))
            cur, used = [], 0
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
    top_points: int = 30,
) -> GlobalResult:
    reports = db.community_reports(level=level)
    if not reports and level is not None:
        reports = db.community_reports(level=None)
    if not reports:
        return GlobalResult(answer="No community reports available. Run ingestion first.")

    all_points: list[dict] = []
    for batch in _batch_reports(reports, batch_tokens):
        try:
            data = llm.chat_json(P.GLOBAL_MAP.format(context=batch, question=question))
        except Exception:
            continue
        for p in data.get("points", []) or []:
            desc = (p.get("description") or "").strip()
            if not desc:
                continue
            try:
                score = int(p.get("score", 0) or 0)
            except (TypeError, ValueError):
                score = 0
            if score > 0:
                all_points.append({"description": desc, "score": score})

    all_points.sort(key=lambda p: p["score"], reverse=True)
    kept = all_points[:top_points]
    if not kept:
        return GlobalResult(
            answer=(
                "The community reports (thematic summaries) do not cover this. "
                "If you're after a specific fact, number, name, or entity, try "
                "`--method local` instead — that searches entities and source text."
            ),
            communities_used=len(reports),
        )

    context = "\n".join(f"- ({p['score']}) {p['description']}" for p in kept)
    answer = llm.chat(P.GLOBAL_REDUCE.format(context=context, question=question))
    return GlobalResult(answer=answer.strip(), points=kept, communities_used=len(reports))
