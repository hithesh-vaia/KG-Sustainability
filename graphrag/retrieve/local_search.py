"""Local search: entity-centric retrieval + answer synthesis.

Pure GraphRAG by default: the ONLY embedding query is the one that picks the seed
entities. Everything else (relationships, neighbours, source chunks, community
reports) is reached by walking the graph from those seeds. Pass
``chunk_search=True`` to additionally run a direct query->chunk vector search
(plain RAG) as a safety net for facts buried in tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..embeddings import embed_one
from ..graphdb import Neo4jClient
from ..llm import LLM
from .. import prompts as P
from .context import pack


@dataclass
class LocalResult:
    answer: str
    entities: list[str] = field(default_factory=list)
    communities: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    context: str = ""


def local_search(
    question: str,
    db: Neo4jClient,
    llm: LLM,
    top_k: int = 15,
    token_budget: int = 11000,
    chunk_search: bool = False,
) -> LocalResult:
    qvec = embed_one(question)
    seeds = db.vector_search_entities(qvec, top_k)
    if not seeds:
        return LocalResult(answer="No entities found in the graph for this question.")
    seed_names = [s["name"] for s in seeds]
    ctx = db.entity_context(seed_names, max_chunks=12)

    def _ent_line(e: dict) -> str:
        typ = e.get("type") or "?"
        dom = f" / {e['domain']}" if e.get("domain") else ""
        props = f"  props={e['properties']}" if e.get("properties") and e["properties"] != "{}" else ""
        return f"- {e['name']} [{typ}{dom}]: {e.get('description') or ''}{props}"

    ent_lines = [
        _ent_line(e)
        for e in (ctx.get("entities") or []) + (ctx.get("neighbors") or [])
        if e and e.get("name")
    ]
    rel_lines = [
        f"- {r['source']} -{r.get('rel_type') or 'RELATED'}-> {r['target']}: {r.get('description') or ''}"
        for r in (ctx.get("relationships") or [])
        if r and r.get("source") and r.get("target")
    ]
    comm = [c for c in (ctx.get("communities") or []) if c and c.get("content")]
    comm.sort(key=lambda c: c.get("rating") or 0, reverse=True)
    comm_lines = [f"- {c['title']}: {c['content']}" for c in comm]

    # chunks reached by walking :MENTIONED_IN from the seed entities (graph)
    chunks_by_id: dict[str, dict] = {}
    for c in ctx.get("chunks") or []:
        if c and c.get("text"):
            chunks_by_id.setdefault(c["id"], c)
    # optional plain-RAG safety net: chunks that match the query directly
    if chunk_search:
        for c in db.vector_search_chunks(qvec, max(top_k, 6)):
            if c and c.get("text"):
                chunks_by_id.setdefault(c["id"], c)
    chunks = list(chunks_by_id.values())
    chunk_lines = [f"- [{c['id']}] {c['text']}" for c in chunks]

    context = pack(
        [
            ("Entities", ent_lines),
            ("Source text", chunk_lines),
            ("Relationships", rel_lines),
            ("Community reports", comm_lines),
        ],
        token_budget,
    )
    answer = llm.chat(P.LOCAL_ANSWER.format(context=context, question=question))
    return LocalResult(
        answer=answer.strip(),
        entities=seed_names,
        communities=[c["id"] for c in comm if c.get("id")],
        chunks=[c["id"] for c in chunks],
        context=context,
    )
