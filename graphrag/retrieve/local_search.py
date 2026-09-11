"""Local GraphRAG search: graph structure + source evidence synthesis.

Retrieval strategy:

1. Vector-search entities to identify KG seed entities.
2. Walk the graph from those entities to retrieve:
   - entities
   - neighbors
   - relationships
   - source chunks
   - community reports
3. Directly vector-search source chunks for query-specific evidence.
4. Merge graph chunks + direct chunks.
5. Give the answer model an explicit evidence hierarchy:
      Source chunks > Entities > Relationships > Community reports
6. Use the graph to establish context and relationships, while using the
   original source chunks to verify exact facts and numerical answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..embeddings import embed_one
from ..graphdb import Neo4jClient
from ..llm import LLM
from .. import prompts as P
from .context import pack
from .rerank import fiscal_years, rerank_seeds, tokens


@dataclass
class LocalResult:
    answer: str
    entities: list[str] = field(default_factory=list)
    communities: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    context: str = ""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_PROP_WHITELIST = (
    "metric_name", "value", "unit", "scope", "reporting_period", "target",
    "target_value", "target_year", "baseline", "baseline_year",
)


def _parse_properties(properties) -> dict:
    """Parse the stored ``properties`` JSON string into a dict; ``{}`` on failure."""
    if isinstance(properties, dict):
        return properties
    if not properties or not isinstance(properties, str):
        return {}
    try:
        parsed = json.loads(properties)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _format_entity(entity: dict) -> str:
    """Format an entity, whitelisting structured properties and flagging them low-trust."""

    name = entity.get("name", "?")
    entity_type = entity.get("type") or "?"
    domain = entity.get("domain")

    domain_text = f" / {domain}" if domain else ""

    description = entity.get("description") or ""

    props = _parse_properties(entity.get("properties"))
    kept = [
        f"{key}={props[key]}"
        for key in _PROP_WHITELIST
        if props.get(key) not in (None, "", [])
    ]

    properties_text = ""
    if kept:
        properties_text = (
            "\n  structured (LOW TRUST — may be mis-extracted; defer to SOURCE "
            "EVIDENCE on any conflict): " + ", ".join(kept)
        )

    return (
        f"- {name} "
        f"[{entity_type}{domain_text}]\n"
        f"  description: {description}"
        f"{properties_text}"
    )


def _format_relationship(relationship: dict) -> str:
    """Format a graph relationship."""

    source = relationship.get("source")
    target = relationship.get("target")
    rel_type = relationship.get("rel_type") or "RELATED"
    description = relationship.get("description") or ""

    return (
        f"- {source} "
        f"-[{rel_type}]-> "
        f"{target}"
        f"{': ' + description if description else ''}"
    )


def _format_chunk(chunk: dict) -> str:
    """Format source evidence with retrieval provenance."""

    chunk_id = chunk.get("id", "?")
    text = (chunk.get("text") or "").strip()
    via = chunk.get("retrieval_source") or "graph"

    return (
        f"- [{chunk_id}] (via {via})\n"
        f"  {text}"
    )


# ---------------------------------------------------------------------------
# Chunk handling
# ---------------------------------------------------------------------------

def _merge_chunks(
    graph_chunks: list[dict],
    direct_chunks: list[dict],
) -> list[dict]:
    """Merge graph and direct retrieval results without duplicates.

    If the same chunk is returned through both retrieval paths, keep one
    copy. Prefer the direct-search version when it contains a similarity
    score because it gives us query-level relevance information.
    """

    merged: dict[str, dict] = {}

    # Graph traversal provides structural relevance.
    for chunk in graph_chunks:
        if not chunk or not chunk.get("text"):
            continue

        chunk_id = chunk.get("id")
        if not chunk_id:
            continue

        chunk = dict(chunk)
        chunk["retrieval_source"] = "graph"
        merged[chunk_id] = chunk

    # Direct chunk search provides query-level relevance.
    for chunk in direct_chunks:
        if not chunk or not chunk.get("text"):
            continue

        chunk_id = chunk.get("id")
        if not chunk_id:
            continue

        existing = merged.get(chunk_id)

        if existing is None:
            chunk = dict(chunk)
            chunk["retrieval_source"] = "direct"
            merged[chunk_id] = chunk
            continue

        # Seen via the graph too: keep the direct score, tag as both.
        combined = dict(chunk)
        combined["retrieval_source"] = "graph+direct"
        if combined.get("score") is None:
            combined["score"] = existing.get("score")
        merged[chunk_id] = combined

    return list(merged.values())


_GRAPH_CHUNK_FLOOR = 0.35


def _lexical_chunk_score(
    chunk: dict,
    q_tokens: set[str],
    q_years: set[str],
    q_nums: set[str],
) -> float:
    text = chunk.get("text") or ""
    c_tok = set(tokens(text))

    s = 0.0
    if q_tokens:
        s += 0.6 * (len(q_tokens & c_tok) / len(q_tokens))
    if q_years and (fiscal_years(text) & q_years):
        s += 0.5
    if q_nums:
        s += 0.25 * len(q_nums & c_tok)
    return s


def _rank_chunks(chunks: list[dict], question: str) -> list[dict]:
    """Blend vector score, lexical overlap and a floor for unscored graph chunks."""

    q_tokens = {t for t in tokens(question) if len(t) > 2}
    q_years = fiscal_years(question)
    q_nums = {t for t in tokens(question) if t.isdigit() and len(t) >= 2}

    scored = []
    for index, chunk in enumerate(chunks):
        if not chunk or not chunk.get("text"):
            continue

        try:
            vec = float(chunk["score"]) if chunk.get("score") is not None else None
        except (TypeError, ValueError):
            vec = None

        lex = _lexical_chunk_score(chunk, q_tokens, q_years, q_nums)
        floor = _GRAPH_CHUNK_FLOOR if vec is None else 0.0
        blended = (vec or 0.0) + 0.8 * lex + floor

        scored.append((blended, -index, chunk))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [chunk for _, _, chunk in scored]


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def local_search(
    question: str,
    db: Neo4jClient,
    llm: LLM,
    top_k: int = 15,
    seed_keep: int = 8,
    token_budget: int = 11000,
    chunk_search: bool = True,
    max_graph_chunks: int = 12,
    max_direct_chunks: int = 12,
) -> LocalResult:

    # =======================================================================
    # 1. QUESTION -> ENTITY VECTOR SEARCH
    # =======================================================================

    qvec = embed_one(question)

    seeds = db.vector_search_entities(
        qvec,
        top_k,
    )

    if not seeds:
        return LocalResult(
            answer="No relevant entities were found in the knowledge graph."
        )

    ranked = rerank_seeds(seeds, question, keep=seed_keep)

    seed_names = [
        entity["name"]
        for entity in ranked
        if entity and entity.get("name")
    ]

    if not seed_names:
        return LocalResult(
            answer="No valid seed entities were found in the knowledge graph."
        )

    # =======================================================================
    # 2. ENTITY -> GRAPH WALK
    #
    # This is the actual GraphRAG portion.
    #
    # We retrieve:
    #   entities
    #   neighbors
    #   relationships
    #   chunks connected to those entities
    #   community reports
    # =======================================================================

    graph_context = db.entity_context(
        seed_names,
        max_chunks=max_graph_chunks,
    )

    # =======================================================================
    # 3. ENTITIES
    # =======================================================================

    entities = (
        (graph_context.get("entities") or [])
    )

    # Deduplicate entities by name.
    entity_map: dict[str, dict] = {}

    for entity in entities:

        if not entity or not entity.get("name"):
            continue

        name = entity["name"]

        if name not in entity_map:
            entity_map[name] = entity

    ent_lines = [
        _format_entity(entity)
        for entity in entity_map.values()
    ]

    # =======================================================================
    # 4. RELATIONSHIPS
    #
    # Relationships are NOT decorative context.
    #
    # They tell the answer model how entities are connected.
    # =======================================================================

    relationships = []

    for relationship in graph_context.get("relationships") or []:

        if not relationship:
            continue

        if (
            relationship.get("source")
            and relationship.get("target")
        ):
            relationships.append(relationship)

    rel_lines = [
        _format_relationship(relationship)
        for relationship in relationships
    ]

    # =======================================================================
    # 5. COMMUNITY REPORTS
    #
    # Useful for broader context, but lower authority than source evidence.
    # =======================================================================

    communities = [
        community
        for community in graph_context.get("communities") or []
        if community and community.get("content")
    ]

    communities.sort(
        key=lambda community: (
            community.get("rating") or 0
        ),
        reverse=True,
    )

    comm_lines = [
        (
            f"- {community.get('title', '?')}: "
            f"{community.get('content', '')}"
        )
        for community in communities
    ]

    # =======================================================================
    # 6. SOURCE CHUNKS FROM GRAPH
    #
    # These chunks are reached through the graph.
    #
    # Example:
    #
    # Question
    #   -> Total Employees entity
    #   -> HAS_DISCLOSURE
    #   -> Employee Disclosure
    #   -> SOURCE_CHUNK
    # =======================================================================

    graph_chunks = [
        chunk
        for chunk in graph_context.get("chunks") or []
        if chunk and chunk.get("text")
    ]

    # =======================================================================
    # 7. DIRECT QUESTION -> CHUNK SEARCH
    #
    # This is the second retrieval path.
    #
    # It catches facts that may not have been represented correctly as
    # entities/relationships, especially:
    #
    #   - BRSR tables
    #   - exact numbers
    #   - percentages
    #   - FY comparisons
    #   - complaints
    #   - Scope 1/2/3 values
    # =======================================================================

    direct_chunks = []

    if chunk_search:

        direct_chunks = db.vector_search_chunks(
            qvec,
            max(top_k, max_direct_chunks),
        )

    # =======================================================================
    # 8. MERGE GRAPH + DIRECT CHUNKS
    # =======================================================================

    chunks = _merge_chunks(
        graph_chunks=graph_chunks,
        direct_chunks=direct_chunks,
    )

    # =======================================================================
    # 9. RANK SOURCE EVIDENCE
    # =======================================================================

    chunks = _rank_chunks(chunks, question)

    # Limit the evidence set before token packing.
    max_chunks = max_graph_chunks + max_direct_chunks

    chunks = chunks[:max_chunks]

    chunk_lines = [
        _format_chunk(chunk)
        for chunk in chunks
    ]

    # =======================================================================
    # 10. BUILD STRUCTURED CONTEXT
    #
    # IMPORTANT:
    #
    # The sections are intentionally separated.
    #
    # Source Evidence = original document evidence
    # Entities        = KG concepts
    # Relationships   = graph structure
    # Communities     = derived summaries
    # =======================================================================

    context = pack(
        [
            (
                "SOURCE EVIDENCE — PRIMARY",
                chunk_lines,
            ),
            (
                "ENTITIES — KNOWLEDGE GRAPH",
                ent_lines,
            ),
            (
                "RELATIONSHIPS — KNOWLEDGE GRAPH",
                rel_lines,
            ),
            (
                "COMMUNITY REPORTS — DERIVED SUMMARIES",
                comm_lines,
            ),
        ],
        token_budget,
    )

    # =======================================================================
    # 11. ANSWER SYNTHESIS
    # =======================================================================

    answer = llm.chat(
        P.LOCAL_ANSWER.format(
            context=context,
            question=question,
        )
    )

    # =======================================================================
    # 12. RETURN
    # =======================================================================

    return LocalResult(
        answer=answer.strip(),

        entities=seed_names,

        communities=[
            community["id"]
            for community in communities
            if community.get("id")
        ],

        chunks=[
            chunk["id"]
            for chunk in chunks
            if chunk.get("id")
        ],

        context=context,
    )
