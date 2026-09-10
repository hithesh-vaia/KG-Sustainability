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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_entity(entity: dict) -> str:
    """Format an entity while preserving useful KG properties."""

    name = entity.get("name", "?")
    entity_type = entity.get("type") or "?"
    domain = entity.get("domain")

    domain_text = f" / {domain}" if domain else ""

    description = entity.get("description") or ""

    properties = entity.get("properties")

    properties_text = ""
    if properties and properties != "{}":
        properties_text = f"\n  properties: {properties}"

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
    """Format source evidence with provenance."""

    chunk_id = chunk.get("id", "?")
    text = (chunk.get("text") or "").strip()

    metadata = []

    if chunk.get("document"):
        metadata.append(f"document={chunk['document']}")

    if chunk.get("page") is not None:
        metadata.append(f"page={chunk['page']}")

    if chunk.get("section"):
        metadata.append(f"section={chunk['section']}")

    if chunk.get("fiscal_year"):
        metadata.append(f"fiscal_year={chunk['fiscal_year']}")

    provenance = ""

    if metadata:
        provenance = " [" + ", ".join(metadata) + "]"

    return (
        f"- [{chunk_id}]{provenance}\n"
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
            merged[chunk_id] = chunk
            continue

        # Prefer the version containing a similarity score.
        if chunk.get("score") is not None:
            merged[chunk_id] = chunk

    return list(merged.values())


def _rank_chunks(chunks: list[dict]) -> list[dict]:
    """Rank chunks using retrieval score when available.

    Direct vector-search results should ideally contain a ``score``.
    Graph-only chunks without scores retain their original order.

    We deliberately do not invent a similarity score here.
    """

    scored = []

    for index, chunk in enumerate(chunks):

        if not chunk or not chunk.get("text"):
            continue

        score = chunk.get("score")

        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        scored.append(
            (
                score is not None,
                score if score is not None else 0.0,
                -index,
                chunk,
            )
        )

    # Chunks with actual vector scores first.
    # Among scored chunks, highest similarity first.
    # Unscored graph chunks preserve retrieval order.
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        ),
        reverse=True,
    )

    return [
        chunk
        for _, _, _, chunk in scored
    ]


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------

def local_search(
    question: str,
    db: Neo4jClient,
    llm: LLM,
    top_k: int = 8,
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

    seed_names = [
        entity["name"]
        for entity in seeds
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
        + (graph_context.get("neighbors") or [])
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

    chunks = _rank_chunks(chunks)

    # Limit the evidence set before token packing.
    max_chunks = max(
        max_graph_chunks,
        max_direct_chunks,
    )

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
