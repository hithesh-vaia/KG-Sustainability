"""End-to-end ingestion: documents -> knowledge graph in Neo4j (+ parquet artifacts)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ..config import CONFIG
from ..chunking import load_documents, chunk_documents
from ..embeddings import embed
from ..llm import LLM
from ..graphdb import Neo4jClient
from .extract import extract_all
from .merge import merge_extractions
from .build import build_graph
from .communities import detect_communities, generate_reports
from ..log import setup_logging, get_logger

log = get_logger()


# MarkItDown emits "<!-- Page number: N -->"; older ingests used "===== PAGE N ====="
_PAGE_MARKER = re.compile(
    r"\n?(?:=====\s*PAGE\s+\d+\s*=====|<!--\s*Page number:\s*\d+\s*-->)\n?",
    re.IGNORECASE,
)


def _strip_markers(text: str) -> str:
    return _PAGE_MARKER.sub("\n", text).strip()


def _write_parquet(name: str, records: list[dict]) -> None:
    path = CONFIG.output_dir / f"{name}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path.name, len(records))


def run_ingest(input_dir: Path | None = None, reset: bool = False, use_cache: bool = True) -> dict:
    CONFIG.ensure_dirs()
    setup_logging("ingest")
    llm = LLM()
    db = Neo4jClient()
    try:
        if reset:
            log.info("resetting Neo4j graph")
            db.reset()
        db.init_schema()

        log.info("loading + chunking documents from %s", input_dir or CONFIG.input_dir)
        docs = load_documents(input_dir)
        if not docs:
            raise RuntimeError(f"No .txt/.md/.pdf/.xlsx files found in {input_dir or CONFIG.input_dir}")
        chunks = chunk_documents(docs)
        log.info("%d docs -> %d chunks", len(docs), len(chunks))

        log.info("extracting entities + relationships (%d chunks, concurrency=%d)",
                 len(chunks), CONFIG.llm_concurrency)
        extractions = extract_all(chunks, llm, use_cache=use_cache)

        log.info("merging + summarizing")
        entities, relationships = merge_extractions(extractions, llm, use_cache=use_cache)
        log.info("%d entities, %d relationships", len(entities), len(relationships))

        log.info("detecting communities (hierarchical Leiden)")
        graph = build_graph(entities, relationships)
        communities = detect_communities(graph)
        log.info("%d communities", len(communities))

        log.info("generating %d community reports", len(communities))
        communities = generate_reports(communities, entities, relationships, llm, use_cache=use_cache)

        log.info("embedding locally (fastembed: %s)", CONFIG.embed_model)
        chunk_vecs = embed([_strip_markers(c.text) for c in chunks])
        entity_vecs = embed(
            [f"{e.name} [{e.entity_type} / {e.community}]: {e.description}" for e in entities]
        )
        comm_vecs = embed([c.full_content or c.title for c in communities])

        log.info("writing to Neo4j")
        db.upsert_chunks(
            [
                {**c.dict(), "embedding": chunk_vecs[i].tolist()}
                for i, c in enumerate(chunks)
            ]
        )
        db.upsert_entities(
            [
                {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "community": e.community,
                    "description": e.description,
                    "properties_json": json.dumps(e.properties, default=str),
                    "provenance_json": json.dumps(e.provenance, default=str),
                    "confidence": e.confidence,
                    "degree": e.degree,
                    "source_chunks": e.source_chunks,
                    "embedding": entity_vecs[i].tolist(),
                }
                for i, e in enumerate(entities)
            ]
        )
        db.upsert_relationships(
            [
                {
                    "key": f"{r.source}|{r.rel_type}|{r.target}",
                    "source": r.source,
                    "target": r.target,
                    "rel_type": r.rel_type,
                    "description": r.description,
                    "strength": r.strength,
                    "confidence": r.confidence,
                    "provenance_json": json.dumps(r.provenance, default=str),
                    "derived": bool(r.properties.get("derived", False)),
                }
                for r in relationships
            ]
        )
        db.upsert_communities(
            [
                {
                    "id": c.id,
                    "level": c.level,
                    "parent": c.parent,
                    "title": c.title,
                    "summary": c.summary,
                    "rating": c.rating,
                    "full_content": c.full_content,
                    "members": c.members,
                    "embedding": comm_vecs[i].tolist(),
                }
                for i, c in enumerate(communities)
            ]
        )

        log.info("writing parquet artifacts")
        _write_parquet("chunks", [c.dict() for c in chunks])
        _write_parquet(
            "entities",
            [
                {"name": e.name, "entity_type": e.entity_type, "community": e.community,
                 "description": e.description,
                 "properties": json.dumps(e.properties, default=str),
                 "confidence": e.confidence, "degree": e.degree,
                 "source_chunks": e.source_chunks}
                for e in entities
            ],
        )
        _write_parquet(
            "relationships",
            [
                {"source": r.source, "target": r.target, "rel_type": r.rel_type,
                 "description": r.description, "strength": r.strength,
                 "confidence": r.confidence, "source_chunks": r.source_chunks}
                for r in relationships
            ],
        )
        _write_parquet(
            "communities",
            [
                {"id": c.id, "level": c.level, "parent": c.parent, "title": c.title,
                 "summary": c.summary, "rating": c.rating, "full_content": c.full_content,
                 "members": c.members}
                for c in communities
            ],
        )

        stats = db.stats()
        stats["llm_usage"] = llm.usage.summary()
        log.info("ingestion complete: %s", stats)
        return stats
    except Exception:
        log.exception("ingestion failed")
        raise
    finally:
        db.close()
