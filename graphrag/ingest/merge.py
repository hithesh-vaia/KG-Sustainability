"""Merge per-chunk extractions into a deduplicated, ontology-typed entity /
relationship set.

Entities are keyed by name (period-scoped types — measurements, emissions … —
are already given period-qualified names in `extract.py`, so different years stay
separate). Description consolidation runs concurrently and is disk-cached.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..config import CONFIG
from ..llm import LLM
from ..log import get_logger
from .. import ontology as O
from .. import prompts as P
from .extract import ChunkExtraction

log = get_logger()

MAX_DESC_BEFORE_SUMMARY = 1


@dataclass
class Entity:
    name: str
    entity_type: str
    community: str
    description: str
    properties: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    confidence: float = 0.0
    source_chunks: list[str] = field(default_factory=list)
    degree: int = 0


@dataclass
class Relationship:
    source: str
    target: str
    rel_type: str
    description: str
    properties: dict = field(default_factory=dict)
    provenance: list = field(default_factory=list)
    confidence: float = 0.0
    strength: float = 1.0
    source_chunks: list[str] = field(default_factory=list)


def _summary_cache_path(kind: str, name: str, descriptions: list[str]):
    key = hashlib.sha256(("|".join([kind, name, *descriptions])).encode()).hexdigest()[:16]
    return CONFIG.cache_dir / f"summary_{key}.json"


def _summarize(llm: LLM, kind: str, name: str, descriptions: list[str], use_cache: bool) -> str:
    cache_file = _summary_cache_path(kind, name, descriptions)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())["summary"]
    joined = "\n".join(f"- {d}" for d in descriptions if d)
    summary = llm.chat(P.DESCRIPTION_SUMMARIZATION.format(kind=kind, name=name, descriptions=joined)).strip()
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"summary": summary}))
    return summary


def _run_summaries(llm: LLM, tasks: list[tuple[str, str, list[str]]], use_cache: bool) -> dict:
    out: dict[str, str] = {}
    if not tasks:
        return out
    log.info("  consolidating %d descriptions (concurrency=%d)", len(tasks), CONFIG.llm_concurrency)
    with ThreadPoolExecutor(max_workers=CONFIG.llm_concurrency) as pool:
        futs = {pool.submit(_summarize, llm, k, n, d, use_cache): n for k, n, d in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            out[futs[fut]] = fut.result()
            if i == len(tasks) or i % 25 == 0:
                log.info("  consolidated %d/%d", i, len(tasks))
    return out


def merge_extractions(
    extractions: list[ChunkExtraction],
    llm: LLM,
    summarize: bool = True,
    use_cache: bool = True,
) -> tuple[list[Entity], list[Relationship]]:
    ents: dict[str, Entity] = {}
    descs: dict[str, list[str]] = {}
    type_votes: dict[str, Counter] = {}
    comm_votes: dict[str, Counter] = {}

    for ex in extractions:
        for e in ex.entities:
            if not e.name:
                continue
            type_votes.setdefault(e.name, Counter())[e.entity_type] += 1
            comm_votes.setdefault(e.name, Counter())[e.community] += 1
            acc = ents.get(e.name)
            if acc is None:
                acc = Entity(name=e.name, entity_type=e.entity_type, community=e.community,
                             description="", properties=dict(e.properties))
                ents[e.name] = acc
                descs[e.name] = []
            acc.properties.update({k: v for k, v in e.properties.items()
                                   if v not in (None, "", [], {})})
            if e.provenance:
                acc.provenance.append(e.provenance)
            acc.confidence = max(acc.confidence, e.confidence)
            for c in [e.source_chunk]:
                if c and c not in acc.source_chunks:
                    acc.source_chunks.append(c)
            if e.description:
                descs[e.name].append(e.description)

    rels: dict[tuple, Relationship] = {}
    for ex in extractions:
        for r in ex.relationships:
            k = (r.source, r.target, r.rel_type)
            acc = rels.get(k)
            if acc is None:
                acc = Relationship(source=r.source, target=r.target, rel_type=r.rel_type,
                                   description=r.description, properties=dict(r.properties))
                rels[k] = acc
            if r.provenance:
                acc.provenance.append(r.provenance)
            acc.confidence = max(acc.confidence, r.confidence)
            if r.source_chunk and r.source_chunk not in acc.source_chunks:
                acc.source_chunks.append(r.source_chunk)

    # entities referenced only by a relationship -> bare node
    for (s, t, _rt) in rels:
        for name in (s, t):
            if name and name not in ents:
                etype = (type_votes[name].most_common(1)[0][0]
                         if type_votes.get(name) else O.DEFAULT_ENTITY_TYPE)
                comm = (comm_votes[name].most_common(1)[0][0]
                        if comm_votes.get(name) else O.DEFAULT_COMMUNITY)
                ents[name] = Entity(name=name, entity_type=etype, community=comm, description="")
                descs[name] = []

    # settle type/community by majority vote across mentions
    for name, ent in ents.items():
        if type_votes.get(name):
            ent.entity_type = type_votes[name].most_common(1)[0][0]
        if comm_votes.get(name):
            ent.community = comm_votes[name].most_common(1)[0][0]

    tasks: list[tuple[str, str, list[str]]] = []
    if summarize:
        for name, dl in descs.items():
            uniq = list(dict.fromkeys(d for d in dl if d))
            if len(uniq) > MAX_DESC_BEFORE_SUMMARY:
                tasks.append(("entity", name, uniq))
    summaries = _run_summaries(llm, tasks, use_cache)

    for name, ent in ents.items():
        uniq = list(dict.fromkeys(d for d in descs.get(name, []) if d))
        ent.description = summaries.get(name) or (uniq[0] if uniq else ent.description)

    entities = list(ents.values())
    relationships = list(rels.values())

    degree: Counter = Counter()
    names = set(ents)
    for r in relationships:
        r.strength = max(r.confidence * 10, 1.0)
        if r.source in names and r.target in names:
            degree[r.source] += 1
            degree[r.target] += 1
    for e in entities:
        e.degree = degree[e.name]

    return entities, relationships
