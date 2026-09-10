"""Merge per-chunk extractions into a deduplicated, ontology-typed entity /
relationship set.

Entities are keyed by name (period-scoped types — measurements, emissions … —
are already given period-qualified names in `extract.py`, so different years stay
separate). Description consolidation runs concurrently and is disk-cached.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..config import CONFIG
from ..llm import LLM
from ..log import get_logger
from .. import ontology as O
from .. import prompts as P
from .extract import ChunkExtraction, split_period_from_name

log = get_logger()

MAX_DESC_BEFORE_SUMMARY = 1

# Legal-form suffixes that make one organisation look like several entities:
# "IDBI BANK" / "IDBI BANK LTD." / "IDBI BANK LIMITED" were three nodes sharing
# 43 relationships between them, which fragments the graph's biggest hub.
_LEGAL_SUFFIX = re.compile(
    r"\s*\b(?:LTD|LIMITED|PVT|PRIVATE|INC|INCORPORATED|LLP|LLC|CORP|CORPORATION|"
    r"COMPANY|PLC|GMBH|AG|SA|NV)\b\.?",
    re.IGNORECASE,
)


def _alias_key(name: str) -> str:
    return re.sub(r"\s+", " ", _LEGAL_SUFFIX.sub("", name).replace(".", " ")).strip().upper()


def build_alias_map(names: list[str]) -> dict[str, str]:
    """Map each legal-form variant of an organisation onto one canonical name.

    Only applied where the variants collapse to the same key; the canonical form
    is the most frequently extracted surface form (ties broken by shortest name).
    """
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(_alias_key(n), []).append(n)
    aliases: dict[str, str] = {}
    for key, variants in groups.items():
        uniq = list(dict.fromkeys(variants))
        if len(uniq) < 2 or not key:
            continue
        counts = Counter(variants)
        canonical = min(uniq, key=lambda v: (-counts[v], len(v), v))
        for v in uniq:
            if v != canonical:
                aliases[v] = canonical
    return aliases


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
    try:
        summary = llm.chat(
            P.DESCRIPTION_SUMMARIZATION.format(kind=kind, name=name, descriptions=joined)
        ).strip()
    except Exception as exc:  # keep the run alive; fall back to the longest raw description
        log.warning("  description summary failed for %s (%s); using raw", name, exc)
        return max((d for d in descriptions if d), key=len, default="")
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


_YEAR = re.compile(r"(\d{4})")


def _period_sort_key(name: str) -> int:
    base, period = split_period_from_name(name)
    m = _YEAR.search(period)
    return int(m.group(1)) if m else -1


# Entity types that represent a figure the reporting organisation is responsible
# for -> get a deterministic owner edge if the LLM did not already draw one.
_OWNED_TYPES: dict[str, str] = {
    "Measurement": "HAS_MEASUREMENT", "Metric": "HAS_MEASUREMENT",
    "Value": "HAS_MEASUREMENT", "ActualValue": "HAS_MEASUREMENT",
    "Intensity": "HAS_MEASUREMENT", "Baseline": "HAS_MEASUREMENT",
    "TargetValue": "HAS_TARGET_VALUE",
    "GHGEmission": "EMITS", "Scope1Emission": "EMITS",
    "Scope2Emission": "EMITS", "Scope3Emission": "EMITS",
    "EnergyConsumption": "USES_ENERGY", "RenewableEnergy": "USES_ENERGY",
    "WaterWithdrawal": "WITHDRAWS_WATER", "WaterConsumption": "USES_WATER",
    "WaterDischarge": "DISCHARGES_WATER",
    "WasteGenerated": "GENERATES_WASTE", "WasteRecycled": "RECYCLES",
    "WasteDisposed": "GENERATES_WASTE",
    "Employee": "EMPLOYS", "Worker": "EMPLOYS", "Workforce": "EMPLOYS",
    "Training": "PROVIDES", "EmployeeGrievance": "RECEIVES_GRIEVANCE_FROM",
    "Revenue": "HAS_REVENUE", "Expense": "HAS_EXPENSE",
}


def link_orphan_figures(
    entities: list[Entity], relationships: list[Relationship], org: str
) -> tuple[list[Relationship], Entity | None]:
    """Connect figure entities that have no owner to the reporting organisation.

    Disclosure-table chunks routinely produce a measurement entity with no
    relationship at all, because the company is never named in that chunk. The
    figure demonstrably belongs to this company's report, so a derived owner edge
    is a safe structural addition (marked derived=True). ~half the graph is
    isolated without this.
    """
    org = org.strip().upper()
    linked = {r.source for r in relationships} | {r.target for r in relationships}
    out: list[Relationship] = []
    for e in entities:
        if e.name in linked or e.entity_type not in _OWNED_TYPES:
            continue
        out.append(Relationship(
            source=org, target=e.name, rel_type=_OWNED_TYPES[e.entity_type],
            description=f"{org} reports this figure ({e.name}).",
            properties={"derived": True}, confidence=1.0, strength=1.0,
        ))
    org_entity = None
    if out and not any(e.name == org for e in entities):
        org_entity = Entity(name=org, entity_type="Company", community="Organization",
                            description=f"{org}, the reporting organisation.")
    return out, org_entity


def link_period_siblings(entities: list[Entity]) -> list[Relationship]:
    """Chain the same metric across reporting periods.

    Half the graph is otherwise isolated: a table chunk yields
    'TOTAL SCOPE 1 EMISSIONS (FY 2024-25)' with no relationship, because the
    owning organisation is never named in that chunk. Its FY 2023-24 twin is
    genuinely the same metric, so we can connect them without inventing a fact.

    Consecutive periods only, so a metric with N years gets N-1 edges rather
    than N^2. Edges are marked derived=True to keep provenance honest.
    """
    groups: dict[str, list[Entity]] = {}
    for e in entities:
        base, period = split_period_from_name(e.name)
        if period and base:
            groups.setdefault((base, e.entity_type), []).append(e)

    out: list[Relationship] = []
    for (base, _etype), members in groups.items():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda m: _period_sort_key(m.name))
        for older, newer in zip(ordered, ordered[1:]):
            if older.name == newer.name:
                continue
            out.append(Relationship(
                source=newer.name, target=older.name,
                rel_type=O.SAME_METRIC_PRIOR_PERIOD,
                description=f"Same metric ({base}) in the preceding reporting period.",
                properties={"derived": True},
                confidence=1.0, strength=1.0,
            ))
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

    # collapse legal-form variants BEFORE folding, so relationship endpoints are
    # rewritten to the canonical name too and the hub keeps all of its edges
    all_names = [e.name for ex in extractions for e in ex.entities if e.name]
    aliases = build_alias_map(all_names)
    if aliases:
        log.info("  aliasing %d legal-form variants (e.g. %s)", len(aliases),
                 next(f"{k!r}->{v!r}" for k, v in aliases.items()))
        for ex in extractions:
            for e in ex.entities:
                e.name = aliases.get(e.name, e.name)
            for r in ex.relationships:
                r.source = aliases.get(r.source, r.source)
                r.target = aliases.get(r.target, r.target)
        for ex in extractions:
            ex.relationships = [r for r in ex.relationships if r.source != r.target]

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

    # deterministic owner edges for figures the LLM left unconnected
    org = CONFIG.reporting_entity or "IDBI Bank"
    owner_edges, org_entity = link_orphan_figures(entities, relationships, org)
    if owner_edges:
        log.info("  linked %d orphan figures to %s (derived owner edges)",
                 len(owner_edges), org.upper())
        relationships.extend(owner_edges)
        if org_entity is not None:
            entities.append(org_entity)
            ents[org_entity.name] = org_entity

    derived = link_period_siblings(entities) if CONFIG.link_period_siblings else []
    if derived:
        log.info("  linked %d period-sibling pairs (derived edges)", len(derived))
        relationships.extend(derived)

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
