"""Per-chunk extraction into the sustainability KG ontology (JSON output),
with a gleaning pass and an on-disk cache (so re-runs don't burn credits)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CONFIG
from ..chunking import Chunk
from ..llm import LLM
from ..log import get_logger
from .. import ontology as O
from .. import prompts as P

log = get_logger()


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str            # ontology entity type, e.g. "Scope2Emission"
    community: str              # ontology domain, e.g. "Environmental"
    description: str            # rendered from properties + source_text
    properties: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    confidence: float = 0.0
    source_chunk: str = ""


@dataclass
class ExtractedRelationship:
    source: str                 # source entity name
    target: str                 # target entity name
    rel_type: str               # ontology relationship, e.g. "USES_FRAMEWORK"
    description: str
    properties: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    confidence: float = 0.0
    source_chunk: str = ""


@dataclass
class ChunkExtraction:
    chunk_id: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().strip('"').strip()).upper()


def _match(value: str, allowed: frozenset[str], default: str) -> str:
    """Case/format-insensitive snap to an allowed ontology term."""
    if not value:
        return default
    v = str(value).strip()
    if v in allowed:
        return v
    low = {a.lower(): a for a in allowed}
    if v.lower() in low:
        return low[v.lower()]
    squished = re.sub(r"[^a-z0-9]", "", v.lower())
    for a in allowed:
        if re.sub(r"[^a-z0-9]", "", a.lower()) == squished:
            return a
    return default


def _period(props: dict) -> str:
    for k in ("reporting_period", "period", "financial_year", "fy", "year"):
        v = props.get(k)
        if v:
            return str(v).strip().upper()
    return ""


def _render_description(props: dict, prov: dict) -> str:
    parts = []
    for k in ("metric_name", "value", "unit", "scope", "reporting_period",
              "target_value", "target_year", "baseline", "methodology"):
        if props.get(k) not in (None, "", []):
            parts.append(f"{k}={props[k]}")
    for k, v in props.items():
        if k not in ("metric_name", "value", "unit", "scope", "reporting_period",
                     "target_value", "target_year", "baseline", "methodology") \
                and v not in (None, "", [], {}):
            parts.append(f"{k}={v}")
    rendered = "; ".join(str(p) for p in parts)
    src = (prov or {}).get("source_text") or ""
    if rendered and src:
        return f"{rendered}  ⟨{src[:200]}⟩"
    return rendered or src[:280]


def parse_extraction(data: dict, chunk_id: str) -> ChunkExtraction:
    """Parse one LLM JSON payload (entities + relationships) into structured records."""
    out = ChunkExtraction(chunk_id=chunk_id)
    if not isinstance(data, dict):
        return out

    id_to_name: dict[str, str] = {}
    for e in data.get("entities", []) or []:
        if not isinstance(e, dict):
            continue
        name = _norm_name(e.get("name", ""))
        if not name:
            continue
        props = e.get("properties") or {}
        prov = e.get("provenance") or {}
        try:
            conf = float(prov.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        etype = _match(e.get("entity_type", ""), O.ENTITY_TYPES, O.DEFAULT_ENTITY_TYPE)
        comm = _match(e.get("community", ""), O.COMMUNITIES, O.DEFAULT_COMMUNITY)
        # keep period-scoped instances (measurements, emissions, ...) uniquely named
        if etype in O.PERIOD_SCOPED_TYPES and isinstance(props, dict):
            per = _period(props)
            if per and per not in name:
                name = f"{name} ({per})"
        if e.get("id"):
            id_to_name[str(e["id"])] = name
        out.entities.append(ExtractedEntity(
            name=name, entity_type=etype, community=comm,
            description=_render_description(props if isinstance(props, dict) else {}, prov),
            properties=props if isinstance(props, dict) else {},
            provenance=prov if isinstance(prov, dict) else {},
            confidence=conf, source_chunk=chunk_id,
        ))
        id_to_name.setdefault(name, name)

    for r in data.get("relationships", []) or []:
        if not isinstance(r, dict):
            continue
        src = id_to_name.get(str(r.get("source_id", "")), _norm_name(r.get("source_id", "")))
        tgt = id_to_name.get(str(r.get("target_id", "")), _norm_name(r.get("target_id", "")))
        if not src or not tgt or src == tgt:
            continue
        prov = r.get("provenance") or {}
        try:
            conf = float(prov.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        rtype = _match(r.get("relationship", ""), O.RELATIONSHIPS, O.DEFAULT_RELATIONSHIP)
        out.relationships.append(ExtractedRelationship(
            source=src, target=tgt, rel_type=rtype,
            description=(prov.get("source_text") or "")[:280],
            properties=r.get("properties") or {},
            provenance=prov if isinstance(prov, dict) else {},
            confidence=conf, source_chunk=chunk_id,
        ))
    return out


def _merge_payloads(payloads: list[dict]) -> dict:
    merged = {"entities": [], "relationships": []}
    for d in payloads:
        if isinstance(d, dict):
            merged["entities"].extend(d.get("entities", []) or [])
            merged["relationships"].extend(d.get("relationships", []) or [])
    return merged


def _cache_path(chunk_id: str) -> Path:
    # v2 = sustainability-ontology JSON extraction (invalidates old delimiter cache)
    return CONFIG.cache_dir / f"extract_v2_{chunk_id}.json"


def extract_chunk(chunk: Chunk, llm: LLM, use_cache: bool = True) -> ChunkExtraction:
    cache_file = _cache_path(chunk.id)
    if use_cache and cache_file.exists():
        payload = json.loads(cache_file.read_text())
        return parse_extraction(payload, chunk.id)

    user = P.EXTRACTION_USER.format(doc_id=chunk.doc_id, input_text=chunk.text)
    payloads: list[dict] = []
    try:
        payloads.append(llm.chat_json(user, system=P.EXTRACTION_SYSTEM))
    except Exception as exc:  # pragma: no cover
        log.warning("  extraction failed for chunk %s: %s", chunk.id, exc)

    for _ in range(CONFIG.max_gleanings):
        if not payloads:
            break
        try:
            extra = llm.chat_json(
                user + "\n\nAlready extracted:\n"
                + json.dumps(_merge_payloads(payloads))[:6000]
                + "\n\n" + P.EXTRACTION_GLEANING,
                system=P.EXTRACTION_SYSTEM,
            )
        except Exception:
            break
        if not (extra.get("entities") or extra.get("relationships")):
            break
        payloads.append(extra)

    merged = _merge_payloads(payloads)
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(merged))
    return parse_extraction(merged, chunk.id)


def extract_all(chunks: list[Chunk], llm: LLM, use_cache: bool = True) -> list[ChunkExtraction]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[ChunkExtraction] = []
    with ThreadPoolExecutor(max_workers=CONFIG.llm_concurrency) as pool:
        futures = [pool.submit(extract_chunk, c, llm, use_cache) for c in chunks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i == len(chunks) or i % 5 == 0:
                log.info("  extracted %d/%d chunks", i, len(chunks))
    return results
