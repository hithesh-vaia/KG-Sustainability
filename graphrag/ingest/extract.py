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
    n = str(name).strip().strip('"').strip()
    n = re.sub(r"\s+-\s+", " ", n)          # "EMPLOYEES - FEMALE" == "EMPLOYEES FEMALE"
    return re.sub(r"\s+", " ", n).upper()


# "FY 2024-25", "FY2024-2025", "(2024-25)", "FY 2024-25*" -> one canonical form
_PERIOD_RE = re.compile(
    r"\(?\s*(?:FY|F\.Y\.|FISCAL\s*YEAR)?\s*(\d{4})\s*[-–—/]\s*(\d{2,4})\s*[*^@#]?\s*\)?",
    re.IGNORECASE,
)


def canon_period(raw: str) -> str:
    """Normalise any reporting-period spelling to 'FY YYYY-YY'.

    The LLM writes the period into the entity NAME sometimes and into
    `properties.reporting_period` other times, in several spellings. Without a
    single canonical form the same fact becomes two nodes -- e.g.
    'TOTAL SCOPE 1 EMISSIONS FY 2024-25' and
    'TOTAL SCOPE 1 EMISSIONS (FY 2024-25)' -- splitting degree and giving the
    answer model two competing siblings.
    """
    if not raw:
        return ""
    m = _PERIOD_RE.search(str(raw))
    if not m:
        return re.sub(r"\s+", " ", str(raw).strip().upper())
    start, end = m.group(1), m.group(2)
    return f"FY {start}-{end[-2:]}"


def split_period_from_name(name: str) -> tuple[str, str]:
    """Return (name without any period token, canonical period found in it)."""
    m = _PERIOD_RE.search(name)
    if not m:
        return name, ""
    base = _PERIOD_RE.sub(" ", name)
    base = re.sub(r"\s*\(\s*\)\s*", " ", base)          # leftover empty parens
    base = re.sub(r"\s+", " ", base).strip(" -–—,")
    return base, canon_period(m.group(0))


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


# "BRSR-FY-2024-25" / "BRSR FY2025-26" / "report_2023-24" -> "FY 2024-25"
_DOC_FY = re.compile(r"(?:FY[\s_-]*)?(\d{4})[\s_-]+(\d{2,4})", re.IGNORECASE)


def doc_period(doc_id: str) -> str:
    """Best-effort reporting period for a document, read from its filename.

    Given to the LLM as a known fact so it can stamp `reporting_period` on
    figures whose chunk doesn't restate the year -- propagation, not inference.
    """
    m = _DOC_FY.search(str(doc_id))
    if not m:
        return "not stated in the filename; use the period given in the text, else null"
    start, end = m.group(1), m.group(2)
    if len(end) == 4:
        end = end[2:]
    return f"FY {start}-{end}"


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
        # the model types most numeric entities as the generic "Measurement";
        # recover a specific type from the name where one is obvious
        etype = O.refine_type(name, etype)
        # the ESG domain is then definitional for a specific type -- a Scope2Emission
        # is Environmental whatever the model claims
        comm = O.domain_for(
            etype, _match(e.get("community", ""), O.COMMUNITIES, O.DEFAULT_COMMUNITY)
        )
        # keep period-scoped instances (measurements, emissions, ...) uniquely named,
        # with ONE canonical spelling of the period wherever the model put it
        if etype in O.PERIOD_SCOPED_TYPES:
            base, name_period = split_period_from_name(name)
            per = canon_period(_period(props) if isinstance(props, dict) else "") or name_period
            name = f"{base} ({per})" if per else base
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
    # v2 = sustainability-ontology JSON extraction (invalidated the delimiter cache)
    # v3 = reporting-period propagation in EXTRACTION_USER (see prompts.py)
    # v4 = explicit column-label period beats the document period (prior-year cols)
    # (v5 was a prompt-side owner-edge mandate -- reverted: it exploded per-chunk
    #  output; the owner edge is now added deterministically in merge.py instead)
    # Bumping the version invalidates the cache without --no-cache, and leaves the
    # previous generation on disk so a prompt change can be rolled back for free.
    return CONFIG.cache_dir / f"extract_v4_{chunk_id}.json"


def _raw_counts(payload: dict) -> tuple[int, int]:
    return len(payload.get("entities") or []), len(payload.get("relationships") or [])


def extract_chunk(chunk: Chunk, llm: LLM, use_cache: bool = True) -> ChunkExtraction:
    tag = f"chunk {chunk.id} [{chunk.doc_id} #{chunk.chunk_index}]"
    cache_file = _cache_path(chunk.id)

    if use_cache and cache_file.exists():
        payload = json.loads(cache_file.read_text())
        res = parse_extraction(payload, chunk.id)
        re_raw, rr_raw = _raw_counts(payload)
        log.info("  %s  cache-hit  raw %de/%dr -> kept %de/%dr",
                 tag, re_raw, rr_raw, len(res.entities), len(res.relationships))
        return res

    user = P.EXTRACTION_USER.format(
        doc_id=chunk.doc_id,
        doc_period=doc_period(chunk.doc_id),
        input_text=chunk.text,
    )
    payloads: list[dict] = []
    primary_ok = False
    try:
        first = llm.chat_json(user, system=P.EXTRACTION_SYSTEM)
        payloads.append(first)
        primary_ok = True
        log.info("  %s  extract   -> %de/%dr", tag, *_raw_counts(first))
    except Exception as exc:  # pragma: no cover
        log.warning("  %s  extraction FAILED (%s): %s", tag, type(exc).__name__, exc)

    if not primary_ok:
        # don't cache a failure — leave it uncached so a later re-run retries it
        return ChunkExtraction(chunk_id=chunk.id)

    for g in range(1, CONFIG.max_gleanings + 1):
        if not payloads:
            break
        try:
            extra = llm.chat_json(
                user + "\n\nAlready extracted:\n"
                + json.dumps(_merge_payloads(payloads))[:6000]
                + "\n\n" + P.EXTRACTION_GLEANING,
                system=P.EXTRACTION_SYSTEM,
            )
        except Exception as exc:
            log.warning("  %s  gleaning %d failed: %s", tag, g, exc)
            break
        ge, gr = _raw_counts(extra)
        if not (ge or gr):
            log.info("  %s  gleaning %d -> nothing new (stop)", tag, g)
            break
        log.info("  %s  gleaning %d -> +%de/+%dr", tag, g, ge, gr)
        payloads.append(extra)

    merged = _merge_payloads(payloads)
    CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(merged))
    res = parse_extraction(merged, chunk.id)
    me_raw, mr_raw = _raw_counts(merged)
    log.info("  %s  parsed     raw %de/%dr -> kept %de/%dr%s",
             tag, me_raw, mr_raw, len(res.entities), len(res.relationships),
             "  (dropped some as invalid/self-loop/dupe)"
             if (me_raw + mr_raw) > (len(res.entities) + len(res.relationships)) else "")
    return res


def extract_all(chunks: list[Chunk], llm: LLM, use_cache: bool = True) -> list[ChunkExtraction]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cached = sum(1 for c in chunks if use_cache and _cache_path(c.id).exists())
    log.info("extract_all: %d chunks (%d cached, %d need the LLM), concurrency=%d, max_gleanings=%d",
             len(chunks), cached, len(chunks) - cached, CONFIG.llm_concurrency, CONFIG.max_gleanings)

    results: list[ChunkExtraction] = []
    with ThreadPoolExecutor(max_workers=CONFIG.llm_concurrency) as pool:
        futures = [pool.submit(extract_chunk, c, llm, use_cache) for c in chunks]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i == len(chunks) or i % 10 == 0:
                log.info("  ...%d/%d chunks done", i, len(chunks))

    tot_e = sum(len(r.entities) for r in results)
    tot_r = sum(len(r.relationships) for r in results)
    empty = sum(1 for r in results if not r.entities and not r.relationships)
    log.info("extract_all done: %d entity mentions, %d relationship mentions across %d chunks "
             "(%d chunks yielded nothing). %s",
             tot_e, tot_r, len(results), empty, llm.usage.summary())
    return results
