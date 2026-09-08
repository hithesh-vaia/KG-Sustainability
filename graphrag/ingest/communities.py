"""Hierarchical Leiden community detection + LLM community reports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import networkx as nx

from ..config import CONFIG
from ..llm import LLM
from ..log import get_logger
from .. import prompts as P
from .merge import Entity, Relationship

log = get_logger()


@dataclass
class Community:
    id: str
    level: int
    parent: str | None
    members: list[str] = field(default_factory=list)
    # report fields (filled later)
    title: str = ""
    summary: str = ""
    rating: float = 0.0
    full_content: str = ""


def detect_communities(graph: nx.Graph) -> list[Community]:
    """Run hierarchical Leiden over each connected component."""
    from graspologic.partition import hierarchical_leiden

    communities: dict[str, Community] = {}
    for comp_nodes in nx.connected_components(graph):
        sub = graph.subgraph(comp_nodes).copy()
        if sub.number_of_edges() == 0:
            continue
        try:
            partitions = hierarchical_leiden(
                sub, max_cluster_size=CONFIG.max_cluster_size, random_seed=42
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("  leiden failed on a component (%s); skipping", exc)
            continue
        for part in partitions:
            cid = f"L{part.level}-{part.cluster}"
            parent = (
                f"L{part.level - 1}-{part.parent_cluster}"
                if part.parent_cluster is not None
                else None
            )
            comm = communities.setdefault(
                cid, Community(id=cid, level=part.level, parent=parent)
            )
            if part.node not in comm.members:
                comm.members.append(part.node)
    return list(communities.values())


def _digest(comm: Community, entities: dict[str, Entity], rels: list[Relationship]) -> str:
    lines = ["-- Entities --"]
    member_set = set(comm.members)
    for name in comm.members:
        e = entities.get(name)
        if e:
            lines.append(f"{e.name} [{e.entity_type} / {e.community}]: {e.description}")
    lines.append("\n-- Relationships --")
    for r in rels:
        if r.source in member_set and r.target in member_set:
            lines.append(f"{r.source} -{r.rel_type}-> {r.target}: {r.description}")
    return "\n".join(lines)


def _report_cache_path(digest: str):
    key = hashlib.sha256(digest.encode()).hexdigest()[:16]
    return CONFIG.cache_dir / f"report_{key}.json"


def _one_report(comm: Community, digest: str, llm: LLM, use_cache: bool) -> Community:
    cache_file = _report_cache_path(digest)
    if use_cache and cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        try:
            data = llm.chat_json(P.COMMUNITY_REPORT.format(input_text=digest))
            CONFIG.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        except Exception as exc:  # pragma: no cover
            log.warning("  report failed for %s: %s", comm.id, exc)
            data = {}

    comm.title = data.get("title", comm.id)
    comm.summary = data.get("summary", "")
    try:
        comm.rating = float(data.get("rating", 0) or 0)
    except (TypeError, ValueError):
        comm.rating = 0.0
    findings = data.get("findings", []) or []
    findings_text = "\n".join(
        f"### {f.get('summary', '')}\n{f.get('explanation', '')}" for f in findings
    )
    comm.full_content = (
        f"# {comm.title}\n\n{comm.summary}\n\n"
        f"Impact rating: {comm.rating} — {data.get('rating_explanation', '')}\n\n"
        f"{findings_text}"
    ).strip()
    return comm


def generate_reports(
    communities: list[Community],
    entities: list[Entity],
    relationships: list[Relationship],
    llm: LLM,
    use_cache: bool = True,
) -> list[Community]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ent_index = {e.name: e for e in entities}
    digests = {c.id: _digest(c, ent_index, relationships) for c in communities}
    with ThreadPoolExecutor(max_workers=CONFIG.llm_concurrency) as pool:
        futures = [
            pool.submit(_one_report, c, digests[c.id], llm, use_cache) for c in communities
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i == len(communities) or i % 10 == 0:
                log.info("  community report %d/%d", i, len(communities))
    return communities
