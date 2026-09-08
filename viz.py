"""Render the knowledge graph to a single interactive HTML file.

    python viz.py                         # entities, coloured by type -> output/graph.html
    python viz.py --color community        # colour by top communities instead
    python viz.py --communities            # community-level map (nodes = communities)
    python viz.py --top 150 --min-degree 3 --open

Nodes are sized by degree; only the most-connected nodes are labelled (the rest
show their label on hover). Hovering a node fades everything except its
neighbours.
"""
from __future__ import annotations

import argparse
import ast
import html
import webbrowser
from collections import Counter
from pathlib import Path

import pandas as pd
from pyvis.network import Network

from graphrag.config import CONFIG

# validated 8-hue categorical palette (dark surface) from the dataviz skill
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500",
           "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER = "#6b7280"
SURFACE = "#15181d"
INK = "#e8e8e8"


def _as_list(v) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return [v]
    try:
        return list(v)
    except TypeError:
        return []


def _legend_html(items: list[tuple[str, str]], title: str) -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};'
        f'display:inline-block"></span>'
        f'<span>{html.escape(name)}</span></div>'
        for name, c in items
    )
    return (
        f'<div style="position:fixed;top:14px;left:14px;z-index:999;'
        f'background:rgba(20,22,26,.92);color:{INK};font:13px/1.3 -apple-system,'
        f'Segoe UI,Roboto,sans-serif;padding:12px 14px;border-radius:10px;'
        f'border:1px solid #2b2f36;max-height:80vh;overflow:auto">'
        f'<div style="font-weight:600;margin-bottom:6px">{html.escape(title)}</div>'
        f'{rows}</div>'
    )


def _new_net() -> Network:
    net = Network(height="100vh", width="100%", bgcolor=SURFACE, font_color=INK, directed=False)
    net.set_options("""
    {
      "nodes": { "borderWidth": 0, "shape": "dot",
                 "font": { "size": 14, "face": "-apple-system, Segoe UI, Roboto",
                           "strokeWidth": 3, "strokeColor": "#15181d" } },
      "edges": { "color": { "color": "#3a3f47", "highlight": "#8b93a1", "opacity": 0.55 },
                 "smooth": { "type": "continuous" }, "width": 0.5 },
      "interaction": { "hover": true, "tooltipDelay": 100,
                       "hideEdgesOnDrag": true, "navigationButtons": true },
      "physics": { "solver": "forceAtlas2Based",
                   "forceAtlas2Based": { "gravitationalConstant": -55,
                                         "centralGravity": 0.008,
                                         "springLength": 110, "springConstant": 0.05,
                                         "damping": 0.5, "avoidOverlap": 0.6 },
                   "stabilization": { "iterations": 300 } }
    }
    """)
    return net


def _inject_legend(path: Path, legend: str) -> None:
    txt = path.read_text()
    path.write_text(txt.replace("<body>", "<body>\n" + legend, 1))


def build_entities(top: int, min_degree: int, color_by: str, level: int,
                   labels: str, out: Path) -> Path:
    ent = pd.read_parquet(CONFIG.output_dir / "entities.parquet")
    rel = pd.read_parquet(CONFIG.output_dir / "relationships.parquet")
    comm = pd.read_parquet(CONFIG.output_dir / "communities.parquet")

    ent = ent[ent["degree"] >= min_degree].sort_values("degree", ascending=False).head(top)
    keep = set(ent["name"])
    n_labelled = {"all": len(ent), "none": 0}.get(labels, min(60, len(ent)))
    label_cut = ent["degree"].nlargest(n_labelled).min() if n_labelled else float("inf")

    if color_by == "leiden":
        ent2c: dict[str, str] = {}
        for _, r in comm[comm["level"] == level].iterrows():
            for m in _as_list(r["members"]):
                ent2c.setdefault(m, r["id"])
        counts = Counter(ent2c[n] for n in keep if n in ent2c)
        top_c = [c for c, _ in counts.most_common(7)]
        cmap = {c: PALETTE[i] for i, c in enumerate(top_c)}
        titles = {c: comm.loc[comm["id"] == c, "title"].iloc[0] if (comm["id"] == c).any() else c
                  for c in top_c}
        legend_items = [(f"{titles[c]}", cmap[c]) for c in top_c] + [("other", OTHER)]
        legend_title = f"Leiden communities (level {level})"

        def node_color(name, _t):
            return cmap.get(ent2c.get(name), OTHER)
    else:
        col = "community" if color_by == "domain" else "entity_type"
        freq = Counter(ent[col])
        present = [t for t, _ in freq.most_common()]
        cmap = {t: PALETTE[i] if i < len(PALETTE) else OTHER for i, t in enumerate(present)}
        shown = present[:len(PALETTE)]
        legend_items = [(t, cmap[t]) for t in shown] + ([("other", OTHER)] if len(present) > len(shown) else [])
        legend_title = "ESG domain" if color_by == "domain" else "Entity type"
        _by_name = dict(zip(ent["name"], ent[col]))

        def node_color(name, _t):
            return cmap.get(_by_name.get(name), OTHER)

    net = _new_net()
    for _, r in ent.iterrows():
        deg = float(r["degree"])
        net.add_node(
            r["name"],
            label=r["name"].title() if deg >= label_cut else "​",
            title=f"<b>{html.escape(str(r['name']).title())}</b> — "
                  f"{r.get('entity_type', '?')} / {r.get('community', '?')} · degree {int(deg)}"
                  f"<br>{html.escape(str(r['description'])[:280])}",
            size=8 + deg ** 0.5 * 5,
            color=node_color(r["name"], None),
        )
    edges = 0
    for _, r in rel.iterrows():
        if r["source"] in keep and r["target"] in keep:
            net.add_edge(r["source"], r["target"], value=float(r["strength"]),
                         title=html.escape(str(r["description"])[:280]))
            edges += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out), notebook=False)
    _inject_legend(out, _legend_html(legend_items, legend_title))
    print(f"wrote {out}  ({len(keep)} entities, {edges} edges)  colour={color_by}")
    return out


def build_communities(level: int, out: Path) -> Path:
    ent = pd.read_parquet(CONFIG.output_dir / "entities.parquet")
    rel = pd.read_parquet(CONFIG.output_dir / "relationships.parquet")
    comm = pd.read_parquet(CONFIG.output_dir / "communities.parquet")
    sub = comm[comm["level"] == level]

    ent2c: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for _, r in sub.iterrows():
        members[r["id"]] = _as_list(r["members"])
        for m in members[r["id"]]:
            ent2c[m] = r["id"]

    cross = Counter()
    for _, r in rel.iterrows():
        a, b = ent2c.get(r["source"]), ent2c.get(r["target"])
        if a and b and a != b:
            cross[tuple(sorted((a, b)))] += 1

    ratings = sub.set_index("id")["rating"].to_dict()
    rmax = max(ratings.values()) if ratings else 1
    net = _new_net()
    for _, r in sub.iterrows():
        n = len(members[r["id"]])
        frac = (ratings.get(r["id"], 0) / rmax) if rmax else 0
        idx = min(int(frac * 4), 4)  # 0..4 -> blue ramp
        ramp = ["#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7"][idx]
        net.add_node(
            r["id"],
            label=html.escape(str(r["title"])[:40]),
            title=f"<b>{html.escape(str(r['title']))}</b><br>{n} entities · "
                  f"impact {r['rating']:.1f}<br>{html.escape(str(r['summary'])[:300])}",
            size=10 + n ** 0.5 * 6,
            color=ramp,
        )
    for (a, b), w in cross.items():
        net.add_edge(a, b, value=w, title=f"{w} cross-community relationships")

    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out), notebook=False)
    _inject_legend(out, _legend_html(
        [("low impact", "#1c5cab"), ("high impact", "#5598e7")],
        f"Communities (level {level}) — size = # entities"))
    print(f"wrote {out}  ({len(sub)} communities, {len(cross)} links)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=250, help="max entities (by degree)")
    p.add_argument("--min-degree", type=int, default=2)
    p.add_argument("--color", choices=["domain", "type", "leiden"], default="domain",
                   help="domain = ESG community, type = entity_type, leiden = detected cluster")
    p.add_argument("--level", type=int, default=1, help="community level")
    p.add_argument("--labels", choices=["all", "top", "none"], default="top",
                   help="which entity names to draw (default: top 60 by degree)")
    p.add_argument("--communities", action="store_true",
                   help="draw the community-level map instead of entities")
    p.add_argument("--out", type=Path, default=CONFIG.output_dir / "graph.html")
    p.add_argument("--open", action="store_true")
    a = p.parse_args()

    if a.communities:
        path = build_communities(a.level, a.out)
    else:
        path = build_entities(a.top, a.min_degree, a.color, a.level, a.labels, a.out)
    if a.open:
        webbrowser.open(path.resolve().as_uri())
