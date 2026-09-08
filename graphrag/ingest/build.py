"""Build an in-memory NetworkX graph (input to community detection)."""
from __future__ import annotations

import networkx as nx

from .merge import Entity, Relationship


def build_graph(entities: list[Entity], relationships: list[Relationship]) -> nx.Graph:
    g = nx.Graph()
    for e in entities:
        g.add_node(e.name, type=e.entity_type, community=e.community,
                   description=e.description, degree=e.degree)
    for r in relationships:
        if r.source in g and r.target in g:
            g.add_edge(r.source, r.target, weight=max(r.strength, 1.0),
                       rel_type=r.rel_type, description=r.description)
    # drop isolated nodes for clustering purposes (kept in Neo4j regardless)
    return g


def largest_component(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    components = sorted(nx.connected_components(g), key=len, reverse=True)
    return g.subgraph(components[0]).copy() if components else g
