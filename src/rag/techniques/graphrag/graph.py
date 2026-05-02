from __future__ import annotations

import networkx as nx

from .extractor import EntityRecord, RelationRecord


def build_graph(
    entities: list[EntityRecord], relations: list[RelationRecord]
) -> nx.Graph:
    """Build an undirected weighted graph from extraction records.

    Entities are deduplicated by lowercased name. Edge weight = sum of per-mention
    weights. Per-node and per-edge descriptions are kept in lists for later summarization.
    """
    g: nx.Graph = nx.Graph()

    by_norm: dict[str, dict] = {}
    for e in entities:
        key = e.name.strip().lower()
        if not key:
            continue
        bucket = by_norm.setdefault(
            key,
            {"name": e.name, "type": e.type, "descriptions": [], "source_chunks": set()},
        )
        if e.description:
            bucket["descriptions"].append(e.description)
        bucket["source_chunks"].update(e.source_chunks)
        # Prefer non-OTHER type if seen
        if bucket["type"] == "OTHER" and e.type != "OTHER":
            bucket["type"] = e.type

    for data in by_norm.values():
        g.add_node(
            data["name"],
            type=data["type"],
            descriptions=data["descriptions"],
            source_chunks=sorted(data["source_chunks"]),
        )

    name_lookup = {n.lower(): n for n in g.nodes}

    edge_data: dict[tuple[str, str], dict] = {}
    for r in relations:
        s_key = r.source.strip().lower()
        t_key = r.target.strip().lower()
        if not s_key or not t_key or s_key == t_key:
            continue
        s_name = name_lookup.get(s_key) or _add_stub(g, r.source, name_lookup)
        t_name = name_lookup.get(t_key) or _add_stub(g, r.target, name_lookup)
        a, b = sorted([s_name, t_name])
        bucket = edge_data.setdefault(
            (a, b), {"weight": 0.0, "descriptions": [], "source_chunks": set()}
        )
        bucket["weight"] += r.weight
        if r.description:
            bucket["descriptions"].append(r.description)
        bucket["source_chunks"].update(r.source_chunks)

    for (a, b), d in edge_data.items():
        g.add_edge(
            a,
            b,
            weight=d["weight"],
            descriptions=d["descriptions"],
            source_chunks=sorted(d["source_chunks"]),
        )
    return g


def _add_stub(g: nx.Graph, name: str, lookup: dict[str, str]) -> str:
    canonical = name.strip()
    g.add_node(canonical, type="OTHER", descriptions=[], source_chunks=[])
    lookup[canonical.lower()] = canonical
    return canonical
