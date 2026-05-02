from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import leidenalg as la
import networkx as nx


@dataclass
class CommunityNode:
    level: int
    community_id: int
    nodes: list[str]
    parent_id: int | None = None


def hierarchical_leiden(
    g: nx.Graph,
    *,
    max_levels: int = 3,
    min_community_size: int = 2,
    seed: int = 42,
) -> list[CommunityNode]:
    """Run Leiden recursively per community to produce a hierarchy.

    Level 0 = top partition over the full graph.
    Level k = partitions inside each level-(k-1) community.
    """
    out: list[CommunityNode] = []
    next_id = [0]

    def alloc_id() -> int:
        i = next_id[0]
        next_id[0] += 1
        return i

    def run(node_subset: list[str], level: int, parent_id: int | None) -> None:
        if not node_subset:
            return
        if level >= max_levels or len(node_subset) < 2 * min_community_size:
            cid = alloc_id()
            out.append(
                CommunityNode(
                    level=level, community_id=cid, nodes=node_subset, parent_id=parent_id
                )
            )
            return

        sub = g.subgraph(node_subset)
        ig_graph = _to_igraph(sub)
        if ig_graph.vcount() == 0:
            return
        weights = (
            ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None
        )
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            weights=weights,
            seed=seed,
        )
        any_emitted = False
        for member_indices in partition:
            members = [ig_graph.vs[i]["name"] for i in member_indices]
            if len(members) < min_community_size and level > 0:
                continue
            cid = alloc_id()
            out.append(
                CommunityNode(
                    level=level, community_id=cid, nodes=members, parent_id=parent_id
                )
            )
            any_emitted = True
            if len(members) >= 2 * min_community_size and level + 1 < max_levels:
                run(members, level + 1, cid)
        if not any_emitted:
            cid = alloc_id()
            out.append(
                CommunityNode(
                    level=level, community_id=cid, nodes=node_subset, parent_id=parent_id
                )
            )

    run(list(g.nodes()), level=0, parent_id=None)
    return out


def _to_igraph(g: nx.Graph) -> ig.Graph:
    nodes = list(g.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    weights = [float(g[u][v].get("weight", 1.0)) for u, v in g.edges()]
    ig_g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_g.vs["name"] = nodes
    if weights:
        ig_g.es["weight"] = weights
    return ig_g
