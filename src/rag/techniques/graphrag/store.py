from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from .community import CommunityNode
from .summarizer import CommunityReport

GRAPH_FILE = "graph.pickle"
COMMUNITIES_FILE = "communities.json"
REPORTS_FILE = "reports.json"


def save_artifacts(
    out_dir: Path,
    *,
    graph: nx.Graph,
    communities: list[CommunityNode],
    reports: list[CommunityReport],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / GRAPH_FILE).open("wb") as f:
        pickle.dump(graph, f)
    with (out_dir / COMMUNITIES_FILE).open("w") as f:
        json.dump([asdict(c) for c in communities], f, ensure_ascii=False, indent=2)
    with (out_dir / REPORTS_FILE).open("w") as f:
        json.dump([asdict(r) for r in reports], f, ensure_ascii=False, indent=2)


def load_artifacts(
    out_dir: Path,
) -> tuple[nx.Graph, list[CommunityNode], list[CommunityReport]]:
    with (out_dir / GRAPH_FILE).open("rb") as f:
        graph = pickle.load(f)
    with (out_dir / COMMUNITIES_FILE).open() as f:
        communities = [CommunityNode(**c) for c in json.load(f)]
    with (out_dir / REPORTS_FILE).open() as f:
        reports = [CommunityReport(**r) for r in json.load(f)]
    return graph, communities, reports
