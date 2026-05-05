"""Local-search anchor index.

GraphRAG/local needs to map a free-form question to a small set of seed
entities ("anchors") in the graph. The Phase 1 retriever did this with a
substring/token heuristic, which silently fails on paraphrased queries
("fastest unstructured search method" never matches "Grover's algorithm").

This module persists a flat list of (name, type, vector, description) tuples
alongside the graph artifact and exposes a numpy-cosine search. It deliberately
does *not* spin up another Qdrant collection so GraphRAG remains usable from a
single artifacts dir, with no extra services to coordinate during loading.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ...embedding import Embedding, build_embedding
from ...logging import log
from .store import ENTITY_INDEX_FILE


@dataclass
class EntityRecord:
    name: str
    type: str
    description: str
    vector: list[float]


class EntityIndex:
    def __init__(self, records: list[EntityRecord]) -> None:
        self.records = records

    @classmethod
    def build(
        cls,
        graph: nx.Graph,
        *,
        embedder: Embedding | None = None,
        max_desc_chars: int = 200,
        batch: int = 64,
    ) -> EntityIndex:
        embedder = embedder or build_embedding()
        nodes = list(graph.nodes(data=True))
        texts: list[str] = []
        meta: list[tuple[str, str, str]] = []
        for name, attrs in nodes:
            etype = attrs.get("type", "OTHER")
            descs = attrs.get("descriptions") or []
            head_desc = descs[0][:max_desc_chars] if descs else ""
            texts.append(f"{name}: {head_desc}" if head_desc else name)
            meta.append((name, etype, head_desc))
        records: list[EntityRecord] = []
        for i in range(0, len(texts), batch):
            vecs = embedder.embed(texts[i : i + batch])
            for (name, etype, desc), vec in zip(meta[i : i + batch], vecs, strict=True):
                records.append(EntityRecord(name=name, type=etype, description=desc, vector=vec))
        log.info("graphrag.entity_index.built", n=len(records))
        return cls(records)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / ENTITY_INDEX_FILE).open("wb") as f:
            pickle.dump(self.records, f)

    @classmethod
    def load(cls, out_dir: Path) -> EntityIndex | None:
        path = out_dir / ENTITY_INDEX_FILE
        if not path.exists():
            return None
        with path.open("rb") as f:
            records = pickle.load(f)
        return cls(records)

    def search(
        self,
        query_vec: list[float],
        *,
        k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[str, float]]:
        qn = math.sqrt(sum(x * x for x in query_vec)) or 1.0
        scored: list[tuple[float, str]] = []
        for r in self.records:
            dot = 0.0
            rn = 0.0
            for a, b in zip(query_vec, r.vector, strict=True):
                dot += a * b
                rn += b * b
            rn = math.sqrt(rn) or 1.0
            score = dot / (qn * rn)
            if score >= min_score:
                scored.append((score, r.name))
        scored.sort(key=lambda t: -t[0])
        return [(name, score) for score, name in scored[:k]]
