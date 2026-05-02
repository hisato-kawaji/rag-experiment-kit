from __future__ import annotations

from pathlib import Path
from typing import Any

from ...types import Answer
from .retriever import GraphRAGRetriever


class GraphRAGPipeline:
    """Pipeline-protocol wrapper around GraphRAGRetriever.

    Modes:
      - "global": map-reduce over community summaries — best for sense-making questions
      - "local":  entity-anchored neighborhood — best for factoid / about-X questions
    """

    def __init__(
        self,
        *,
        mode: str = "global",
        artifacts_dir: Path | str = Path("data/graphrag"),
        top_k: int = 5,
        **_: Any,
    ) -> None:
        self.name = f"graphrag/{mode}"
        self.mode = mode
        self.top_k = top_k
        self._retriever = GraphRAGRetriever(Path(artifacts_dir))

    def answer(self, query: str) -> Answer:
        if self.mode == "global":
            return self._retriever.global_search(query)
        if self.mode == "local":
            return self._retriever.local_search(query, top_chunks=self.top_k)
        raise ValueError(f"Unknown graphrag mode: {self.mode!r}")
