from __future__ import annotations

from typing import Any

from .base import Pipeline
from .baseline import BaselineRAG


def build_pipeline(name: str, **kwargs: Any) -> Pipeline:
    if name == "baseline":
        return BaselineRAG(**kwargs)
    if name == "graphrag":
        # Lazy import — keeps `baseline` runs free of leidenalg/igraph load cost.
        from ..techniques.graphrag.pipeline import GraphRAGPipeline

        return GraphRAGPipeline(**kwargs)
    raise ValueError(f"Unknown pipeline: {name!r}")


__all__ = ["BaselineRAG", "Pipeline", "build_pipeline"]
