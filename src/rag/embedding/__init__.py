from __future__ import annotations

from ..config import Settings
from .base import Embedding
from .ollama_embed import OllamaEmbedding


def build_embedding(
    settings: Settings | None = None, *, model: str | None = None
) -> Embedding:
    settings = settings or Settings()
    return OllamaEmbedding(
        model=model or settings.ollama_model_embed,
        host=settings.ollama_host,
    )


__all__ = ["Embedding", "OllamaEmbedding", "build_embedding"]
