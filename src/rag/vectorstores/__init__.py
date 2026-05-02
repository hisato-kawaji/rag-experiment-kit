from __future__ import annotations

from ..config import Settings
from .base import VectorStore
from .qdrant_store import QdrantStore


def build_vectorstore(
    settings: Settings | None = None, *, collection: str | None = None
) -> VectorStore:
    settings = settings or Settings()
    return QdrantStore(
        url=settings.qdrant_url,
        collection=collection or settings.qdrant_collection,
        api_key=settings.qdrant_api_key or None,
    )


__all__ = ["VectorStore", "QdrantStore", "build_vectorstore"]
