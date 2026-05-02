from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..types import Chunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """Strict interface — Phase 2 plugs in Weaviate/LanceDB/pgvector behind this."""

    name: str

    def ensure_collection(self, dim: int) -> None: ...

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(
        self,
        vector: list[float],
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def get_by_ids(self, ids: list[str]) -> list[Chunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
