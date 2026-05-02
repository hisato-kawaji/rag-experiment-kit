from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..types import Chunk, RetrievedChunk


class QdrantStore:
    def __init__(self, url: str, collection: str, api_key: str | None = None) -> None:
        self.name = f"qdrant/{collection}"
        self.collection = collection
        self._client = QdrantClient(url=url, api_key=api_key or None)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant requires UUID or unsigned int IDs — derive deterministically.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def ensure_collection(self, dim: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        if any(c.embedding is None for c in chunks):
            raise ValueError("All chunks must have embeddings before upsert")
        points = [
            qm.PointStruct(
                id=self._point_id(c.id),
                vector=c.embedding,  # type: ignore[arg-type]
                payload={
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    **c.metadata,
                },
            )
            for c in chunks
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        vector: list[float],
        k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        flt = self._make_filter(where) if where else None
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=k,
            query_filter=flt,
            with_payload=True,
        )
        out: list[RetrievedChunk] = []
        for h in hits:
            payload = dict(h.payload or {})
            text = payload.pop("text", "")
            chunk = Chunk(
                id=payload.pop("chunk_id", str(h.id)),
                doc_id=payload.pop("doc_id", ""),
                text=text,
                metadata=payload,
            )
            out.append(RetrievedChunk(chunk=chunk, score=h.score))
        return out

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        if not ids:
            return []
        records = self._client.retrieve(
            collection_name=self.collection,
            ids=[self._point_id(i) for i in ids],
            with_payload=True,
        )
        out: list[Chunk] = []
        for r in records:
            payload = dict(r.payload or {})
            text = payload.pop("text", "")
            out.append(
                Chunk(
                    id=payload.pop("chunk_id", str(r.id)),
                    doc_id=payload.pop("doc_id", ""),
                    text=text,
                    metadata=payload,
                )
            )
        return out

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count

    def reset(self) -> None:
        if self.collection in {c.name for c in self._client.get_collections().collections}:
            self._client.delete_collection(self.collection)

    @staticmethod
    def _make_filter(where: dict[str, Any]) -> qm.Filter:
        must = [
            qm.FieldCondition(key=k, match=qm.MatchValue(value=v)) for k, v in where.items()
        ]
        return qm.Filter(must=must)
