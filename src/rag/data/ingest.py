"""Orchestrator: source.fetch() → chunk_text → (optional enrich) → embed → vectorstore.upsert."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..embedding import build_embedding
from ..logging import log
from ..types import Chunk, Document
from ..vectorstores import build_vectorstore
from .chunking import chunk_text
from .sources import build_source


@dataclass
class IngestStats:
    docs: int
    chunks: int


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return f"{doc_id}::chunk{idx:04d}::{h}"


def ingest(
    source: str,
    kwargs: dict[str, Any],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    embed_batch: int = 32,
    upsert_batch: int = 128,
    enrich_chunk: Callable[[Document, str], str] | None = None,
) -> IngestStats:
    """If `enrich_chunk` is given, it is called per chunk before embedding/storage.

    Used by Contextual Retrieval (Anthropic 2024) to prepend a 1-2 sentence
    document-aware context to each chunk so the embedded text becomes more
    self-contained.
    """
    src = build_source(source)
    embedder = build_embedding()
    vectorstore = build_vectorstore()
    vectorstore.ensure_collection(embedder.dim)

    n_docs = 0
    n_chunks = 0
    pending: list[Chunk] = []

    def flush(batch: list[Chunk]) -> None:
        if not batch:
            return
        vectors = embedder.embed([c.text for c in batch])
        for c, v in zip(batch, vectors, strict=True):
            c.embedding = v
        vectorstore.upsert(batch)

    for doc in src.fetch(**kwargs):
        n_docs += 1
        pieces = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            text = enrich_chunk(doc, piece) if enrich_chunk else piece
            pending.append(
                Chunk(
                    id=_chunk_id(doc.id, i, text),
                    doc_id=doc.id,
                    text=text,
                    metadata={**doc.metadata, "chunk_index": i},
                )
            )
            n_chunks += 1
            if len(pending) >= embed_batch:
                flush(pending[:embed_batch])
                pending = pending[embed_batch:]
            if len(pending) >= upsert_batch:
                flush(pending)
                pending = []

    flush(pending)
    log.info("ingest.done", source=source, docs=n_docs, chunks=n_chunks)
    return IngestStats(docs=n_docs, chunks=n_chunks)
