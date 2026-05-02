"""Core dataclasses passed between layers (data → vector → retriever → pipeline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    text: str
    contexts: list[RetrievedChunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
