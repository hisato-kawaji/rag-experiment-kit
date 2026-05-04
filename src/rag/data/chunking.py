"""Plain character-window chunker with overlap.

We deliberately avoid token-based chunking here so the data layer has no LLM
dependency. Token-aware variants live in `rag/techniques/*/chunker.py`.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            snap = text.rfind(" ", start, end)
            if snap > start + step // 2:
                end = snap
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = start + step
        start = next_start
    return chunks
