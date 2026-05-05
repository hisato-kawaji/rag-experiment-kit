"""Contextual Retrieval enricher (Anthropic 2024).

We don't replace the chunker — we enrich each chunk with an LLM-generated
context blurb prepended to the chunk text. The enriched text is what gets
embedded and stored.

Anthropic's blog reports prompt caching is what makes this cheap (the document
block is reused across all chunks of that document). For this Phase 1 wiring
we issue a plain call per chunk; switching to cached calls is a follow-up
that lives in the LLM adapter, not here.
"""

from __future__ import annotations

from collections.abc import Callable

from ...llm import LLM, build_llm
from ...logging import log
from ...types import Document
from .prompts import CONTEXT_PROMPT


def make_contextual_enricher(
    llm: LLM | None = None,
    *,
    max_doc_chars: int = 12000,
    max_context_tokens: int = 120,
) -> Callable[[Document, str], str]:
    """Return an `enrich_chunk(doc, chunk_text) -> str` callable.

    The callable prepends a 1-2 sentence document-aware context to each chunk
    before it is embedded.
    """
    llm = llm or build_llm()

    def enrich(doc: Document, chunk_text: str) -> str:
        try:
            ctx = llm.complete(
                CONTEXT_PROMPT.format(
                    doc=doc.text[:max_doc_chars],
                    chunk=chunk_text,
                ),
                temperature=0.0,
                max_tokens=max_context_tokens,
            ).strip()
        except Exception as e:
            log.warning("contextual.enrich.failed", doc_id=doc.id, error=str(e))
            return chunk_text
        if not ctx:
            return chunk_text
        return f"{ctx}\n\n{chunk_text}"

    return enrich
