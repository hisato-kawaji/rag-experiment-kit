"""Contextual Retrieval pipeline.

The retrieval side is plain dense search — Anthropic's contribution is at
*indexing* time (chunks get an LLM-generated document-level context prepended
before embedding). This pipeline therefore subclasses BaselineRAG only to give
the run a distinct `name` for metric attribution and run-dir naming.

Usage:
    QDRANT_COLLECTION=rag_contextual task ingest -- --chunker contextual ...
    QDRANT_COLLECTION=rag_contextual task run -- --pipeline contextual_retrieval
"""

from __future__ import annotations

from ...pipelines.baseline import BaselineRAG


class ContextualRetrievalPipeline(BaselineRAG):
    name = "contextual_retrieval"
