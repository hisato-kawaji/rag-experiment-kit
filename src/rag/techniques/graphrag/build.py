from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...llm import build_llm
from ...logging import log
from ...types import Chunk
from ...vectorstores import build_vectorstore
from ...vectorstores.qdrant_store import QdrantStore
from .community import hierarchical_leiden
from .extractor import extract_all
from .graph import build_graph
from .store import save_artifacts
from .summarizer import summarize_communities


@dataclass
class BuildResult:
    n_chunks: int
    n_entities: int
    n_relations: int
    n_communities: int


def fetch_all_chunks(vectorstore: QdrantStore, batch: int = 200) -> list[Chunk]:
    """Stream every chunk out of Qdrant via the scroll API."""
    client = vectorstore._client  # type: ignore[attr-defined]
    out: list[Chunk] = []
    next_offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=vectorstore.collection,
            limit=batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
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
        if next_offset is None:
            break
    return out


def build_graphrag_index(
    out_dir: Path = Path("data/graphrag"),
    *,
    max_chunks: int | None = None,
    leiden_max_levels: int = 3,
    extraction_workers: int = 4,
    summary_workers: int = 4,
) -> BuildResult:
    vs = build_vectorstore()
    if not isinstance(vs, QdrantStore):
        raise RuntimeError(
            "GraphRAG build currently only supports QdrantStore "
            "(needs scroll API). Implement fetch_all_chunks for your store."
        )
    chunks = fetch_all_chunks(vs)
    if max_chunks:
        chunks = chunks[:max_chunks]
    log.info("graphrag.build.chunks", n=len(chunks))
    if not chunks:
        raise RuntimeError(
            "No chunks in vector store. Run `task ingest -- ...` first."
        )

    llm = build_llm()
    entities, relations = extract_all(chunks, llm=llm, workers=extraction_workers)
    g = build_graph(entities, relations)
    log.info(
        "graphrag.build.graph", nodes=g.number_of_nodes(), edges=g.number_of_edges()
    )
    if g.number_of_nodes() == 0:
        raise RuntimeError("Extraction produced no entities.")

    communities = hierarchical_leiden(g, max_levels=leiden_max_levels)
    log.info("graphrag.build.communities", n=len(communities))

    reports = summarize_communities(g, communities, llm=llm, workers=summary_workers)
    save_artifacts(out_dir, graph=g, communities=communities, reports=reports)
    log.info("graphrag.build.saved", out_dir=str(out_dir))

    return BuildResult(
        n_chunks=len(chunks),
        n_entities=g.number_of_nodes(),
        n_relations=g.number_of_edges(),
        n_communities=len(reports),
    )
