from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..llm import build_llm
from ..logging import log
from ..techniques.graphrag.build import fetch_all_chunks
from ..techniques.graphrag.extractor import _safe_parse_json
from ..types import Chunk
from ..vectorstores import build_vectorstore
from ..vectorstores.qdrant_store import QdrantStore

SYNTH_PROMPT = """You are creating an evaluation question-and-answer pair from a passage.

Passage:
\"\"\"
{text}
\"\"\"

Write ONE question that is fully answerable from the passage above (no outside knowledge),
and the SHORTEST faithful answer derived from it.

Return STRICT JSON in this exact schema (no commentary):
{{"question": "...", "ground_truth": "..."}}
"""


def _gen_one(chunk: Chunk, llm) -> dict[str, Any] | None:
    raw = llm.complete(
        SYNTH_PROMPT.format(text=chunk.text[:3000]),
        json_mode=True,
        temperature=0.2,
        max_tokens=512,
    )
    data = _safe_parse_json(raw)
    if not data or not data.get("question") or not data.get("ground_truth"):
        return None
    return {
        "question": str(data["question"]).strip(),
        "ground_truth": str(data["ground_truth"]).strip(),
        "reference_doc_id": chunk.doc_id,
        "reference_chunk_id": chunk.id,
        "reference_text": chunk.text[:1000],
    }


def generate_eval_set(
    n: int = 30, seed: int = 42, workers: int = 4
) -> list[dict[str, Any]]:
    """Sample chunks from the store and ask the LLM to write Q+ground_truth from each."""
    vs = build_vectorstore()
    if not isinstance(vs, QdrantStore):
        raise RuntimeError("synth.generate_eval_set requires QdrantStore for now")
    chunks = fetch_all_chunks(vs)
    if not chunks:
        raise RuntimeError("No chunks in store. Run `task ingest -- ...` first.")
    rng = random.Random(seed)
    sampled = rng.sample(chunks, min(len(chunks), max(n * 2, 10)))
    llm = build_llm()
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_gen_one, c, llm) for c in sampled]
        for fut in as_completed(futures):
            try:
                pair = fut.result()
            except Exception as e:
                log.warning("synth.error", error=str(e))
                continue
            if pair:
                out.append(pair)
            if len(out) >= n:
                break
    log.info("synth.done", n=len(out))
    return out[:n]
