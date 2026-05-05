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

MULTIHOP_PROMPT = """You are creating a MULTI-HOP evaluation question that requires combining information from TWO passages.

Passage A:
\"\"\"
{text_a}
\"\"\"

Passage B:
\"\"\"
{text_b}
\"\"\"

Write ONE question that:
- Is fully answerable using BOTH passages together (no outside knowledge).
- Cannot be answered from either passage alone — the answer requires linking facts across the two.

Return STRICT JSON (no commentary):
{{"question": "...", "ground_truth": "..."}}
"""

ABSTRACT_PROMPT = """You are creating an ABSTRACT evaluation question over a longer document.

Document:
\"\"\"
{text}
\"\"\"

Write ONE high-level question (e.g. "Summarize the main approaches discussed",
"What is the central theme of this document?") together with a faithful 2-3
sentence answer grounded in the document.

The question must NOT be answerable from a single short paragraph — it should
require synthesizing the whole document.

Return STRICT JSON (no commentary):
{{"question": "...", "ground_truth": "..."}}
"""


def _gen_factoid(chunk: Chunk, llm) -> dict[str, Any] | None:
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
        "question_type": "factoid",
        "reference_doc_id": chunk.doc_id,
        "reference_chunk_id": chunk.id,
        "reference_text": chunk.text[:1000],
    }


def _gen_multi_hop(chunk_a: Chunk, chunk_b: Chunk, llm) -> dict[str, Any] | None:
    raw = llm.complete(
        MULTIHOP_PROMPT.format(text_a=chunk_a.text[:1800], text_b=chunk_b.text[:1800]),
        json_mode=True,
        temperature=0.3,
        max_tokens=512,
    )
    data = _safe_parse_json(raw)
    if not data or not data.get("question") or not data.get("ground_truth"):
        return None
    return {
        "question": str(data["question"]).strip(),
        "ground_truth": str(data["ground_truth"]).strip(),
        "question_type": "multi_hop",
        "reference_doc_id": f"{chunk_a.doc_id}+{chunk_b.doc_id}",
        "reference_chunk_id": f"{chunk_a.id}+{chunk_b.id}",
        "reference_text": (chunk_a.text[:600] + "\n---\n" + chunk_b.text[:600]),
    }


def _gen_abstract(doc_chunks: list[Chunk], llm) -> dict[str, Any] | None:
    # Concatenate up to ~6000 chars from a single doc so the model can see
    # something resembling the whole document.
    joined = "\n\n".join(c.text for c in doc_chunks)[:6000]
    if len(joined) < 500:
        return None
    raw = llm.complete(
        ABSTRACT_PROMPT.format(text=joined),
        json_mode=True,
        temperature=0.3,
        max_tokens=512,
    )
    data = _safe_parse_json(raw)
    if not data or not data.get("question") or not data.get("ground_truth"):
        return None
    return {
        "question": str(data["question"]).strip(),
        "ground_truth": str(data["ground_truth"]).strip(),
        "question_type": "abstract",
        "reference_doc_id": doc_chunks[0].doc_id,
        "reference_chunk_id": ",".join(c.id for c in doc_chunks[:3]),
        "reference_text": joined[:1500],
    }


def _by_doc(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    by: dict[str, list[Chunk]] = {}
    for c in chunks:
        by.setdefault(c.doc_id, []).append(c)
    return by


def generate_eval_set(n: int = 30, seed: int = 42, workers: int = 4) -> list[dict[str, Any]]:
    """Single-passage factoid Q&A pairs (back-compat with Phase 1)."""
    return generate_testset(n=n, seed=seed, workers=workers, style="simple")


def generate_testset(
    n: int = 30,
    seed: int = 42,
    workers: int = 4,
    *,
    style: str = "simple",
    ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate a labelled eval set.

    style="simple" — all factoid (legacy behavior).
    style="mixed"  — split across factoid / multi_hop / abstract by `ratios`
                     (default 50:30:20). Each pair gets a `question_type` field.
    """
    if ratios is None:
        ratios = {"factoid": 0.5, "multi_hop": 0.3, "abstract": 0.2}

    vs = build_vectorstore()
    if not isinstance(vs, QdrantStore):
        raise RuntimeError("synth requires QdrantStore for now")
    chunks = fetch_all_chunks(vs)
    if not chunks:
        raise RuntimeError("No chunks in store. Run `task ingest -- ...` first.")
    rng = random.Random(seed)
    llm = build_llm()

    if style == "simple":
        return _generate_factoid(chunks, n=n, rng=rng, llm=llm, workers=workers)
    if style != "mixed":
        raise ValueError(f"Unknown synth style: {style!r}")

    n_factoid = round(n * ratios.get("factoid", 0))
    n_multi = round(n * ratios.get("multi_hop", 0))
    n_abstract = max(0, n - n_factoid - n_multi)
    log.info("synth.mixed.plan", factoid=n_factoid, multi_hop=n_multi, abstract=n_abstract)

    out: list[dict[str, Any]] = []
    if n_factoid:
        out.extend(_generate_factoid(chunks, n=n_factoid, rng=rng, llm=llm, workers=workers))
    if n_multi:
        out.extend(_generate_multi_hop(chunks, n=n_multi, rng=rng, llm=llm, workers=workers))
    if n_abstract:
        out.extend(_generate_abstract(chunks, n=n_abstract, rng=rng, llm=llm, workers=workers))
    log.info(
        "synth.done",
        n=len(out),
        types={
            t: sum(1 for r in out if r.get("question_type") == t)
            for t in {"factoid", "multi_hop", "abstract"}
        },
    )
    return out


def _generate_factoid(
    chunks: list[Chunk], *, n: int, rng: random.Random, llm, workers: int
) -> list[dict[str, Any]]:
    sampled = rng.sample(chunks, min(len(chunks), max(n * 2, 10)))
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_gen_factoid, c, llm) for c in sampled]
        for fut in as_completed(futures):
            try:
                pair = fut.result()
            except Exception as e:
                log.warning("synth.factoid.error", error=str(e))
                continue
            if pair:
                out.append(pair)
            if len(out) >= n:
                break
    return out[:n]


def _generate_multi_hop(
    chunks: list[Chunk], *, n: int, rng: random.Random, llm, workers: int
) -> list[dict[str, Any]]:
    """Pair chunks within the same doc when possible (real link), else cross-doc."""
    by_doc = _by_doc(chunks)
    pairs: list[tuple[Chunk, Chunk]] = []
    rich_docs = [cs for cs in by_doc.values() if len(cs) >= 2]
    while len(pairs) < n * 2 and rich_docs:
        cs = rng.choice(rich_docs)
        a, b = rng.sample(cs, 2)
        pairs.append((a, b))
        if len(pairs) > 4 * n:
            break
    while len(pairs) < n * 2:
        a, b = rng.sample(chunks, 2)
        if a.doc_id == b.doc_id:
            continue
        pairs.append((a, b))
        if len(pairs) > 6 * n:
            break

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_gen_multi_hop, a, b, llm) for a, b in pairs]
        for fut in as_completed(futures):
            try:
                pair = fut.result()
            except Exception as e:
                log.warning("synth.multi_hop.error", error=str(e))
                continue
            if pair:
                out.append(pair)
            if len(out) >= n:
                break
    return out[:n]


def _generate_abstract(
    chunks: list[Chunk], *, n: int, rng: random.Random, llm, workers: int
) -> list[dict[str, Any]]:
    by_doc = _by_doc(chunks)
    docs = list(by_doc.values())
    rng.shuffle(docs)
    docs = docs[: max(n * 2, len(docs))]
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_gen_abstract, d, llm) for d in docs]
        for fut in as_completed(futures):
            try:
                pair = fut.result()
            except Exception as e:
                log.warning("synth.abstract.error", error=str(e))
                continue
            if pair:
                out.append(pair)
            if len(out) >= n:
                break
    return out[:n]
