from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ...llm import LLM, build_llm
from ...logging import log
from ...types import Chunk
from .prompts import EXTRACTION_PROMPT, EXTRACTION_SYSTEM


@dataclass
class EntityRecord:
    name: str
    type: str
    description: str
    source_chunks: list[str] = field(default_factory=list)


@dataclass
class RelationRecord:
    source: str
    target: str
    description: str
    weight: float = 1.0
    source_chunks: list[str] = field(default_factory=list)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _safe_parse_json(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def extract_from_chunk(
    chunk: Chunk, llm: LLM, max_input_chars: int = 4000
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    raw = llm.complete(
        EXTRACTION_PROMPT.format(text=chunk.text[:max_input_chars]),
        system=EXTRACTION_SYSTEM,
        json_mode=True,
        temperature=0.0,
        max_tokens=2048,
    )
    data = _safe_parse_json(raw)
    if not data:
        log.warning("graphrag.extract.json-error", chunk_id=chunk.id, head=raw[:200])
        return [], []

    entities: list[EntityRecord] = []
    for e in data.get("entities", []) or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        entities.append(
            EntityRecord(
                name=name,
                type=str(e.get("type", "OTHER")).strip().upper(),
                description=str(e.get("description", "")).strip(),
                source_chunks=[chunk.id],
            )
        )

    relations: list[RelationRecord] = []
    for r in data.get("relationships", []) or []:
        s = str(r.get("source", "")).strip()
        t = str(r.get("target", "")).strip()
        if not s or not t or s.lower() == t.lower():
            continue
        try:
            weight = float(r.get("weight", 1) or 1)
        except (TypeError, ValueError):
            weight = 1.0
        relations.append(
            RelationRecord(
                source=s,
                target=t,
                description=str(r.get("description", "")).strip(),
                weight=weight,
                source_chunks=[chunk.id],
            )
        )
    return entities, relations


def extract_all(
    chunks: list[Chunk],
    llm: LLM | None = None,
    workers: int = 4,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    llm = llm or build_llm()
    log.info("graphrag.extract.start", n_chunks=len(chunks), workers=workers)
    all_e: list[EntityRecord] = []
    all_r: list[RelationRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_from_chunk, c, llm): c for c in chunks}
        for i, fut in enumerate(as_completed(futures), 1):
            chunk = futures[fut]
            try:
                e, r = fut.result()
                all_e.extend(e)
                all_r.extend(r)
            except Exception as exc:
                log.warning("graphrag.extract.error", chunk_id=chunk.id, error=str(exc))
            if i % 10 == 0 or i == len(chunks):
                log.info("graphrag.extract.progress", done=i, total=len(chunks))
    log.info(
        "graphrag.extract.done",
        chunks=len(chunks),
        entities=len(all_e),
        relations=len(all_r),
    )
    return all_e, all_r
