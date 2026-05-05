from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ...llm import LLM, build_llm
from ...logging import log
from ...types import Chunk
from .prompts import EXTRACTION_PROMPT, EXTRACTION_SYSTEM, GLEANING_PROMPT


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


def _format_prior(entities: list[EntityRecord], relations: list[RelationRecord]) -> tuple[str, str]:
    ent_lines = [f"- {e.name} ({e.type})" for e in entities] or ["(none)"]
    rel_lines = [f"- {r.source} -> {r.target}: {r.description[:80]}" for r in relations] or [
        "(none)"
    ]
    return "\n".join(ent_lines), "\n".join(rel_lines)


def _glean_once(
    chunk: Chunk,
    llm: LLM,
    prior_entities: list[EntityRecord],
    prior_relations: list[RelationRecord],
    max_input_chars: int = 4000,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    ent_str, rel_str = _format_prior(prior_entities, prior_relations)
    raw = llm.complete(
        GLEANING_PROMPT.format(
            prior_entities=ent_str,
            prior_relationships=rel_str,
            text=chunk.text[:max_input_chars],
        ),
        system=EXTRACTION_SYSTEM,
        json_mode=True,
        temperature=0.0,
        max_tokens=2048,
    )
    data = _safe_parse_json(raw)
    if not data:
        return [], []
    new_e: list[EntityRecord] = []
    seen = {e.name.lower() for e in prior_entities}
    for e in data.get("entities", []) or []:
        name = str(e.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        new_e.append(
            EntityRecord(
                name=name,
                type=str(e.get("type", "OTHER")).strip().upper(),
                description=str(e.get("description", "")).strip(),
                source_chunks=[chunk.id],
            )
        )
    new_r: list[RelationRecord] = []
    rel_seen = {(r.source.lower(), r.target.lower()) for r in prior_relations}
    rel_seen |= {(t, s) for s, t in rel_seen}
    for r in data.get("relationships", []) or []:
        s = str(r.get("source", "")).strip()
        t = str(r.get("target", "")).strip()
        if not s or not t or s.lower() == t.lower():
            continue
        if (s.lower(), t.lower()) in rel_seen:
            continue
        try:
            weight = float(r.get("weight", 1) or 1)
        except (TypeError, ValueError):
            weight = 1.0
        new_r.append(
            RelationRecord(
                source=s,
                target=t,
                description=str(r.get("description", "")).strip(),
                weight=weight,
                source_chunks=[chunk.id],
            )
        )
    return new_e, new_r


def extract_with_gleaning(
    chunk: Chunk,
    llm: LLM,
    *,
    max_passes: int = 1,
    max_input_chars: int = 4000,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    """Initial extraction + up to (max_passes - 1) gleaning passes per GraphRAG §2.2.

    `max_passes=1` is the legacy single-pass behavior (default).
    """
    entities, relations = extract_from_chunk(chunk, llm, max_input_chars=max_input_chars)
    for pass_idx in range(2, max_passes + 1):
        new_e, new_r = _glean_once(chunk, llm, entities, relations, max_input_chars=max_input_chars)
        if not new_e and not new_r:
            log.info("graphrag.glean.converged", chunk_id=chunk.id, pass_=pass_idx)
            break
        entities.extend(new_e)
        relations.extend(new_r)
    return entities, relations


def extract_all(
    chunks: list[Chunk],
    llm: LLM | None = None,
    workers: int = 4,
    *,
    gleaning_passes: int = 1,
) -> tuple[list[EntityRecord], list[RelationRecord]]:
    llm = llm or build_llm()
    log.info(
        "graphrag.extract.start",
        n_chunks=len(chunks),
        workers=workers,
        gleaning_passes=gleaning_passes,
    )
    all_e: list[EntityRecord] = []
    all_r: list[RelationRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(extract_with_gleaning, c, llm, max_passes=gleaning_passes): c for c in chunks
        }
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
