from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ...embedding import build_embedding
from ...llm import LLM, build_llm
from ...logging import log
from ...tracing import traced_span
from ...types import Answer, Chunk, RetrievedChunk
from ...vectorstores import build_vectorstore
from .extractor import _safe_parse_json
from .prompts import GLOBAL_MAP_PROMPT, GLOBAL_REDUCE_PROMPT, LOCAL_PROMPT
from .store import load_artifacts
from .summarizer import CommunityReport


class GraphRAGRetriever:
    def __init__(self, artifacts_dir: Path, llm: LLM | None = None) -> None:
        self.graph, self.communities, self.reports = load_artifacts(artifacts_dir)
        self.llm = llm or build_llm()
        self.reports_by_id = {r.community_id: r for r in self.reports}
        # Prefer level 1 if present (smaller, more focused communities)
        levels = {r.level for r in self.reports}
        self.preferred_level = 1 if 1 in levels else (max(levels) if levels else 0)

    # ---------- global (map-reduce over community summaries) ----------
    def global_search(
        self,
        question: str,
        *,
        level: int | None = None,
        map_workers: int = 4,
        top_n_for_reduce: int = 8,
    ) -> Answer:
        chosen_level = self.preferred_level if level is None else level
        candidates = [r for r in self.reports if r.level == chosen_level] or self.reports
        log.info("graphrag.global.map", n=len(candidates), level=chosen_level)

        partials: list[tuple[int, CommunityReport, str]] = []
        with (
            traced_span(
                "graphrag.global.map",
                level=chosen_level,
                n_candidates=len(candidates),
                workers=map_workers,
            ),
            ThreadPoolExecutor(max_workers=map_workers) as ex,
        ):
            futures = {ex.submit(self._map_one, r, question): r for r in candidates}
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    h, ans = fut.result()
                    if h > 0 and ans:
                        partials.append((h, r, ans))
                except Exception as e:
                    log.warning("graphrag.global.map-error", cid=r.community_id, error=str(e))

        partials.sort(key=lambda t: -t[0])
        top = partials[:top_n_for_reduce]
        if not top:
            return Answer(
                text="I don't know based on the provided sources.",
                contexts=[],
                metadata={"pipeline": "graphrag/global", "n_helpful": 0},
            )

        partials_block = "\n\n".join(
            f"[community: {r.title} (helpfulness={h})]\n{ans}" for h, r, ans in top
        )
        final = self.llm.complete(
            GLOBAL_REDUCE_PROMPT.format(question=question, partials=partials_block),
            temperature=0.0,
            max_tokens=800,
        )

        contexts = [
            RetrievedChunk(
                chunk=Chunk(
                    id=f"community::{r.community_id}",
                    doc_id=f"community::{r.title}",
                    text=r.summary,
                    metadata={"helpfulness": h, "level": r.level, "findings": r.findings},
                ),
                score=h / 100.0,
            )
            for h, r, _ in top
        ]
        return Answer(
            text=final,
            contexts=contexts,
            metadata={
                "pipeline": "graphrag/global",
                "level": chosen_level,
                "n_helpful": len(top),
                "llm": self.llm.name,
            },
        )

    def _map_one(self, report: CommunityReport, question: str) -> tuple[int, str]:
        report_block = f"# {report.title}\n\n{report.summary}\n\n## Findings\n" + "\n".join(
            f"- {f}" for f in report.findings
        )
        with traced_span(
            "graphrag.global.map_one",
            community_id=report.community_id,
            level=report.level,
        ):
            raw = self.llm.complete(
                GLOBAL_MAP_PROMPT.format(report=report_block, question=question),
                json_mode=True,
                temperature=0.0,
                max_tokens=512,
            )
        data = _safe_parse_json(raw)
        if not data:
            return 0, ""
        try:
            h = int(data.get("helpfulness", 0) or 0)
        except (TypeError, ValueError):
            h = 0
        return max(0, min(100, h)), str(data.get("partial_answer", "") or "")

    # ---------- local (entity-anchored) ----------
    def local_search(
        self,
        question: str,
        *,
        top_entities: int = 5,
        n_neighborhood: int = 12,
        top_chunks: int = 5,
    ) -> Answer:
        anchors = self._anchor_entities(question, top_entities)
        if not anchors:
            log.info("graphrag.local.no-anchor", question=question[:60])
        neighborhood = self._expand_neighborhood(anchors, n_neighborhood)
        chunks = self._supporting_chunks(question, anchors | neighborhood, top_chunks)

        entities_block = (
            "\n".join(
                f"- {e} ({self.graph.nodes[e].get('type', 'OTHER')})"
                for e in anchors
                if e in self.graph
            )
            or "(none — falling back to similarity-only retrieval)"
        )

        nb_lines: list[str] = []
        for u in anchors:
            if u not in self.graph:
                continue
            for v in self.graph.neighbors(u):
                d = self.graph.edges[u, v]
                desc = (d.get("descriptions") or [""])[0][:150]
                nb_lines.append(f"- {u} ↔ {v}: {desc}")
                if len(nb_lines) >= n_neighborhood:
                    break
            if len(nb_lines) >= n_neighborhood:
                break
        neighborhood_block = "\n".join(nb_lines) or "(none)"
        chunks_block = "\n\n".join(f"[{c.chunk.doc_id}] {c.chunk.text}" for c in chunks) or "(none)"

        prompt = LOCAL_PROMPT.format(
            entities=entities_block,
            neighborhood=neighborhood_block,
            chunks=chunks_block,
            question=question,
        )
        text = self.llm.complete(prompt, temperature=0.0, max_tokens=800)
        return Answer(
            text=text,
            contexts=chunks,
            metadata={
                "pipeline": "graphrag/local",
                "anchors": list(anchors),
                "n_neighbors": len(neighborhood),
                "llm": self.llm.name,
            },
        )

    # ---------- helpers ----------
    def _anchor_entities(self, question: str, k: int) -> set[str]:
        """Heuristic: lowercased substring/token match of entity names against the question.

        Phase 2 will replace this with a dedicated entity-vector index for robust matching.
        """
        q = question.lower()
        scored: list[tuple[int, str]] = []
        for n in self.graph.nodes:
            ln = n.lower()
            if not ln:
                continue
            score = 0
            if ln in q:
                score = 4
            else:
                tokens = [t for t in ln.split() if len(t) > 3]
                score = sum(1 for t in tokens if t in q)
            if score > 0:
                scored.append((score, n))
        scored.sort(key=lambda t: -t[0])
        return {n for _, n in scored[:k]}

    def _expand_neighborhood(self, anchors: set[str], n: int) -> set[str]:
        out: set[str] = set()
        for a in anchors:
            if a not in self.graph:
                continue
            for v in self.graph.neighbors(a):
                out.add(v)
                if len(out) >= n:
                    return out
        return out

    def _supporting_chunks(self, question: str, ents: set[str], k: int) -> list[RetrievedChunk]:
        embedding = build_embedding()
        vs = build_vectorstore()
        qv = embedding.embed([question])[0]
        candidates = vs.search(qv, k=max(k * 4, 20))
        if not ents:
            return candidates[:k]
        ent_lower = {e.lower() for e in ents}

        def mention_count(text: str) -> int:
            tl = text.lower()
            return sum(1 for e in ent_lower if e in tl)

        candidates.sort(key=lambda rc: (-mention_count(rc.chunk.text), -rc.score))
        return candidates[:k]
