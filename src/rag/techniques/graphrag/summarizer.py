from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import networkx as nx

from ...llm import LLM, build_llm
from ...logging import log
from .community import CommunityNode
from .extractor import _safe_parse_json
from .prompts import COMMUNITY_SUMMARY_PROMPT


@dataclass
class CommunityReport:
    community_id: int
    level: int
    title: str
    summary: str
    findings: list[str]
    nodes: list[str]


def summarize_communities(
    g: nx.Graph,
    communities: list[CommunityNode],
    llm: LLM | None = None,
    *,
    workers: int = 4,
    max_entities: int = 50,
    max_relationships: int = 80,
) -> list[CommunityReport]:
    llm = llm or build_llm()
    log.info("graphrag.summarize.start", n=len(communities))
    out: list[CommunityReport] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_summarize_one, g, c, llm, max_entities, max_relationships): c
            for c in communities
        }
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                rep = fut.result()
                if rep:
                    out.append(rep)
            except Exception as exc:
                log.warning(
                    "graphrag.summarize.error", community=c.community_id, error=str(exc)
                )
    log.info("graphrag.summarize.done", n=len(out))
    return out


def _summarize_one(
    g: nx.Graph,
    community: CommunityNode,
    llm: LLM,
    max_e: int,
    max_r: int,
) -> CommunityReport | None:
    nodes = community.nodes[:max_e]
    entity_lines = []
    for n in nodes:
        node = g.nodes.get(n, {})
        descs = node.get("descriptions") or []
        first = (descs[0] if descs else "")[:200]
        entity_lines.append(f"- {n} ({node.get('type', 'OTHER')}): {first}")
    entities_block = "\n".join(entity_lines)

    edge_lines: list[str] = []
    for u, v, d in g.subgraph(nodes).edges(data=True):
        if len(edge_lines) >= max_r:
            break
        descs = d.get("descriptions") or []
        first = (descs[0] if descs else "")[:200]
        edge_lines.append(
            f"- {u} ↔ {v}: {first} (weight={d.get('weight', 1):.1f})"
        )
    relationships_block = "\n".join(edge_lines) if edge_lines else "(none)"

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities=entities_block, relationships=relationships_block
    )
    raw = llm.complete(prompt, json_mode=True, temperature=0.0, max_tokens=1024)
    data = _safe_parse_json(raw)
    if not data:
        return None
    return CommunityReport(
        community_id=community.community_id,
        level=community.level,
        title=str(data.get("title", "")).strip() or f"community-{community.community_id}",
        summary=str(data.get("summary", "")).strip(),
        findings=[str(f).strip() for f in (data.get("findings") or [])],
        nodes=community.nodes,
    )
