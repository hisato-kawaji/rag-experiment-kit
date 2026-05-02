# GraphRAG (Edge et al., 2024) — implementation notes

PDF: [./paper.pdf](./paper.pdf)
Module: `src/rag/techniques/graphrag/`

## Pipeline summary
1. `extractor.extract_all` — per-chunk LLM extraction of entities + relationships, JSON output
2. `graph.build_graph` — dedupe by lowercased name, accumulate edge weights and descriptions
3. `community.hierarchical_leiden` — recursive Leiden; level 0 partitions the full graph, level k partitions each level-(k-1) community
4. `summarizer.summarize_communities` — LLM writes `{title, summary, findings}` per community
5. `retriever.GraphRAGRetriever` — `global_search` (map-reduce over summaries) and `local_search` (anchor entities → neighborhood → chunks)

## Where this implementation deviates from the paper
- **No gleaning loop** — paper iterates "are there more entities?" prompts to push recall. We do a single pass.
- **Naive entity anchoring** — substring/token match against entity names. Paper uses an entity vector index.
- **LLM self-rated helpfulness** — paper applies softer ranking; we trust the model's 0-100 score.

These are intentional Phase 1 trade-offs to keep the first end-to-end run cheap. See `meta.yaml: phase2_followups`.

## Reading map (paper section → code file)
- §2.1 Source documents → text chunks → `src/rag/data/chunking.py`
- §2.2 Element instance extraction → `extractor.py`
- §2.3 Element summaries → currently inline in `graph.py` (descriptions list per node/edge)
- §2.4 Graph communities → `community.py`
- §2.5 Community summaries → `summarizer.py`
- §3 Global query → `retriever.global_search`
- §3 (extension) Local query → `retriever.local_search`
