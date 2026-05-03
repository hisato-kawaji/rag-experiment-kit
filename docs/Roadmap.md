# Roadmap

## Phase 1 — Foundation ✅ (完了 2026-05-03)

| 区分 | 内容 |
|---|---|
| LLM | Ollama (`llama3.1:8b` default) / vLLM (OpenAI 互換) を Protocol 経由で交換可能 |
| Embedding | `nomic-embed-text` via Ollama |
| Vector DB | Qdrant (Docker) — 抽象 IF は最初から strict |
| Graph | NetworkX in-memory + leidenalg で hierarchical Leiden community detection |
| 取得元 | Wikipedia API + trafilatura (任意 web URL) |
| 自前手法 | **GraphRAG** (Edge et al. arXiv:2404.16130) — extractor / graph / community / summarizer / global+local retriever |
| Baseline | dense retrieval + LLM answer |
| 評価 | Ragas (faithfulness / answer_relevancy / context_precision / context_recall) + synth QA generator |
| UI | Streamlit query inspector (side-by-side compare) |
| ワンコマンド | Taskfile: `setup, ingest, build-graph, synth-eval, run, compare, ui, new-paper` |
| 論文管理 | `papers/<name>_<id>/{paper.pdf, meta.yaml, NOTES.md}` |
| ノートブック | jupytext で .py ↔ .ipynb (Colab 互換) |

**初回デモ結果** (Wikipedia "Quantum computing", 5 質問):

| metric | baseline | graphrag/global |
|---|---|---|
| answer_relevancy | 0.859 | **0.889** |
| context_precision | 1.000 | 1.000 |
| context_recall | 0.740 | **1.000** |
| elapsed | 43s | 836s |

---

## Phase 2 — Robust + Extensible (in progress)

[Milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/1) — 7 issues

Phase 1 で実害が出た所を直し、観測可能性を上げる。

### P1 (優先実装) — 「困った」を直す
- **#1** [Swappable Ragas judge LLM](https://github.com/hisato-kawaji/rag-experiment-kit/issues/1) — `faithfulness` の NaN 解消 (judge を Anthropic / OpenAI 切替)
- **#2** [Entity vector index for GraphRAG local](https://github.com/hisato-kawaji/rag-experiment-kit/issues/2) — substring 一致を embedding 検索に置換
- **#3** [Wikipedia 関連度フィルタ + GraphRAG gleaning](https://github.com/hisato-kawaji/rag-experiment-kit/issues/3) — 無関係 link 除外 + extraction recall +20%

### P2 (基盤強化)
- **#4** [Multi-VDB アダプタ (Weaviate / LanceDB / pgvector)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/4)
- **#5** [MLflow run tracking](https://github.com/hisato-kawaji/rag-experiment-kit/issues/5)
- **#6** [Phoenix tracing](https://github.com/hisato-kawaji/rag-experiment-kit/issues/6)
- **#7** [`/implement-paper` Claude slash command](https://github.com/hisato-kawaji/rag-experiment-kit/issues/7)

**推奨着手順**: #1 → #2 → #3 を一気にやる ("修復週") → 残りは並列可

---

## Phase 3 — Research Velocity

[Milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/2) — 8 issues

新手法・追加 retrieval・eval 拡張・観察 UI で「論文 1 本を 1 日で取り込み定量検証する」体制を作る。

### tech-paper (新しい論文の自前実装)
- **#8** [Contextual Retrieval (Anthropic 2024)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/8) — P1。最もコスパ良い
- **#9** [HippoRAG (arXiv:2405.14831)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/9) — P2。Personalized PageRank
- **#10** [RAPTOR (arXiv:2401.18059)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/10) — P2。階層 summary tree
- **#11** [ColBERT late interaction (PLAID)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/11) — P3。reranker 用途

### infra / 拡張
- **#12** [Hybrid retrieval (BM25 + dense + reranker, RRF)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/12) — P2
- **#13** [Synth eval 拡張 (Ragas TestsetGenerator, multi-hop, abstract)](https://github.com/hisato-kawaji/rag-experiment-kit/issues/13) — P2
- **#14** [Docling for PDF/HTML/DOCX](https://github.com/hisato-kawaji/rag-experiment-kit/issues/14) — P3
- **#15** [Streamlit run comparison dashboard](https://github.com/hisato-kawaji/rag-experiment-kit/issues/15) — P3

**推奨着手順**: #8 (Contextual) → #12 (Hybrid) → #13 (eval 拡張) で「ベースライン強化 + 評価精度向上」を先にやる → その後 #9/#10/#11 のテクニック比較に進む

---

## Phase 4 以降 (まだ issue 化していない検討事項)

- Multi-modal RAG (image / table 質問対応)
- 自前学習 reranker / embedding fine-tuning ループ
- Production deployment テンプレ (Docker compose → Kubernetes)
- 中規模日本語コーパスでの動作検証 + 日本語 embedding (multilingual-e5, ruri-large)
- Feedback loop: ユーザー評価 (👍/👎) → preference dataset → DPO で fine-tune
