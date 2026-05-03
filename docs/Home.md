## rag-experiment-kit

論文駆動型 RAG 研究/開発環境。**1 論文 = 1 ブランチ = 1 config = 1 module** を貫き、最新の RAG 手法を素早く取り込み、Ragas で定量検証するためのキット。

## このリポジトリの設計思想

1. **論文 → 実装までのリードタイム最小化**: `task new-paper` で skeleton を 30 秒で生成、`/implement-paper` Claude slash command (Phase 2) で論文 PDF からスケルトン充填まで半自動
2. **フレームワーク非依存の薄さ**: LlamaIndex は "部品の供給源" として使うが、orchestration は自前 Python。`Pipeline` / `VectorStore` / `LLM` Protocol を strict に切ってあるのでロックインしない
3. **常時評価**: 各 pipeline 変更を Ragas (faithfulness, answer relevancy, context precision/recall) で測定、`task compare` で横並び
4. **観察可能性**: Phoenix トレース (Phase 2) で各 LLM call / retrieval の詳細を可視化
5. **Multi-VDB**: Qdrant がデフォルトだが、Phase 2 で Weaviate / LanceDB / pgvector に同データを並列投入し性能比較

## 始めかた

```bash
brew install ollama go-task
brew services start ollama
open -a Docker
git clone https://github.com/hisato-kawaji/rag-experiment-kit
cd rag-experiment-kit
cp .env.example .env
task setup
task ingest -- --source wikipedia --query "Quantum computing" --limit 10
task build-graph -- --max-chunks 60
task synth-eval -- --n 10
task run -- --pipeline baseline
task run -- --pipeline graphrag
task compare
task ui
```

詳細は [README](https://github.com/hisato-kawaji/rag-experiment-kit#readme) を参照。

## ナビゲーション

- **[[Roadmap]]** — Phase 1 完成内容 + Phase 2/3 の全 issue 一覧
- **[[Architecture]]** — システム構成、レイヤ責務、主要設計判断
- **[[Working-Principles]]** — 作業方針 (PR / commit / 実装ルール)
- **[[Adding-a-Paper]]** — 新しい論文を取り込む手順 (1 論文 1 module の作り方)

## 状態スナップショット (2026-05-03)

- Phase 1 ✅ 完了 (baseline + 自前 GraphRAG + Ragas + Streamlit + Wikipedia ingest)
- 初回 demo 実行: baseline vs graphrag/global で **context_recall 0.74 → 1.00 (+0.26)**, **answer_relevancy 0.86 → 0.89** を確認
- Phase 2 = 7 issues (#1-#7) — judge LLM swap, entity vector index, multi-VDB ほか
- Phase 3 = 8 issues (#8-#15) — Contextual Retrieval, HippoRAG, RAPTOR, hybrid, ほか
