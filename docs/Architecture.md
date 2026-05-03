# Architecture

## レイヤと責務

```
┌─────────────────────────────────────────────────────────┐
│  CLI (rag.cli) / UI (Streamlit)                         │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  Pipelines (rag.pipelines)                              │
│   - BaselineRAG, GraphRAGPipeline, ...                  │
│   - Pipeline Protocol: name + answer(query) -> Answer   │
└─────────────────────────────────────────────────────────┘
        │                        │
┌──────────────────┐   ┌─────────────────────────────────┐
│ Retrievers       │   │ Techniques (rag.techniques)      │
│  (Phase 2)       │   │  - graphrag/                     │
│  dense/bm25/RRF  │   │  - hipporag/, raptor/, ...       │
└──────────────────┘   └─────────────────────────────────┘
        │                        │
┌─────────────────────────────────────────────────────────┐
│  Vectorstores (rag.vectorstores)                        │
│   - QdrantStore, [Phase 2: Weaviate/LanceDB/pgvector]   │
│   - VectorStore Protocol: ensure/upsert/search/...      │
└─────────────────────────────────────────────────────────┘
                          │
┌──────────────────────┐ ┌──────────────────────────────┐
│ LLM (rag.llm)        │ │ Embedding (rag.embedding)    │
│ - OllamaLLM, VLLM    │ │ - OllamaEmbedding            │
│ - LLM Protocol       │ │ - Embedding Protocol         │
└──────────────────────┘ └──────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Data (rag.data)                                        │
│   - Sources: Wikipedia, web, [local files Phase 2]      │
│   - Chunking, ingest orchestrator                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Evaluation (rag.evaluation)                            │
│   - synth.py (Q+GT 生成)                                │
│   - ragas_eval.py (judge LLM 経由で 4 metric 計測)       │
│   - runner.py (run_pipeline_against_eval, 結果永続化)    │
└─────────────────────────────────────────────────────────┘
```

## 主要 Protocol

すべての差し替え点は `Protocol` で定義。新しい backend 追加は Protocol を満たすクラスを書くだけで済む。

| Protocol | 定義 | 実装例 |
|---|---|---|
| `LLM` | `complete(prompt, **) / acomplete(...)` | OllamaLLM, VLLMClient |
| `Embedding` | `embed(texts) / aembed(texts) / dim` | OllamaEmbedding |
| `VectorStore` | `ensure_collection / upsert / search / get_by_ids / count / reset` | QdrantStore |
| `Source` | `name / fetch(**)` → Iterator[Document] | WikipediaSource, WebSource |
| `Pipeline` | `name / answer(query)` → Answer | BaselineRAG, GraphRAGPipeline |
| `Retriever` (Phase 2) | `retrieve(query, k)` → list[RetrievedChunk] | DenseRetriever, BM25Retriever, HybridRetriever |
| `Reranker` (Phase 3) | `rerank(query, candidates, k)` → list[RetrievedChunk] | ColBERTReranker, BGEReranker |
| `Parser` (Phase 3) | `parse(path)` → Document | DoclingParser |

## 主要設計判断と理由

### 1. LlamaIndex を「部品の供給源」として使う(orchestration は自前)
- ✅ LlamaIndex の chunking, embeddings adapter, vector store wrapper は便利なので import OK
- ❌ LlamaIndex の `QueryEngine`, `IngestionPipeline`, `Settings` シングルトンは使わない
- 理由: 高層抽象に乗ると論文の細部 (gleaning loop, hierarchical leiden levels, helpfulness scoring 等) を表現できなくなる

### 2. 1 論文 = 1 ディレクトリ = 1 Pipeline class
- `papers/<name>_<id>/` (PDF + meta + 実装メモ)
- `src/rag/techniques/<name>/` (実装)
- `configs/pipelines/<name>.yaml` (パラメタ)
- 三者を `task new-paper` で同時に scaffold

### 3. Vector store の抽象化は最初から strict
- 現在は QdrantStore のみだが、`VectorStore` Protocol は最小 6 メソッドに絞り、新 backend 追加が単純な mapping 作業で済む
- Phase 2 の Multi-VDB (#4) はこの土台の上で 3 アダプタ追加するだけ

### 4. 評価は eval set と pipeline run を分離
- `task synth-eval` で生成した `experiments/eval_set.jsonl` を全 pipeline run が共有
- `task run --pipeline X` で pipeline 別 run dir に answers + Ragas score を保存
- `task compare` で複数 run の横並び
- これで「同じ質問で複数手法を比較」が定量的に取れる

### 5. CLI は Typer 一本、コマンドは lazy import
- `rag/cli.py` の各サブコマンドは関連モジュールを **関数内で import** (例: `def cmd_build_graph: from .techniques.graphrag.build import ...`)
- 重い deps (graspologic, leidenalg) は graphrag を呼ぶ時だけロード
- baseline 走るだけなら graphrag deps が無くても動作 (将来モジュラー split を簡単に)

### 6. ステップを増やさない原則 (Taskfile)
- 新しいタスクは 1 コマンドで完了する形でしか追加しない
- `task ingest -- --reset --source wiki --query X` のように **副作用 (reset) もフラグ**化して 1 コマンドに収める

### 7. 設定は `.env` (pydantic-settings) で 1 ファイル
- Hydra も dep にあるが Phase 1 は使っていない
- パイプライン固有 config (configs/pipelines/*.yaml) は CLI 引数で上書き可
- 環境変数は credential や URL 等の "デプロイ間で変わるもの" のみ

## ディレクトリレイアウト

```
rag-experiment-kit/
├── pyproject.toml          # uv で管理、optional extras 多め
├── Taskfile.yml            # ワンコマンド窓口
├── docker-compose.yml      # qdrant default, neo4j/phoenix optional
├── .env.example
├── configs/
│   └── pipelines/{baseline,graphrag,...}.yaml
├── src/rag/
│   ├── llm/{base,ollama,vllm}.py
│   ├── embedding/{base,ollama_embed}.py
│   ├── vectorstores/{base,qdrant_store}.py
│   ├── data/{chunking,ingest}.py
│   ├── data/sources/{base,wikipedia,web}.py
│   ├── pipelines/{base,baseline}.py
│   ├── techniques/graphrag/   # ★ 自前実装
│   ├── evaluation/{synth,ragas_eval,runner}.py
│   ├── scaffolding.py         # task new-paper の中身
│   ├── config.py types.py logging.py cli.py
├── ui/app.py                  # Streamlit
├── papers/<name>_<id>/        # PDF + meta + 実装メモ
├── notebooks/                 # jupytext .py
├── experiments/               # eval_set.jsonl + runs/<run_id>/
└── tests/
```
