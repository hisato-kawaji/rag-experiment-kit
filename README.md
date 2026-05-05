# rag-experiment-kit

Paper-driven RAG research environment. One-command flows for ingestion, indexing, evaluation, and side-by-side technique comparison.

📚 **Docs**: [Home](docs/Home.md) · [Roadmap](docs/Roadmap.md) · [Architecture](docs/Architecture.md) · [Working Principles](docs/Working-Principles.md) · [Adding a paper](docs/Adding-a-Paper.md)
🗺️ **Plan**: [Phase 2 milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/1) · [Phase 3 milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/2) · [All issues](https://github.com/hisato-kawaji/rag-experiment-kit/issues)

## What's here (Phase 1)

- **LLM / embed**: Ollama default (`llama3.1:8b`, `nomic-embed-text`); vLLM (OpenAI-compatible) as drop-in alt via `.env`
- **Vector store**: Qdrant (Docker). Abstract `VectorStore` interface so Phase 2 can plug in Weaviate / LanceDB / pgvector
- **Graph**: NetworkX in-memory + leidenalg for hierarchical Leiden community detection. Optional Neo4j sink for browser visualization
- **Data**: Wikipedia + trafilatura web crawler (general web text)
- **Techniques**: dense-retrieval baseline, **self-built GraphRAG** (Edge et al. arXiv:2404.16130) under `src/rag/techniques/graphrag/`
- **Eval**: Ragas (faithfulness, answer relevancy, context precision/recall) + optional Phoenix tracing
- **UI**: Streamlit query inspector with pipeline switcher and side-by-side compare

## First-time setup

Install once:
```bash
brew install ollama go-task jupytext
brew services start ollama          # or: ollama serve &
open -a Docker                      # start Docker Desktop
```

Then in this repo:
```bash
cp .env.example .env
task setup     # uv sync + docker compose up qdrant + ollama pull
```

## Daily flow

```bash
task ingest -- --source wikipedia --query "Quantum computing" --limit 30
task build-graph                                 # GraphRAG indexing
task synth-eval -- --n 30                        # synth QA from corpus
task run -- --pipeline baseline                  # eval baseline
task run -- --pipeline graphrag                  # eval graphrag
task compare                                     # Ragas diff report
task ui                                          # Streamlit inspector
```

One-off query:
```bash
task query -- --pipeline graphrag --q "What are the main approaches to quantum error correction?"
```

## Switching the Ragas judge LLM

Generation pipelines always use `OLLAMA_MODEL_LLM`. The Ragas **judge** (the
LLM that scores faithfulness / context precision / etc.) can be swapped
independently — useful when llama3.1:8b times out on judge prompts and metrics
come back as `NaN`.

```bash
# Anthropic (recommended for stable faithfulness):
uv sync --extra judge-anthropic
# .env:
#   RAGAS_JUDGE_BACKEND=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
#   RAGAS_JUDGE_MODEL=claude-haiku-4-5-20251001   # optional; empty -> default

# OpenAI:
uv sync --extra judge-openai
# .env:
#   RAGAS_JUDGE_BACKEND=openai
#   OPENAI_API_KEY=sk-...
#   RAGAS_JUDGE_MODEL=gpt-4o-mini

# Back to Ollama (default):
#   RAGAS_JUDGE_BACKEND=ollama
```

`task run -- --pipeline baseline` reads `RAGAS_JUDGE_BACKEND` at evaluation
time, so no code changes are needed.

## Layout

```
configs/        # Hydra YAML — pipeline = 1 file
src/rag/
  llm/          # Ollama / vLLM adapters behind a Protocol
  embedding/    # Embedding adapters
  vectorstores/ # VectorStore Protocol + Qdrant adapter
  data/         # sources/, parsers/, synthesizers/, chunking
  retrievers/   # dense / hybrid (Phase 2)
  techniques/   # ★ self-built advanced methods. graphrag/ is the first.
  pipelines/    # baseline.py, graphrag.py — assemble the above
  evaluation/   # Ragas wrapper
  cli.py        # `rag` Typer entrypoint
ui/app.py       # Streamlit inspector
papers/         # paper PDFs + meta.yaml + impl notes (1 paper = 1 dir)
notebooks/      # jupytext-synced .py — open in Colab as-is
experiments/    # MLflow-style run dirs (Phase 2 adds MLflow)
```

## Adding a new paper

```bash
task new-paper -- --id 2410.05229 --name hipporag
# scaffolds papers/hipporag_2410.05229/{paper.pdf, meta.yaml, NOTES.md}
# scaffolds src/rag/techniques/hipporag/ with the standard Technique skeleton
# scaffolds configs/pipelines/hipporag.yaml
```

## Phase 2/3 backlog

- Multi vector DB adapters (Weaviate, LanceDB, pgvector)
- `/implement-paper` Claude slash command
- MLflow run tracking + comparison dashboard
- Additional techniques: HippoRAG, Contextual Retrieval, RAPTOR, ColBERT-PLAID
- Docling-based PDF/HTML parsers (extra: `uv sync --extra docling`)
