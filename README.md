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

## Prerequisites

- macOS / Linux with **8GB+ RAM** (16GB recommended for `llama3.1:8b`)
- **20GB free disk** (5GB for the LLM, ~3GB for embeddings + qdrant index, headroom for experiments)
- Python 3.11+ (3.12 used in CI/dev; managed via `.python-version` + uv)
- Docker (for Qdrant container)

## First-time setup

```bash
brew install ollama go-task            # `go-task` is optional — see "Without Taskfile" below
brew services start ollama             # or: ollama serve &
open -a Docker                         # start Docker Desktop, wait for it to be ready
```

Then in this repo:
```bash
cp .env.example .env                   # edit if you want a different model / vLLM
task setup                             # uv sync + docker compose up qdrant + ollama pull
```

`task setup` pulls `llama3.1:8b` (~5GB) + `nomic-embed-text` (~280MB) on first run — takes 5–10 min depending on bandwidth.

## Quickstart (10 min, small dataset, baseline only)

Smoke test the whole stack without waiting for GraphRAG indexing.

```bash
task ingest -- --reset --source wikipedia --query "Quantum computing" --limit 5
task query  -- --pipeline baseline --q "What is quantum entanglement?"
```

Expected: an English answer that cites `[wikipedia:Quantum computing]` with 5 retrieved contexts. ~30 sec ingest + ~10 sec query.

## Full demo (~40 min, baseline vs GraphRAG with eval)

The first end-to-end pass that verifies everything end-to-end.

```bash
# 1. Fresh ingest of 5 Wikipedia pages (~30 sec, ~130 chunks)
task ingest      -- --reset --source wikipedia --query "Quantum computing" --limit 5

# 2. Build the GraphRAG index — start small, this is the slow step
#    (entity extraction + community detection + per-community LLM summaries)
task build-graph -- --max-chunks 30 --levels 2     # 15–20 min on local 8B

# 3. Generate a small synth eval set from the ingested corpus (~30 sec)
task synth-eval  -- --n 5

# 4. Run baseline against the eval set (~5 min — answers + Ragas)
task run         -- --pipeline baseline

# 5. Run GraphRAG/global against the eval set (~10–15 min)
#    Slower because each query maps over all community summaries.
task run         -- --pipeline graphrag

# 6. Side-by-side Ragas comparison
task compare

# 7. Interactive inspector (browser auto-opens)
task ui
```

### Demo result snapshot

A representative run on the corpus above (5 questions, llama3.1:8b judge):

| metric              | baseline | graphrag/global | Δ                |
|---------------------|---------:|----------------:|-----------------:|
| answer_relevancy    |    0.859 |       **0.889** | +0.030           |
| context_precision   |    1.000 |           1.000 | tie              |
| context_recall      |    0.740 |       **1.000** | **+0.260**       |
| faithfulness        |      NaN |             NaN | (8B judge timeout — see [#1](https://github.com/hisato-kawaji/rag-experiment-kit/issues/1)) |
| elapsed             |      43s |            836s |                  |

GraphRAG global wins context_recall (+26pt) — community summaries cover related topics that flat retrieval misses. Cost: ~19× slower per query.

## Common one-off commands

```bash
# Single query, choose pipeline + mode
task query -- --pipeline baseline --q "..."
task query -- --pipeline graphrag --q "..."         # default mode = global
# graphrag mode = local (entity-anchored neighborhood, faster):
uv run rag query --pipeline graphrag --q "..." -p graphrag        # via direct CLI; mode flag = "-p" for pipeline name

# Verify settings resolved from .env
task -- info                                        # or: uv run rag info

# Reset Qdrant collection without re-installing
uv run python -c "from rag.vectorstores import build_vectorstore; build_vectorstore().reset()"

# Just the docker services
task docker-up                                       # qdrant only
task docker-up-all                                   # + neo4j (graph viz) + phoenix (tracing)
task docker-down
```

## Without `task` (Taskfile) installed

Every Taskfile target is a thin wrapper around `uv run rag <subcommand>`. If you don't have `go-task`:

| Taskfile               | Direct equivalent                                                  |
|------------------------|--------------------------------------------------------------------|
| `task setup`           | `uv sync --extra dev --extra notebooks` then `docker compose up -d qdrant` then `ollama pull llama3.1:8b nomic-embed-text` |
| `task ingest -- ...`   | `uv run rag ingest ...`                                            |
| `task build-graph -- ...` | `uv run rag build-graph ...`                                    |
| `task synth-eval -- ...` | `uv run rag synth-eval ...`                                      |
| `task run -- ...`      | `uv run rag run ...`                                               |
| `task query -- ...`    | `uv run rag query ...`                                             |
| `task compare`         | `uv run rag compare`                                               |
| `task ui`              | `uv run streamlit run ui/app.py`                                   |
| `task new-paper -- ...`| `uv run rag new-paper ...`                                         |

`uv run rag --help` lists everything.

## Adding a new paper

```bash
task new-paper -- --id 2405.14831 --name hipporag
# Scaffolds:
#   papers/hipporag_2405.14831/{paper.pdf, meta.yaml, NOTES.md}     # auto-DLs PDF
#   src/rag/techniques/hipporag/{__init__.py, pipeline.py}
#   configs/pipelines/hipporag.yaml
```

Then implement `src/rag/techniques/hipporag/` to satisfy the `Pipeline` Protocol (`name`, `answer(query) -> Answer`). Step-by-step recipe in [docs/Adding-a-Paper.md](docs/Adding-a-Paper.md).

Phase 2 will half-automate this via the `/implement-paper` Claude slash command ([#7](https://github.com/hisato-kawaji/rag-experiment-kit/issues/7)).

## Layout

```
configs/        # YAML pipeline configs (currently CLI-driven; Hydra wiring deferred to Phase 2)
src/rag/
  llm/          # Ollama / vLLM adapters behind a Protocol
  embedding/    # Embedding adapters
  vectorstores/ # VectorStore Protocol + Qdrant adapter
  data/         # sources/, chunking, ingest orchestrator
  pipelines/    # baseline.py — assemble llm + retriever + prompt
  techniques/   # ★ self-built advanced methods. graphrag/ is the first.
  evaluation/   # synth.py + ragas_eval.py + runner.py
  scaffolding.py    # `task new-paper` implementation
  cli.py        # `rag` Typer entrypoint
ui/app.py       # Streamlit inspector
papers/         # paper PDFs + meta.yaml + impl notes (1 paper = 1 dir)
notebooks/      # jupytext-synced .py — open in Colab as-is
experiments/    # eval_set.jsonl + runs/<run_id>/{answers.jsonl, summary.json}
docs/           # Roadmap / Architecture / Working Principles / Adding-a-Paper
```

## Troubleshooting

| Symptom | Check / fix |
|---|---|
| `Error: could not connect to ollama server` | `curl http://localhost:11434/api/tags`; if down: `ollama serve &` (the brew service can sometimes fail to bind) |
| `Cannot connect to the Docker daemon` | Open Docker Desktop, wait until status icon is "running", then `task docker-up` |
| `qdrant_client ... AttributeError 'search'` | Ancient bug, fixed in 9b402f5 — pull latest `main` |
| Ragas `faithfulness: NaN` | Judge LLM (local 8B) is timing out on multi-step verification. Use fewer eval questions, or wait for [#1 swappable judge](https://github.com/hisato-kawaji/rag-experiment-kit/issues/1) to use Anthropic/OpenAI as judge |
| GraphRAG build appears stuck | It's slow. Verify activity: `ps aux \| grep ollama` (ollama runner should be active), and `tail -f` your tee'd log if you piped one. Each chunk needs 1 LLM call; 30 chunks × ~8 sec ≈ 4 min for extraction alone |
| GraphRAG/global query takes 3+ min | Expected on 8B local. Each query maps over N communities (often 20–80). Use `--pipeline graphrag` with `mode=local` for entity-anchored query (~10× faster). Phase 2 [#2](https://github.com/hisato-kawaji/rag-experiment-kit/issues/2) improves local quality |
| `wikipediaapi` fetches unrelated linked pages (e.g. "16-bit computing") | Known: `_traverse` walks links alphabetically. Workaround: pick a more specific seed or `--limit 1`. Fixed in [#3](https://github.com/hisato-kawaji/rag-experiment-kit/issues/3) (relevance filter) |
| `task: command not found` | Use `uv run rag <cmd>` directly — see "Without Taskfile" table above |

## What's next

- [Phase 2 milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/1) (7 issues): swappable judge LLM, entity vector index, multi-VDB adapters, MLflow, Phoenix tracing, `/implement-paper` slash command
- [Phase 3 milestone](https://github.com/hisato-kawaji/rag-experiment-kit/milestone/2) (8 issues): Contextual Retrieval, HippoRAG, RAPTOR, ColBERT, hybrid retrieval (BM25+dense+RRF), expanded eval, Docling for PDFs, comparison dashboard

Recommended starting issue: [#1 Swappable Ragas judge LLM](https://github.com/hisato-kawaji/rag-experiment-kit/issues/1) — fastest path to stable eval numbers.
