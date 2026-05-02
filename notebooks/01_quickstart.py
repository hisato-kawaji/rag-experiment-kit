# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # RAG dev env — quickstart
#
# End-to-end walkthrough: ingest → graph build → query (baseline vs GraphRAG) → eval.
#
# **Local**: assumes `task setup` already ran (qdrant up, ollama models pulled).
# **Colab**: uncomment the install cell, then point env vars to a remote Qdrant
# (Qdrant Cloud free tier or a Cloudflare Tunnel to your local instance).

# %% [markdown]
# ## 0. (Colab only) install + env

# %%
# !pip install -q ollama qdrant-client trafilatura wikipedia-api networkx \
#   python-igraph leidenalg ragas datasets langchain-ollama \
#   llama-index-core pydantic-settings structlog typer rich httpx \
#   pyyaml pandas
# !pip install -q -e git+https://github.com/<you>/rag.git#egg=rag

# %%
import os
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from rag.config import Settings
Settings().model_dump()

# %% [markdown]
# ## 1. Ingest

# %%
from rag.data import ingest

stats = ingest("wikipedia", {"query": "Quantum computing", "limit": 10})
stats

# %% [markdown]
# ## 2. Build the GraphRAG index

# %%
from pathlib import Path
from rag.techniques.graphrag import build_graphrag_index

result = build_graphrag_index(out_dir=Path("data/graphrag"), max_chunks=80)
result

# %% [markdown]
# ## 3. Query — baseline vs GraphRAG

# %%
from rag.pipelines import build_pipeline

baseline = build_pipeline("baseline", top_k=5)
graphrag_global = build_pipeline("graphrag", mode="global")
graphrag_local = build_pipeline("graphrag", mode="local", top_k=5)

q = "What are the main approaches to quantum error correction?"

# %%
print("=== BASELINE ===")
print(baseline.answer(q).text)

# %%
print("=== GRAPHRAG / GLOBAL ===")
print(graphrag_global.answer(q).text)

# %%
print("=== GRAPHRAG / LOCAL ===")
print(graphrag_local.answer(q).text)

# %% [markdown]
# ## 4. Evaluate (small synth set)

# %%
import json
from rag.evaluation import generate_eval_set
from rag.evaluation.runner import run_pipeline_against_eval, latest_runs_summary

pairs = generate_eval_set(n=5)
eval_path = Path("experiments/eval_set.jsonl")
eval_path.parent.mkdir(parents=True, exist_ok=True)
eval_path.write_text("\n".join(json.dumps(p) for p in pairs))

run_pipeline_against_eval(
    pipeline=baseline,
    eval_set=eval_path,
    out_dir=Path("experiments/runs"),
    skip_eval=False,
)

run_pipeline_against_eval(
    pipeline=graphrag_global,
    eval_set=eval_path,
    out_dir=Path("experiments/runs"),
    skip_eval=False,
)

# %%
latest_runs_summary(Path("experiments/runs"))
