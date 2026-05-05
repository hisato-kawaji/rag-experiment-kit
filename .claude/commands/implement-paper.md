---
description: Read a paper PDF and populate src/rag/techniques/<name>/ to satisfy the Pipeline protocol.
argument-hint: <name>  (the technique name, e.g. hipporag)
---

You are implementing a RAG technique from a paper into this repository.

## Argument

`$ARGUMENTS` is the technique name (matches the directory name under
`src/rag/techniques/` and the prefix of `papers/<name>_<id>/`).

## Steps

1. **Locate the paper.** Find the paper directory:
   `papers/$ARGUMENTS_*/` (the `*` is the arXiv id). Read:
   - `papers/$ARGUMENTS_*/paper.pdf` — the actual paper
   - `papers/$ARGUMENTS_*/meta.yaml` — current metadata
   - `papers/$ARGUMENTS_*/NOTES.md` — current notes

2. **Read the GraphRAG implementation as a reference for code style and module
   decomposition.** Look at `src/rag/techniques/graphrag/` — note how it splits
   into `prompts.py` / `extractor.py` / `graph.py` / `community.py` /
   `summarizer.py` / `retriever.py` / `pipeline.py` / `build.py` / `store.py`.
   Mirror this decomposition where it fits the new technique. Don't force
   modules that don't apply.

3. **Identify the two algorithms in the paper:**
   - The **indexing-time** algorithm (what does the paper do at corpus build
     time? entity extraction? hierarchical summarization? clustering? prompt
     enrichment?).
   - The **query-time** algorithm (what does the paper do when a question
     comes in? graph traversal? PPR? tree descent? retriever fusion?).
   Write a 1-paragraph summary of each in `papers/$ARGUMENTS_*/NOTES.md`
   under `## Indexing` and `## Query`.

4. **Populate the technique modules** under
   `src/rag/techniques/$ARGUMENTS/`:
   - `prompts.py` — every LLM prompt the algorithm needs, named in
     SCREAMING_SNAKE_CASE, with the exact f-string placeholders.
   - `<algorithm-specific>.py` — one module per substep
     (e.g. `extractor.py`, `cluster.py`, `tree.py`, `ranker.py`). Each
     function/class takes its inputs as plain dataclasses or stdlib types
     and returns plain dataclasses or stdlib types. No global state.
   - `pipeline.py` — the `XPipeline` class that conforms to
     `rag.pipelines.base.Pipeline`. It must have:
     - `name = "$ARGUMENTS"` (class attribute)
     - `__init__(self, *, top_k=5, **_)` that loads any persisted artifacts
     - `answer(self, query: str) -> Answer` that returns a real `Answer` or
       raises `NotImplementedError("step X not done")` for parts you
       intentionally defer.
   - `build.py` (if the technique has an indexing step that writes
     artifacts to disk, like graphrag does). Use the same `data/<name>/`
     output convention.
   - `store.py` (if there is anything to persist between build and query).
   - Add an `__init__.py` that exports the `XPipeline` class.

5. **Register the pipeline** in `src/rag/pipelines/__init__.py` —
   add a branch in `build_pipeline(name, **kwargs)`:
   ```python
   if name == "$ARGUMENTS":
       from ..techniques.$ARGUMENTS import XPipeline
       return XPipeline(**kwargs)
   ```

6. **If the technique has a build step**, add a CLI command in
   `src/rag/cli.py`:
   ```bash
   task build-$ARGUMENTS
   ```
   that calls the build entry point.

7. **Update `papers/$ARGUMENTS_*/meta.yaml`:**
   - `status: implemented` (or `partial` if step-3 returned NotImplementedError
     in places)
   - `key_ideas:` — 3-5 bullets, each one a single sentence summarizing one
     specific algorithmic insight (not generic praise)
   - `phase1_simplifications:` — every place you intentionally diverged
     from the paper (e.g. "swapped UMAP+GMM for plain GaussianMixture",
     "single-pass extraction without gleaning", "dense-only — no PPR yet")
   - `reading_map:` — `section_X.Y` -> `module_path.function_name` pairs
     pointing the reader from the paper to the code

8. **Sanity-check:**
   ```bash
   uv run rag query --pipeline $ARGUMENTS --q "test"
   ```
   This must not raise `ImportError` or `AttributeError`. A
   `NotImplementedError` from inside the pipeline is OK at this stage and
   tells you which substep to work on next.

   Then:
   ```bash
   uv run pytest -q
   uv run ruff check src/rag/techniques/$ARGUMENTS/
   uv run ruff format src/rag/techniques/$ARGUMENTS/
   ```

## Constraints

- Match the repo's house style: pure functions where possible, dataclasses
  for records, `Protocol`s for swappable parts, structured logging via
  `from ...logging import log`.
- Don't pull in heavy deps unannounced. If you need `umap-learn`,
  `scikit-learn`, `colbert-ai`, etc., add them to `pyproject.toml`'s
  `[project.optional-dependencies]` (a new extra named after the
  technique) and import them lazily inside the function that needs them.
- Don't introduce `LlamaIndex.QueryEngine` / `IngestionPipeline` /
  `Settings` singleton. We use LlamaIndex only as a parts library
  (chunking, embeddings adapter) when convenient.
- If a paper algorithm is too large to fit in this run, leave clearly
  scoped `NotImplementedError` raises with a comment saying which
  substep is deferred.

## Deliverable

When you finish, summarize what you produced:
- Files created / modified (paths only).
- Which paper sections each module corresponds to (1-line each).
- What you intentionally simplified.
- The exact command(s) the user should run to try it.
