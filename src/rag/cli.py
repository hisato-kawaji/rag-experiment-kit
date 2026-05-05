"""Single CLI entrypoint exposed as `rag` (project script) and `python -m rag.cli`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import Settings
from .logging import log, setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@app.callback()
def _root(verbose: bool = typer.Option(False, "-v", "--verbose")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


# ---------- ingest ----------
@app.command("ingest")
def cmd_ingest(
    source: str = typer.Option("wikipedia", "--source", "-s"),
    query: str = typer.Option("Artificial intelligence", "--query", "-q"),
    limit: int = typer.Option(20, "--limit", "-n"),
    urls: Optional[str] = typer.Option(None, "--urls", help="comma-separated URLs (web source)"),
    chunk_size: int = typer.Option(800),
    chunk_overlap: int = typer.Option(100),
    reset: bool = typer.Option(False, "--reset", help="Drop collection first"),
    related_only: bool = typer.Option(
        False,
        "--related-only",
        help="(wikipedia) drop link targets whose summary cosine-sim to seed < --threshold",
    ),
    threshold: float = typer.Option(
        0.4, "--threshold", help="(wikipedia) cosine-sim cutoff for --related-only"
    ),
) -> None:
    """Pull docs → chunk → embed → upsert into Qdrant."""
    from .data import ingest as _ingest
    from .vectorstores import build_vectorstore

    if reset:
        vs = build_vectorstore()
        vs.reset()
        console.print(f"[yellow]reset[/yellow] {vs.name}")

    if source == "wikipedia":
        kwargs = {"query": query, "limit": limit}
    elif source == "web":
        if not urls:
            raise typer.BadParameter("--urls required for web source")
        kwargs = {"urls": [u.strip() for u in urls.split(",") if u.strip()]}
    else:
        raise typer.BadParameter(f"Unknown source: {source}")

    stats = _ingest(
        source,
        kwargs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        related_only=related_only,
        relevance_threshold=threshold,
    )
    console.print(f"[green]ingested[/green] docs={stats.docs} chunks={stats.chunks}")


# ---------- build-graph ----------
@app.command("build-graph")
def cmd_build_graph(
    out_dir: Path = typer.Option(Path("data/graphrag"), "--out"),
    max_chunks: int = typer.Option(0, "--max-chunks", help="0 = use all"),
    leiden_max_levels: int = typer.Option(3, "--levels"),
    gleaning_passes: int = typer.Option(
        1,
        "--gleaning-passes",
        help="GraphRAG §2.2 gleaning loop passes. 1 = single-pass (default)",
    ),
) -> None:
    """Build the GraphRAG index from chunks already in the vector store."""
    from .techniques.graphrag.build import build_graphrag_index

    out = build_graphrag_index(
        out_dir=out_dir,
        max_chunks=max_chunks or None,
        leiden_max_levels=leiden_max_levels,
        gleaning_passes=gleaning_passes,
    )
    console.print(
        f"[green]graph built[/green] entities={out.n_entities} "
        f"relations={out.n_relations} communities={out.n_communities} → {out_dir}"
    )


# ---------- synth-eval ----------
@app.command("synth-eval")
def cmd_synth_eval(
    n: int = typer.Option(30, "--n"),
    out: Path = typer.Option(Path("experiments/eval_set.jsonl"), "--out"),
) -> None:
    """Generate synthetic Q+reference-context pairs from the ingested corpus."""
    from .evaluation.synth import generate_eval_set

    pairs = generate_eval_set(n=n)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    console.print(f"[green]wrote[/green] {len(pairs)} eval pairs → {out}")


# ---------- run ----------
@app.command("run")
def cmd_run(
    pipeline: str = typer.Option("baseline", "--pipeline", "-p"),
    eval_set: Path = typer.Option(Path("experiments/eval_set.jsonl"), "--eval-set"),
    out_dir: Path = typer.Option(Path("experiments/runs"), "--out"),
    top_k: int = typer.Option(5, "--top-k"),
    skip_eval: bool = typer.Option(False, "--skip-eval"),
) -> None:
    """Run a pipeline against the eval set; persist run outputs + Ragas scores."""
    from .pipelines import build_pipeline
    from .evaluation.runner import run_pipeline_against_eval

    pipe = build_pipeline(pipeline, top_k=top_k)
    run_id = run_pipeline_against_eval(
        pipeline=pipe, eval_set=eval_set, out_dir=out_dir, skip_eval=skip_eval
    )
    console.print(f"[green]run[/green] {pipeline} → {out_dir / run_id}")


# ---------- compare ----------
@app.command("compare")
def cmd_compare(
    runs_dir: Path = typer.Option(Path("experiments/runs"), "--runs"),
) -> None:
    """Print a side-by-side metric table across the most recent runs."""
    from .evaluation.runner import latest_runs_summary

    rows = latest_runs_summary(runs_dir)
    if not rows:
        console.print("[yellow]no runs found[/yellow]")
        return
    table = Table(
        "pipeline",
        "run_id",
        "n",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    for r in rows:
        table.add_row(
            r["pipeline"],
            r["run_id"],
            str(r["n"]),
            f"{r.get('faithfulness', float('nan')):.3f}",
            f"{r.get('answer_relevancy', float('nan')):.3f}",
            f"{r.get('context_precision', float('nan')):.3f}",
            f"{r.get('context_recall', float('nan')):.3f}",
        )
    console.print(table)


# ---------- query ----------
@app.command("query")
def cmd_query(
    q: str = typer.Option(..., "--q"),
    pipeline: str = typer.Option("baseline", "--pipeline", "-p"),
    top_k: int = typer.Option(5, "--top-k"),
) -> None:
    """One-off query — print answer + retrieved contexts."""
    from .pipelines import build_pipeline

    pipe = build_pipeline(pipeline, top_k=top_k)
    answer = pipe.answer(q)
    console.rule("[bold cyan]Answer")
    console.print(answer.text)
    console.rule("[bold cyan]Contexts")
    for i, ctx in enumerate(answer.contexts, 1):
        console.print(
            f"[dim]{i}. score={ctx.score:.3f} doc={ctx.chunk.doc_id} chunk={ctx.chunk.id}[/dim]"
        )
        console.print(ctx.chunk.text[:300] + ("…" if len(ctx.chunk.text) > 300 else ""))
        console.print()


# ---------- new-paper ----------
@app.command("new-paper")
def cmd_new_paper(
    arxiv_id: str = typer.Option(..., "--id"),
    name: str = typer.Option(..., "--name"),
) -> None:
    """Scaffold papers/<name>_<id>/ + src/rag/techniques/<name>/ + configs/pipelines/<name>.yaml."""
    from .scaffolding import scaffold_paper

    paths = scaffold_paper(arxiv_id=arxiv_id, name=name)
    console.print("[green]scaffolded:[/green]")
    for p in paths:
        console.print(f"  {p}")


# ---------- info ----------
@app.command("info")
def cmd_info() -> None:
    """Print resolved settings (sanity check)."""
    s = Settings()
    table = Table("key", "value")
    for k, v in s.model_dump().items():
        if "key" in k.lower() or "password" in k.lower():
            v = "***" if v else ""
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
