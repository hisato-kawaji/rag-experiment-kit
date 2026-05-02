from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging import log
from ..pipelines import Pipeline
from .ragas_eval import evaluate_ragas


def run_pipeline_against_eval(
    *,
    pipeline: Pipeline,
    eval_set: Path,
    out_dir: Path,
    skip_eval: bool = False,
) -> str:
    if not eval_set.exists():
        raise FileNotFoundError(
            f"Eval set not found: {eval_set}. Run `task synth-eval` first."
        )
    eval_rows = [
        json.loads(line) for line in eval_set.read_text().splitlines() if line.strip()
    ]
    log.info("run.start", pipeline=pipeline.name, n=len(eval_rows))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{pipeline.name.replace('/', '_')}-{ts}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for i, row in enumerate(eval_rows, 1):
        q = row["question"]
        try:
            ans = pipeline.answer(q)
        except Exception as e:
            log.warning("run.answer-error", q=q[:60], error=str(e))
            continue
        results.append(
            {
                "question": q,
                "ground_truth": row.get("ground_truth", ""),
                "answer": ans.text,
                "contexts": [c.chunk.text for c in ans.contexts],
                "context_doc_ids": [c.chunk.doc_id for c in ans.contexts],
                "metadata": ans.metadata,
            }
        )
        if i % 5 == 0:
            log.info(
                "run.progress",
                done=i,
                total=len(eval_rows),
                elapsed=f"{time.time() - t0:.1f}s",
            )

    (run_dir / "answers.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "pipeline": pipeline.name,
        "n": len(results),
        "elapsed_sec": round(time.time() - t0, 2),
        "ts": ts,
    }
    if not skip_eval and results:
        try:
            scores = evaluate_ragas(results)
            summary.update(scores)
        except Exception as e:
            log.warning("run.eval-error", error=str(e))
            summary["eval_error"] = str(e)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    log.info("run.done", **{k: v for k, v in summary.items() if k != "ts"})
    return run_id


def latest_runs_summary(runs_dir: Path) -> list[dict[str, Any]]:
    if not runs_dir.exists():
        return []
    entries: list[dict[str, Any]] = []
    for d in sorted(runs_dir.iterdir()):
        f = d / "summary.json"
        if not f.exists():
            continue
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            continue
    return entries
