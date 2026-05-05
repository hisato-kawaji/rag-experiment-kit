"""Optional MLflow run tracking.

A thin wrapper around mlflow.start_run / log_params / log_metrics /
log_artifact. Turned on by `MLFLOW_ENABLED=true` in .env. Importing mlflow
is deferred so plain runs (default settings) don't pay the import cost.

The CLI always writes its own `experiments/runs/<run_id>/` directory with
answers.jsonl + summary.json — MLflow is purely additional, used for
parameter sweeps and time-series UI.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..config import Settings
from ..logging import log


class MLflowTracker:
    """Context-manager-style wrapper.

    Usage:
        tracker = build_tracker(settings)
        with tracker.run(name=run_id):
            tracker.log_params({...})
            tracker.log_metrics({...})
            tracker.log_artifact(path)
    """

    def __init__(self, *, tracking_uri: str, experiment: str = "rag-experiment-kit") -> None:
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._active = False

    def start(self, *, run_name: str | None = None) -> None:
        self._mlflow.start_run(run_name=run_name)
        self._active = True
        try:
            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
            self._mlflow.set_tag("git_commit_sha", sha[:12])
        except Exception:
            pass

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        self._mlflow.log_params({k: str(v) for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self._active:
            return
        clean: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN
                continue
            clean[k] = fv
        if clean:
            self._mlflow.log_metrics(clean)

    def log_artifact(self, path: str | Path) -> None:
        if not self._active:
            return
        self._mlflow.log_artifact(str(path))

    def end(self) -> None:
        if self._active:
            self._mlflow.end_run()
            self._active = False


class _NoopTracker:
    """No-op fallback used when MLflow is disabled or import fails."""

    def start(self, *, run_name: str | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, path: str | Path) -> None: ...
    def end(self) -> None: ...


def build_tracker(settings: Settings | None = None) -> MLflowTracker | _NoopTracker:
    s = settings or Settings()
    if not s.mlflow_enabled:
        return _NoopTracker()
    try:
        return MLflowTracker(
            tracking_uri=s.mlflow_tracking_uri,
            experiment=s.mlflow_experiment,
        )
    except ImportError:
        log.warning(
            "mlflow.import-failed",
            hint="Install with: uv sync --extra mlflow",
        )
        return _NoopTracker()
    except Exception as e:
        log.warning("mlflow.init-failed", error=str(e))
        return _NoopTracker()


__all__ = ["MLflowTracker", "build_tracker"]
