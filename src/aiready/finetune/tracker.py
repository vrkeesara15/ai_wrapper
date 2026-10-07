"""Experiment tracker (plugin kind ``experiment_tracker``): settings and scores of every training run.

``docstore`` keeps runs in the platform's state database (no extra infrastructure). ``mlflow`` forwards to
the client's MLflow server when it has one (imported lazily). Either way a run records the base revision,
dataset version/hash, adapter settings and the release-gate scores, so any adapter can be reproduced.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.errors import NotFound
from aiready.core.plugins import create, register
from aiready.core.stores import DocStore, get_docstore

NS = "finetune.runs"


class Run(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    experiment: str
    name: str = ""
    status: str = "running"  # running | finished | failed
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    started_at: float = Field(default_factory=time.time)
    ended_at: float | None = None
    external_id: str | None = None


class ExperimentTracker(Protocol):
    def start_run(self, experiment: str, *, name: str = "", params: dict[str, Any] | None = None,
                  tags: dict[str, str] | None = None) -> Run: ...
    def log_params(self, run_id: str, params: dict[str, Any]) -> None: ...
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, run_id: str, name: str, uri: str) -> None: ...
    def end_run(self, run_id: str, status: str = "finished") -> Run: ...
    def get_run(self, run_id: str) -> Run: ...
    def list_runs(self, experiment: str) -> list[Run]: ...


@register("experiment_tracker", "docstore")
class DocStoreTracker:
    def __init__(self, store: DocStore | None = None):
        self._store = store

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def _save(self, run: Run) -> Run:
        self.store.put(NS, run.id, run.model_dump())
        return run

    def start_run(self, experiment: str, *, name: str = "", params: dict[str, Any] | None = None,
                  tags: dict[str, str] | None = None) -> Run:
        return self._save(Run(experiment=experiment, name=name, params=params or {}, tags=tags or {}))

    def get_run(self, run_id: str) -> Run:
        raw = self.store.get(NS, run_id)
        if raw is None:
            raise NotFound(f"run {run_id} not found")
        return Run.model_validate(raw)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        run = self.get_run(run_id)
        run.params.update(params)
        self._save(run)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        run = self.get_run(run_id)
        run.metrics.update({k: float(v) for k, v in metrics.items()})
        self._save(run)

    def log_artifact(self, run_id: str, name: str, uri: str) -> None:
        run = self.get_run(run_id)
        run.artifacts[name] = uri
        self._save(run)

    def end_run(self, run_id: str, status: str = "finished") -> Run:
        run = self.get_run(run_id)
        run.status, run.ended_at = status, time.time()
        return self._save(run)

    def list_runs(self, experiment: str) -> list[Run]:
        return sorted((Run.model_validate(v) for v in self.store.find(NS, experiment=experiment)), key=lambda r: r.started_at)

    def best_run(self, experiment: str, metric: str, *, higher_is_better: bool = True) -> Run | None:
        runs = [r for r in self.list_runs(experiment) if metric in r.metrics]
        if not runs:
            return None
        return (max if higher_is_better else min)(runs, key=lambda r: r.metrics[metric])


@register("experiment_tracker", "mlflow")
class MlflowTracker(DocStoreTracker):
    """Mirrors every call to MLflow and keeps the DocStore copy (the release gate reads runs from there)."""

    def __init__(self, tracking_uri: str, store: DocStore | None = None):
        super().__init__(store)
        import mlflow  # optional dependency

        self._mlflow = mlflow
        self._client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def start_run(self, experiment: str, *, name: str = "", params: dict[str, Any] | None = None,
                  tags: dict[str, str] | None = None) -> Run:
        exp = self._client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else self._client.create_experiment(experiment)
        mr = self._client.create_run(exp_id, run_name=name or None, tags=tags or {})
        run = super().start_run(experiment, name=name, params=params, tags=tags)
        run.external_id = mr.info.run_id
        for k, v in (params or {}).items():
            self._client.log_param(mr.info.run_id, k, str(v)[:500])
        return self._save(run)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        super().log_params(run_id, params)
        ext = self.get_run(run_id).external_id
        for k, v in params.items():
            self._client.log_param(ext, k, str(v)[:500])

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        super().log_metrics(run_id, metrics)
        ext = self.get_run(run_id).external_id
        for k, v in metrics.items():
            self._client.log_metric(ext, k, float(v))

    def end_run(self, run_id: str, status: str = "finished") -> Run:
        run = super().end_run(run_id, status)
        self._client.set_terminated(run.external_id, status="FINISHED" if status == "finished" else "FAILED")
        return run


def make_tracker(spec: dict[str, Any] | None) -> ExperimentTracker:
    return create("experiment_tracker", spec or {"type": "docstore"})
