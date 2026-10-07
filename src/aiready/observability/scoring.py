"""Live scorers: sampled production answers scored for grounding, citations, refusals and PII leaks.

"Sampled answers scored for grounding" (observability slide) and "Traces and live scores - sampled from
production" (feedback loop). Sampling is deterministic on the trace id, so every replica agrees on which
requests are scored and a re-delivered batch is not scored twice with different results. Scores feed:

* the ``aiready_quality_score`` gauge (rolling mean per app / feature / scorer) and the quality trend;
* the review queue, when the overall score is below ``low_score_threshold`` or a safety scorer fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import SpanRecord, metrics, span
from aiready.evals.scorers import Passage, Scorer, ScoreInput, ScoreResult, build_scorers
from aiready.observability.config import ScoringConfig
from aiready.observability.review import ReviewItem, ReviewQueue
from aiready.observability.traces import attr, match_any, span_user

log = logging.getLogger(__name__)
NS = "obs.scores"


class AnswerSample(BaseModel):
    trace_id: str
    span: str
    ts: float
    question: str = ""
    answer: str = ""
    passages: list[Passage] = Field(default_factory=list)
    cited_sources: list[str] = Field(default_factory=list)
    cited_chunks: list[str] = Field(default_factory=list)
    found: bool | None = None
    app: str = "-"
    feature: str = "-"
    model: str = "-"
    user: str | None = None


class ScoreRecord(BaseModel):
    trace_id: str
    ts: float
    app: str
    feature: str
    model: str
    overall: float
    scores: dict[str, float]
    failed: list[str]
    details: dict[str, Any] = Field(default_factory=dict)
    enqueued: bool = False


def sampled(trace_id: str, rate: float) -> bool:
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    h = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < rate


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            return [v]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def extract_sample(spans: list[SpanRecord], answer_span_names: list[str]) -> AnswerSample | None:
    """Find the answer span of a trace (the RAG / agent / assistant span holding ``answer``)."""
    cand = [s for s in spans if attr(s, "answer") is not None and match_any(s.name, answer_span_names)]
    cand = cand or [s for s in spans if attr(s, "answer") is not None]
    if not cand:
        return None
    s = sorted(cand, key=lambda x: x.start)[-1]
    passages = []
    for i, p in enumerate(_as_list(attr(s, "passages"))):
        if isinstance(p, dict):
            chunk = p.get("chunk") if isinstance(p.get("chunk"), dict) else p
            passages.append(Passage(id=str(chunk.get("id", "")), doc_id=str(chunk.get("doc_id", "")),
                                    text=str(chunk.get("text", "")), marker=str(p.get("marker") or i + 1)))
        else:
            passages.append(Passage(id=str(i + 1), text=str(p), marker=str(i + 1)))
    cites = _as_list(attr(s, "citations"))
    ctx: dict[str, str] = {}
    for sp in spans:
        for k in ("app", "feature", "model"):
            v = attr(sp, k)
            if v is not None and k not in ctx:
                ctx[k] = str(v)
    return AnswerSample(
        trace_id=s.trace_id, span=s.name, ts=s.start, question=str(attr(s, "question", "query", default="")),
        answer=str(attr(s, "answer")), passages=passages,
        cited_sources=[str(c.get("doc_id")) for c in cites if isinstance(c, dict) and c.get("doc_id")],
        cited_chunks=[str(c.get("chunk_id")) for c in cites if isinstance(c, dict) and c.get("chunk_id")],
        found=attr(s, "found"), app=str(attr(s, "app", default=ctx.get("app", "-"))),
        feature=str(attr(s, "feature", default=ctx.get("feature", "-"))),
        model=str(attr(s, "model", default=ctx.get("model", "-"))),
        user=next((u for u in map(span_user, spans) if u), None),
    )


class LiveScorer:
    def __init__(self, cfg: ScoringConfig, review: ReviewQueue, *, store: DocStore | None = None,
                 service: str = "observability", judge_principal: Principal | None = None):
        self.cfg = cfg
        self.review = review
        self._store = store
        self.service = service
        specs = list(cfg.scorers)
        if cfg.llm_judge.get("enabled"):
            specs.append({"type": "llm_judge", **{k: v for k, v in cfg.llm_judge.items() if k != "enabled"}})
        self.scorers: list[Scorer] = build_scorers(specs)
        self.judge_principal = judge_principal
        self._rolling: dict[tuple[str, str, str], float] = {}

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    async def maybe_score(self, spans: list[SpanRecord]) -> ScoreRecord | None:
        if not self.cfg.enabled or not spans or not sampled(spans[0].trace_id, self.cfg.sample_rate):
            return None
        sample = extract_sample(spans, self.cfg.answer_span_names)
        if sample is None or self.store.get(NS, sample.trace_id) is not None:
            return None
        return await self.score(sample)

    async def score(self, sample: AnswerSample) -> ScoreRecord:
        with span("observability.live_score", trace_of=sample.trace_id, feature=sample.feature), \
                metrics.time_stage(self.service, "live_score"):
            inp = ScoreInput(question=sample.question, answer=sample.answer, passages=sample.passages,
                             cited_sources=sample.cited_sources, cited_chunks=sample.cited_chunks, found=sample.found,
                             principal=self.judge_principal)
            results: list[ScoreResult] = []
            for sc in self.scorers:
                try:
                    results.append(await sc.score(inp))
                except Exception as exc:  # a broken scorer must not stop the others
                    log.warning("live scorer %s failed: %s", sc.name, exc)
            used = [r for r in results if not r.skipped]
            overall = sum(r.score for r in used) / len(used) if used else 1.0
            failed = [r.scorer for r in used if not r.passed]
            safety_failed = [r.scorer for r in used if not r.passed and getattr(self._by_name(r.scorer), "safety", False)]
            rec = ScoreRecord(trace_id=sample.trace_id, ts=sample.ts, app=sample.app, feature=sample.feature,
                              model=sample.model, overall=round(overall, 4), scores={r.scorer: r.score for r in used},
                              failed=failed, details={r.scorer: r.details for r in used if r.details})
            for r in used:
                self._update_gauge(sample, r.scorer, r.score)
            self._update_gauge(sample, "overall", rec.overall)
            if overall < self.cfg.low_score_threshold or safety_failed:
                self.review.enqueue(ReviewItem(
                    trace_id=sample.trace_id, source="live_score", app=sample.app, feature=sample.feature,
                    reason=(f"overall {rec.overall:.2f} < {self.cfg.low_score_threshold}" if overall < self.cfg.low_score_threshold
                            else f"safety scorer failed: {safety_failed}"),
                    question=sample.question, answer=sample.answer, scores=rec.scores,
                    sources=sorted({p.doc_id for p in sample.passages if p.doc_id}),
                ))
                rec.enqueued = True
            self.store.put(NS, sample.trace_id, rec.model_dump())
            return rec

    def _by_name(self, name: str) -> Scorer | None:
        return next((s for s in self.scorers if s.name == name), None)

    def _update_gauge(self, sample: AnswerSample, scorer: str, value: float, alpha: float = 0.1) -> None:
        key = (sample.app, sample.feature, scorer)
        prev = self._rolling.get(key)
        cur = value if prev is None else (1 - alpha) * prev + alpha * value
        self._rolling[key] = cur
        metrics.quality.labels(self.service, sample.app, sample.feature, scorer).set(cur)

    # ------------------------------------------------------------------ reporting
    def records(self, since: float | None = None, until: float | None = None) -> list[ScoreRecord]:
        recs = [ScoreRecord.model_validate(v) for v in self.store.values(NS)]
        return [r for r in recs if (since is None or r.ts >= since) and (until is None or r.ts <= until)]

    def trend(self, *, since: float | None = None, app: str | None = None, feature: str | None = None,
              model: str | None = None, bucket: str = "day") -> list[dict[str, Any]]:
        """Mean score per day (or hour) and scorer: the "quality scores over time" gauge."""
        fmt = "%Y-%m-%dT%H:00" if bucket == "hour" else "%Y-%m-%d"
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in self.records(since=since):
            if app and r.app != app or feature and r.feature != feature or model and r.model != model:
                continue
            b = datetime.fromtimestamp(r.ts, tz=timezone.utc).strftime(fmt)
            acc[b]["overall"].append(r.overall)
            for k, v in r.scores.items():
                acc[b][k].append(v)
        return [{"bucket": b, "samples": len(v["overall"]),
                 **{k: round(sum(x) / len(x), 4) for k, x in v.items()}} for b, v in sorted(acc.items())]


def now() -> float:
    return time.time()
