"""Observability service: trace store, viewer and replay, cost reports, live scores, review queue, feedback,
alerts and dashboard data - the "Store" and "Use" columns of the observability architecture slide.

Access rules:
  * every signed-in user may read their *own* traces and send feedback on them;
  * ``access.viewer_roles`` read all traces of their tenant; ``replay_roles`` may replay; ``report_roles`` see
    cost/latency/quality reports; ``reviewer_roles`` work the review queue.
  * a replay re-issues the recorded request through the gateway under the *replaying admin's* identity (never
    the original user's), is policy-checked as ``trace.replay`` and audited with both trace ids.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.plugins import create
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import SpanRecord, current_trace_id, metrics, recorder, span
from aiready.core.types import ChatRequest
from aiready.observability.alerts import AlertEvent, AlertManager, Notifier
from aiready.observability.config import ObservabilityConfig, load_config
from aiready.observability.feedback import FeedbackCapture, FeedbackIn, FeedbackRecord
from aiready.observability.governance import authorize, require_any_role
from aiready.observability.masking import RegexMasker, SpanMasker
from aiready.observability.review import ReviewItem, ReviewLabel, ReviewQueue
from aiready.observability.scoring import AnswerSample, LiveScorer, ScoreRecord, extract_sample, sampled
from aiready.observability.traces import (
    TraceStore,
    TraceSummary,
    TraceTree,
    build_tree,
    group_by_trace,
    recorded_request,
    summarize,
)
from aiready.observability.usage import (
    ChargebackReport,
    StageLatency,
    UsageRecord,
    UsageRow,
    aggregate,
    cache_hit_rate,
    chargeback,
    latency_report,
    month_bounds,
    usage_records,
)

log = logging.getLogger(__name__)
SERVICE = "observability"
REPLAY_NS = "obs.replays"


class ObservabilityService:
    def __init__(self, cfg: ObservabilityConfig | None = None, *, store: TraceStore | None = None,
                 docstore: DocStore | None = None, notifiers: dict[str, Notifier] | None = None):
        self.cfg = cfg or ObservabilityConfig()
        self.store: TraceStore = store or create("trace_store", self.cfg.store.spec() or {"type": "sql"})
        self._docstore = docstore
        self.masker = SpanMasker(RegexMasker(self.cfg.masking.extra_patterns, disabled=self.cfg.masking.disabled_entities),
                                 use_guardrails=self.cfg.masking.use_guardrails)
        self.review = ReviewQueue(docstore, causes=self.cfg.review.causes, service=SERVICE)
        self.feedback = FeedbackCapture(self.review, self.masker, store=docstore,
                                        negative_rating_threshold=self.cfg.review.negative_rating_threshold, service=SERVICE)
        self.scorer = LiveScorer(self.cfg.scoring, self.review, store=docstore, service=SERVICE)
        self.alerts = AlertManager(self.cfg, self, notifiers=notifiers, store=docstore, service=SERVICE)
        self._local: deque[SpanRecord] = deque(maxlen=50_000)
        self._bg: asyncio.Task[None] | None = None

    @classmethod
    def from_config(cls) -> ObservabilityService:
        return cls(load_config())

    @property
    def docstore(self) -> DocStore:
        return self._docstore or get_docstore()

    # ================================================================== ingest
    async def ingest(self, spans: list[SpanRecord], principal: Principal | None = None) -> dict[str, Any]:
        """Mask, store, then live-score traces that now contain an answer. Not wrapped in a span itself, so
        the service's own telemetry does not feed back into the store it is writing."""
        if len(spans) > self.cfg.ingest.max_batch:
            raise ValidationFailed(f"batch of {len(spans)} spans exceeds max_batch {self.cfg.ingest.max_batch}")
        t0 = time.perf_counter()
        masked: list[SpanRecord] = []
        for s in spans:
            data = s.model_dump()
            data["attributes"] = await self.masker.mask_value(s.attributes)
            data["events"] = await self.masker.mask_value(s.events)
            masked.append(SpanRecord.model_validate(data))
        n = self.store.add(masked)
        metrics.stage.labels(SERVICE, "ingest").observe(time.perf_counter() - t0)
        scored = 0
        for trace_id, batch in group_by_trace(masked).items():
            if extract_sample(batch, self.cfg.scoring.answer_span_names) is None:
                continue
            try:
                if await self.scorer.maybe_score(self.store.trace(trace_id)):
                    scored += 1
            except Exception:  # scoring is best-effort; the spans are stored either way
                log.exception("live scoring failed for %s", trace_id)
        return {"accepted": n, "scored": scored}

    async def submit_answer(self, sample: AnswerSample, principal: Principal | None = None, *,
                            force: bool = False) -> ScoreRecord | None:
        """Explicit path for emitters that do not put answers in span attributes (privacy by default): the
        answer is masked, then live-scored when its trace is sampled (``force`` scores it regardless)."""
        if not force and not sampled(sample.trace_id, self.cfg.scoring.sample_rate):
            return None
        if self.scorer.store.get("obs.scores", sample.trace_id) is not None:
            return None
        sample = sample.model_copy(update={
            "question": await self.masker.mask_text(sample.question),
            "answer": await self.masker.mask_text(sample.answer),
            "passages": [p.model_copy(update={"text": await self.masker.mask_text(p.text)}) for p in sample.passages],
        })
        return await self.scorer.score(sample)

    def install_local_exporter(self) -> None:
        """Capture spans of this process (single-process pilot). They are flushed by the background loop."""
        recorder.exporters.append(self._local.append)

    async def flush_local(self) -> int:
        batch = []
        while self._local:
            batch.append(self._local.popleft())
        batch = [s for s in batch if not s.name.startswith("observability.")]
        if batch:
            await self.ingest(batch)
        return len(batch)

    # ================================================================== traces
    def _can_view(self, principal: Principal, summary: TraceSummary, spans: list[SpanRecord]) -> bool:
        tenants = {str(s.attributes["tenant"]) for s in spans if s.attributes.get("tenant") is not None}
        if tenants and principal.tenant not in tenants:
            return False
        if principal.has_role(*self.cfg.access.viewer_roles):
            return True
        return summary.user == principal.sub

    async def get_trace(self, trace_id: str, principal: Principal) -> TraceTree:
        spans = self.store.trace(trace_id)
        if not spans:
            raise NotFound(f"trace {trace_id} not found")
        tree = build_tree(trace_id, spans)
        if not self._can_view(principal, tree.summary, spans):
            raise NotFound(f"trace {trace_id} not found")  # do not reveal that someone else's trace exists
        tree.replays = list(self.docstore.get(REPLAY_NS, trace_id, []) or [])
        get_audit().for_principal(principal, f"{SERVICE}.trace.view", resource=trace_id, decision="allow",
                                  own=tree.summary.user == principal.sub)
        return tree

    async def search_traces(self, principal: Principal, *, app: str | None = None, feature: str | None = None,
                            user: str | None = None, since: float | None = None, until: float | None = None,
                            limit: int = 50) -> list[TraceSummary]:
        viewer = principal.has_role(*self.cfg.access.viewer_roles)
        if not viewer:
            if user and user != principal.sub:
                raise PermissionDenied("you can only search your own traces")
            user = principal.sub
        ids = self.store.find_traces(app=app, feature=feature, user=user, since=since, until=until, limit=limit)
        out = []
        for tid in ids:
            spans = self.store.trace(tid)
            summary = summarize(tid, spans)
            if self._can_view(principal, summary, spans):
                out.append(summary)
        return out

    async def replay(self, trace_id: str, principal: Principal, *, model: str | None = None) -> dict[str, Any]:
        require_any_role(principal, self.cfg.access.replay_roles, what="replaying a request")
        spans = self.store.trace(trace_id)
        if not spans or not self._can_view(principal, summarize(trace_id, spans), spans):
            raise NotFound(f"trace {trace_id} not found")
        found = recorded_request(spans)
        if found is None:
            raise ValidationFailed(f"trace {trace_id} holds no recorded gateway request (span attribute 'request')")
        src_span, raw = found
        summary = summarize(trace_id, spans)
        await authorize(principal, "trace.replay", service=SERVICE,
                        resource={"type": "trace", "id": trace_id, "original_user": summary.user},
                        context={"feature": summary.feature, "app": summary.app})
        body = {k: v for k, v in raw.items() if k in ChatRequest.model_fields}
        req = ChatRequest.model_validate(body)
        req.stream = False
        if model:
            req.model = model
        req.metadata = {**(req.metadata or {}), "replay_of": trace_id, "replayed_by": principal.sub,
                        "feature": summary.feature or req.metadata.get("feature", "replay")}
        from aiready.core.wiring import connect

        with span("observability.replay", replay_of=trace_id, user=principal.sub, model=req.model) as sp, \
                metrics.time_stage(SERVICE, "replay"):
            new_tid = current_trace_id()
            try:
                resp = await connect("gateway").chat(req, principal)
            except Exception as exc:
                sp.error(exc)
                get_audit().for_principal(principal, f"{SERVICE}.replay", trace_id=new_tid, resource=trace_id,
                                          decision="failed", error=str(exc))
                raise
        replays = list(self.docstore.get(REPLAY_NS, trace_id, []) or [])
        if new_tid:
            replays.append(new_tid)
            self.docstore.put(REPLAY_NS, trace_id, replays)
        get_audit().for_principal(principal, f"{SERVICE}.replay", trace_id=new_tid, resource=trace_id,
                                  decision="replayed", original_user=summary.user, source_span=src_span.name,
                                  model=resp.model)
        original_answer = next((s.attributes.get("answer") for s in spans if s.attributes.get("answer")), None)
        return {"original_trace_id": trace_id, "replay_trace_id": new_tid, "response": resp.model_dump(),
                "original_answer": original_answer, "note": "the recorded request was masked before storage"}

    # ================================================================== feedback & review
    async def record_feedback(self, fb: FeedbackIn, principal: Principal) -> FeedbackRecord:
        spans = self.store.trace(fb.trace_id)
        question = answer = None
        if spans:
            summary = summarize(fb.trace_id, spans)
            if summary.user and summary.user != principal.sub and not principal.has_role(*self.cfg.access.viewer_roles):
                raise PermissionDenied("feedback can only be given on your own requests")
            sample = extract_sample(spans, self.cfg.scoring.answer_span_names)
            if sample:
                question, answer = sample.question, sample.answer
                fb.feature = fb.feature or sample.feature
        return await self.feedback.record(fb, principal, question=question, answer=answer)

    async def review_list(self, principal: Principal, status: str | None = "open", source: str | None = None) -> list[ReviewItem]:
        require_any_role(principal, self.cfg.access.reviewer_roles, what="the review queue")
        return self.review.list(status=status, source=source)

    async def review_label(self, item_id: str, label: ReviewLabel, principal: Principal) -> ReviewItem:
        require_any_role(principal, self.cfg.access.reviewer_roles, what="the review queue")
        return self.review.label(item_id, label, principal)

    async def review_enqueue(self, item: ReviewItem, principal: Principal) -> ReviewItem:
        require_any_role(principal, self.cfg.access.reviewer_roles, what="the review queue")
        item.question = await self.masker.mask_text(item.question) if item.question else None
        item.answer = await self.masker.mask_text(item.answer) if item.answer else None
        return self.review.enqueue(item)

    async def golden_candidates(self, principal: Principal, *, mark_exported: bool = False) -> list[dict[str, Any]]:
        require_any_role(principal, self.cfg.access.reviewer_roles, what="exporting golden-set candidates")
        return self.review.golden_candidates(mark_exported=mark_exported, principal=principal)

    # ================================================================== reports
    def window_spans(self, since: float, until: float) -> list[SpanRecord]:
        return self.store.spans(since=since, until=until)

    def window_usage(self, since: float, until: float) -> list[UsageRecord]:
        return usage_records(self.window_spans(since, until), pricing=self.cfg.pricing, teams=self.cfg.teams)

    def window_scores(self, since: float, until: float) -> list[ScoreRecord]:
        return self.scorer.records(since, until)

    def _window(self, since: float | None, until: float | None, default_days: float = 7) -> tuple[float, float]:
        until = until or time.time()
        return since or until - default_days * 86400, until

    async def usage(self, principal: Principal, *, by: list[str], since: float | None = None,
                    until: float | None = None) -> list[UsageRow]:
        require_any_role(principal, self.cfg.access.report_roles, what="cost reports")
        s, u = self._window(since, until)
        return aggregate(self.window_usage(s, u), by)

    async def chargeback(self, principal: Principal, period: str | None = None) -> ChargebackReport:
        require_any_role(principal, self.cfg.access.report_roles, what="chargeback reports")
        period, s, u = month_bounds(period)
        report = chargeback(self.window_usage(s, u), period=period, since=s, until=u, currency=self.cfg.currency,
                            budgets=self.cfg.budgets)
        get_audit().for_principal(principal, f"{SERVICE}.chargeback", resource=period, decision="generated",
                                  total=report.total_cost)
        return report

    async def latency(self, principal: Principal, *, since: float | None = None, until: float | None = None) -> list[StageLatency]:
        require_any_role(principal, self.cfg.access.report_roles, what="latency reports")
        s, u = self._window(since, until, 1)
        return latency_report(self.window_spans(s, u), self.cfg.latency_budget_ms, self.cfg.stage_map)

    async def quality(self, principal: Principal, *, since: float | None = None, app: str | None = None,
                      feature: str | None = None, model: str | None = None, bucket: str = "day") -> list[dict[str, Any]]:
        require_any_role(principal, self.cfg.access.report_roles, what="quality reports")
        return self.scorer.trend(since=since, app=app, feature=feature, model=model, bucket=bucket)

    async def evaluate_alerts(self) -> list[AlertEvent]:
        return await self.alerts.evaluate()

    async def dashboard_data(self, principal: Principal, *, days: float = 7) -> dict[str, Any]:
        require_any_role(principal, self.cfg.access.report_roles, what="the dashboard")
        until = time.time()
        since = until - days * 86400
        spans = self.window_spans(since, until)
        recs = usage_records(spans, pricing=self.cfg.pricing, teams=self.cfg.teams)
        return {
            "generated_at": until, "days": days, "currency": self.cfg.currency,
            "requests": len({r.trace_id for r in recs}), "total_cost": round(sum(r.cost for r in recs), 4),
            "spend_by_feature": [r.model_dump() for r in aggregate(recs, ["feature"])],
            "spend_by_team": [r.model_dump() for r in aggregate(recs, ["team"])],
            "spend_by_day": sorted((r.model_dump() for r in aggregate(recs, ["day"])), key=lambda r: r["key"]["day"]),
            "latency": [r.model_dump() for r in latency_report(spans, self.cfg.latency_budget_ms, self.cfg.stage_map)],
            "quality": self.scorer.trend(since=since),
            "cache_hit_rate": cache_hit_rate(recs, spans),
            "review_queue": self.review.size(),
            "review_causes": self.review.cause_summary(since),
            "alerts": self.alerts.active(),
        }

    # ================================================================== lifecycle
    async def start(self) -> None:
        if self.cfg.ingest.capture_local:
            self.install_local_exporter()
        self.alerts.start()
        if self._bg is None:
            self._bg = asyncio.get_running_loop().create_task(self._housekeeping())

    async def stop(self) -> None:
        await self.alerts.stop()
        if self._bg:
            self._bg.cancel()
            self._bg = None

    async def _housekeeping(self) -> None:
        last_purge = 0.0
        while True:
            try:
                if self.cfg.ingest.capture_local:
                    await self.flush_local()
                if time.time() - last_purge > 3600:
                    self.store.purge(time.time() - self.cfg.store.retention_days * 86400)
                    last_purge = time.time()
            except Exception:
                log.exception("observability housekeeping failed")
            await asyncio.sleep(self.cfg.ingest.flush_interval_seconds)
