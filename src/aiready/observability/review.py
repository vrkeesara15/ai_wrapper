"""Review queue: people check low scores and negative feedback.

Release pipeline slide, step 5: "Weekly failure review - people sort the causes" and "New test cases - each
failure becomes a test". An item arrives from a live scorer, from user feedback, or from an evaluation run.
A reviewer labels the cause (retrieval, prompt, model, data, policy - configurable) and, usually, what the
right answer was; labelled items are exported as golden-set candidates in the evals format, so the failure
is checked on every later release.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics

NS = "obs.review"
Status = Literal["open", "labelled", "dismissed", "exported"]


class ReviewItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str | None = None
    source: Literal["live_score", "feedback", "eval", "manual"] = "manual"
    reason: str = ""
    app: str | None = None
    feature: str | None = None
    question: str | None = None  # masked
    answer: str | None = None  # masked
    scores: dict[str, float] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)  # doc ids the answer used
    status: Status = "open"
    cause: str | None = None
    notes: str | None = None
    expected_answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expect_refusal: bool | None = None
    make_test: bool = False
    labelled_by: str | None = None
    created_at: float = Field(default_factory=time.time)
    labelled_at: float | None = None


class ReviewLabel(BaseModel):
    cause: str | None = None
    notes: str | None = None
    expected_answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expect_refusal: bool | None = None
    make_test: bool = True  # "each failure becomes a test"
    dismiss: bool = False  # not a failure (false alarm)


class ReviewQueue:
    def __init__(self, store: DocStore | None = None, causes: list[str] | None = None, service: str = "observability"):
        self._store = store
        self.causes = causes or ["retrieval", "prompt", "model", "data", "policy"]
        self.service = service

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def enqueue(self, item: ReviewItem) -> ReviewItem:
        """Add an item; one open item per (trace, source) so a noisy trace is not reviewed twice."""
        if item.trace_id:
            for existing in self.store.find(NS, trace_id=item.trace_id, source=item.source):
                if existing.get("status") == "open":
                    return ReviewItem.model_validate(existing)
        self.store.put(NS, item.id, item.model_dump())
        self._gauge()
        return item

    def get(self, item_id: str) -> ReviewItem:
        raw = self.store.get(NS, item_id)
        if raw is None:
            raise NotFound(f"review item {item_id} not found")
        return ReviewItem.model_validate(raw)

    def list(self, status: str | None = "open", source: str | None = None, limit: int = 200) -> list[ReviewItem]:
        items = [ReviewItem.model_validate(v) for v in self.store.values(NS)]
        items = [i for i in items if (status is None or i.status == status) and (source is None or i.source == source)]
        return sorted(items, key=lambda i: -i.created_at)[:limit]

    def size(self, status: str = "open") -> int:
        return sum(1 for v in self.store.values(NS) if v.get("status") == status)

    def label(self, item_id: str, label: ReviewLabel, principal: Principal) -> ReviewItem:
        item = self.get(item_id)
        if not label.dismiss and label.cause not in self.causes:
            raise ValidationFailed(f"cause must be one of {self.causes}", details={"causes": self.causes})
        item.status = "dismissed" if label.dismiss else "labelled"
        item.cause = None if label.dismiss else label.cause
        item.notes = label.notes
        item.expected_answer = label.expected_answer or item.expected_answer
        item.must_include = label.must_include or item.must_include
        item.expected_sources = label.expected_sources or item.expected_sources
        item.expect_refusal = label.expect_refusal
        item.make_test = label.make_test and not label.dismiss
        item.labelled_by, item.labelled_at = principal.sub, time.time()
        self.store.put(NS, item.id, item.model_dump())
        get_audit().for_principal(principal, f"{self.service}.review.label", trace_id=item.trace_id,
                                  resource=item.id, decision=item.status, cause=item.cause, make_test=item.make_test)
        self._gauge()
        return item

    def cause_summary(self, since: float | None = None) -> dict[str, int]:
        """Input to the weekly failure review: how many labelled failures per cause."""
        c: Counter[str] = Counter()
        for v in self.store.values(NS):
            if v.get("status") in ("labelled", "exported") and v.get("cause") and (since is None or (v.get("labelled_at") or 0) >= since):
                c[v["cause"]] += 1
        return dict(c)

    def golden_candidates(self, *, mark_exported: bool = False, principal: Principal | None = None) -> list[dict[str, Any]]:
        """Labelled failures as cases in the golden-set format (``aiready.evals.golden.EvalCase``)."""
        out = []
        for item in self.list(status="labelled", limit=10_000):
            if not item.make_test or not item.question:
                continue
            case: dict[str, Any] = {"id": f"review-{item.id[:12]}", "question": item.question,
                                    "tags": ["from-review", f"cause:{item.cause}"] + ([item.feature] if item.feature else []),
                                    "metadata": {"trace_id": item.trace_id, "source": item.source, "notes": item.notes}}
            if item.expected_answer:
                case["expected_answer"] = item.expected_answer
            if item.must_include:
                case["must_include"] = item.must_include
            if item.expected_sources:
                case["expected_sources"] = item.expected_sources
            if item.expect_refusal is not None:
                case["expect_refusal"] = item.expect_refusal
            if not any(k in case for k in ("expected_answer", "must_include", "expected_sources", "expect_refusal")):
                continue  # a test without an expectation would always pass
            out.append(case)
            if mark_exported:
                item.status = "exported"
                self.store.put(NS, item.id, item.model_dump())
        if principal is not None:
            get_audit().for_principal(principal, f"{self.service}.review.export", decision="exported" if mark_exported else "listed",
                                      count=len(out))
        return out

    def _gauge(self) -> None:
        metrics.queue_depth.labels(self.service, "review").set(self.size())
