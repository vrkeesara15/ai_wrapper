"""Feedback capture: ratings and edits from the assistant widget, linked to the trace they are about.

A rating alone says *that* an answer was wrong; the user's edit often says *what* was right, which is the
cheapest source of expected answers for the golden set. Negative feedback goes straight to the review queue.
Free text (comment, edit) is masked before storage like everything else here.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from aiready.core.audit import get_audit
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.observability.masking import SpanMasker
from aiready.observability.review import ReviewItem, ReviewQueue

NS = "obs.feedback"


class FeedbackIn(BaseModel):
    """Body of ``POST /v1/feedback``. Give a thumbs value, a 1-5 rating, an edit, or any combination."""

    trace_id: str
    thumbs: Literal["up", "down"] | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    edit: str | None = None  # the answer as the user corrected it
    comment: str | None = None
    app: str | None = None
    feature: str | None = None

    @model_validator(mode="after")
    def _something(self) -> "FeedbackIn":
        if self.thumbs is None and self.rating is None and not self.edit and not self.comment:
            raise ValueError("feedback needs thumbs, rating, edit or comment")
        return self


class FeedbackRecord(FeedbackIn):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    user: str
    tenant: str = "default"
    ts: float = Field(default_factory=time.time)
    negative: bool = False


class FeedbackCapture:
    def __init__(self, review: ReviewQueue, masker: SpanMasker, *, negative_rating_threshold: int = 2,
                 store: DocStore | None = None, service: str = "observability"):
        self.review, self.masker = review, masker
        self.threshold = negative_rating_threshold
        self._store = store
        self.service = service

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def is_negative(self, fb: FeedbackIn) -> bool:
        return fb.thumbs == "down" or (fb.rating is not None and fb.rating <= self.threshold)

    async def record(self, fb: FeedbackIn, principal: Principal, *, question: str | None = None,
                     answer: str | None = None) -> FeedbackRecord:
        rec = FeedbackRecord(**fb.model_dump(), user=principal.sub, tenant=principal.tenant, negative=self.is_negative(fb))
        rec.app = rec.app or principal.app
        rec.comment = await self.masker.mask_text(rec.comment) if rec.comment else None
        rec.edit = await self.masker.mask_text(rec.edit) if rec.edit else None
        self.store.put(NS, rec.id, rec.model_dump())
        get_audit().for_principal(principal, f"{self.service}.feedback", trace_id=fb.trace_id, resource=rec.id,
                                  decision="negative" if rec.negative else "positive", rating=rec.rating,
                                  thumbs=rec.thumbs, has_edit=bool(rec.edit))
        if rec.negative or rec.edit:
            self.review.enqueue(ReviewItem(
                trace_id=fb.trace_id, source="feedback", app=rec.app, feature=rec.feature,
                reason=("negative rating" if rec.negative else "user edited the answer") + (f": {rec.comment}" if rec.comment else ""),
                question=question, answer=answer, expected_answer=rec.edit,
            ))
        return rec

    def for_trace(self, trace_id: str) -> list[FeedbackRecord]:
        return [FeedbackRecord.model_validate(v) for v in self.store.find(NS, trace_id=trace_id)]

    def recent(self, since: float | None = None) -> list[FeedbackRecord]:
        recs = [FeedbackRecord.model_validate(v) for v in self.store.values(NS)]
        return [r for r in recs if since is None or r.ts >= since]
