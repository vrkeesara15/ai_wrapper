"""Access reviews: who may use which model and which data, signed off by a person.

A report is *computed*, not hand-written: for every group and role in the client's directory mapping, the
active policy bundle is evaluated against every approved model (for each data class) and every reviewed
data collection. The audit log supplies the last time each member actually used a model, so grants nobody
used for ``stale_after_days`` are flagged, as are grants on deprecated or disabled models and empty groups.

The reviewer records a decision per entry (keep / revoke) and signs the report; both are audited. Revoke
decisions are listed for the client's identity team (group membership lives in their directory, not here).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import span
from aiready.governance.config import AccessReviewConfig
from aiready.governance.registry import RegistryService

NS_REPORTS = "access_reviews"
DAY = 86400.0

Decide = Callable[[PolicyInput], Awaitable[PolicyDecision]]


class ReviewEntry(BaseModel):
    id: str
    subject_type: Literal["group", "role"]
    subject: str
    members: list[str]
    resource_type: Literal["model", "collection"]
    resource: str
    allowed: bool
    data_classes: list[str] = Field(default_factory=list)  # for models: classes the subject may send
    rules: list[str] = Field(default_factory=list)
    last_used: dict[str, float | None] = Field(default_factory=dict)  # member -> last use (models only)
    stale_members: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    decision: Literal["keep", "revoke"] | None = None


class AccessReport(BaseModel):
    id: str
    created_at: float
    created_by: str
    policy_version: str
    status: Literal["open", "signed"] = "open"
    entries: list[ReviewEntry] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    signed_by: str | None = None
    signed_at: float | None = None
    comment: str = ""
    revocations: list[dict[str, Any]] = Field(default_factory=list)


class AccessReviewService:
    def __init__(
        self,
        decide: Decide,
        registry: RegistryService,
        cfg: AccessReviewConfig | None = None,
        store: DocStore | None = None,
        policy_version: Callable[[], str] | None = None,
    ):
        self.decide = decide
        self.registry = registry
        self.cfg = cfg or AccessReviewConfig()
        self._store = store
        self.policy_version = policy_version or (lambda: "")

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def _subjects(self) -> list[tuple[str, str, list[str]]]:
        subs = [("group", g, sorted(m)) for g, m in sorted(self.cfg.group_members.items())]
        subs += [("role", r, sorted(m)) for r, m in sorted(self.cfg.role_members.items())]
        return subs

    @staticmethod
    def _usage_index(since: float) -> dict[tuple[str, str], float]:
        """(actor, model) -> last use, from model-related audit records."""
        idx: dict[tuple[str, str], float] = {}
        for rec in get_audit().read(since=since):
            names: set[str] = set()
            for key in ("model", "resolved_model", "route"):
                v = rec.details.get(key)
                if isinstance(v, str):
                    names.add(v.split("@", 1)[0])
            if rec.resource:
                r = rec.resource.removeprefix("model:")
                names.add(r.split("@", 1)[0])
            for n in names:
                for actor in filter(None, (rec.actor, rec.on_behalf_of)):
                    k = (actor, n)
                    if rec.ts > idx.get(k, 0):
                        idx[k] = rec.ts
        return idx

    async def generate(self, principal: Principal) -> AccessReport:
        with span("governance.access_review.generate"):
            now = time.time()
            stale_cutoff = now - self.cfg.stale_after_days * DAY
            usage = self._usage_index(since=now - max(self.cfg.stale_after_days * 4, 365) * DAY)
            models = [m for m in self.registry.list_models() if m.status != "draft"]
            entries: list[ReviewEntry] = []
            for stype, subject, members in self._subjects():
                p = {
                    "sub": f"review:{stype}:{subject}",
                    "roles": [subject] if stype == "role" else [],
                    "groups": [subject] if stype == "group" else [],
                    "tenant": principal.tenant,
                }
                for m in models:
                    allowed_classes: list[str] = []
                    rules: set[str] = set()
                    for dc in self.cfg.data_classes:
                        d = await self.decide(
                            PolicyInput(
                                action="model.invoke",
                                principal=p,
                                resource={"model": m.name, **m.facts()},
                                context={"data_class": dc, "review": True},
                            )
                        )
                        if d.allow:
                            allowed_classes.append(dc)
                            rules.update(d.matched_rules)
                    if not allowed_classes:
                        continue
                    last = {u: usage.get((u, m.name)) for u in members}
                    stale = [u for u, ts in last.items() if ts is None or ts < stale_cutoff]
                    flags = []
                    if m.status == "deprecated":
                        flags.append("model_deprecated")
                    if m.disabled:
                        flags.append("model_disabled")
                    if not members:
                        flags.append("empty_subject")
                    if stale:
                        flags.append("stale_grants")
                    entries.append(
                        ReviewEntry(
                            id=uuid.uuid4().hex[:12],
                            subject_type=stype,
                            subject=subject,
                            members=members,
                            resource_type="model",
                            resource=m.key,
                            allowed=True,
                            data_classes=allowed_classes,
                            rules=sorted(rules),
                            last_used=last,
                            stale_members=stale,
                            flags=flags,
                        )
                    )
                for coll, tags in sorted(self.cfg.collections.items()):
                    d = await self.decide(
                        PolicyInput(
                            action="data.read",
                            principal=p,
                            resource={"collection": coll, "access_tags": tags},
                            context={"review": True},
                        )
                    )
                    if d.allow:
                        entries.append(
                            ReviewEntry(
                                id=uuid.uuid4().hex[:12],
                                subject_type=stype,
                                subject=subject,
                                members=members,
                                resource_type="collection",
                                resource=coll,
                                allowed=True,
                                rules=d.matched_rules,
                                flags=[] if members else ["empty_subject"],
                            )
                        )
            report = AccessReport(
                id=uuid.uuid4().hex,
                created_at=now,
                created_by=principal.sub,
                policy_version=self.policy_version(),
                entries=entries,
            )
            report.summary = {
                "entries": len(entries),
                "subjects": len(self._subjects()),
                "models": len(models),
                "flagged": sum(1 for e in entries if e.flags),
                "stale_grants": sum(len(e.stale_members) for e in entries),
                "previous_review_overdue": self.overdue(),
            }
            self.store.put(NS_REPORTS, report.id, report.model_dump(mode="json"))
            get_audit().for_principal(
                principal,
                "governance.access_review.generate",
                action="access_review",
                resource=f"access_review:{report.id}",
                decision="open",
                **report.summary,
            )
            return report

    def get(self, report_id: str) -> AccessReport:
        raw = self.store.get(NS_REPORTS, report_id)
        if raw is None:
            raise NotFound(f"access review {report_id} not found")
        return AccessReport.model_validate(raw)

    def list(self) -> list[AccessReport]:
        return sorted(
            (AccessReport.model_validate(v) for v in self.store.values(NS_REPORTS)),
            key=lambda r: r.created_at,
            reverse=True,
        )

    def sign_off(
        self, report_id: str, principal: Principal, decisions: dict[str, str] | None = None, comment: str = ""
    ) -> AccessReport:
        """Record keep/revoke per entry and sign. Unlisted entries are kept. A signed report is final."""
        with span("governance.access_review.sign_off"):
            report = self.get(report_id)
            if report.status == "signed":
                raise ValidationFailed("access review already signed")
            if principal.is_service_only:
                raise PermissionDenied("a person must sign the access review")
            decisions = decisions or {}
            ids = {e.id for e in report.entries}
            unknown = set(decisions) - ids
            if unknown:
                raise ValidationFailed(f"unknown entries: {sorted(unknown)}")
            for e in report.entries:
                d = decisions.get(e.id, "keep")
                if d not in ("keep", "revoke"):
                    raise ValidationFailed(f"decision for {e.id} must be keep or revoke")
                e.decision = d  # type: ignore[assignment]
                if d == "revoke":
                    report.revocations.append(
                        {
                            "entry": e.id,
                            "subject_type": e.subject_type,
                            "subject": e.subject,
                            "resource": e.resource,
                            "members": e.stale_members or e.members,
                        }
                    )
            report.status, report.signed_by, report.signed_at, report.comment = (
                "signed",
                principal.sub,
                time.time(),
                comment,
            )
            self.store.put(NS_REPORTS, report.id, report.model_dump(mode="json"))
            get_audit().for_principal(
                principal,
                "governance.access_review.sign_off",
                action="access_review",
                resource=f"access_review:{report.id}",
                decision="signed",
                revocations=len(report.revocations),
                comment=comment,
            )
            return report

    def overdue(self) -> bool:
        """True when no signed review exists within the review period."""
        cutoff = time.time() - self.cfg.review_period_days * DAY
        return not any(r.status == "signed" and (r.signed_at or 0) >= cutoff for r in self.list())
