"""Approval store and inbox: "Reads run freely. Writes, payments and deletions wait for a person."

Shared by the MCP tool executor (which *creates* requests when a write is attempted and *consumes* them when
the approved call is replayed) and the agent runtime (which pauses a run until the request is decided).
Requests live in the shared DocStore (namespace ``approvals``), so the MCP server, the agent workers and the
approval inbox may run in different pods.

Rules enforced here, not in the callers:

* **binding** - an approval covers exactly one tool, one argument set (SHA-256 of canonical JSON) and one
  requesting user; replaying it with other arguments or as someone else is refused;
* **single use** - an approved request executes once;
* **expiry** - undecided requests expire (``approvals.ttl_seconds``); approved ones must be used before
  ``approvals.use_within_seconds``;
* **four eyes** - the requester can never approve their own request (``four_eyes: true`` by default);
* **routing** - which roles may decide is configured per tool/effect (first matching rule wins);
* **tenant isolation** - approvers only see and decide requests of their own tenant;
* every creation and decision is written to the audit log.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import hashlib
import json
import threading
import time
import uuid
import weakref
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import NotFound, PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span

log = logging.getLogger(__name__)
NAMESPACE = "approvals"
_PENDING: set[asyncio.Task[Any]] = set()
Status = Literal["pending", "approved", "rejected", "expired", "used", "cancelled"]


def args_hash(arguments: dict[str, Any] | None) -> str:
    """Canonical hash of tool arguments: key order and whitespace never change it."""
    canonical = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ApprovalRule(BaseModel):
    """Routes requests to approver roles. ``tool`` is a glob; ``effects`` empty means any."""

    tool: str = "*"
    effects: list[str] = Field(default_factory=list)
    approver_roles: list[str] = Field(default_factory=lambda: ["approver"])
    ttl_seconds: float | None = None


class ApprovalsConfig(BaseModel):
    ttl_seconds: float = 86400.0  # an undecided request expires after a day
    use_within_seconds: float = 3600.0  # an approved request must be executed within an hour
    four_eyes: bool = True
    default_approver_roles: list[str] = Field(default_factory=lambda: ["approver"])
    admin_roles: list[str] = Field(default_factory=lambda: ["admin"])  # may see every request of the tenant
    rules: list[ApprovalRule] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    id: str = Field(default_factory=lambda: "apr_" + uuid.uuid4().hex)
    status: Status = "pending"
    tool: str
    effect: str = "write"
    arguments: dict[str, Any] = Field(default_factory=dict)  # shown to the approver: they approve *this* call
    args_hash: str
    requester: str
    requester_name: str | None = None
    tenant: str = "default"
    app: str = "default"
    actor: str | None = None  # agent acting for the user, if any
    run_id: str | None = None
    step: str | None = None
    reason: str | None = None
    data_class: str | None = None
    approver_roles: list[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    expires_at: float
    decided_by: str | None = None
    decided_at: float | None = None
    comment: str | None = None
    used_at: float | None = None
    trace_id: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status == "pending" and self.expires_at > time.time()


Listener = Callable[[ApprovalRequest], Any]
# weak references: a listener whose owner (e.g. an agent runtime) is gone stops being called
_LISTENERS: list[Callable[[], Listener | None]] = []


def _ref(listener: Listener) -> Callable[[], Listener | None]:
    if hasattr(listener, "__self__") and hasattr(listener, "__func__"):
        return weakref.WeakMethod(listener)  # type: ignore[arg-type]
    try:
        return weakref.ref(listener)
    except TypeError:
        return lambda: listener


class ApprovalStore:
    def __init__(self, docstore: DocStore | None = None, config: ApprovalsConfig | None = None):
        self._docs = docstore
        self.config = config or ApprovalsConfig()
        self._lock = threading.RLock()

    @property
    def docs(self) -> DocStore:
        return self._docs or get_docstore()

    # ------------------------------------------------------------------ routing
    def route(self, tool: str, effect: str, policy_roles: list[str] | None = None) -> tuple[list[str], float]:
        """Approver roles: a matching ``approvals.rules`` entry (most specific, set per tool) wins; then the
        roles named by the policy decision's ``approval_roles`` obligation; then the configured default."""
        for rule in self.config.rules:
            if fnmatch.fnmatchcase(tool, rule.tool) and (not rule.effects or effect in rule.effects):
                return list(rule.approver_roles), rule.ttl_seconds or self.config.ttl_seconds
        if policy_roles:
            return list(policy_roles), self.config.ttl_seconds
        return list(self.config.default_approver_roles), self.config.ttl_seconds

    # ------------------------------------------------------------------ persistence
    def _save(self, req: ApprovalRequest) -> None:
        self.docs.put(NAMESPACE, req.id, req.model_dump())

    def _load(self, approval_id: str) -> ApprovalRequest:
        raw = self.docs.get(NAMESPACE, approval_id)
        if raw is None:
            raise NotFound(f"approval {approval_id} not found")
        req = ApprovalRequest.model_validate(raw)
        if req.status == "pending" and req.expires_at <= time.time():
            req.status = "expired"
            self._save(req)
        return req

    def subscribe(self, listener: Listener) -> None:
        """Called with the request after every decision (the agent runtime resumes paused runs this way).

        Listeners are process-wide so that every store instance (one per configuration version) reaches them.
        """
        ref = _ref(listener)
        if not any(r() == listener for r in _LISTENERS):
            _LISTENERS.append(ref)

    def unsubscribe(self, listener: Listener) -> None:
        _LISTENERS[:] = [r for r in _LISTENERS if r() is not None and r() != listener]

    def _notify(self, req: ApprovalRequest) -> None:
        _LISTENERS[:] = [r for r in _LISTENERS if r() is not None]
        for cb in [r() for r in _LISTENERS]:
            if cb is None:
                continue
            try:
                res = cb(req)
                if asyncio.iscoroutine(res):
                    try:
                        task = asyncio.get_running_loop().create_task(res)
                    except RuntimeError:  # no event loop (sync caller): the run is resumed by its next poll
                        res.close()
                        continue
                    _PENDING.add(task)
                    task.add_done_callback(_PENDING.discard)
            except Exception:  # a listener must not undo a recorded decision
                log.exception("approval listener failed")

    # ------------------------------------------------------------------ lifecycle
    def create(self, *, tool: str, effect: str, arguments: dict[str, Any], principal: Principal,
               run_id: str | None = None, step: str | None = None, reason: str | None = None,
               data_class: str | None = None, approver_roles: list[str] | None = None) -> ApprovalRequest:
        from aiready.core.telemetry import current_trace_id

        roles, ttl = self.route(tool, effect, approver_roles)
        req = ApprovalRequest(tool=tool, effect=effect, arguments=arguments, args_hash=args_hash(arguments),
                              requester=principal.sub, requester_name=principal.name, tenant=principal.tenant,
                              app=principal.app, actor=principal.actor, run_id=run_id, step=step, reason=reason,
                              data_class=data_class, approver_roles=roles, expires_at=time.time() + ttl,
                              trace_id=current_trace_id())
        with span("approvals.create", tool=tool, effect=effect, approval_id=req.id):
            self._save(req)
            get_audit().for_principal(principal, "approval.request", action="tool.call", resource=tool,
                                      decision="pending", approval_id=req.id, effect=effect,
                                      args_hash=req.args_hash, run_id=run_id, approver_roles=roles)
        metrics.policy.labels("approvals", "approval.request", "pending").inc()
        return req

    def get(self, approval_id: str, principal: Principal | None = None) -> ApprovalRequest:
        req = self._load(approval_id)
        if principal is not None and not self._can_view(req, principal):
            raise NotFound(f"approval {approval_id} not found")  # do not reveal other tenants' requests
        return req

    def _can_view(self, req: ApprovalRequest, p: Principal) -> bool:
        if req.tenant != p.tenant:
            return False
        return req.requester == p.sub or p.has_role(*req.approver_roles) or p.has_role(*self.config.admin_roles)

    def list_pending(self, principal: Principal) -> list[ApprovalRequest]:
        """The inbox: open requests this person may decide (never their own when four-eyes is on)."""
        out = []
        for raw in self.docs.values(NAMESPACE):
            req = ApprovalRequest.model_validate(raw)
            if req.tenant != principal.tenant or req.status != "pending":
                continue
            if req.expires_at <= time.time():
                req.status = "expired"
                self._save(req)
                continue
            if self.config.four_eyes and req.requester == principal.sub:
                continue
            if principal.has_role(*req.approver_roles) or principal.has_role(*self.config.admin_roles):
                out.append(req)
        return sorted(out, key=lambda r: r.created_at)

    def list_requested(self, principal: Principal, status: str | None = None) -> list[ApprovalRequest]:
        out = [ApprovalRequest.model_validate(r) for r in self.docs.values(NAMESPACE)]
        out = [r for r in out if r.tenant == principal.tenant and r.requester == principal.sub
               and (status is None or r.status == status)]
        return sorted(out, key=lambda r: r.created_at)

    def decide(self, approval_id: str, principal: Principal, *, approve: bool,
               comment: str | None = None) -> ApprovalRequest:
        with self._lock, span("approvals.decide", approval_id=approval_id, approve=approve):
            req = self._load(approval_id)
            if req.tenant != principal.tenant:
                raise NotFound(f"approval {approval_id} not found")
            if req.status != "pending":
                raise ValidationFailed(f"approval {approval_id} is {req.status}, not pending")
            if self.config.four_eyes and req.requester == principal.sub:
                get_audit().for_principal(principal, "approval.decision", action="approve", resource=req.tool,
                                          decision="denied_self_approval", approval_id=req.id)
                raise PermissionDenied("four-eyes rule: you cannot approve your own request")
            if not (principal.has_role(*req.approver_roles) or principal.has_role(*self.config.admin_roles)):
                get_audit().for_principal(principal, "approval.decision", action="approve", resource=req.tool,
                                          decision="denied_role", approval_id=req.id)
                raise PermissionDenied(f"approving {req.tool} requires one of roles {req.approver_roles}")
            req.status = "approved" if approve else "rejected"
            req.decided_by, req.decided_at, req.comment = principal.sub, time.time(), comment
            if approve:  # the approved call must run soon, while the approver's judgement still holds
                req.expires_at = time.time() + self.config.use_within_seconds
            self._save(req)
            get_audit().for_principal(principal, "approval.decision", action="approve", resource=req.tool,
                                      decision=req.status, approval_id=req.id, requester=req.requester,
                                      args_hash=req.args_hash, run_id=req.run_id, comment=comment)
        metrics.policy.labels("approvals", "approval.decision", req.status).inc()
        self._notify(req)
        return req

    def cancel(self, approval_id: str, principal: Principal) -> ApprovalRequest:
        with self._lock:
            req = self._load(approval_id)
            if req.requester != principal.sub or req.tenant != principal.tenant:
                raise PermissionDenied("only the requester can cancel a request")
            if req.status == "pending":
                req.status = "cancelled"
                self._save(req)
                get_audit().for_principal(principal, "approval.decision", resource=req.tool, decision="cancelled",
                                          approval_id=req.id)
        return req

    def consume(self, approval_id: str, *, tool: str, arguments: dict[str, Any],
                principal: Principal) -> ApprovalRequest:
        """Check that an approval authorises *this* call and mark it used. Raises if it does not."""
        with self._lock:
            req = self._load(approval_id)
            problem = None
            if req.tenant != principal.tenant or req.requester != principal.sub:
                problem = "approval belongs to another user"
            elif req.tool != tool:
                problem = f"approval is for tool {req.tool}, not {tool}"
            elif req.args_hash != args_hash(arguments):
                problem = "arguments differ from the approved call"
            elif req.status == "used":
                problem = "approval was already used"
            elif req.status != "approved":
                problem = f"approval is {req.status}"
            elif req.expires_at <= time.time():
                problem = "approval expired before use"
            if problem:
                get_audit().for_principal(principal, "approval.use", resource=tool, decision="refused",
                                          approval_id=approval_id, reason=problem)
                raise PermissionDenied(problem)
            req.status, req.used_at = "used", time.time()
            self._save(req)
            get_audit().for_principal(principal, "approval.use", resource=tool, decision="used",
                                      approval_id=approval_id, args_hash=req.args_hash)
            return req


_store: ApprovalStore | None = None
_store_key: tuple[int, str] | None = None
_store_lock = threading.Lock()


def get_approval_store() -> ApprovalStore:
    """Process-wide store configured from the ``approvals`` section; rebuilt when the configuration changes."""
    global _store, _store_key
    cfg = get_config()
    key = (id(cfg), cfg.version)
    with _store_lock:
        if _store is None or (_store_key is not None and _store_key != key):
            _store = ApprovalStore(config=cfg.section("approvals", ApprovalsConfig))
            _store_key = key
        return _store


def set_approval_store(store: ApprovalStore | None) -> None:
    """Install a store (tests / embedding). An installed store is kept across configuration changes."""
    global _store, _store_key
    with _store_lock:
        _store, _store_key = store, None
