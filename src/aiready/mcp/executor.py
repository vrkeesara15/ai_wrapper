"""Tool executor: the single choke point every tool call passes through, from agents and outside assistants alike.

    call(name, arguments, principal, approval_id=None)
      1. resolve the tool; check the caller holds a required role
      2. validate the arguments against the tool's JSON schema (a malformed call never reaches the app)
      3. policy ``tool.call`` (resource: tool name, effect, data class) -> deny / require approval / allow
      4. approval gate: reads run freely; writes, payments and deletions create an approval request and return
         ``ToolResult(ok=False, approval_id=...)``. Replaying the call with an *approved* id that matches the
         same tool, the same arguments (hash) and the same user executes it exactly once
      5. execute with a timeout, as the user (their token is forwarded - no shared service account)
      6. cap the result size, audit ``tool.call`` (arguments hash, effect, decision), trace span, metrics

Expected failures (unknown tool, denied, invalid arguments, upstream error) come back as ``ToolResult`` with
``ok=False`` and an ``error`` message so a model can read them and recover; nothing raises past this layer
except programming errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import jsonschema

from aiready.agents.approvals import ApprovalStore, args_hash, get_approval_store
from aiready.core.audit import get_audit
from aiready.core.errors import AIReadyError
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput, ToolDefinition, ToolResult
from aiready.core.telemetry import metrics, span
from aiready.core.wiring import connect
from aiready.mcp.config import ToolsConfig
from aiready.mcp.registry import CURRENT_APPROVAL, ToolRegistry

log = logging.getLogger(__name__)
SERVICE = "mcp"


def principal_view(p: Principal) -> dict[str, Any]:
    return {"sub": p.sub, "tenant": p.tenant, "app": p.app, "roles": p.roles, "groups": p.groups,
            "scopes": p.scopes, "actor": p.actor, "service": p.service}


def visible(defn: ToolDefinition, principal: Principal) -> bool:
    return not defn.required_roles or principal.has_role(*defn.required_roles)


class ToolExecutorService:
    """In-process implementation of ``core.interfaces.ToolExecutor``."""

    def __init__(self, registry: ToolRegistry, cfg: ToolsConfig, *, approvals: ApprovalStore | None = None):
        self.registry = registry
        self.cfg = cfg
        self._approvals = approvals

    @property
    def approvals(self) -> ApprovalStore:
        return self._approvals or get_approval_store()

    # ------------------------------------------------------------------ listing
    async def list_tools(self, principal: Principal) -> list[ToolDefinition]:
        """Tools the caller may see: role-filtered, so an assistant never even learns of a tool it cannot use."""
        await self.registry.ensure_loaded()
        return [t.definition for t in self.registry.all() if visible(t.definition, principal)]

    # ------------------------------------------------------------------ calling
    async def call(self, name: str, arguments: dict[str, Any], principal: Principal, *,
                   approval_id: str | None = None, context: dict[str, Any] | None = None) -> ToolResult:
        await self.registry.ensure_loaded()
        arguments = dict(arguments or {})
        context = context or {}
        ahash = args_hash(arguments)
        audit = get_audit()
        with span("mcp.call", tool=name, approval_id=approval_id) as sp, metrics.time_stage(SERVICE, "tool.call"):
            tool = self.registry.get(name)
            if tool is None:
                audit.for_principal(principal, "tool.call", action="tool.call", resource=name, decision="unknown_tool",
                                    args_hash=ahash)
                return ToolResult(ok=False, error=f"unknown tool '{name}'")
            defn = tool.definition
            sp.set(effect=defn.effect, data_class=str(defn.data_class.value))

            if not visible(defn, principal):
                return self._deny(principal, defn, ahash, f"tool '{name}' requires one of roles {defn.required_roles}",
                                  "denied_role")

            try:
                jsonschema.validate(arguments, defn.input_schema)
            except jsonschema.ValidationError as exc:
                where = "/".join(str(p) for p in exc.absolute_path) or "arguments"
                audit.for_principal(principal, "tool.call", action="tool.call", resource=name,
                                    decision="invalid_arguments", args_hash=ahash, effect=defn.effect)
                return ToolResult(ok=False, error=f"invalid arguments for {name}: {where}: {exc.message}")
            except jsonschema.SchemaError as exc:
                log.error("tool %s has an invalid input schema: %s", name, exc)
                return ToolResult(ok=False, error=f"tool {name} is misconfigured")

            decision = await connect("policy").decide(PolicyInput(
                action="tool.call", principal=principal_view(principal),
                resource={"tool": name, "name": name, "effect": defn.effect, "data_class": defn.data_class.value,
                          "source": defn.source, "required_roles": defn.required_roles},
                context={"args_hash": ahash, "approval_id": approval_id, "app": principal.app, **context},
            ))
            metrics.policy.labels(SERVICE, "tool.call", decision.effect if decision.allow else "deny").inc()
            if not decision.allow and decision.effect != "require_approval":
                return self._deny(principal, defn, ahash, "; ".join(decision.reasons) or "denied by policy",
                                  "denied_policy", policy_version=decision.policy_version,
                                  matched_rules=decision.matched_rules)

            needs_approval = defn.effect != "read" or decision.effect == "require_approval" \
                or bool(decision.obligations.get("require_approval"))
            if needs_approval:
                if not approval_id:
                    req = self.approvals.create(tool=name, effect=defn.effect, arguments=arguments,
                                                principal=principal, run_id=context.get("run_id"),
                                                step=context.get("step"), data_class=defn.data_class.value,
                                                reason=context.get("reason"),
                                                approver_roles=decision.obligations.get("approval_roles"))
                    audit.for_principal(principal, "tool.call", action="tool.call", resource=name,
                                        decision="approval_required", args_hash=ahash, effect=defn.effect,
                                        approval_id=req.id, policy_version=decision.policy_version)
                    sp.set(decision="approval_required")
                    return ToolResult(ok=False, approval_id=req.id,
                                      error=f"'{name}' is a {defn.effect} action and waits for approval",
                                      content={"status": "pending_approval", "approval_id": req.id,
                                               "approver_roles": req.approver_roles, "expires_at": req.expires_at})
                try:
                    self.approvals.consume(approval_id, tool=name, arguments=arguments, principal=principal)
                except AIReadyError as exc:
                    return self._deny(principal, defn, ahash, f"approval {approval_id} not usable: {exc.message}",
                                      "approval_refused")

            return await self._execute(tool, defn, arguments, principal, ahash, approval_id, sp,
                                       obligations=decision.obligations)

    async def _execute(self, tool: Any, defn: ToolDefinition, arguments: dict[str, Any], principal: Principal,
                       ahash: str, approval_id: str | None, sp: Any, *, obligations: dict[str, Any]) -> ToolResult:
        audit = get_audit()
        timeout = float(obligations.get("timeout_seconds") or defn.timeout_seconds or self.cfg.default_timeout_seconds)
        t0 = time.perf_counter()
        attempts = 1 + (self.cfg.http_retries if defn.effect == "read" else 0)  # never retry a write blindly
        last_error = ""
        content: Any = None
        succeeded = False
        for attempt in range(attempts):
            token = CURRENT_APPROVAL.set(approval_id)
            try:
                content = await asyncio.wait_for(tool.invoke(arguments, principal), timeout=timeout)
                succeeded = True
                break
            except TimeoutError:
                last_error = f"tool {defn.name} timed out after {timeout:g}s"
            except AIReadyError as exc:
                last_error = exc.message
                if exc.status_code < 500:  # a 4xx from the app will not get better on retry
                    break
            except Exception as exc:  # a broken tool must not take the executor down
                log.exception("tool %s failed", defn.name)
                last_error = f"tool {defn.name} failed: {type(exc).__name__}"
            finally:
                CURRENT_APPROVAL.reset(token)
            if attempt < attempts - 1:
                await asyncio.sleep(0.2 * 2 ** attempt)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        if not succeeded:
            audit.for_principal(principal, "tool.call", action="tool.call", resource=defn.name, decision="error",
                                args_hash=ahash, effect=defn.effect, approval_id=approval_id, error=last_error,
                                latency_ms=elapsed_ms)
            sp.set(decision="error")
            sp.error(last_error)
            return ToolResult(ok=False, error=last_error)

        if obligations.get("mask_pii"):
            content = await mask_content(content)
        content, truncated = cap_result(content, int(obligations.get("max_result_bytes") or self.cfg.max_result_bytes))
        audit.for_principal(principal, "tool.call", action="tool.call", resource=defn.name, decision="executed",
                            args_hash=ahash, effect=defn.effect, approval_id=approval_id, latency_ms=elapsed_ms,
                            truncated=truncated, data_class=defn.data_class.value)
        sp.set(decision="executed", truncated=truncated, latency_ms=elapsed_ms)
        return ToolResult(ok=True, content=content)

    def _deny(self, principal: Principal, defn: ToolDefinition, ahash: str, reason: str, decision: str,
              **extra: Any) -> ToolResult:
        get_audit().for_principal(principal, "tool.call", action="tool.call", resource=defn.name, decision=decision,
                                  args_hash=ahash, effect=defn.effect, reason=reason, **extra)
        return ToolResult(ok=False, error=reason)


async def mask_content(content: Any) -> Any:
    """Policy obligation ``mask_pii``: personal data in a tool result is masked before any model sees it.
    If the guardrails service is unavailable the result is withheld rather than passed through unmasked."""
    try:
        guard = connect("guardrails")
    except Exception as exc:
        log.error("mask_pii obligation but guardrails unavailable: %s", exc)
        return {"withheld": True, "reason": "personal data could not be masked"}
    is_text = isinstance(content, str)
    raw = content if is_text else json.dumps(content, default=str, ensure_ascii=False)
    masked = (await guard.mask(raw)).text
    if is_text:
        return masked
    try:
        return json.loads(masked)
    except ValueError:
        return masked


def cap_result(content: Any, max_bytes: int) -> tuple[Any, bool]:
    """Keep tool output within ``max_bytes`` of JSON so one call cannot flood a model's context window."""
    try:
        raw = json.dumps(content, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        raw = str(content)
    if len(raw.encode()) <= max_bytes:
        return content, False
    cut = raw.encode()[:max_bytes].decode(errors="ignore")
    return {"truncated": True, "max_bytes": max_bytes, "preview": cut}, True
