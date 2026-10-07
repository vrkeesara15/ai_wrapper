"""Alert rules: quality, latency, spend, error rate and pipeline lag, evaluated periodically.

Rules are configuration (``observability.alerts.rules``), notifiers are plugins (kind ``notifier``: ``log``,
``webhook``, ``slack``, ``email``; or ``class: pkg.mod:Notifier``). A notification is sent when a rule
*changes state* (ok -> firing, firing -> resolved), not on every evaluation, so an incident pages once.
Firing state is persisted, so a restart does not re-page for an alert already sent.

The same rules are exported as Prometheus alert rules (``exports.prometheus_rules``) for clients that alert
from their own monitoring stack instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.plugins import create, register
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import SpanRecord
from aiready.observability.config import AlertRuleConfig, ObservabilityConfig
from aiready.observability.governance import service_principal
from aiready.observability.traces import attr
from aiready.observability.usage import UsageRecord, latency_report, month_bounds

log = logging.getLogger(__name__)
NS = "obs.alerts"

DEFAULT_THRESHOLDS = {"quality": 0.7, "error_rate": 0.05, "pipeline_lag": 3600.0}


class AlertEvent(BaseModel):
    rule: str
    kind: str
    state: str  # firing | resolved
    severity: str
    value: float | None
    threshold: float | None
    message: str
    scope: dict[str, str] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)


class Notifier(Protocol):
    async def send(self, event: AlertEvent) -> None: ...


@register("notifier", "log")
class LogNotifier:
    async def send(self, event: AlertEvent) -> None:
        (log.warning if event.state == "firing" else log.info)("ALERT %s %s: %s", event.state, event.rule, event.message)


@register("notifier", "memory")
class MemoryNotifier:
    """Collects events (tests, the dashboard's recent alerts)."""

    def __init__(self) -> None:
        self.events: list[AlertEvent] = []

    async def send(self, event: AlertEvent) -> None:
        self.events.append(event)


@register("notifier", "webhook")
class WebhookNotifier:
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 5.0):
        self.url, self.headers, self.timeout = url, headers or {}, timeout

    async def send(self, event: AlertEvent) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(self.url, json=event.model_dump(), headers=self.headers)
            r.raise_for_status()


@register("notifier", "slack")
class SlackNotifier:
    """Slack incoming webhook (or any Slack-compatible endpoint, e.g. Mattermost)."""

    def __init__(self, webhook_url: str, channel: str | None = None, timeout: float = 5.0):
        self.url, self.channel, self.timeout = webhook_url, channel, timeout

    async def send(self, event: AlertEvent) -> None:
        icon = ":red_circle:" if event.state == "firing" else ":large_green_circle:"
        payload: dict[str, Any] = {"text": f"{icon} *{event.rule}* {event.state} ({event.severity}): {event.message}"}
        if self.channel:
            payload["channel"] = self.channel
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            (await c.post(self.url, json=payload)).raise_for_status()


@register("notifier", "email")
class EmailNotifier:
    """Plain SMTP (the client's relay). The password is a secret reference, e.g. ``${secret:smtp_password}``."""

    def __init__(self, host: str, to: list[str], sender: str = "aiready-alerts@localhost", port: int = 25,
                 username: str | None = None, password: str | None = None, starttls: bool = False):
        self.host, self.port, self.to, self.sender = host, port, to, sender
        self.username, self.password, self.starttls = username, password, starttls

    def _send_sync(self, event: AlertEvent) -> None:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = f"[AI-Ready {event.severity}] {event.rule} {event.state}"
        msg["From"], msg["To"] = self.sender, ", ".join(self.to)
        msg.set_content(f"{event.message}\n\n{json.dumps(event.model_dump(), indent=2, default=str)}")
        with smtplib.SMTP(self.host, self.port, timeout=10) as s:
            if self.starttls:
                s.starttls()
            if self.username:
                s.login(self.username, self.password or "")
            s.send_message(msg)

    async def send(self, event: AlertEvent) -> None:
        await asyncio.to_thread(self._send_sync, event)


class AlertData(Protocol):
    """What the evaluator needs from the service (kept narrow so it can be tested with plain lists)."""

    def window_spans(self, since: float, until: float) -> list[SpanRecord]: ...
    def window_usage(self, since: float, until: float) -> list[UsageRecord]: ...
    def window_scores(self, since: float, until: float) -> list[Any]: ...


def _in_scope(values: dict[str, Any], scope: dict[str, str]) -> bool:
    return all(str(values.get(k)) == v for k, v in scope.items() if k not in ("stage", "scorer"))


class AlertManager:
    def __init__(self, cfg: ObservabilityConfig, data: AlertData, *, notifiers: dict[str, Notifier] | None = None,
                 store: DocStore | None = None, service: str = "observability"):
        self.cfg = cfg
        self.data = data
        self.notifiers = notifiers if notifiers is not None else {
            name: create("notifier", spec) for name, spec in cfg.alerts.notifiers.items()}
        self._store = store
        self.service = service
        self._task: asyncio.Task[None] | None = None

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    # ------------------------------------------------------------------ rule evaluation
    def measure(self, rule: AlertRuleConfig, now: float | None = None) -> tuple[float | None, float | None, int]:
        """(value, threshold, samples) for one rule."""
        now = now or time.time()
        since = now - rule.window_minutes * 60
        if rule.kind == "quality":
            recs = [r for r in self.data.window_scores(since, now) if _in_scope(r.model_dump(), rule.scope)]
            scorer = rule.scope.get("scorer", "overall")
            vals = [r.overall if scorer == "overall" else r.scores[scorer] for r in recs
                    if scorer == "overall" or scorer in r.scores]
            th = rule.threshold if rule.threshold is not None else DEFAULT_THRESHOLDS["quality"]
            return (sum(vals) / len(vals) if vals else None), th, len(vals)
        if rule.kind == "latency":
            stage = rule.scope.get("stage", "first_token")
            spans = [s for s in self.data.window_spans(since, now) if _in_scope(s.attributes, rule.scope)]
            rows = {r.stage: r for r in latency_report(spans, self.cfg.latency_budget_ms, self.cfg.stage_map)}
            row = rows.get(stage)
            th = rule.threshold if rule.threshold is not None else self.cfg.latency_budget_ms.get(stage)
            return (row.p95_ms if row and row.count else None), th, (row.count if row else 0)
        if rule.kind == "spend":
            if (rule.period or "month") == "month":
                _, since, _ = month_bounds(time.strftime("%Y-%m", time.gmtime(now)))
            recs = [r for r in self.data.window_usage(since, now) if _in_scope(r.model_dump(), rule.scope)]
            th = rule.threshold
            if th is None:
                for dim in ("team", "feature", "app"):
                    if dim in rule.scope:
                        th = self.cfg.budgets.get(f"{dim}:{rule.scope[dim]}")
                        break
            return sum(r.cost for r in recs), th, len(recs)
        if rule.kind == "error_rate":
            recs = [r for r in self.data.window_usage(since, now) if _in_scope(r.model_dump(), rule.scope)]
            th = rule.threshold if rule.threshold is not None else DEFAULT_THRESHOLDS["error_rate"]
            return (sum(r.error for r in recs) / len(recs) if recs else None), th, len(recs)
        if rule.kind == "pipeline_lag":
            lags = [float(v) for s in self.data.window_spans(since, now)
                    if (v := attr(s, "lag_seconds")) is not None and _in_scope(s.attributes, rule.scope)]
            th = rule.threshold if rule.threshold is not None else DEFAULT_THRESHOLDS["pipeline_lag"]
            return (max(lags) if lags else None), th, len(lags)
        raise ValueError(f"unknown alert kind '{rule.kind}'")

    @staticmethod
    def breached(kind: str, value: float, threshold: float) -> bool:
        return value < threshold if kind == "quality" else value > threshold

    async def evaluate(self, now: float | None = None) -> list[AlertEvent]:
        """Evaluate every rule once; returns the state changes (and notifies them)."""
        events: list[AlertEvent] = []
        for rule in self.cfg.alerts.rules:
            try:
                value, th, n = self.measure(rule, now)
            except Exception as exc:
                log.warning("alert rule %s failed: %s", rule.name, exc)
                continue
            firing_now = value is not None and th is not None and n >= rule.min_samples and self.breached(rule.kind, value, th)
            state = self.store.get(NS, rule.name, {}) or {}
            was_firing = state.get("state") == "firing"
            if firing_now == was_firing:
                continue
            cmp = "below" if rule.kind == "quality" else "above"
            ev = AlertEvent(
                rule=rule.name, kind=rule.kind, state="firing" if firing_now else "resolved", severity=rule.severity,
                value=None if value is None else round(value, 4), threshold=th, scope=rule.scope,
                message=(f"{rule.kind} {value:.4g} {cmp} threshold {th:.4g} over {rule.window_minutes} min ({n} samples)"
                         if firing_now and value is not None and th is not None else f"{rule.kind} back within threshold"),
            )
            self.store.put(NS, rule.name, {"state": ev.state, "since": ev.ts, "last": ev.model_dump()})
            get_audit().for_principal(service_principal(self.service), f"{self.service}.alert", resource=rule.name,
                                      decision=ev.state, value=ev.value, threshold=th, kind=rule.kind)
            await self._notify(rule, ev)
            events.append(ev)
        return events

    async def _notify(self, rule: AlertRuleConfig, ev: AlertEvent) -> None:
        targets = rule.notifiers or list(self.notifiers)
        for name in targets:
            n = self.notifiers.get(name)
            if n is None:
                log.warning("alert %s: unknown notifier %s", rule.name, name)
                continue
            try:
                await n.send(ev)
            except Exception as exc:  # one broken channel must not hide the alert from the others
                log.error("notifier %s failed for %s: %s", name, rule.name, exc)

    def active(self) -> list[dict[str, Any]]:
        return [v for _, v in self.store.list(NS) if v.get("state") == "firing"]

    # ------------------------------------------------------------------ background loop
    def start(self) -> None:
        if self._task is None and self.cfg.alerts.enabled and self.cfg.alerts.rules:
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.evaluate()
            except Exception:
                log.exception("alert evaluation failed")
            await asyncio.sleep(self.cfg.alerts.interval_seconds)
