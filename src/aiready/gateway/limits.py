"""Step 2 - Limit: rate limits and token budgets.

State lives in the shared KV store (Redis with several replicas) so the data plane itself stays stateless.

* Rate limits use a *sliding window counter*: the count of the current fixed window plus the previous window's
  count weighted by how much of it still overlaps the sliding window. Two keys per subject, O(1), and accurate
  to within a few percent - good enough to protect model capacity without storing every timestamp.
* Budgets accumulate tokens and cost per subject and period (day / month, UTC). They are checked before the
  call with the prompt estimate and charged with the real usage after it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aiready.core.errors import BudgetExceeded, RateLimited
from aiready.core.stores import KVStore
from aiready.gateway.config import BudgetRule, RateLimitRule


@dataclass
class Subjects:
    """The values each limit scope resolves to for one request."""

    user: str
    team: str
    app: str
    feature: str
    tenant: str

    def value(self, scope: str) -> str:
        return getattr(self, scope)

    def as_dict(self) -> dict[str, str]:
        return {"user": self.user, "team": self.team, "app": self.app, "feature": self.feature, "tenant": self.tenant}


def _applies(match: str, value: str) -> bool:
    return match == "*" or match == value


class RateLimiter:
    def __init__(self, kv: KVStore):
        self.kv = kv

    def check(self, rules: dict[str, RateLimitRule], subjects: Subjects, operation: str,
              now: float | None = None) -> list[dict[str, Any]]:
        """Count this request against every applicable rule; raise ``RateLimited`` if any is exhausted.

        All rules are evaluated before any counter is incremented, so a refused request does not consume quota
        in the rules it did not break.
        """
        now = now or time.time()
        pending: list[tuple[str, float]] = []
        status: list[dict[str, Any]] = []
        for name, rule in rules.items():
            if not rule.enabled or (rule.operations and operation not in rule.operations):
                continue
            subject = subjects.value(rule.scope)
            if not _applies(rule.match, subject):
                continue
            w = rule.window_seconds
            idx = math.floor(now / w)
            cur_key = f"rl:{name}:{rule.scope}:{subject}:{idx}"
            prev = float(self.kv.get(f"rl:{name}:{rule.scope}:{subject}:{idx - 1}") or 0)
            cur = float(self.kv.get(cur_key) or 0)
            elapsed = (now - idx * w) / w
            estimated = cur + prev * (1 - elapsed)
            if estimated + 1 > rule.requests:
                retry_after = max(0.1, round((idx + 1) * w - now, 2))
                raise RateLimited(f"rate limit '{name}' exceeded for {rule.scope} '{subject}'",
                                  details={"rule": name, "scope": rule.scope, "limit": rule.requests,
                                           "window_seconds": w, "retry_after": retry_after})
            pending.append((cur_key, w * 2))
            status.append({"rule": name, "scope": rule.scope, "limit": rule.requests,
                           "remaining": max(0, int(rule.requests - estimated - 1))})
        for key, ttl in pending:
            self.kv.incr(key, 1, ttl=ttl)
        return status


def period_key(period: str, now: float | None = None) -> tuple[str, float]:
    """(period id, seconds until it ends) in UTC."""
    dt = datetime.fromtimestamp(now or time.time(), tz=UTC)
    if period == "month":
        nxt = datetime(dt.year + (dt.month == 12), dt.month % 12 + 1, 1, tzinfo=UTC)
        return dt.strftime("%Y-%m"), (nxt - dt).total_seconds()
    nxt_day = datetime(dt.year, dt.month, dt.day, tzinfo=UTC).timestamp() + 86400
    return dt.strftime("%Y-%m-%d"), nxt_day - dt.timestamp()


class BudgetTracker:
    def __init__(self, kv: KVStore):
        self.kv = kv

    @staticmethod
    def _keys(name: str, rule: BudgetRule, subject: str, now: float | None) -> tuple[str, str, float]:
        pid, remaining = period_key(rule.period, now)
        base = f"budget:{name}:{rule.scope}:{subject}:{pid}"
        return base + ":tokens", base + ":cost", remaining + 3600

    def _applicable(self, rules: dict[str, BudgetRule], subjects: Subjects):
        for name, rule in rules.items():
            if rule.enabled and _applies(rule.match, subjects.value(rule.scope)):
                yield name, rule, subjects.value(rule.scope)

    def check(self, rules: dict[str, BudgetRule], subjects: Subjects, estimated_tokens: int,
              now: float | None = None) -> list[dict[str, Any]]:
        """Refuse when a blocking budget would be exceeded by this request. Returns warnings for ``warn`` rules."""
        warnings: list[dict[str, Any]] = []
        for name, rule, subject in self._applicable(rules, subjects):
            tk, ck, _ = self._keys(name, rule, subject, now)
            used_tokens = float(self.kv.get(tk) or 0)
            used_cost = float(self.kv.get(ck) or 0)
            over_tokens = rule.max_tokens is not None and used_tokens + estimated_tokens > rule.max_tokens
            over_cost = rule.max_cost is not None and used_cost >= rule.max_cost
            if not (over_tokens or over_cost):
                continue
            detail = {"budget": name, "scope": rule.scope, "subject": subject, "period": rule.period,
                      "used_tokens": int(used_tokens), "max_tokens": rule.max_tokens,
                      "used_cost": round(used_cost, 6), "max_cost": rule.max_cost}
            if rule.action == "block":
                raise BudgetExceeded(f"{rule.period} budget '{name}' exhausted for {rule.scope} '{subject}'",
                                     details=detail)
            warnings.append(detail)
        return warnings

    def charge(self, rules: dict[str, BudgetRule], subjects: Subjects, tokens: int, cost: float,
               now: float | None = None) -> None:
        if tokens <= 0 and cost <= 0:
            return
        for name, rule, subject in self._applicable(rules, subjects):
            tk, ck, ttl = self._keys(name, rule, subject, now)
            self.kv.incr(tk, tokens, ttl=ttl)
            if cost:
                self.kv.incr(ck, cost, ttl=ttl)

    def status(self, rules: dict[str, BudgetRule], subjects: Subjects, now: float | None = None) -> list[dict[str, Any]]:
        out = []
        for name, rule, subject in self._applicable(rules, subjects):
            tk, ck, _ = self._keys(name, rule, subject, now)
            out.append({"budget": name, "scope": rule.scope, "subject": subject, "period": rule.period,
                        "used_tokens": int(float(self.kv.get(tk) or 0)), "max_tokens": rule.max_tokens,
                        "used_cost": round(float(self.kv.get(ck) or 0), 6), "max_cost": rule.max_cost,
                        "action": rule.action})
        return out
