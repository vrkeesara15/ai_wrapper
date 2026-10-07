"""Release gate: compare scorecards to the baseline and to absolute thresholds.

Release pipeline slide, step 2: "Evaluation suite - score must not drop", "Safety tests - injection and data
leaks", "Cost and latency budget". The gate is a pure function of scorecards + ``evals.gate`` configuration,
so CI, the control plane and a person reading the report reach the same verdict. ``run_gate`` adds the side
effects: audit record, registry entry (best-effort), and an exit code for CI.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.identity import Principal
from aiready.evals.runner import Scorecard
from aiready.observability.governance import service_principal

log = logging.getLogger(__name__)


class GateThresholds(BaseModel):
    max_drop: float = 0.0  # overall score may drop at most this much vs the baseline ("must not drop")
    per_scorer_max_drop: float | None = 0.02  # per-scorer tolerance (None = not checked)
    min_overall: float | None = None
    min_pass_rate: float | None = None
    min_scorer: dict[str, float] = Field(default_factory=dict)  # absolute floor per scorer mean
    safety_pass_rate: float = 1.0  # safety suites: share of cases that must pass
    max_latency_p95_ms: float | None = None
    max_cost_per_case: float | None = None
    max_errors: int = 0


class GateConfig(GateThresholds):
    require_baseline: bool = False  # fail when no baseline exists (set true once the first release is in)
    suites: dict[str, GateThresholds] = Field(default_factory=dict)  # per-suite overrides (merged on defaults)

    def for_suite(self, suite: str) -> GateThresholds:
        base = GateThresholds.model_validate(self.model_dump(exclude={"require_baseline", "suites"}))
        if suite in self.suites:
            override = self.suites[suite].model_dump(exclude_unset=True)
            return base.model_copy(update=override)
        return base


class GateCheck(BaseModel):
    suite: str
    name: str
    passed: bool
    actual: float | None = None
    expected: float | None = None
    message: str = ""


class GateResult(BaseModel):
    release: str
    passed: bool
    checks: list[GateCheck]
    baselines: dict[str, str | None] = Field(default_factory=dict)  # suite -> baseline release
    registry_recorded: bool = False

    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def to_markdown(self) -> str:
        lines = [f"# Release gate {self.release}: {'PASSED' if self.passed else 'BLOCKED'}", "",
                 "| Suite | Check | Result | Actual | Expected | Note |", "|---|---|---|---|---|---|"]
        for c in self.checks:
            fmt = lambda v: "-" if v is None else f"{v:.4g}"
            lines.append(f"| {c.suite} | {c.name} | {'pass' if c.passed else '**FAIL**'} | {fmt(c.actual)} | "
                         f"{fmt(c.expected)} | {c.message} |")
        return "\n".join(lines) + "\n"


def load_gate_config() -> GateConfig:
    from aiready.core.config import get_config

    return get_config().section("evals.gate", GateConfig)


def evaluate_gate(cards: list[Scorecard], baselines: dict[str, Scorecard | None] | None = None,
                  cfg: GateConfig | None = None, *, release: str | None = None) -> GateResult:
    cfg = cfg or GateConfig()
    baselines = baselines or {}
    checks: list[GateCheck] = []

    def add(suite: str, name: str, ok: bool, actual: float | None, expected: float | None, msg: str = "") -> None:
        checks.append(GateCheck(suite=suite, name=name, passed=bool(ok), actual=actual, expected=expected, message=msg))

    for card in cards:
        th = cfg.for_suite(card.suite)
        s = card.suite
        if card.safety:
            add(s, "safety_pass_rate", card.pass_rate >= th.safety_pass_rate, card.pass_rate, th.safety_pass_rate,
                f"{len(card.failed_cases())} safety case(s) failed: {[c.case_id for c in card.failed_cases()][:10]}"
                if card.failed_cases() else "all safety cases passed")
        add(s, "errors", card.n_errors <= th.max_errors, card.n_errors, th.max_errors)
        if th.min_overall is not None:
            add(s, "min_overall", card.overall >= th.min_overall, card.overall, th.min_overall)
        if th.min_pass_rate is not None:
            add(s, "min_pass_rate", card.pass_rate >= th.min_pass_rate, card.pass_rate, th.min_pass_rate)
        for scorer, floor in th.min_scorer.items():
            agg = card.per_scorer.get(scorer)
            if agg is None or agg.n == 0:
                add(s, f"min_scorer.{scorer}", False, None, floor, "scorer did not run on any case")
            else:
                add(s, f"min_scorer.{scorer}", agg.mean >= floor, agg.mean, floor)
        if th.max_latency_p95_ms is not None:
            lat = card.latency_p95_ms
            add(s, "latency_p95_ms", lat is not None and lat <= th.max_latency_p95_ms, lat, th.max_latency_p95_ms,
                "" if lat is not None else "no latency measured")
        if th.max_cost_per_case is not None and card.cost_per_case is not None:
            add(s, "cost_per_case", card.cost_per_case <= th.max_cost_per_case, card.cost_per_case, th.max_cost_per_case)
        base = baselines.get(card.suite)
        if base is None:
            if cfg.require_baseline:
                add(s, "baseline", False, None, None, "no baseline scorecard (require_baseline is on)")
            continue
        add(s, "no_drop.overall", card.overall >= base.overall - th.max_drop - 1e-9, card.overall,
            round(base.overall - th.max_drop, 6), f"baseline {base.release}: {base.overall:.4f}")
        if th.per_scorer_max_drop is not None:
            for name, agg in base.per_scorer.items():
                cur = card.per_scorer.get(name)
                if agg.n == 0 or cur is None or cur.n == 0:
                    continue
                add(s, f"no_drop.{name}", cur.mean >= agg.mean - th.per_scorer_max_drop - 1e-9, cur.mean,
                    round(agg.mean - th.per_scorer_max_drop, 6), f"baseline {base.release}: {agg.mean:.4f}")
    rel = release or (cards[0].release if cards else "unknown")
    return GateResult(release=rel, passed=bool(cards) and all(c.passed for c in checks), checks=checks,
                      baselines={c.suite: (baselines.get(c.suite).release if baselines.get(c.suite) else None) for c in cards})


async def record_gate(result: GateResult, *, principal: Principal | None = None,
                      artifact: dict[str, Any] | None = None) -> GateResult:
    """Audit the verdict and, when a registry is deployed, attach it to the release / prompt / model version.
    The registry call is duck-typed and best-effort: the audit log is the system of record for the verdict."""
    actor = principal or service_principal("evals")
    get_audit().for_principal(actor, "evals.gate", resource=result.release, decision="pass" if result.passed else "block",
                              failures=[f"{c.suite}:{c.name}" for c in result.failures()], baselines=result.baselines,
                              artifact=artifact or {})
    try:
        from aiready.core.wiring import connect

        reg = connect("registry")
        for method in ("record_evaluation", "record_gate", "add_evaluation"):
            fn = getattr(reg, method, None)
            if fn is None:
                continue
            res = fn(release=result.release, passed=result.passed, report=result.model_dump(), artifact=artifact or {},
                     principal=actor)
            if inspect.isawaitable(res):
                await res
            result.registry_recorded = True
            break
    except Exception as exc:
        log.info("registry not updated with gate result (%s)", exc)
    return result


def exit_code(result: GateResult) -> int:
    """0 = release may proceed, 1 = blocked. CI fails the pipeline on non-zero."""
    return 0 if result.passed else 1
