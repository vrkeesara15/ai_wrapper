"""Offline evaluation runner: suite x target -> scorecard, with a history per release.

"Offline runner - blocks a release that scores lower" and "Scorecard - trend for each release" (observability
slide). The runner only *measures*; ``gate.py`` decides. Each case runs concurrently under its persona's
identity, every scorer that applies scores it, and the scorecard aggregates per scorer and per tag so a drop
can be traced to a kind of question rather than just a number.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.core.telemetry import metrics, span
from aiready.evals.golden import EvalCase, EvalSuite, Persona, validate_suite
from aiready.evals.scorers import (
    DEFAULT_QA_SCORERS,
    DEFAULT_SAFETY_SCORERS,
    ScoreInput,
    Scorer,
    ScoreResult,
    build_scorers,
)
from aiready.evals.targets import Target, TargetOutput, build_target, timed
from aiready.observability.governance import service_principal
from aiready.observability.masking import RegexMasker
from aiready.observability.usage import percentile

log = logging.getLogger(__name__)
SERVICE = "evals"
NS = "evals.scorecards"


class CaseResult(BaseModel):
    case_id: str
    task: str = "qa"
    tags: list[str] = Field(default_factory=list)
    persona: str | None = None
    answer: str = ""  # masked, truncated
    scores: dict[str, ScoreResult] = Field(default_factory=dict)
    score: float = 0.0  # mean of applicable scorers
    passed: bool = False
    error: str | None = None
    blocked: str | None = None
    latency_ms: float | None = None
    cost: float | None = None
    trace_id: str | None = None
    weight: float = 1.0


class ScorerAggregate(BaseModel):
    mean: float
    pass_rate: float
    n: int
    skipped: int = 0


class Scorecard(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    suite: str
    suite_version: str = "1"
    safety: bool = False
    release: str
    target: str
    started_at: float
    finished_at: float
    n_cases: int
    n_errors: int
    overall: float
    pass_rate: float
    per_scorer: dict[str, ScorerAggregate]
    per_tag: dict[str, dict[str, float]]
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    cost_total: float | None = None
    cost_per_case: float | None = None
    cases: list[CaseResult]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def to_markdown(self) -> str:
        lines = [f"# Scorecard: {self.suite} @ {self.release}", "",
                 f"Target `{self.target}` · {self.n_cases} cases · {self.n_errors} errors · "
                 f"{'safety suite' if self.safety else 'quality suite'}", "",
                 "| Metric | Value |", "|---|---|",
                 f"| Overall score | {self.overall:.3f} |", f"| Case pass rate | {self.pass_rate:.1%} |"]
        if self.latency_p95_ms is not None:
            lines.append(f"| Latency p50 / p95 | {self.latency_p50_ms:.0f} / {self.latency_p95_ms:.0f} ms |")
        if self.cost_total is not None:
            lines.append(f"| Cost total / per case | {self.cost_total:.4f} / {self.cost_per_case:.4f} |")
        lines += ["", "## Scorers", "", "| Scorer | Mean | Pass rate | Cases | Skipped |", "|---|---|---|---|---|"]
        for name, a in sorted(self.per_scorer.items()):
            lines.append(f"| {name} | {a.mean:.3f} | {a.pass_rate:.1%} | {a.n} | {a.skipped} |")
        if self.per_tag:
            lines += ["", "## By tag", "", "| Tag | " + " | ".join(sorted(self.per_scorer)) + " |",
                      "|---|" + "---|" * len(self.per_scorer)]
            for tag, vals in sorted(self.per_tag.items()):
                lines.append(f"| {tag} | " + " | ".join(f"{vals[s]:.2f}" if s in vals else "-" for s in sorted(self.per_scorer)) + " |")
        failed = self.failed_cases()
        if failed:
            lines += ["", f"## Failed cases ({len(failed)})", ""]
            for c in failed[:50]:
                why = c.error or ", ".join(f"{k}={v.score:.2f}" for k, v in c.scores.items() if not v.passed and not v.skipped)
                lines.append(f"- `{c.case_id}` ({', '.join(c.tags) or 'untagged'}): {why}")
        return "\n".join(lines) + "\n"


class ScorecardHistory:
    """Scorecards per suite and release (DocStore), plus optional JSON/markdown files for CI artifacts."""

    def __init__(self, store: DocStore | None = None, output_dir: str | Path | None = None):
        self._store = store
        self.output_dir = Path(output_dir) if output_dir else None

    @property
    def store(self) -> DocStore:
        return self._store or get_docstore()

    def save(self, card: Scorecard) -> None:
        self.store.put(NS, f"{card.suite}/{card.release}", card.model_dump())
        if self.output_dir:
            d = self.output_dir / card.suite
            d.mkdir(parents=True, exist_ok=True)
            safe = card.release.replace("/", "_")
            (d / f"{safe}.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
            (d / f"{safe}.md").write_text(card.to_markdown(), encoding="utf-8")

    def get(self, suite: str, release: str) -> Scorecard | None:
        raw = self.store.get(NS, f"{suite}/{release}")
        return Scorecard.model_validate(raw) if raw else None

    def list(self, suite: str) -> list[Scorecard]:
        cards = [Scorecard.model_validate(v) for v in self.store.values(NS, prefix=f"{suite}/")]
        return sorted([c for c in cards if c.suite == suite], key=lambda c: c.finished_at)

    def latest(self, suite: str, *, exclude_release: str | None = None) -> Scorecard | None:
        cards = [c for c in self.list(suite) if c.release != exclude_release]
        return cards[-1] if cards else None

    def baseline(self, suite: str, ref: str | None, *, current_release: str | None = None) -> Scorecard | None:
        """``latest`` (previous release), a release name, or a path to a scorecard JSON."""
        if not ref or ref == "none":
            return None
        if ref == "latest":
            return self.latest(suite, exclude_release=current_release)
        p = Path(ref)
        if p.suffix == ".json" and p.exists():
            return Scorecard.model_validate_json(p.read_text(encoding="utf-8"))
        return self.get(suite, ref)

    def trend(self, suite: str) -> list[dict[str, Any]]:
        return [{"release": c.release, "finished_at": c.finished_at, "overall": c.overall, "pass_rate": c.pass_rate,
                 **{f"scorer.{k}": v.mean for k, v in c.per_scorer.items()}} for c in self.list(suite)]


def persona_principal(persona: Persona, *, actor: str = "agent:eval-runner") -> Principal:
    """A signed token for the persona when a signing key is available (HTTP targets need one), else an
    in-process principal. The actor claim marks every call as made by the evaluation runner."""
    try:
        from aiready.core.identity import get_token_service

        ts = get_token_service()
        tok = ts.issue(sub=persona.sub, tenant=persona.tenant, app=persona.app, roles=persona.roles,
                       groups=persona.groups, actor=actor, ttl=3600)
        return ts.verify(tok)
    except Exception:
        return Principal(sub=persona.sub, tenant=persona.tenant, app=persona.app, roles=persona.roles,
                         groups=persona.groups, actor=actor)


class EvalRunner:
    def __init__(self, target: Target | dict[str, Any] | str, *, scorers: list[Scorer] | None = None,
                 concurrency: int = 4, history: ScorecardHistory | None = None, mask_answers: bool = True,
                 max_answer_chars: int = 2000):
        self.target = build_target(target)
        self.scorers = scorers
        self.concurrency = max(1, concurrency)
        self.history = history or ScorecardHistory()
        self.masker = RegexMasker() if mask_answers else None
        self.max_answer_chars = max_answer_chars

    def _scorers_for(self, suite: EvalSuite) -> list[Scorer]:
        if self.scorers is not None:
            return self.scorers
        specs = suite.scorers or (DEFAULT_SAFETY_SCORERS if suite.safety else DEFAULT_QA_SCORERS)
        return build_scorers(specs, suite.thresholds)

    async def _run_case(self, suite: EvalSuite, case: EvalCase, scorers: list[Scorer], sem: asyncio.Semaphore) -> CaseResult:
        persona = suite.persona_for(case)
        principal = persona_principal(persona)
        pname = case.persona if isinstance(case.persona, str) else persona.sub
        async with sem:
            with span("evals.case", suite=suite.name, case=case.id, persona=persona.sub) as sp:
                try:
                    out: TargetOutput = await timed(self.target, case, principal)
                except Exception as exc:
                    sp.error(exc)
                    log.warning("case %s failed: %s", case.id, exc)
                    return CaseResult(case_id=case.id, task=case.task, tags=case.tags, persona=pname,
                                      error=f"{type(exc).__name__}: {exc}", weight=case.weight)
        inp = ScoreInput(question=case.question, answer=out.answer, passages=out.passages,
                         cited_sources=out.cited_sources, cited_chunks=out.cited_chunks, found=out.found,
                         parsed=out.parsed, tool_calls=out.tool_calls, latency_ms=out.latency_ms, cost=out.cost,
                         case=case, principal=principal)
        results: dict[str, ScoreResult] = {}
        for sc in scorers:
            if case.scorers is not None and sc.name not in case.scorers:
                continue
            try:
                results[sc.name] = await sc.score(inp)
            except Exception as exc:
                results[sc.name] = ScoreResult(scorer=sc.name, score=0.0, passed=False, details={"error": str(exc)})
        used = [r for r in results.values() if not r.skipped]
        answer = out.answer[: self.max_answer_chars]
        return CaseResult(
            case_id=case.id, task=case.task, tags=case.tags, persona=pname,
            answer=self.masker.mask(answer) if self.masker else answer, scores=results,
            score=round(sum(r.score for r in used) / len(used), 4) if used else 1.0,
            passed=all(r.passed for r in used), blocked=out.blocked, latency_ms=out.latency_ms, cost=out.cost,
            trace_id=out.trace_id, weight=case.weight,
        )

    async def run(self, suite: EvalSuite, *, release: str, operator: Principal | None = None,
                  metadata: dict[str, Any] | None = None, save: bool = True) -> Scorecard:
        problems = validate_suite(suite)
        if problems:
            from aiready.core.errors import ValidationFailed

            raise ValidationFailed(f"suite '{suite.name}' is invalid", details={"problems": problems})
        prep = getattr(self.target, "prepare", None)
        if prep:
            await prep(suite)
        scorers = self._scorers_for(suite)
        sem = asyncio.Semaphore(self.concurrency)
        started = time.time()
        with span("evals.run", suite=suite.name, release=release, target=self.target.name), \
                metrics.time_stage(SERVICE, "run"):
            cases = await asyncio.gather(*(self._run_case(suite, c, scorers, sem) for c in suite.cases))
        card = aggregate(suite, list(cases), release=release, target=self.target.name, started_at=started,
                         metadata=metadata or {})
        if save:
            self.history.save(card)
        actor = operator or service_principal(SERVICE)
        get_audit().for_principal(actor, f"{SERVICE}.run", resource=f"{suite.name}@{release}", decision="completed",
                                  overall=card.overall, pass_rate=card.pass_rate, n_cases=card.n_cases,
                                  n_errors=card.n_errors, target=card.target, scorecard_id=card.id)
        for name, agg in card.per_scorer.items():
            metrics.quality.labels(SERVICE, "evals", suite.name, name).set(agg.mean)
        return card


def aggregate(suite: EvalSuite, cases: list[CaseResult], *, release: str, target: str, started_at: float,
              metadata: dict[str, Any] | None = None) -> Scorecard:
    per: dict[str, list[ScoreResult]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    tag_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        for name, r in c.scores.items():
            if r.skipped:
                skipped[name] += 1
                continue
            per[name].append(r)
            for t in c.tags or ["untagged"]:
                tag_vals[t][name].append(r.score)
        if c.error:  # an errored case counts as zero for every scorer it would have used
            for t in c.tags or ["untagged"]:
                tag_vals[t]["_errors"].append(1.0)
    per_scorer = {n: ScorerAggregate(mean=round(sum(r.score for r in rs) / len(rs), 4),
                                     pass_rate=round(sum(r.passed for r in rs) / len(rs), 4), n=len(rs),
                                     skipped=skipped.get(n, 0)) for n, rs in per.items()}
    for n, k in skipped.items():
        per_scorer.setdefault(n, ScorerAggregate(mean=0.0, pass_rate=1.0, n=0, skipped=k))
    wsum = sum(c.weight for c in cases) or 1.0
    overall = sum((0.0 if c.error else c.score) * c.weight for c in cases) / wsum
    lat = [c.latency_ms for c in cases if c.latency_ms is not None]
    costs = [c.cost for c in cases if c.cost is not None]
    return Scorecard(
        suite=suite.name, suite_version=suite.version, safety=suite.safety, release=release, target=target,
        started_at=started_at, finished_at=time.time(), n_cases=len(cases), n_errors=sum(1 for c in cases if c.error),
        overall=round(overall, 4), pass_rate=round(sum(c.passed for c in cases) / len(cases), 4) if cases else 0.0,
        per_scorer=per_scorer,
        per_tag={t: {n: round(sum(v) / len(v), 4) for n, v in vals.items() if n != "_errors"} for t, vals in tag_vals.items()},
        latency_p50_ms=round(percentile(lat, 50), 2) if lat else None,
        latency_p95_ms=round(percentile(lat, 95), 2) if lat else None,
        cost_total=round(sum(costs), 6) if costs else None,
        cost_per_case=round(sum(costs) / len(costs), 6) if costs else None,
        cases=cases, metadata=metadata or {},
    )
