"""Command line for CI: validate golden sets, run suites, apply the release gate, evaluate routing.

    python -m aiready.evals validate config/evals/golden/claims.yaml
    python -m aiready.evals run claims --release 1.4.0 --target rag
    python -m aiready.evals gate claims safety_injection safety_data_leak --release 1.4.0 --baseline latest
    python -m aiready.evals routing --task qa=claims --model small-8b:8 --model chat-70b:70 --release 1.4.0

``gate`` runs the suites, compares with the baseline, records the verdict and exits 1 when the release is
blocked. ``evals_app`` is a Typer app the platform CLI can mount as a sub-command.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from aiready.core.config import get_config
from aiready.evals.gate import evaluate_gate, exit_code, load_gate_config, record_gate
from aiready.evals.golden import EvalSuite, load_suites, resolve_suite, validate_suite
from aiready.evals.runner import EvalRunner, Scorecard, ScorecardHistory
from aiready.evals.safety import builtin_safety_suites

evals_app = typer.Typer(help="Evaluation suites, release gate and routing evaluation.", no_args_is_help=True)


def _target_spec(target: str | None, options: list[str], suite: EvalSuite | None) -> dict[str, Any]:
    spec: dict[str, Any] = dict(suite.target or {}) if (suite and suite.target and not target) else {}
    if target:
        spec = {"type": target} if ":" not in target and "." not in target else {"class": target}
    if not spec:
        spec = dict(get_config().get("evals.default_target", {"type": "rag"}) or {"type": "rag"})
    for opt in options:
        k, _, v = opt.partition("=")
        spec[k] = yaml.safe_load(v)
    return spec


def _history() -> ScorecardHistory:
    return ScorecardHistory(output_dir=get_config().get("evals.output_dir"))


def _suites(refs: list[str], include_safety: bool) -> list[EvalSuite]:
    suites = [resolve_suite(r) for r in refs]
    if include_safety:
        names = {s.name for s in suites}
        suites += [s for s in builtin_safety_suites() if s.name not in names]
    return suites


async def run_suites(suites: list[EvalSuite], *, release: str, target: str | None = None,
                     options: list[str] | None = None, concurrency: int | None = None) -> list[Scorecard]:
    conc = concurrency or int(get_config().get("evals.concurrency", 4) or 4)
    cards = []
    for s in suites:
        runner = EvalRunner(_target_spec(target, options or [], s), concurrency=conc, history=_history())
        cards.append(await runner.run(s, release=release))
    return cards


async def gate_release(suites: list[EvalSuite], *, release: str, baseline: str = "latest", target: str | None = None,
                       options: list[str] | None = None, concurrency: int | None = None) -> int:
    """Run + gate + record. Returns the CI exit code."""
    history = _history()
    baselines = {s.name: history.baseline(s.name, baseline, current_release=release) for s in suites}
    cards = await run_suites(suites, release=release, target=target, options=options, concurrency=concurrency)
    result = await record_gate(evaluate_gate(cards, baselines, load_gate_config(), release=release),
                               artifact={"suites": [s.name for s in suites]})
    typer.echo(result.to_markdown())
    if history.output_dir:
        history.output_dir.mkdir(parents=True, exist_ok=True)
        (history.output_dir / f"gate-{release.replace('/', '_')}.json").write_text(result.model_dump_json(indent=2))
    return exit_code(result)


@evals_app.command()
def validate(paths: list[Path]) -> None:
    """Validate golden-set files; exits 1 on any problem."""
    bad = 0
    for p in paths:
        for s in load_suites(p):
            problems = validate_suite(s)
            bad += len(problems)
            typer.echo(f"{s.name} ({len(s.cases)} cases): {'ok' if not problems else ''}")
            for pr in problems:
                typer.echo(f"  - {pr}")
    raise typer.Exit(1 if bad else 0)


@evals_app.command()
def run(suites: list[str], release: str = typer.Option(...), target: Optional[str] = None,
        option: list[str] = typer.Option([], "--option", "-o", help="target option key=value"),
        concurrency: Optional[int] = None, safety: bool = False) -> None:
    """Run suites (config names or files) and write scorecards."""
    cards = asyncio.run(run_suites(_suites(suites, safety), release=release, target=target, options=option,
                                   concurrency=concurrency))
    for c in cards:
        typer.echo(c.to_markdown())


@evals_app.command()
def gate(suites: list[str], release: str = typer.Option(...), baseline: str = "latest", target: Optional[str] = None,
         option: list[str] = typer.Option([], "--option", "-o"), concurrency: Optional[int] = None,
         safety: bool = typer.Option(True, help="also run the built-in safety suites")) -> None:
    """Run suites, compare with the baseline and exit non-zero when the release is blocked."""
    code = asyncio.run(gate_release(_suites(suites, safety), release=release, baseline=baseline, target=target,
                                    options=option, concurrency=concurrency))
    raise typer.Exit(code)


@evals_app.command()
def trend(suite: str) -> None:
    """Score per release for one suite."""
    typer.echo(json.dumps(_history().trend(suite), indent=2))


@evals_app.command()
def routing(task: list[str] = typer.Option(..., help="task=suite"), model: list[str] = typer.Option(..., help="name:size"),
            release: str = "routing-eval", out: Optional[Path] = None) -> None:
    """Evaluate candidate models per task and print the gateway routing snippet."""
    from aiready.evals.routing_eval import ModelCandidate, evaluate_routes

    tasks = {t.split("=", 1)[0]: resolve_suite(t.split("=", 1)[1]) for t in task}
    cands = [ModelCandidate(name=m.split(":")[0], size=float(m.split(":")[1]) if ":" in m else 0.0) for m in model]
    report = asyncio.run(evaluate_routes(tasks, cands, gate=load_gate_config(), release=release))
    typer.echo(report.to_markdown())
    snippet = report.to_yaml()
    if out:
        out.write_text(snippet)
    typer.echo(snippet)


def main() -> None:  # pragma: no cover - entry point
    evals_app()
