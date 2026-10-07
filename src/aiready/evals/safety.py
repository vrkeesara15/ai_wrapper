"""Built-in safety suites: prompt injection (direct and indirect via poisoned documents), data leaks across
users, PII echo and jailbreaks.

They ship as YAML in ``config/evals/safety/`` (wrapped in ``evals.suites`` like every other suite, with
``safety: true``), so a client extends them with its own attack cases in configuration. The release gate
requires a 100% pass rate on every safety suite: one leaked record is one too many.

Persona note for data-leak tests: the persona names (``alice``/``bob``) are evaluation identities; the client
maps them to real test accounts and seeds the referenced records in the staging copy (``forbidden`` lists the
strings that identify the other user's data).
"""

from __future__ import annotations

from pathlib import Path

from aiready.core.identity import Principal
from aiready.evals.golden import EvalSuite, load_suites, suites_from_config
from aiready.evals.runner import EvalRunner, Scorecard
from aiready.evals.targets import Target


def safety_dirs() -> list[Path]:
    """Where built-in safety YAML is looked up: the configured dir, the config dir, the repository copy."""
    dirs: list[Path] = []
    try:
        from aiready.core.config import get_config

        cfg = get_config()
        if cfg.get("evals.safety_dir"):
            dirs.append(Path(cfg.get("evals.safety_dir")))
        dirs.append(cfg.config_dir / "evals" / "safety")
    except Exception:
        pass
    dirs.append(Path(__file__).resolve().parents[3] / "config" / "evals" / "safety")
    return dirs


def builtin_safety_suites(directory: str | Path | None = None) -> list[EvalSuite]:
    """Safety suites from configuration (already merged into ``evals.suites``) or, when the configuration tree
    does not hold them (tests, a trimmed config), from the YAML files directly."""
    from_cfg = [s for s in suites_from_config().values() if s.safety]
    if from_cfg and directory is None:
        return from_cfg
    for d in ([Path(directory)] if directory else safety_dirs()):
        if d.is_dir():
            suites = [s for s in load_suites(d) if s.safety]
            if suites:
                return suites
    return []


async def run_safety(target: Target | dict | str, *, release: str, operator: Principal | None = None,
                     suites: list[EvalSuite] | None = None, concurrency: int = 4) -> list[Scorecard]:
    runner = EvalRunner(target, concurrency=concurrency)
    return [await runner.run(s, release=release, operator=operator) for s in (suites or builtin_safety_suites())]
