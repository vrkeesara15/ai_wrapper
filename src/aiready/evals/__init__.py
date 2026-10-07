"""Evaluation: golden sets, scorers, offline runner with scorecards per release, the release gate, built-in
safety suites and routing evaluation ("eval pass list" per task).

Not a long-running service: it runs in CI and from the control plane. Scorers are plugins (kind ``scorer``)
and are reused by the observability service's live scorers.
"""

from aiready.evals.golden import EvalCase, EvalSuite, Persona, load_suite, load_suites, validate_suite

__all__ = ["EvalCase", "EvalSuite", "Persona", "load_suite", "load_suites", "validate_suite"]
