"""Golden set format, loader and validator.

A suite is a list of cases built from real client questions ("a test set built from real client questions
runs before every release"). Files are YAML or JSONL and live in git next to prompts and routing rules, so
an evaluation set is versioned with the release it gates.

YAML files under ``config/`` are deep-merged into the configuration tree, so suites shipped there are wrapped
in ``evals.suites.<name>`` - that keeps several suites from colliding, and makes ``get_config()`` the suite
registry (hot-reloaded like any other config)::

    evals:
      suites:
        claims:
          description: Claims assistant questions
          personas: {adjuster: {sub: eval-adjuster, roles: [adjuster], groups: [claims-team]}}
          cases:
            - id: claims-001
              question: How many days does a customer have to report a claim?
              must_include: ["30 days"]
              expected_sources: [doc-claims-policy]
              persona: adjuster
              tags: [policy]

A standalone file may also contain the suite mapping directly, a bare list of cases, or JSONL (one case per
line; suite-level fields can be given to the loader).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from aiready.core.errors import ValidationFailed


class Persona(BaseModel):
    """Identity a case runs under, so access tests run as a real user with real groups (no service account)."""

    sub: str = "eval-user"
    tenant: str = "default"
    app: str = "default"
    roles: list[str] = Field(default_factory=lambda: ["user"])
    groups: list[str] = Field(default_factory=list)


class ContextDocument(BaseModel):
    """A document placed in front of the model for this case (e.g. a poisoned page for indirect injection).
    The gateway target puts these in the prompt as numbered passages, the RAG target can seed them."""

    id: str
    text: str
    title: str | None = None
    access_tags: list[str] = Field(default_factory=lambda: ["public"])


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer: str | None = None
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)  # doc ids
    expect_refusal: bool | None = None  # True: the right answer is "not found" / a refusal
    json_schema: dict[str, Any] | None = None
    task: str = "qa"
    tags: list[str] = Field(default_factory=list)
    persona: Persona | str | None = None  # inline persona or a name from the suite's personas
    documents: list[ContextDocument] = Field(default_factory=list)
    canaries: list[str] = Field(default_factory=list)  # strings that prove an injection worked
    forbidden: list[str] = Field(default_factory=list)  # another user's data that must never appear
    forbidden_sources: list[str] = Field(default_factory=list)  # doc ids the persona must not retrieve
    forbidden_tools: list[str] = Field(default_factory=list)  # tools an injected instruction tries to call
    max_latency_ms: float | None = None
    max_cost: float | None = None
    scorers: list[str] | None = None  # restrict to these scorer names for this case
    history: list[dict[str, Any]] = Field(default_factory=list)
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if not self.question.strip():
            raise ValueError("question is empty")
        return self


class EvalSuite(BaseModel):
    name: str
    version: str = "1"
    description: str = ""
    safety: bool = False  # every case must pass (release gate: 100%)
    target: dict[str, Any] | None = None  # default target spec for this suite
    personas: dict[str, Persona] = Field(default_factory=dict)
    scorers: list[dict[str, Any] | str] = Field(default_factory=list)  # default scorer specs
    thresholds: dict[str, float] = Field(default_factory=dict)  # scorer -> pass threshold override
    cases: list[EvalCase] = Field(default_factory=list)
    source: str | None = None  # file it was loaded from

    def persona_for(self, case: EvalCase) -> Persona:
        p = case.persona
        if isinstance(p, Persona):
            return p
        if isinstance(p, str):
            if p not in self.personas:
                raise ValidationFailed(f"case {case.id}: unknown persona '{p}'")
            return self.personas[p]
        return self.personas.get("default", Persona())


# --------------------------------------------------------------------------- loading


def _suite_from_mapping(data: dict[str, Any], *, name: str, source: str | None) -> list[EvalSuite]:
    if "evals" in data and isinstance(data["evals"], dict) and "suites" in data["evals"]:
        return [EvalSuite.model_validate({"name": n, **s, "source": source}) for n, s in data["evals"]["suites"].items()]
    if "cases" in data:
        return [EvalSuite.model_validate({"name": data.get("name", name), **data, "source": source})]
    raise ValidationFailed(f"{source}: expected 'evals.suites', a suite with 'cases', or a list of cases")


def load_suites(path: str | Path, **suite_fields: Any) -> list[EvalSuite]:
    """Load every suite defined in a YAML/JSON/JSONL file or a directory of them."""
    p = Path(path)
    if p.is_dir():
        out: list[EvalSuite] = []
        for f in sorted(list(p.rglob("*.y*ml")) + list(p.rglob("*.jsonl")) + list(p.rglob("*.json"))):
            out.extend(load_suites(f, **suite_fields))
        return out
    if not p.exists():
        raise ValidationFailed(f"golden set not found: {p}")
    try:
        if p.suffix == ".jsonl":
            cases = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [EvalSuite.model_validate({"name": suite_fields.pop("name", p.stem), **suite_fields,
                                              "cases": cases, "source": str(p)})]
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw) if p.suffix == ".json" else yaml.safe_load(raw)
        if isinstance(data, list):
            return [EvalSuite.model_validate({"name": suite_fields.pop("name", p.stem), **suite_fields,
                                              "cases": data, "source": str(p)})]
        if not isinstance(data, dict):
            raise ValidationFailed(f"{p}: empty or invalid golden set")
        suites = _suite_from_mapping(data, name=p.stem, source=str(p))
        return [s.model_copy(update=suite_fields) for s in suites] if suite_fields else suites
    except ValidationError as exc:
        raise ValidationFailed(f"{p}: invalid golden set: {exc}") from exc
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValidationFailed(f"{p}: cannot parse: {exc}") from exc


def load_suite(path: str | Path, name: str | None = None) -> EvalSuite:
    suites = load_suites(path)
    if name:
        for s in suites:
            if s.name == name:
                return s
        raise ValidationFailed(f"suite '{name}' not in {path}; found {[s.name for s in suites]}")
    if len(suites) != 1:
        raise ValidationFailed(f"{path} holds {len(suites)} suites; pass a name")
    return suites[0]


def suites_from_config() -> dict[str, EvalSuite]:
    """Suites registered in the configuration tree (``evals.suites``)."""
    from aiready.core.config import get_config

    raw = get_config().get("evals.suites", {}) or {}
    return {n: EvalSuite.model_validate({"name": n, **s}) for n, s in raw.items()}


def resolve_suite(ref: str) -> EvalSuite:
    """A suite by config name, or ``path[#name]`` to a file."""
    path, _, name = ref.partition("#")
    if Path(path).exists():
        return load_suite(path, name or None)
    suites = suites_from_config()
    if ref in suites:
        return suites[ref]
    from aiready.evals.safety import builtin_safety_suites

    for s in builtin_safety_suites():
        if s.name == ref:
            return s
    raise ValidationFailed(f"unknown suite '{ref}'")


def validate_suite(suite: EvalSuite) -> list[str]:
    """Problems that would make results meaningless. Empty list = valid."""
    errors: list[str] = []
    seen: set[str] = set()
    if not suite.cases:
        errors.append(f"suite '{suite.name}' has no cases")
    for c in suite.cases:
        if c.id in seen:
            errors.append(f"duplicate case id '{c.id}'")
        seen.add(c.id)
        if isinstance(c.persona, str) and c.persona not in suite.personas:
            errors.append(f"case {c.id}: unknown persona '{c.persona}'")
        has_expectation = any([c.expected_answer, c.must_include, c.must_not_include, c.expected_sources,
                               c.expect_refusal is not None, c.json_schema, c.canaries, c.forbidden,
                               c.forbidden_sources, c.max_latency_ms, c.max_cost])
        if not has_expectation:
            errors.append(f"case {c.id}: no expectation (expected answer, must_include, sources, refusal...)")
        if c.json_schema is not None:
            try:
                import jsonschema

                jsonschema.Draft202012Validator.check_schema(c.json_schema)
            except Exception as exc:
                errors.append(f"case {c.id}: invalid json_schema: {exc}")
        if c.expect_refusal and c.expected_sources:
            errors.append(f"case {c.id}: expects a refusal but also expected sources")
        if suite.safety and not (c.canaries or c.forbidden or c.forbidden_sources or c.expect_refusal
                                 or c.must_not_include or c.forbidden_tools):
            errors.append(f"case {c.id}: safety case without a safety expectation")
    return errors


def dump_cases_yaml(suite_name: str, cases: list[dict[str, Any]]) -> str:
    """YAML in the config-registry shape, e.g. for golden-set candidates exported from the review queue."""
    return yaml.safe_dump({"evals": {"suites": {suite_name: {"cases": cases}}}}, sort_keys=False, allow_unicode=True)
