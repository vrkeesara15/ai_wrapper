"""Workflow registry: versioned step graphs defined in YAML (``config/workflows/*.yaml``).

A workflow is either an explicit graph of steps or one of the four orchestration patterns, which expand into a
graph ("LLM orchestration" slide):

* ``single_call``     - one model call;
* ``retrieval_chain`` - retrieve what the user may see, then answer with citations (default for the assistant);
* ``router``          - a small model classifies the request, then the matching workflow runs as a sub-agent;
* ``agent_loop``      - plan (model picks a tool) -> act (call it) -> check (results, policy, approval) ->
  decide (done, or take the next step).

Step types: ``llm``, ``tool``, ``retrieve``, ``query``, ``approval``, ``branch``, ``subagent``, ``end``; more
plug in with ``plugins.register("agent_step", "<type>")`` or ``type: {class: pkg.mod:Step}`` - no fork.

    workflows:
      claims_triage:
        version: "3"
        description: Route a claim question
        pattern: router
        routes:
          billing: {workflow: billing_agent, description: invoices, payments}
          policy:  {workflow: retrieval_chain, description: questions about policy wording}
        default: policy

Older versions stay available side by side as ``claims_triage@2``; a run is pinned to the version (and the
graph snapshot) it started with, so a configuration change never alters a run half-way.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aiready.core.config import get_config
from aiready.core.errors import ConfigError, NotFound

STEP_TYPES = {"llm", "tool", "retrieve", "query", "approval", "branch", "subagent", "end"}


class StepDef(BaseModel):
    """One node. Type-specific settings are given inline (extra keys) and read through ``params``."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str | dict[str, Any]  # a registered type name, or {class: module:Class} for a custom step
    next: str | None = None
    retries: int | None = None  # override of agents.retry.max_attempts for this step
    on_error: str | None = None  # step to go to when the step fails for good (default: the run fails)
    timeout_seconds: float | None = None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    @property
    def type_name(self) -> str:
        return self.type if isinstance(self.type, str) else str(self.type.get("class", "custom"))


class WorkflowDef(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    version: str = "1"
    description: str = ""
    pattern: str | None = None
    start: str | None = None
    steps: list[StepDef] = Field(default_factory=list)
    limits: dict[str, Any] = Field(default_factory=dict)
    agent: str | None = None  # profile in agents.profiles (prompt, model, tool allow-list)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})

    def step(self, step_id: str) -> StepDef:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise NotFound(f"workflow {self.name}@{self.version} has no step '{step_id}'")

    @property
    def first(self) -> str:
        return self.start or self.steps[0].id

    def validate_graph(self) -> WorkflowDef:
        if not self.steps:
            raise ConfigError(f"workflow {self.name}: no steps")
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"workflow {self.name}: duplicate step ids")
        targets: list[str | None] = [self.start]
        for s in self.steps:
            targets += [s.next, s.on_error]
            p = s.params
            targets += [c.get("next") for c in p.get("cases", []) if isinstance(c, dict)]
            targets += [p.get("default") if s.type == "branch" else None, p.get("on_reject")]
        missing = sorted({t for t in targets if t and t not in ids})
        if missing:
            raise ConfigError(f"workflow {self.name}: unknown step(s) {missing}")
        if not any(s.type == "end" for s in self.steps):
            raise ConfigError(f"workflow {self.name}: needs at least one 'end' step")
        return self


# =========================================================================== patterns


def _single_call(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "answer", "type": "llm", "prompt": p.get("prompt", "{{input.message}}"), "system": p.get("system"),
         "model": p.get("model"), "stream": p.get("stream", True), "next": "done"},
        {"id": "done", "type": "end", "output": "{{steps.answer.text}}"},
    ]


def _retrieval_chain(p: dict[str, Any]) -> list[dict[str, Any]]:
    common = {"question": p.get("question", "{{input.message}}"), "collection": p.get("collection"),
              "top_n": p.get("top_n")}
    if p.get("generate", "rag") == "rag":  # the RAG service answers, checks grounding and cites
        return [
            {"id": "retrieve", "type": "retrieve", "answer": True, **common, "next": "done"},
            {"id": "done", "type": "end", "output": "{{steps.retrieve.text}}",
             "citations": "{{steps.retrieve.citations}}"},
        ]
    return [
        {"id": "retrieve", "type": "retrieve", "answer": False, **common, "next": "answer"},
        {"id": "answer", "type": "llm", "context_from": "retrieve", "prompt": "{{input.message}}",
         "system": p.get("system"), "model": p.get("model"), "stream": p.get("stream", True), "next": "done"},
        {"id": "done", "type": "end", "output": "{{steps.answer.text}}", "citations": "{{steps.retrieve.citations}}"},
    ]


def _router(p: dict[str, Any]) -> list[dict[str, Any]]:
    routes: dict[str, Any] = p.get("routes") or {}
    if not routes:
        raise ConfigError("router pattern needs 'routes: {label: {workflow: ...}}'")
    labels = {k: (v if isinstance(v, dict) else {"workflow": v}) for k, v in routes.items()}
    default = p.get("default") or next(iter(labels))
    steps: list[dict[str, Any]] = [
        {"id": "classify", "type": "llm", "model": p.get("model", "small"), "prompt": "{{input.message}}",
         "classify": {k: v.get("description", "") for k, v in labels.items()}, "default_label": default,
         "next": "route"},
        {"id": "route", "type": "branch",
         "cases": [{"when": {"var": "steps.classify.label", "equals": k}, "next": f"run_{k}"} for k in labels],
         "default": f"run_{default}"},
    ]
    for k, v in labels.items():
        steps.append({"id": f"run_{k}", "type": "subagent", "workflow": v["workflow"],
                      "input": v.get("input", {"message": "{{input.message}}", "history": "{{input.history}}"}),
                      "next": "done"})
    steps.append({"id": "done", "type": "end", "output": "{{last.answer}}", "citations": "{{last.citations}}"})
    return steps


def _agent_loop(p: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "plan", "type": "llm", "conversation": True, "tools": p.get("tools", ["*"]),
         "system": p.get("system"), "model": p.get("model"), "prompt": p.get("prompt", "{{input.message}}"),
         "next": "decide"},
        {"id": "decide", "type": "branch",
         "cases": [{"when": {"var": "steps.plan.tool_calls", "truthy": True}, "next": "act"}], "default": "done"},
        {"id": "act", "type": "tool", "from": "tool_calls", "next": "plan"},
        {"id": "done", "type": "end", "output": "{{steps.plan.text}}", "validate": p.get("validate", True)},
    ]


PATTERNS = {"single_call": _single_call, "retrieval_chain": _retrieval_chain, "router": _router,
            "agent_loop": _agent_loop}

BUILTIN_WORKFLOWS: dict[str, dict[str, Any]] = {
    "single_call": {"pattern": "single_call", "description": "One model call."},
    "retrieval_chain": {"pattern": "retrieval_chain", "description": "Answer from the knowledge base with citations."},
    "agent_loop": {"pattern": "agent_loop", "description": "Plan, call tools, check, decide - until done."},
}


def build_workflow(name: str, raw: dict[str, Any]) -> WorkflowDef:
    raw = copy.deepcopy(raw)
    raw.setdefault("name", name)
    raw["version"] = str(raw.get("version", "1"))
    pattern = raw.get("pattern")
    if pattern and not raw.get("steps"):
        if pattern not in PATTERNS:
            raise ConfigError(f"workflow {name}: unknown pattern '{pattern}' (have {sorted(PATTERNS)})")
        raw["steps"] = [{k: v for k, v in s.items() if v is not None} for s in PATTERNS[pattern](raw)]
    return WorkflowDef.model_validate(raw).validate_graph()


def _version_key(v: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"[.\-]", v))


class WorkflowRegistry:
    """Resolves ``name`` (latest or pinned version) or ``name@version`` to a validated graph."""

    def __init__(self, tree: dict[str, Any] | None = None, *, pins: dict[str, str] | None = None):
        self.pins = pins or {}
        self._defs: dict[str, dict[str, WorkflowDef]] = {}
        self.load(tree if tree is not None else (get_config().get("workflows") or {}))

    def load(self, tree: dict[str, Any]) -> None:
        defs: dict[str, dict[str, WorkflowDef]] = {}
        for name, raw in {**BUILTIN_WORKFLOWS, **(tree or {})}.items():
            if not isinstance(raw, dict):
                raise ConfigError(f"workflow {name} must be a mapping")
            base, _, ver = name.partition("@")
            if ver:
                raw = {**raw, "version": raw.get("version", ver)}
            wf = build_workflow(base, raw)
            defs.setdefault(base, {})[wf.version] = wf
        self._defs = defs

    def get(self, name: str, version: str | None = None) -> WorkflowDef:
        base, _, ver = name.partition("@")
        versions = self._defs.get(base)
        if not versions:
            raise NotFound(f"unknown workflow '{base}'")
        ver = version or ver or self.pins.get(base)
        if ver:
            if ver not in versions:
                raise NotFound(f"workflow '{base}' has no version '{ver}' (have {sorted(versions)})")
            return versions[ver]
        return versions[max(versions, key=_version_key)]

    def list(self) -> list[dict[str, Any]]:
        return [{"name": n, "versions": sorted(v, key=_version_key),
                 "latest": max(v, key=_version_key), "description": v[max(v, key=_version_key)].description}
                for n, v in sorted(self._defs.items())]


# =========================================================================== templates and conditions

_TEMPLATE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def lookup(scope: dict[str, Any], path: str) -> Any:
    node: Any = scope
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit():
            idx = int(part)
            node = node[idx] if idx < len(node) else None
        else:
            node = getattr(node, part, None)
        if node is None:
            return None
    return node


def render(template: Any, scope: dict[str, Any]) -> Any:
    """``{{ path }}`` substitution. A string that is exactly one placeholder keeps the value's type."""
    if isinstance(template, str):
        full = _TEMPLATE.fullmatch(template.strip())
        if full:
            return lookup(scope, full.group(1))
        return _TEMPLATE.sub(lambda m: _str(lookup(scope, m.group(1))), template)
    if isinstance(template, dict):
        return {k: render(v, scope) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, scope) for v in template]
    return template


def _str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        import json

        return json.dumps(v, default=str)
    return str(v)


def evaluate(cond: Any, scope: dict[str, Any]) -> bool:
    """Small, safe condition language (no eval): ``{var, equals|not_equals|in|contains|exists|truthy}``,
    combined with ``{all: [...]}``, ``{any: [...]}``, ``{not: {...}}``."""
    if isinstance(cond, bool):
        return cond
    if not isinstance(cond, dict):
        raise ConfigError(f"invalid condition {cond!r}")
    if "all" in cond:
        return all(evaluate(c, scope) for c in cond["all"])
    if "any" in cond:
        return any(evaluate(c, scope) for c in cond["any"])
    if "not" in cond:
        return not evaluate(cond["not"], scope)
    value = lookup(scope, cond["var"]) if "var" in cond else None
    if "equals" in cond:
        return _norm(value) == _norm(render(cond["equals"], scope))
    if "not_equals" in cond:
        return _norm(value) != _norm(render(cond["not_equals"], scope))
    if "in" in cond:
        return _norm(value) in [_norm(x) for x in render(cond["in"], scope)]
    if "contains" in cond:
        return value is not None and render(cond["contains"], scope) in value
    if "exists" in cond:
        return (value is not None) == bool(cond["exists"])
    if "truthy" in cond:
        return bool(value) == bool(cond["truthy"])
    return bool(value)


def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v
