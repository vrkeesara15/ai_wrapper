"""Policy as code: who may use which model, tool and data, under which conditions.

Rules are YAML (one file per domain in ``config/policies/``), reviewed in git like any other code, and
hot-reloaded into running services. Each rule names the actions it covers (globs), an effect and a ``when``
condition written in the safe expression language of ``aiready.governance.expressions`` (never ``eval``)::

    policies:
      tool:
        version: 1.2.0
        rules:
          - id: tool.write.approval
            description: Writes, payments and deletions wait for a person.
            action: tool.call
            effect: require_approval
            when: {resource.effect: [write, payment, delete], context.approved: {ne: true}}
            obligations: {approver_roles: [supervisor]}

Combining is **deny-overrides**: any matching ``deny`` wins, then ``require_approval``, then ``allow``.
When no rule matches, the default effect for the action's longest configured prefix applies (deny for
sensitive actions). ``obligate`` rules only add obligations (``mask_pii``, ``max_tokens``, ``require_citations``,
``log_full_prompt``...) and never change the effect, so an obligation can be attached to every decision
without accidentally granting access.

A loaded bundle is versioned by its semver label plus a content hash, so every decision records exactly
which rules produced it. ``explain()`` returns the evaluation trace of every rule that covers the action.

Engines are pluggable (``policy_engine`` kind): ``local`` (this module) and ``opa`` (a client's Open Policy
Agent server, for teams that prefer Rego). Rule sources are pluggable too (``policy_source`` kind): the
merged configuration, a directory of YAML files, or inline documents.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

from aiready.core.config import get_config, load_tree
from aiready.core.interfaces import PolicyDecision, PolicyInput
from aiready.core.plugins import create, register
from aiready.governance.config import PolicyConfig
from aiready.governance.expressions import Condition, PolicyCompileError, compile_condition

log = logging.getLogger(__name__)

Effect = Literal["allow", "deny", "require_approval"]
RULE_EFFECTS = ("allow", "deny", "require_approval", "obligate")
_RULE_KEYS = {"id", "description", "action", "actions", "effect", "when", "obligations", "reason", "enabled"}


# =========================================================================== rules and bundles


@dataclass
class Rule:
    id: str
    domain: str
    actions: list[str]
    effect: str
    condition: Condition
    obligations: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    reason: str = ""

    def covers(self, action: str) -> bool:
        return any(fnmatch.fnmatchcase(action, a) for a in self.actions)

    @property
    def message(self) -> str:
        return self.reason or self.description or self.id


def compile_rule(spec: dict[str, Any], domain: str, index: int) -> Rule | None:
    where = f"policies.{domain}.rules[{index}]"
    if not isinstance(spec, dict):
        raise PolicyCompileError(f"{where}: a rule must be a mapping")
    unknown = set(spec) - _RULE_KEYS
    if unknown:
        raise PolicyCompileError(f"{where}: unknown keys {sorted(unknown)}")
    rid = spec.get("id")
    if not rid or not isinstance(rid, str):
        raise PolicyCompileError(f"{where}: rule needs a string 'id'")
    if spec.get("enabled") is False:
        return None
    actions = spec.get("action", spec.get("actions"))
    actions = [actions] if isinstance(actions, str) else actions
    if not actions or not all(isinstance(a, str) and a for a in actions):
        raise PolicyCompileError(f"{where} ({rid}): 'action' must be a glob or a list of globs")
    effect = spec.get("effect")
    if effect not in RULE_EFFECTS:
        raise PolicyCompileError(f"{where} ({rid}): effect must be one of {RULE_EFFECTS}, got {effect!r}")
    obligations = spec.get("obligations") or {}
    if not isinstance(obligations, dict):
        raise PolicyCompileError(f"{where} ({rid}): obligations must be a mapping")
    return Rule(
        id=rid,
        domain=domain,
        actions=list(actions),
        effect=effect,
        condition=compile_condition(spec.get("when"), f"{where}({rid}).when"),
        obligations=obligations,
        description=str(spec.get("description", "")),
        reason=str(spec.get("reason", "")),
    )


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class BundleInfo(BaseModel):
    version: str  # "<label>+<hash12>"
    label: str
    hash: str
    loaded_at: float
    rules: int
    domains: dict[str, dict[str, Any]] = Field(
        default_factory=dict
    )  # domain -> {version, rules, description}
    defaults: dict[str, str] = Field(default_factory=dict)


class PolicyBundle:
    """An immutable, compiled set of rules. Swapped atomically on reload."""

    def __init__(
        self, documents: dict[str, Any], *, defaults: dict[str, str] | None = None, label: str = "0.0.0"
    ):
        self.documents = copy.deepcopy(documents)
        self.defaults = dict(defaults or {"": "deny"})
        for prefix, eff in self.defaults.items():
            if eff not in ("allow", "deny", "require_approval"):
                raise PolicyCompileError(f"default effect for '{prefix}' must be allow/deny/require_approval")
        self.rules: list[Rule] = []
        seen: set[str] = set()
        domains: dict[str, dict[str, Any]] = {}
        for domain in sorted(self.documents):
            doc = self.documents[domain] or {}
            if not isinstance(doc, dict):
                raise PolicyCompileError(f"policies.{domain} must be a mapping with 'rules'")
            raw_rules = doc.get("rules") or []
            if not isinstance(raw_rules, list):
                raise PolicyCompileError(f"policies.{domain}.rules must be a list")
            count = 0
            for i, spec in enumerate(raw_rules):
                rule = compile_rule(spec, domain, i)
                if rule is None:
                    continue
                if rule.id in seen:
                    raise PolicyCompileError(f"duplicate rule id '{rule.id}' (policies.{domain})")
                seen.add(rule.id)
                self.rules.append(rule)
                count += 1
            domains[domain] = {
                "version": str(doc.get("version", "")),
                "rules": count,
                "description": str(doc.get("description", "")),
            }
        self.hash = hashlib.sha256(
            _canonical({"documents": self.documents, "defaults": self.defaults}).encode()
        ).hexdigest()
        self.label = label
        self.info = BundleInfo(
            version=f"{label}+{self.hash[:12]}",
            label=label,
            hash=self.hash,
            loaded_at=time.time(),
            rules=len(self.rules),
            domains=domains,
            defaults=self.defaults,
        )

    @property
    def version(self) -> str:
        return self.info.version

    def default_for(self, action: str) -> tuple[str, str]:
        """Longest matching prefix wins; returns (prefix, effect)."""
        best = ""
        for prefix in self.defaults:
            if action.startswith(prefix) and len(prefix) >= len(best):
                best = prefix
        return best, self.defaults.get(best, "deny")


# =========================================================================== combining


def merge_obligations(target: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge obligations so the most restrictive wins.

    booleans OR together (any rule asking to mask PII masks), numbers take the minimum (the tightest
    max_tokens), lists union, mappings merge recursively, strings keep the first value set.
    """
    for k, v in new.items():
        if k not in target:
            target[k] = copy.deepcopy(v)
            continue
        cur = target[k]
        if isinstance(cur, bool) and isinstance(v, bool):
            target[k] = cur or v
        elif isinstance(cur, (int, float)) and isinstance(v, (int, float)) and not isinstance(cur, bool):
            target[k] = min(cur, v)
        elif isinstance(cur, list) and isinstance(v, list):
            target[k] = cur + [x for x in v if x not in cur]
        elif isinstance(cur, dict) and isinstance(v, dict):
            target[k] = merge_obligations(dict(cur), v)
    return target


def normalize_input(inp: PolicyInput) -> dict[str, Any]:
    """Build the document rules are evaluated against, with derived principal facts."""
    principal = dict(inp.principal or {})
    sub = principal.get("sub")
    roles = list(principal.get("roles") or [])
    groups = list(principal.get("groups") or [])
    principal["roles"], principal["groups"] = roles, groups
    # Derived facts are always recomputed: a caller cannot assert them.
    principal["authenticated"] = bool(sub)
    principal["is_service"] = bool(isinstance(sub, str) and sub.startswith("svc:"))
    if "access_groups" not in principal:
        ag = {"public"} | set(groups) | {f"role:{r}" for r in roles}
        if sub:
            ag.add(f"user:{sub}")
        principal["access_groups"] = sorted(ag)
    return {
        "action": inp.action,
        "principal": principal,
        "resource": dict(inp.resource or {}),
        "context": dict(inp.context or {}),
    }


def evaluate_bundle(
    bundle: PolicyBundle, inp: PolicyInput, *, with_trace: bool = False
) -> tuple[PolicyDecision, list[dict[str, Any]], dict[str, Any] | None]:
    doc = normalize_input(inp)
    trace: list[dict[str, Any]] = []
    matched: list[Rule] = []
    for rule in bundle.rules:
        if not rule.covers(inp.action):
            continue
        if with_trace:
            t = rule.condition.trace(doc)
            ok = t.result
            trace.append(
                {
                    "rule": rule.id,
                    "domain": rule.domain,
                    "effect": rule.effect,
                    "matched": ok,
                    "description": rule.description,
                    "condition": t.to_dict(),
                }
            )
        else:
            ok = rule.condition.evaluate(doc)
        if ok:
            matched.append(rule)

    denies = [r for r in matched if r.effect == "deny"]
    approvals = [r for r in matched if r.effect == "require_approval"]
    allows = [r for r in matched if r.effect == "allow"]
    obligates = [r for r in matched if r.effect == "obligate"]
    default_info: dict[str, Any] | None = None
    obligations: dict[str, Any] = {}

    if denies:
        effect, deciding = "deny", denies
        reasons = [r.message for r in denies]
    elif approvals:
        effect, deciding = "require_approval", approvals
        reasons = [r.message for r in approvals]
    elif allows:
        effect, deciding = "allow", allows
        reasons = [r.message for r in allows]
    else:
        prefix, effect = bundle.default_for(inp.action)
        deciding = []
        reasons = [f"no rule matched '{inp.action}'; default {effect} for prefix '{prefix or '*'}'"]
        default_info = {"prefix": prefix, "effect": effect}
    if effect != "deny":
        for r in [*approvals, *allows, *obligates]:
            merge_obligations(obligations, r.obligations)
    decision = PolicyDecision(
        allow=effect == "allow",
        effect=effect,
        reasons=reasons,  # type: ignore[arg-type]
        matched_rules=[r.id for r in deciding] + [r.id for r in obligates if effect != "deny"],
        obligations=obligations,
        policy_version=bundle.version,
    )
    return decision, trace, default_info


# =========================================================================== sources


class PolicySource:
    """Where rule documents come from. ``fingerprint()`` must be cheap: it is checked on the hot path."""

    name = "source"

    def load(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def fingerprint(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@register("policy_source", "config")
class ConfigPolicySource(PolicySource):
    """Rules from the merged configuration tree (``config/policies/*.yaml`` merge under ``policies``).

    Hot reload comes for free: the config store already watches the files, and its version changes.
    """

    name = "config"

    def __init__(self, path: str = "policies"):
        self.path = path

    def load(self) -> dict[str, Any]:
        data = get_config().get(self.path, {}) or {}
        if not isinstance(data, dict):
            raise PolicyCompileError(f"configuration '{self.path}' must be a mapping of domain -> rules")
        return data

    def fingerprint(self) -> str:
        store = get_config()
        return f"{id(store)}:{store.version}"


@register("policy_source", "directory")
class DirectoryPolicySource(PolicySource):
    """Rules from YAML files outside the config tree (e.g. a policy repo mounted read-only).

    A file holds either ``policies: {domain: {...}}`` or a bare ``{version, rules}`` document whose domain is
    the file name. Files under a ``tests`` directory are ignored (they hold policy test cases).
    """

    name = "directory"

    def __init__(self, paths: list[str] | str, reload_interval: float = 2.0):
        self.paths = [Path(p) for p in ([paths] if isinstance(paths, str) else paths)]
        self.reload_interval = reload_interval
        self._checked = 0.0
        self._fp = ""

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for p in self.paths:
            if p.is_file():
                out.append(p)
            elif p.is_dir():
                out.extend(f for f in sorted(p.rglob("*.y*ml")) if "tests" not in f.relative_to(p).parts)
        return out

    def load(self) -> dict[str, Any]:
        docs: dict[str, Any] = {}
        for f in self._files():
            try:
                data = yaml.safe_load(f.read_text()) or {}
            except yaml.YAMLError as exc:
                raise PolicyCompileError(f"invalid YAML in {f}: {exc}") from exc
            if isinstance(data, dict) and isinstance(data.get("policies"), dict):
                for domain, doc in data["policies"].items():
                    docs[domain] = _merge_domain(docs.get(domain), doc)
            elif isinstance(data, dict) and "rules" in data:
                docs[f.stem] = _merge_domain(docs.get(f.stem), data)
        return docs

    def fingerprint(self) -> str:
        now = time.monotonic()
        if self._fp and now - self._checked < self.reload_interval:
            return self._fp
        self._checked = now
        parts = [f"{f}:{f.stat().st_mtime_ns}:{f.stat().st_size}" for f in self._files() if f.exists()]
        self._fp = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return self._fp


@register("policy_source", "inline")
class InlinePolicySource(PolicySource):
    """Rules passed as data (tests, embedding the engine, the policy test harness)."""

    name = "inline"

    def __init__(self, policies: dict[str, Any]):
        self.policies = policies

    def load(self) -> dict[str, Any]:
        return self.policies

    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(self.policies).encode()).hexdigest()


def _merge_domain(existing: Any, doc: Any) -> Any:
    if not existing:
        return copy.deepcopy(doc)
    merged = dict(existing)
    merged.update({k: v for k, v in doc.items() if k != "rules"})
    merged["rules"] = list(existing.get("rules") or []) + list(doc.get("rules") or [])
    return merged


# =========================================================================== engines


BundleListener = Callable[[BundleInfo | None, BundleInfo], None]


@register("policy_engine", "local")
class LocalPolicyEngine:
    """The built-in engine. Thread-safe; evaluation is pure CPU work (microseconds per rule).

    The bundle is rebuilt whenever a source fingerprint changes. A bundle that fails to compile is rejected
    and the previous one stays active, so a typo in a policy file can never open access or take the
    platform down.
    """

    def __init__(
        self,
        sources: list[PolicySource | dict[str, Any]] | None = None,
        *,
        defaults: dict[str, str] | None = None,
        version: str = "0.0.0",
        reload_interval: float = 2.0,
    ):
        self.sources: list[PolicySource] = [
            s if isinstance(s, PolicySource) else _make_source(s, reload_interval)
            for s in (sources if sources is not None else [ConfigPolicySource()])
        ]
        self.defaults = defaults if defaults is not None else {"": "deny"}
        self.label = version
        self._lock = threading.RLock()
        self._listeners: list[BundleListener] = []
        self.last_error: str | None = None
        self._fingerprint = self._fingerprints()
        self.bundle = self._build()  # fail fast at start: an invalid bundle never serves traffic

    @classmethod
    def from_config(cls, cfg: PolicyConfig) -> LocalPolicyEngine:
        return cls(
            [dict(s) for s in cfg.sources],
            defaults=dict(cfg.defaults),
            version=cfg.version,
            reload_interval=cfg.reload_interval,
        )

    # ------------------------------------------------------------------ loading
    def _fingerprints(self) -> str:
        return "|".join(s.fingerprint() for s in self.sources)

    def _build(self) -> PolicyBundle:
        docs: dict[str, Any] = {}
        for src in self.sources:
            for domain, doc in src.load().items():
                docs[domain] = _merge_domain(docs.get(domain), doc or {})
        return PolicyBundle(docs, defaults=self.defaults, label=self.label)

    def on_change(self, listener: BundleListener) -> None:
        self._listeners.append(listener)

    def reload(self, *, force: bool = False) -> bool:
        """Rebuild when a source changed. Returns True when a new bundle became active."""
        with self._lock:
            fp = self._fingerprints()
            if not force and fp == self._fingerprint:
                return False
            self._fingerprint = fp
            try:
                new = self._build()
            except Exception as exc:
                self.last_error = str(exc)
                log.error("policy bundle rejected, keeping %s: %s", self.bundle.version, exc)
                return False
            self.last_error = None
            if new.hash == self.bundle.hash:
                return False
            old, self.bundle = self.bundle, new
        log.info("policy bundle %s active (%d rules)", new.version, len(new.rules))
        for cb in list(self._listeners):
            try:
                cb(old.info, new.info)
            except Exception:
                log.exception("policy bundle listener failed")
        return True

    def _current(self) -> PolicyBundle:
        try:
            self.reload()
        except Exception:  # a failing fingerprint (e.g. unreadable mount) keeps the active bundle
            log.exception("policy reload check failed")
        return self.bundle

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, inp: PolicyInput) -> PolicyDecision:
        return evaluate_bundle(self._current(), inp)[0]

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        return self.evaluate(inp)

    async def explain(self, inp: PolicyInput) -> dict[str, Any]:
        bundle = self._current()
        decision, trace, default = evaluate_bundle(bundle, inp, with_trace=True)
        return {
            "decision": decision.model_dump(),
            "policy_version": bundle.version,
            "rules": trace,
            "default_applied": default,
            "input": normalize_input(inp),
        }

    def bundles(self) -> dict[str, Any]:
        b = self._current()
        return {
            "active": b.info.model_dump(),
            "last_error": self.last_error,
            "sources": [s.name for s in self.sources],
        }


def _make_source(spec: dict[str, Any], reload_interval: float) -> PolicySource:
    spec = dict(spec)
    if spec.get("type") == "directory":
        spec.setdefault("reload_interval", reload_interval)
    return create("policy_source", spec)


@register("policy_engine", "opa")
class OpaPolicyEngine:
    """Adapter for a client's Open Policy Agent server (Rego policies).

    Queries ``POST {url}/v1/data/{package}`` with ``{"input": <normalised input>}``. The rule may return a
    boolean or ``{allow, effect, reasons, obligations, version}``. Any failure is a **deny** (fail closed): a
    policy engine outage must never become an open door.
    """

    def __init__(
        self,
        url: str,
        package: str = "aiready/authz",
        timeout: float = 2.0,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.url = url.rstrip("/")
        self.package = package.strip("/").replace(".", "/")
        self.timeout, self.headers, self.transport = timeout, headers or {}, transport
        self.label = f"opa:{self.package}"

    async def _query(self, inp: PolicyInput, explain: bool = False) -> Any:
        params = {"explain": "full"} if explain else None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as c:
            r = await c.post(
                f"{self.url}/v1/data/{self.package}",
                json={"input": normalize_input(inp)},
                params=params,
                headers=self.headers,
            )
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _to_decision(body: Any, label: str) -> PolicyDecision:
        result = body.get("result") if isinstance(body, dict) else None
        if isinstance(result, bool):
            return PolicyDecision(
                allow=result,
                effect="allow" if result else "deny",
                reasons=[f"opa: {'allow' if result else 'deny'}"],
                policy_version=label,
            )
        if isinstance(result, dict):
            effect = result.get("effect") or ("allow" if result.get("allow") else "deny")
            if effect not in ("allow", "deny", "require_approval"):
                effect = "deny"
            return PolicyDecision(
                allow=effect == "allow",
                effect=effect,
                reasons=list(result.get("reasons") or [f"opa: {effect}"]),
                matched_rules=list(result.get("matched_rules") or []),
                obligations=dict(result.get("obligations") or {}),
                policy_version=str(result.get("version") or label),
            )
        return PolicyDecision(
            allow=False,
            effect="deny",
            reasons=["opa: undefined decision (default deny)"],
            policy_version=label,
        )

    async def decide(self, inp: PolicyInput) -> PolicyDecision:
        try:
            return self._to_decision(await self._query(inp), self.label)
        except (httpx.HTTPError, ValueError) as exc:
            log.error("OPA query failed, denying: %s", exc)
            return PolicyDecision(
                allow=False,
                effect="deny",
                reasons=[f"policy engine unavailable: {exc}"],
                policy_version=self.label,
            )

    async def explain(self, inp: PolicyInput) -> dict[str, Any]:
        try:
            body = await self._query(inp, explain=True)
        except (httpx.HTTPError, ValueError) as exc:
            return {"decision": (await self.decide(inp)).model_dump(), "error": str(exc)}
        return {
            "decision": self._to_decision(body, self.label).model_dump(),
            "policy_version": self.label,
            "opa": body,
        }

    def bundles(self) -> dict[str, Any]:
        return {"active": {"version": self.label, "engine": "opa", "url": self.url}, "last_error": None}

    def on_change(self, listener: BundleListener) -> None:  # bundles are managed by OPA itself
        return None

    def reload(self, *, force: bool = False) -> bool:
        return False


def build_engine(cfg: PolicyConfig | None = None) -> Any:
    """Create the configured engine (``governance.policy.engine``)."""
    cfg = cfg or get_config().section("governance.policy", PolicyConfig)
    spec = dict(cfg.engine or {"type": "local"})
    if spec.get("type", "local") == "local" and "class" not in spec:
        return LocalPolicyEngine.from_config(cfg)
    return create("policy_engine", spec)


# =========================================================================== policy test harness


class PolicyTestFailure(BaseModel):
    file: str
    name: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    problems: list[str]


class PolicyTestReport(BaseModel):
    total: int = 0
    passed: int = 0
    failures: list[PolicyTestFailure] = Field(default_factory=list)
    policy_version: str = ""

    @property
    def ok(self) -> bool:
        return not self.failures and self.total > 0

    def summary(self) -> str:
        lines = [f"policy tests: {self.passed}/{self.total} passed (bundle {self.policy_version})"]
        for f in self.failures:
            lines.append(f"  FAIL {f.file} :: {f.name}: {'; '.join(f.problems)}")
        return "\n".join(lines)


def _load_cases(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    files = [path] if path.is_file() else sorted(path.rglob("*.y*ml"))
    for f in files:
        data = yaml.safe_load(f.read_text()) or {}
        cases = data if isinstance(data, list) else (data.get("policy_tests") or data.get("tests") or [])
        for case in cases:
            yield str(f), case


def _check_case(case: dict[str, Any], decision: PolicyDecision) -> tuple[dict[str, Any], list[str]]:
    expect = case.get("expect")
    if isinstance(expect, str):
        expect = {"effect": expect}
    if not isinstance(expect, dict) or not expect:
        return {}, ["test case has no 'expect'"]
    problems: list[str] = []
    if "effect" in expect and decision.effect != expect["effect"]:
        problems.append(f"effect {decision.effect} != expected {expect['effect']}")
    if "allow" in expect and decision.allow != expect["allow"]:
        problems.append(f"allow {decision.allow} != expected {expect['allow']}")
    for k, v in (expect.get("obligations") or {}).items():
        if decision.obligations.get(k) != v:
            problems.append(f"obligation {k}={decision.obligations.get(k)!r} != expected {v!r}")
    for rid in expect.get("matched_rules") or []:
        if rid not in decision.matched_rules:
            problems.append(f"rule {rid} did not decide (got {decision.matched_rules})")
    for rid in expect.get("not_matched_rules") or []:
        if rid in decision.matched_rules:
            problems.append(f"rule {rid} unexpectedly decided")
    return expect, problems


def engine_from_tree(tree: dict[str, Any]) -> LocalPolicyEngine:
    """A local engine over a configuration tree (no global state touched)."""
    pcfg = PolicyConfig.model_validate((tree.get("governance") or {}).get("policy") or {})
    return LocalPolicyEngine(
        [InlinePolicySource(tree.get("policies") or {})], defaults=dict(pcfg.defaults), version=pcfg.version
    )


def run_policy_tests(
    path: str | Path, *, engine: LocalPolicyEngine | None = None, config_dir: str | Path | None = None
) -> PolicyTestReport:
    """Run YAML policy test cases (CI gate for policy changes).

    Each case is ``{name, input: {action, principal, resource, context}, expect: <effect> | {effect, allow,
    obligations, matched_rules, not_matched_rules}}``, listed under ``policy_tests:`` in each file. The engine
    defaults to the policies in ``config_dir`` (or the active configuration).
    """
    if engine is None:
        if config_dir is not None:
            engine = engine_from_tree(load_tree(Path(config_dir)))
        else:
            engine = LocalPolicyEngine.from_config(get_config().section("governance.policy", PolicyConfig))
    report = PolicyTestReport(policy_version=engine.bundle.version)
    for file, case in _load_cases(Path(path)):
        report.total += 1
        name = str(case.get("name", f"case-{report.total}"))
        try:
            decision = engine.evaluate(PolicyInput.model_validate(case.get("input") or {}))
        except Exception as exc:
            report.failures.append(
                PolicyTestFailure(
                    file=file, name=name, expected={}, actual={}, problems=[f"invalid input: {exc}"]
                )
            )
            continue
        expected, problems = _check_case(case, decision)
        if problems:
            report.failures.append(
                PolicyTestFailure(
                    file=file, name=name, expected=expected, actual=decision.model_dump(), problems=problems
                )
            )
        else:
            report.passed += 1
    return report


async def run_policy_cases(
    cases: list[dict[str, Any]], engine: Any, *, source: str = "request"
) -> PolicyTestReport:
    """Run test cases given as data against any engine with an async ``decide`` (local or OPA)."""
    info = engine.bundles().get("active") or {}
    report = PolicyTestReport(policy_version=str(info.get("version", "")))
    for i, case in enumerate(cases):
        report.total += 1
        name = str(case.get("name", f"case-{i + 1}"))
        try:
            decision = await engine.decide(PolicyInput.model_validate(case.get("input") or {}))
        except Exception as exc:
            report.failures.append(
                PolicyTestFailure(
                    file=source, name=name, expected={}, actual={}, problems=[f"invalid input: {exc}"]
                )
            )
            continue
        expected, problems = _check_case(case, decision)
        if problems:
            report.failures.append(
                PolicyTestFailure(
                    file=source, name=name, expected=expected, actual=decision.model_dump(), problems=problems
                )
            )
        else:
            report.passed += 1
    return report
