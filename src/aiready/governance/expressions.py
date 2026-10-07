"""A small, safe condition language for policy rules.

Policies are data written by security teams, reviewed in git and hot-reloaded into running services, so the
condition language must never execute code: there is no ``eval``, no attribute access and no function calls.
A condition is a YAML tree compiled once (at bundle load, where mistakes are rejected with a clear error) into
a closure tree that only reads values from the policy input.

Grammar::

    cond     := mapping | list                        # a list means "all of"
    mapping  := {all: [cond...]} | {any: [cond...]} | {not: cond}
              | {<path>: test, <path>: test, ...}     # several keys in one mapping are all required
    test     := scalar                                # shorthand for {eq: scalar}
              | list                                  # shorthand for {in: list}
              | {<op>: value, ...}                    # several operators are all required
    value    := literal | {ref: <path>}               # compare with another field of the input
    path     := principal.<...> | resource.<...> | context.<...> | action

Operators: eq, ne, in, not_in, contains, not_contains, any_in, all_in, glob, regex, gt, gte, lt, lte,
exists, empty.

Missing fields resolve to ``None``. Every operator is total: a type mismatch evaluates to False rather than
raising, so a malformed request can never crash the engine into an "allow".
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from aiready.core.errors import ConfigError

ROOTS = ("principal", "resource", "context", "action")
MAX_REGEX_LENGTH = 512
_MISSING = object()


class PolicyCompileError(ConfigError):
    """A policy rule is malformed. Raised at load time so a broken bundle never replaces a working one."""

    code = "policy_compile_error"


# --------------------------------------------------------------------------- value access


def resolve_path(doc: dict[str, Any], path: str) -> Any:
    """Read a dotted path from the input document; lists accept numeric segments. Missing -> None."""
    node: Any = doc
    for part in path.split("."):
        if isinstance(node, dict):
            node = node.get(part, _MISSING)
        elif isinstance(node, (list, tuple)) and part.isdigit():
            idx = int(part)
            node = node[idx] if idx < len(node) else _MISSING
        else:
            return None
        if node is _MISSING:
            return None
    return node


def _check_path(path: Any, where: str) -> str:
    if not isinstance(path, str) or not path:
        raise PolicyCompileError(f"{where}: field path must be a non-empty string, got {path!r}")
    root = path.split(".", 1)[0]
    if root not in ROOTS:
        raise PolicyCompileError(f"{where}: field '{path}' must start with one of {list(ROOTS)}")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+(\.[A-Za-z0-9_\-]+)*", path):
        raise PolicyCompileError(f"{where}: invalid field path '{path}'")
    return path


Getter = Callable[[dict[str, Any]], Any]


def _value(spec: Any, where: str) -> Getter:
    """Literal or ``{ref: path}``."""
    if isinstance(spec, dict):
        if set(spec) == {"ref"}:
            path = _check_path(spec["ref"], where)
            return lambda doc: resolve_path(doc, path)
        raise PolicyCompileError(f"{where}: an operand mapping must be {{ref: <path>}}, got {spec!r}")
    return lambda doc: spec


# --------------------------------------------------------------------------- operators


def _as_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple, set, frozenset)):
        return list(v)
    return [v]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _cmp(a: Any, b: Any, fn: Callable[[float, float], bool]) -> bool:
    x, y = _num(a), _num(b)
    return x is not None and y is not None and fn(x, y)


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return isinstance(expected, str) and expected in actual
    if isinstance(actual, (list, tuple, set)):
        return expected in actual
    if isinstance(actual, dict):
        return expected in actual
    return False


def _eq(a: Any, b: Any) -> bool:
    # str-Enums (e.g. DataClass.PERSONAL) compare equal to their value, so plain equality is enough. Booleans
    # are kept distinct from 1/0 so ``approved: true`` cannot be satisfied by a counter.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return bool(a == b)


@lru_cache(maxsize=512)
def _regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _regex_match(actual: Any, pattern: Any) -> bool:
    return isinstance(actual, str) and isinstance(pattern, str) and _regex(pattern).search(actual) is not None


def _glob(actual: Any, patterns: Any) -> bool:
    if not isinstance(actual, str):
        return any(_glob(a, patterns) for a in actual) if isinstance(actual, (list, tuple)) else False
    return any(isinstance(p, str) and fnmatch.fnmatchcase(actual, p) for p in _as_list(patterns))


OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": _eq,
    "ne": lambda a, b: not _eq(a, b),
    "in": lambda a, b: any(_eq(a, x) for x in _as_list(b)),
    "not_in": lambda a, b: not any(_eq(a, x) for x in _as_list(b)),
    "contains": _contains,
    "not_contains": lambda a, b: not _contains(a, b),
    "any_in": lambda a, b: any(any(_eq(x, y) for y in _as_list(b)) for x in _as_list(a)),
    "all_in": lambda a, b: (
        bool(_as_list(a)) and all(any(_eq(x, y) for y in _as_list(b)) for x in _as_list(a))
    ),
    "glob": _glob,
    "regex": _regex_match,
    "gt": lambda a, b: _cmp(a, b, lambda x, y: x > y),
    "gte": lambda a, b: _cmp(a, b, lambda x, y: x >= y),
    "lt": lambda a, b: _cmp(a, b, lambda x, y: x < y),
    "lte": lambda a, b: _cmp(a, b, lambda x, y: x <= y),
    "exists": lambda a, b: (a is not None) == bool(b),
    "empty": lambda a, b: (a is None or (hasattr(a, "__len__") and len(a) == 0)) == bool(b),
}


# --------------------------------------------------------------------------- compiled tree


@dataclass
class Trace:
    """Evaluation trace for ``explain()``: what was compared and why it matched or not."""

    kind: str
    result: bool
    detail: dict[str, Any] = field(default_factory=dict)
    children: list[Trace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "result": self.result, **self.detail}
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out


class Condition:
    """A compiled condition: ``evaluate(doc)`` for the hot path, ``trace(doc)`` for explanations."""

    def evaluate(self, doc: dict[str, Any]) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def trace(self, doc: dict[str, Any]) -> Trace:  # pragma: no cover - interface
        raise NotImplementedError


class _Always(Condition):
    def evaluate(self, doc: dict[str, Any]) -> bool:
        return True

    def trace(self, doc: dict[str, Any]) -> Trace:
        return Trace("always", True)


class _All(Condition):
    def __init__(self, items: list[Condition]):
        self.items = items

    def evaluate(self, doc: dict[str, Any]) -> bool:
        return all(c.evaluate(doc) for c in self.items)

    def trace(self, doc: dict[str, Any]) -> Trace:
        kids = [c.trace(doc) for c in self.items]
        return Trace("all", all(k.result for k in kids), children=kids)


class _Any(Condition):
    def __init__(self, items: list[Condition]):
        self.items = items

    def evaluate(self, doc: dict[str, Any]) -> bool:
        return any(c.evaluate(doc) for c in self.items)

    def trace(self, doc: dict[str, Any]) -> Trace:
        kids = [c.trace(doc) for c in self.items]
        return Trace("any", any(k.result for k in kids), children=kids)


class _Not(Condition):
    def __init__(self, item: Condition):
        self.item = item

    def evaluate(self, doc: dict[str, Any]) -> bool:
        return not self.item.evaluate(doc)

    def trace(self, doc: dict[str, Any]) -> Trace:
        kid = self.item.trace(doc)
        return Trace("not", not kid.result, children=[kid])


class _Test(Condition):
    def __init__(self, path: str, op: str, operand: Getter, raw: Any):
        self.path, self.op, self.operand, self.raw = path, op, operand, raw
        self.fn = OPERATORS[op]

    def evaluate(self, doc: dict[str, Any]) -> bool:
        try:
            return bool(self.fn(resolve_path(doc, self.path), self.operand(doc)))
        except Exception:  # total by design: never let a bad input raise through the engine
            return False

    def trace(self, doc: dict[str, Any]) -> Trace:
        actual = resolve_path(doc, self.path)
        expected = self.operand(doc)
        return Trace(
            "test",
            self.evaluate(doc),
            {"field": self.path, "op": self.op, "expected": expected, "actual": actual},
        )


def compile_condition(spec: Any, where: str = "when") -> Condition:
    """Compile a YAML condition tree. Raises ``PolicyCompileError`` on anything outside the grammar."""
    if spec is None or spec == {} or spec == []:
        return _Always()
    if isinstance(spec, list):
        return _All([compile_condition(s, f"{where}[{i}]") for i, s in enumerate(spec)])
    if not isinstance(spec, dict):
        raise PolicyCompileError(
            f"{where}: a condition must be a mapping or a list, got {type(spec).__name__} {spec!r} "
            "(expressions are data, not code)"
        )
    parts: list[Condition] = []
    for key, val in spec.items():
        if key == "all":
            if not isinstance(val, list):
                raise PolicyCompileError(f"{where}.all must be a list")
            parts.append(_All([compile_condition(s, f"{where}.all[{i}]") for i, s in enumerate(val)]))
        elif key == "any":
            if not isinstance(val, list):
                raise PolicyCompileError(f"{where}.any must be a list")
            parts.append(_Any([compile_condition(s, f"{where}.any[{i}]") for i, s in enumerate(val)]))
        elif key == "not":
            parts.append(_Not(compile_condition(val, f"{where}.not")))
        else:
            parts.extend(_compile_tests(_check_path(key, where), val, f"{where}.{key}"))
    return parts[0] if len(parts) == 1 else _All(parts)


def _compile_tests(path: str, spec: Any, where: str) -> list[Condition]:
    if isinstance(spec, dict) and set(spec) == {"ref"}:
        return [_Test(path, "eq", _value(spec, where), spec)]
    if isinstance(spec, list):
        return [_Test(path, "in", _value(spec, where), spec)]
    if not isinstance(spec, dict):
        return [_Test(path, "eq", _value(spec, where), spec)]
    out: list[Condition] = []
    for op, operand in spec.items():
        if op not in OPERATORS:
            raise PolicyCompileError(f"{where}: unknown operator '{op}'. Allowed: {sorted(OPERATORS)}")
        if op == "regex":
            if not isinstance(operand, str) or len(operand) > MAX_REGEX_LENGTH:
                raise PolicyCompileError(f"{where}: regex must be a string under {MAX_REGEX_LENGTH} chars")
            try:
                _regex(operand)
            except re.error as exc:
                raise PolicyCompileError(f"{where}: invalid regex {operand!r}: {exc}") from exc
        if op in ("exists", "empty") and not isinstance(operand, bool):
            raise PolicyCompileError(f"{where}: '{op}' takes true or false")
        out.append(_Test(path, op, _value(operand, where), operand))
    if not out:
        raise PolicyCompileError(f"{where}: empty test")
    return out
