"""Search filters shared by every index backend.

A ``SearchFilter`` is applied *before* scoring: access first, then metadata. Backends translate it to their
native filter (Qdrant payload filter, SQL ``WHERE``, OpenSearch ``bool.filter``) or, for in-memory indexes,
evaluate ``matches`` on each stored chunk payload.

Access rule: a chunk is visible only if ``set(chunk.access_tags) & set(access_groups)``. A chunk without tags
is visible to nobody (fail closed). ``public`` is in every authenticated principal's ``access_groups``, so a
chunk tagged ``public`` is visible to all signed-in users.

Metadata filter syntax (``filters`` in a search request):

    {"doc_id": "d1"}                       equality
    {"source": ["wiki", "jira"]}           any of
    {"updated_at": {"gte": 1700000000}}    range: gt / gte / lt / lte
    {"data_class": {"in": [...]}}          any-of; {"ne": x} / {"nin": [...]} exclude
    {"metadata.region": "eu"}              under chunk.metadata (bare unknown keys are looked up there too)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aiready.core.errors import ValidationFailed

# Chunk fields stored at the top level of every payload; any other filter key refers to chunk.metadata.
TOP_LEVEL_FIELDS = frozenset(
    {
        "id",
        "doc_id",
        "title",
        "source",
        "source_uri",
        "section",
        "position",
        "data_class",
        "owner",
        "updated_at",
        "access_tags",
    }
)
OPERATORS = frozenset({"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte"})


@dataclass
class SearchFilter:
    # None = no access filter: internal maintenance only, never a user search
    access_groups: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, cond in self.metadata.items():
            if isinstance(cond, dict):
                bad = set(cond) - OPERATORS
                if bad:
                    raise ValidationFailed(f"unknown filter operator(s) {sorted(bad)} on '{key}'")


def field_path(key: str) -> tuple[str, ...]:
    """Payload path of a filter key: top-level chunk field or a key under ``metadata``."""
    if key.startswith("metadata."):
        return ("metadata", key[len("metadata.") :])
    if key in TOP_LEVEL_FIELDS:
        return (key,)
    return ("metadata", key)


def _get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for p in path:
        if not isinstance(node, dict):
            return None
        node = node.get(p)
    return node


def _norm(v: Any) -> Any:
    return v.value if hasattr(v, "value") and not isinstance(v, (str, int, float)) else v


def _eq(actual: Any, expected: Any) -> bool:
    actual, expected = _norm(actual), _norm(expected)
    if isinstance(actual, list):
        return expected in actual
    return actual == expected


def condition_matches(actual: Any, cond: Any) -> bool:
    if isinstance(cond, dict):
        for op, val in cond.items():
            if op == "eq" and not _eq(actual, val):
                return False
            if op == "ne" and _eq(actual, val):
                return False
            if op == "in" and not any(_eq(actual, v) for v in val):
                return False
            if op == "nin" and any(_eq(actual, v) for v in val):
                return False
            if op in ("gt", "gte", "lt", "lte"):
                if actual is None:
                    return False
                try:
                    a, b = float(actual), float(val)
                except (TypeError, ValueError):
                    a, b = str(actual), str(val)  # ISO dates compare as strings
                if (
                    (op == "gt" and not a > b)
                    or (op == "gte" and not a >= b)
                    or (op == "lt" and not a < b)
                    or (op == "lte" and not a <= b)
                ):
                    return False
        return True
    if isinstance(cond, (list, tuple, set)):
        return any(_eq(actual, v) for v in cond)
    return _eq(actual, cond)


def access_allowed(tags: list[str] | None, groups: list[str] | None) -> bool:
    if groups is None:
        return True
    return bool(set(tags or ()) & set(groups))


def matches(payload: dict[str, Any], flt: SearchFilter | None) -> bool:
    """Evaluate a filter against a stored chunk payload (``Chunk.model_dump()`` without vector)."""
    if flt is None:
        return True
    if not access_allowed(payload.get("access_tags"), flt.access_groups):
        return False
    return all(condition_matches(_get(payload, field_path(k)), c) for k, c in flt.metadata.items())
