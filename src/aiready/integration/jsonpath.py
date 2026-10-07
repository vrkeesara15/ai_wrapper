"""A deliberately small JSONPath subset used to map legacy JSON responses.

Supported: ``$`` root, ``.name`` / ``name`` members, ``[n]`` indexes (negative allowed) and ``[*]`` wildcards,
e.g. ``$.user.login``, ``roles[*]``, ``$.items[*].claim_number``. That covers mapping "who am I" responses and
selecting fields from API results without pulling in a JSONPath dependency or allowing arbitrary expressions
in configuration.
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN = re.compile(r"\.?([^.\[\]]+)|\[(\*|-?\d+)\]")
_MISSING = object()


def _tokens(path: str) -> list[str | int]:
    p = path.strip()
    p = p.removeprefix("$")
    out: list[str | int] = []
    pos = 0
    while pos < len(p):
        m = _TOKEN.match(p, pos)
        if not m:
            raise ValueError(f"invalid path '{path}' at position {pos}")
        if m.group(1) is not None:
            out.append(m.group(1))
        else:
            idx = m.group(2)
            out.append("*" if idx == "*" else int(idx))
        pos = m.end()
    return out


def has_wildcard(path: str) -> bool:
    return "*" in _tokens(path)


def extract(data: Any, path: str | None, default: Any = None) -> Any:
    """Value at ``path``; a list when the path has a wildcard; ``default`` when absent."""
    if path is None or path.strip() in ("", "$"):
        return data
    tokens = _tokens(path)
    wildcard = "*" in tokens
    results = _walk(data, tokens)
    if wildcard:
        return [r for r in results if r is not _MISSING]
    return results[0] if results and results[0] is not _MISSING else default


def _walk(node: Any, tokens: list[str | int]) -> list[Any]:
    if not tokens:
        return [node]
    head, rest = tokens[0], tokens[1:]
    if head == "*":
        if isinstance(node, list):
            items = node
        elif isinstance(node, dict):
            items = list(node.values())
        else:
            return []
        out: list[Any] = []
        for item in items:
            out.extend(_walk(item, rest))
        return out
    if isinstance(head, int):
        if isinstance(node, list) and -len(node) <= head < len(node):
            return _walk(node[head], rest)
        return [_MISSING]
    if isinstance(node, dict) and head in node:
        return _walk(node[head], rest)
    return [_MISSING]
