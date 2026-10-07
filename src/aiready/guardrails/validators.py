"""Output validation: JSON schema (with repair), length, banned content.

Models often wrap JSON in markdown fences, add a sentence before it or leave a trailing comma. A bounded,
deterministic repair (strip fences, cut to the outermost object/array, remove trailing commas, normalise
smart quotes) recovers most of these without a second model call; anything still invalid is reported as a
``schema`` finding so the caller can retry or fall back.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from aiready.core.interfaces import Finding

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _outermost(text: str) -> str | None:
    """The first balanced {...} or [...] in the text (string-aware)."""
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None
    stack: list[str] = []
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return text[start : i + 1]
    return None


def parse_json(text: str) -> tuple[Any, list[str]]:
    """Parse JSON, repairing common model formatting mistakes. Returns (value, repairs applied).

    Raises ``ValueError`` when the text cannot be recovered.
    """
    try:
        return json.loads(text), []
    except (ValueError, TypeError):
        pass
    repairs: list[str] = []
    candidate = text or ""
    m = _FENCE.search(candidate)
    if m:
        candidate = m.group(1)
        repairs.append("stripped_code_fence")
    candidate = candidate.strip()
    if candidate.translate(_SMART) != candidate:
        candidate = candidate.translate(_SMART)
        repairs.append("normalised_quotes")
    inner = _outermost(candidate)
    if inner is not None and inner != candidate:
        candidate = inner
        repairs.append("extracted_json")
    fixed = _TRAILING_COMMA.sub(r"\1", candidate)
    if fixed != candidate:
        candidate = fixed
        repairs.append("removed_trailing_commas")
    try:
        return json.loads(candidate), repairs
    except ValueError as exc:
        raise ValueError(f"not valid JSON after repair ({', '.join(repairs) or 'none'}): {exc}") from exc


def validate_schema(value: Any, schema: dict[str, Any]) -> list[Finding]:
    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
    except jsonschema.SchemaError as exc:
        return [Finding(check="schema", severity="high", message=f"invalid schema: {exc.message}")]
    validator = validator_cls(schema)
    out = []
    for err in sorted(validator.iter_errors(value), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "$"
        out.append(Finding(check="schema", severity="high", message=f"{where}: {err.message}"))
    return out


def validate_json(text: str, schema: dict[str, Any] | None) -> tuple[Any, list[Finding], list[str]]:
    """(parsed value or None, findings, repairs)."""
    try:
        value, repairs = parse_json(text)
    except ValueError as exc:
        return None, [Finding(check="schema", severity="high", message=str(exc))], []
    findings = validate_schema(value, schema) if schema else []
    return value, findings, repairs


def check_length(text: str, max_chars: int | None, check: str = "length") -> list[Finding]:
    if max_chars is not None and len(text or "") > max_chars:
        return [
            Finding(
                check=check,
                severity="high",
                message=f"{len(text)} characters exceeds the limit of {max_chars}",
            )
        ]
    return []


def check_banned(text: str, patterns: list[str]) -> list[Finding]:
    out = []
    for p in patterns:
        try:
            m = re.search(p, text or "", re.IGNORECASE)
        except re.error:
            m = re.search(re.escape(p), text or "", re.IGNORECASE)
        if m:
            out.append(
                Finding(check="banned", severity="high", span=m.span(), message=f"banned content: {p}")
            )
    return out
