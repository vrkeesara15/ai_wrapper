"""Reversible masking with typed placeholders.

``Alice (alice@example.com) paid with 4111 1111 1111 1111`` becomes
``Alice (<EMAIL_1>) paid with <CREDIT_CARD_1>``. The same value always gets the same placeholder within one
text, so the model can still reason about "the same email twice". The placeholder -> value map is returned
to the caller and kept in memory only; ``unmask`` restores values for flows that are authorised to see them
(e.g. showing the user their own data in the answer).

Typed placeholders keep answers useful (the model knows a value is an email) without the value ever leaving
the boundary or entering logs, caches or training data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from aiready.core.interfaces import Finding
from aiready.core.types import DataClass
from aiready.guardrails.detectors import SECRET_ENTITIES

_PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9_]*)_(\d+)>")


def mask_text(
    text: str,
    findings: Iterable[Finding],
    *,
    entities: Iterable[str] | None = None,
    mapping: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Replace finding spans with placeholders. Returns (masked text, placeholder -> original).

    Pass an existing ``mapping`` to keep numbering consistent across several texts of one conversation.
    """
    allowed = set(entities) if entities is not None else None
    mapping = dict(mapping or {})
    by_value = {v: k for k, v in mapping.items()}
    counters: dict[str, int] = {}
    for ph in mapping:
        m = _PLACEHOLDER.fullmatch(ph)
        if m:
            counters[m.group(1)] = max(counters.get(m.group(1), 0), int(m.group(2)))
    spans = sorted(
        {
            (f.span[0], f.span[1], f.entity or f.check.upper())
            for f in findings
            if f.span
            and f.span[1] > f.span[0]
            and f.check in ("pii", "secret")
            and (allowed is None or (f.entity or "") in allowed)
        }
    )
    # drop overlaps (detectors already resolve them; this guards against combined detector output)
    clean: list[tuple[int, int, str]] = []
    for s in spans:
        if not clean or s[0] >= clean[-1][1]:
            clean.append(s)
    out, pos = [], 0
    for start, end, entity in clean:
        value = text[start:end]
        ph = by_value.get(value)
        if ph is None:
            counters[entity] = counters.get(entity, 0) + 1
            ph = f"<{entity}_{counters[entity]}>"
            mapping[ph], by_value[value] = value, ph
        out.append(text[pos:start])
        out.append(ph)
        pos = end
    out.append(text[pos:])
    return "".join(out), mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore placeholders. Unknown placeholders are left as they are."""
    return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text or "")


def mask_object(obj: Any, mask_fn: Any) -> Any:
    """Apply ``mask_fn(str) -> str`` to every string inside a JSON-like structure."""
    if isinstance(obj, str):
        return mask_fn(obj)
    if isinstance(obj, list):
        return [mask_object(x, mask_fn) for x in obj]
    if isinstance(obj, dict):
        return {k: mask_object(v, mask_fn) for k, v in obj.items()}
    return obj


def classify(findings: Iterable[Finding], floor: DataClass | str | None = None) -> DataClass:
    """Data class implied by the findings: RESTRICTED for secrets, PERSONAL for personal data."""
    cls = DataClass(floor) if floor else DataClass.INTERNAL
    for f in findings:
        if f.check == "secret" or (f.entity or "") in SECRET_ENTITIES:
            cls = DataClass.max(cls, DataClass.RESTRICTED)
        elif f.check == "pii":
            cls = DataClass.max(cls, DataClass.PERSONAL)
    return cls
