"""Personal-data masking applied *before* anything is stored.

The observability architecture slide puts "personal data masked before storage" in the collector: traces,
feedback and review items are kept for months and read by many people, so they must never hold raw personal
data. The guardrails service owns the authoritative detector (``connect("guardrails").mask``) - it may run a
NER model and client-specific recognisers. It is used when available. The built-in regex masker below always
runs as well, as a second line of defence and as the fallback when guardrails is not deployed (pilot, tests,
or a guardrails outage, which must not stop trace ingestion).

The same masker is reused by the fine-tuning dataset builder (training data is masked again even when the
logs claim to be masked) and by the ``pii_leak`` evaluation scorer (detection only).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

log = logging.getLogger(__name__)


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(nums)):
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class PiiPattern:
    entity: str
    regex: re.Pattern[str]
    validator: Any = None  # optional callable(str) -> bool to cut false positives (e.g. Luhn for cards)


# Ordered: more specific patterns first so an IBAN is not half-eaten by the phone pattern.
DEFAULT_PATTERNS: list[PiiPattern] = [
    PiiPattern("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    PiiPattern("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")),
    PiiPattern("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _luhn_ok),
    PiiPattern("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    PiiPattern("PHONE", re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?:[\s.-]\d{2,4})?(?![\w-])")),
    PiiPattern("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


@dataclass
class PiiMatch:
    entity: str
    start: int
    end: int
    text: str


class RegexMasker:
    """Deterministic, dependency-free PII detector/masker.

    ``extra_patterns`` (entity -> regex) lets a client add identifiers specific to its domain (policy numbers,
    member ids) purely in configuration.
    """

    def __init__(self, extra_patterns: dict[str, str] | None = None, placeholder: str = "[{entity}]",
                 disabled: Iterable[str] = ()):
        off = {d.upper() for d in disabled}
        self.patterns = [p for p in DEFAULT_PATTERNS if p.entity not in off]
        for entity, rx in (extra_patterns or {}).items():
            self.patterns.insert(0, PiiPattern(entity.upper(), re.compile(rx)))
        self.placeholder = placeholder

    def find(self, text: str) -> list[PiiMatch]:
        found: list[PiiMatch] = []
        taken: list[tuple[int, int]] = []
        for p in self.patterns:
            for m in p.regex.finditer(text or ""):
                if p.validator and not p.validator(m.group(0)):
                    continue
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                found.append(PiiMatch(p.entity, m.start(), m.end(), m.group(0)))
        return sorted(found, key=lambda x: x.start)

    def mask(self, text: str) -> str:
        if not text:
            return text
        out, pos = [], 0
        for m in self.find(text):
            out.append(text[pos:m.start])
            out.append(self.placeholder.format(entity=m.entity))
            pos = m.end
        out.append(text[pos:])
        return "".join(out)

    def contains_pii(self, text: str) -> bool:
        return bool(self.find(text))


class SpanMasker:
    """Masks every string inside nested attributes (dicts, lists) - the shape spans and feedback arrive in.

    Uses ``connect("guardrails").mask`` when ``use_guardrails`` is on and the service is reachable, then the
    regex masker. A failing guardrails call disables it for ``retry_after`` seconds instead of slowing every
    ingest batch down with timeouts.
    """

    #: attribute keys that never hold free text and are skipped (ids, numbers, model names)
    SAFE_KEYS = frozenset({"trace_id", "span_id", "parent_id", "model", "app", "feature", "team", "status",
                           "service", "route", "stage", "doc_id", "chunk_id", "prompt_version"})

    def __init__(self, regex: RegexMasker | None = None, *, use_guardrails: bool = True, retry_after: float = 60.0,
                 max_string: int = 20000):
        self.regex = regex or RegexMasker()
        self.use_guardrails = use_guardrails
        self.retry_after = retry_after
        self.max_string = max_string
        self._guardrails_down_until = 0.0

    async def _guardrails_mask(self, text: str) -> str:
        if not self.use_guardrails or time.time() < self._guardrails_down_until:
            return text
        try:
            from aiready.core.wiring import connect

            res = await connect("guardrails").mask(text)
            return res.text if hasattr(res, "text") else str(res)
        except Exception as exc:  # guardrails optional for ingestion; regex below still masks
            log.debug("guardrails mask unavailable, using regex only: %s", exc)
            self._guardrails_down_until = time.time() + self.retry_after
            return text

    async def mask_text(self, text: str) -> str:
        if not text:
            return text
        if len(text) > self.max_string:
            text = text[: self.max_string] + "...[truncated]"
        text = await self._guardrails_mask(text)
        return self.regex.mask(text)

    async def mask_value(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            return value if key in self.SAFE_KEYS else await self.mask_text(value)
        if isinstance(value, dict):
            return {k: await self.mask_value(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [await self.mask_value(v, key) for v in value]
        return value
