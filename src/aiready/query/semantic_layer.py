"""Semantic layer: business terms and approved metrics (``config/semantic_layer.yaml``).

An approved metric is a fixed, reviewed SQL statement with bound parameters, a description, synonyms and
the roles allowed to use it. The router always prefers a matching metric to generated SQL (slide
"Structured data access"): the number a manager sees is then the number finance agreed on.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from aiready.core.errors import PermissionDenied, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.textutil import tokenize
from aiready.query.config import Metric, SemanticLayerConfig


def _stem(tok: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(tok) > 4 and tok.endswith(suffix):
            return tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tok


def terms_of(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text.replace("_", " "))}


class SemanticLayer:
    def __init__(self, config: SemanticLayerConfig | None = None):
        cfg = config or SemanticLayerConfig()
        self.terms = dict(cfg.terms)
        self.metrics: dict[str, Metric] = {}
        for name, m in cfg.metrics.items():
            self.metrics[name] = m.model_copy(update={"name": m.name or name})

    def get(self, name: str) -> Metric | None:
        return self.metrics.get(name)

    @staticmethod
    def allowed(metric: Metric, principal: Principal) -> bool:
        return not metric.allowed_roles or principal.has_role(*metric.allowed_roles)

    def visible(self, principal: Principal) -> list[Metric]:
        return [m for m in self.metrics.values() if self.allowed(m, principal)]

    def match(self, question: str, principal: Principal | None = None) -> tuple[Metric, float] | None:
        """Best metric for a question: share of a phrase's words (name or synonym) found in the question."""
        q = terms_of(question)
        if not q:
            return None
        best: tuple[Metric, float] | None = None
        for m in self.metrics.values():
            if principal is not None and not self.allowed(m, principal):
                continue
            for phrase in [m.name, *m.synonyms]:
                p = terms_of(phrase)
                if not p:
                    continue
                score = len(p & q) / len(p)
                if best is None or score > best[1] or (score == best[1] and len(p) > len(terms_of(best[0].name))):
                    best = (m, score)
        return best

    def bind_params(self, metric: Metric, params: dict[str, Any], reserved: set[str]) -> dict[str, Any]:
        """Validate and coerce request parameters against the metric's declaration.

        Reserved names (``tenant``, ``user``, ``app``) come from the verified principal and can never be
        supplied by the caller - otherwise a request could read another tenant's numbers.
        """
        clash = reserved & set(params)
        if clash:
            raise PermissionDenied(f"parameters {sorted(clash)} are set from your identity and cannot be passed")
        unknown = set(params) - set(metric.params)
        if unknown:
            raise ValidationFailed(f"unknown parameters for metric '{metric.name}': {sorted(unknown)}")
        out: dict[str, Any] = {}
        for name, spec in metric.params.items():
            if name in params and params[name] is not None:
                val = _coerce(name, params[name], spec.type)
            elif spec.default is not None:
                val = _coerce(name, spec.default, spec.type)
            elif spec.required:
                raise ValidationFailed(f"metric '{metric.name}' needs parameter '{name}' ({spec.description})")
            else:
                val = None
            if spec.allowed_values is not None and val is not None and val not in spec.allowed_values:
                raise ValidationFailed(f"parameter '{name}' must be one of {spec.allowed_values}")
            out[name] = val
        return out

    def missing_required(self, metric: Metric, params: dict[str, Any]) -> list[str]:
        return [n for n, s in metric.params.items() if s.required and s.default is None and n not in params]

    def describe_terms(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.terms.items()) or "(none)"


def _coerce(name: str, value: Any, typ: str) -> Any:
    try:
        if typ == "integer":
            return int(value)
        if typ == "number":
            return float(value)
        if typ == "boolean":
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
        if typ == "date":
            return date.fromisoformat(str(value)).isoformat()
        text = str(value)
        if len(text) > 500 or re.search(r"[\x00-\x08]", text):
            raise ValueError("invalid string")
        return text
    except (TypeError, ValueError) as exc:
        raise ValidationFailed(f"parameter '{name}' must be a {typ}") from exc
