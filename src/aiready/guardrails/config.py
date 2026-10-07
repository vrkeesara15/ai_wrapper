"""Configuration models for the guardrails service (section ``guardrails``).

A *profile* says which detectors run on inputs, outputs and retrieved documents, at which severity a
finding blocks, and what happens to personal data (mask / block / allow). ``guardrails.defaults`` is the
base profile; ``guardrails.apps.<app>`` overrides any part of it for one application.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "never": 99}


class Profile(BaseModel):
    input_detectors: list[str] = Field(default_factory=lambda: ["injection", "pii", "toxicity"])
    output_detectors: list[str] = Field(default_factory=lambda: ["pii", "toxicity"])
    document_detectors: list[str] = Field(default_factory=lambda: ["injection_document"])
    # A finding of this check at or above this severity blocks ("never" disables blocking for a check).
    block_on: dict[str, str] = Field(
        default_factory=lambda: {
            "injection": "high",
            "toxicity": "high",
            "guard_model": "high",
            "length": "low",
        }
    )
    # Personal data in inputs: mask (typed placeholders), block, or allow untouched.
    pii_action: Literal["mask", "block", "allow"] = "mask"
    # Secrets (keys, tokens, private keys) in inputs: mask or block. Never sent to a model as-is.
    secret_action: Literal["mask", "block"] = "mask"
    # Personal data in model outputs: mask, block or allow; secrets in outputs always invalidate the answer.
    output_pii_action: Literal["mask", "block", "allow"] = "mask"
    # Entity types allowed to appear in outputs untouched (e.g. [EMAIL] for a CRM assistant).
    output_allowed_entities: list[str] = Field(default_factory=list)
    # Only these entity types are masked (None = every detected entity).
    mask_entities: list[str] | None = None
    max_input_chars: int | None = 32000
    max_output_chars: int | None = 20000
    # Regexes that must never appear in an answer (case-insensitive).
    banned_output: list[str] = Field(default_factory=list)
    # Retrieved documents: injection score at/above this drops (or flags) the passage.
    document_threshold: float = 0.4
    document_action: Literal["drop", "flag"] = "drop"
    # Audit every screen with findings, not only blocks.
    audit_findings: bool = False


class GuardrailsConfig(BaseModel):
    # Detector instances by name -> plugin spec (kind `detector`). Profiles refer to these names.
    detectors: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "injection": {"type": "injection", "threshold": 0.5},
            "injection_document": {"type": "injection", "threshold": 0.4, "mode": "document"},
            "pii": {"type": "pii"},
            "toxicity": {"type": "toxicity"},
        }
    )
    defaults: Profile = Field(default_factory=Profile)
    # Per-app overrides, merged onto `defaults` (dicts merge, other values replace).
    apps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Roles allowed to unmask placeholders back to the original values (audited).
    unmask_roles: list[str] = Field(default_factory=lambda: ["pii-reader"])
