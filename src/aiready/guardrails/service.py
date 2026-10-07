"""Guardrail service: screen inputs, screen retrieved documents, validate outputs, mask personal data.

Implements ``core.interfaces.Guardrails``. Detectors are composed per application from configuration
(``guardrails.defaults`` + ``guardrails.apps.<app>``) and run concurrently. The service decides:

* **block** - a finding at or above the profile's ``block_on`` severity for its check (injection, toxicity,
  guard model...), or personal data / secrets when the profile says ``block``;
* **mask** - personal data and secrets replaced by typed placeholders (reversible, map kept in memory);
* **classify** - the data class implied by the findings (PERSONAL, RESTRICTED) which the gateway uses for
  routing and egress.

Every outcome is counted in ``aiready_guardrail_total{check,outcome}``; blocks are written to the audit log
(entity types and scores only - never the detected values).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.config import deep_merge, get_config
from aiready.core.errors import ConfigError, PermissionDenied
from aiready.core.identity import Principal
from aiready.core.interfaces import Finding, ScreenResult, ValidationResult
from aiready.core.plugins import create
from aiready.core.telemetry import metrics, span
from aiready.guardrails.config import SEVERITY_RANK, GuardrailsConfig, Profile
from aiready.guardrails.detectors import PiiDetector
from aiready.guardrails.masking import classify, mask_object, mask_text, unmask
from aiready.guardrails.validators import check_banned, check_length, validate_json

log = logging.getLogger(__name__)
SERVICE = "guardrails"


class GuardrailService:
    def __init__(
        self,
        cfg: GuardrailsConfig | None = None,
        *,
        detectors: dict[str, Any] | None = None,
        watch_config: bool = False,
    ):
        self.cfg = cfg or GuardrailsConfig()
        self._detectors: dict[str, Any] = dict(detectors or {})
        self._injected = set(self._detectors)
        self._profiles: dict[str, Profile] = {}
        self._config_version: str | None = None
        self._watch = watch_config

    @classmethod
    def from_config(cls) -> GuardrailService:
        svc = cls(get_config().section("guardrails", GuardrailsConfig), watch_config=True)
        svc._config_version = get_config().version
        return svc

    # ------------------------------------------------------------------ configuration
    def _refresh(self) -> None:
        """Pick up configuration changes (detectors, profiles) without a restart."""
        if not self._watch:
            return
        version = get_config().version
        if version != self._config_version:
            try:
                self.cfg = get_config().section("guardrails", GuardrailsConfig)
            except ConfigError as exc:
                log.error("guardrails config rejected, keeping previous: %s", exc)
                return
            self._config_version = version
            self._profiles.clear()
            self._detectors = {k: v for k, v in self._detectors.items() if k in self._injected}

    def profile(self, app: str | None) -> Profile:
        self._refresh()
        key = app or "default"
        if key not in self._profiles:
            base = self.cfg.defaults.model_dump()
            override = self.cfg.apps.get(key) or {}
            merged = deep_merge(base, {k: v for k, v in override.items() if not isinstance(v, list)})
            merged.update({k: v for k, v in override.items() if isinstance(v, list)})  # lists replace
            self._profiles[key] = Profile.model_validate(merged)
        return self._profiles[key]

    def detector(self, name: str) -> Any:
        if name not in self._detectors:
            spec = self.cfg.detectors.get(name)
            if spec is None:
                spec = {"type": name}  # a registered detector used by name without extra settings
            self._detectors[name] = create("detector", spec)
        return self._detectors[name]

    # ------------------------------------------------------------------ helpers
    async def _run(self, names: list[str], text: str, context: dict[str, Any]) -> list[Finding]:
        async def one(name: str) -> list[Finding]:
            try:
                det = self.detector(name)
                return await det.detect(text, context=context)
            except ConfigError:
                raise
            except Exception as exc:  # a broken detector is visible, never silently "clean"
                log.exception("detector %s failed", name)
                metrics.guardrail.labels(SERVICE, name, "error").inc()
                return [Finding(check="detector_error", severity="low", message=f"{name}: {exc}")]

        results = await asyncio.gather(*(one(n) for n in names))
        return [f for batch in results for f in batch]

    @staticmethod
    def _blocking(findings: list[Finding], profile: Profile) -> list[Finding]:
        out = []
        for f in findings:
            threshold = profile.block_on.get(f.check, "never")
            if SEVERITY_RANK.get(f.severity, 0) >= SEVERITY_RANK.get(threshold, 99):
                out.append(f)
        return out

    @staticmethod
    def _count(stage: str, findings: list[Finding], outcome: str) -> None:
        checks = sorted({f.check for f in findings}) or [stage]
        for c in checks:
            metrics.guardrail.labels(SERVICE, c, outcome if findings else "pass").inc()

    @staticmethod
    def _audit(
        event: str,
        principal: Principal | None,
        decision: str,
        findings: list[Finding],
        context: dict[str, Any],
        **extra: Any,
    ) -> None:
        details = {
            "checks": sorted({f.check for f in findings}),
            "entities": sorted({f.entity for f in findings if f.entity}),
            "max_score": max((f.score or 0.0 for f in findings), default=0.0),
            "rules": sorted({f.message for f in findings if f.check == "injection"})[:20],
            "feature": context.get("feature"),
            "source": context.get("source"),
            **extra,
        }
        audit = get_audit()
        if principal is not None:
            audit.for_principal(principal, event, action="guardrails.screen", decision=decision, **details)
        else:
            audit.record(
                event,
                actor=str(context.get("actor") or "anonymous"),
                app=str(context.get("app") or "default"),
                action="guardrails.screen",
                decision=decision,
                **details,
            )

    # ------------------------------------------------------------------ Guardrails protocol
    async def screen_input(
        self, text: str, *, principal: Principal | None = None, context: dict[str, Any] | None = None
    ) -> ScreenResult:
        context = dict(context or {})
        app = context.get("app") or (principal.app if principal else None)
        profile = self.profile(app)
        if principal is not None:
            context.setdefault("principal", principal)
        with (
            span("guardrails.screen_input", app=app, chars=len(text or "")) as sp,
            metrics.time_stage(SERVICE, "screen_input"),
        ):
            findings = check_length(text, profile.max_input_chars)
            if not findings:
                findings = await self._run(profile.input_detectors, text or "", context)
            blocking = self._blocking(findings, profile)
            pii = [f for f in findings if f.check == "pii"]
            secrets = [f for f in findings if f.check == "secret"]
            if pii and profile.pii_action == "block":
                blocking += pii
            if secrets and profile.secret_action == "block":
                blocking += secrets
            to_mask = ([*pii] if profile.pii_action == "mask" else []) + (
                secrets if profile.secret_action == "mask" else []
            )
            masked, mapping = (
                mask_text(text or "", to_mask, entities=profile.mask_entities)
                if to_mask
                else (text or "", {})
            )
            data_class = classify(findings, context.get("data_class"))
            allowed = not blocking
            outcome = "blocked" if not allowed else ("masked" if mapping else "flagged")
            self._count("input", findings, outcome)
            sp.set(
                allowed=allowed,
                checks=sorted({f.check for f in findings}),
                data_class=data_class.value,
                masked=len(mapping),
            )
            if not allowed:
                self._audit("guardrails.input_blocked", principal, "blocked", blocking, context)
            elif findings and profile.audit_findings:
                self._audit("guardrails.input_flagged", principal, outcome, findings, context)
            return ScreenResult(
                allowed=allowed,
                text=masked,
                findings=findings,
                data_class=data_class,
                masked_entities=mapping,
            )

    async def screen_document(
        self,
        text: str,
        *,
        source: str | None = None,
        principal: Principal | None = None,
        context: dict[str, Any] | None = None,
    ) -> ScreenResult:
        """Screen retrieved or ingested content for *indirect* prompt injection.

        ``allowed=False`` means the passage must not be placed in a prompt (``document_action: drop``); with
        ``flag`` it is returned allowed but with findings so the caller can label it as untrusted.
        """
        context = {**(context or {}), "source": source, "stage": "document"}
        app = context.get("app") or (principal.app if principal else None)
        profile = self.profile(app)
        with (
            span("guardrails.screen_document", source=source) as sp,
            metrics.time_stage(SERVICE, "screen_document"),
        ):
            findings = await self._run(profile.document_detectors, text or "", context)
            inj = [f for f in findings if f.check == "injection"]
            score = max((f.score or 0.0 for f in inj), default=0.0)
            hit = score >= profile.document_threshold or bool(self._blocking(findings, profile))
            allowed = not (hit and profile.document_action == "drop")
            pii = [f for f in findings if f.check in ("pii", "secret")]
            data_class = classify(pii, context.get("data_class"))
            self._count("document", findings, "blocked" if not allowed else ("flagged" if hit else "pass"))
            sp.set(allowed=allowed, injection_score=score)
            if hit:
                self._audit(
                    "guardrails.document_flagged",
                    principal,
                    "dropped" if not allowed else "flagged",
                    findings,
                    context,
                    injection_score=score,
                )
            return ScreenResult(allowed=allowed, text=text or "", findings=findings, data_class=data_class)

    async def validate_output(
        self, text: str, *, schema: dict[str, Any] | None = None, context: dict[str, Any] | None = None
    ) -> ValidationResult:
        context = dict(context or {})
        principal = context.get("principal") if isinstance(context.get("principal"), Principal) else None
        app = context.get("app") or (principal.app if principal else None)
        profile = self.profile(app)
        with (
            span("guardrails.validate_output", app=app, chars=len(text or ""), schema=bool(schema)) as sp,
            metrics.time_stage(SERVICE, "validate_output"),
        ):
            findings: list[Finding] = check_length(text, profile.max_output_chars)
            parsed: Any = None
            repairs: list[str] = []
            if schema is not None:
                parsed, schema_findings, repairs = validate_json(text or "", schema)
                findings += schema_findings
            findings += check_banned(text or "", profile.banned_output)
            detected = await self._run(profile.output_detectors, text or "", context)

            known = {str(v) for v in (context.get("known_values") or [])}
            allowed_entities = set(profile.output_allowed_entities) | set(context.get("allow_entities") or [])
            leaks: list[Finding] = []
            pii_out: list[Finding] = []
            for f in detected:
                value = (text or "")[f.span[0] : f.span[1]] if f.span else ""
                if f.check == "secret":
                    leaks.append(
                        Finding(
                            check="leak",
                            severity="high",
                            entity=f.entity,
                            span=f.span,
                            score=f.score,
                            message=f"secret in output: {f.entity}",
                        )
                    )
                elif f.check == "pii" and f.entity not in allowed_entities and value not in known:
                    pii_out.append(f)
                elif f.check not in ("pii",):
                    findings.append(f)
            findings += leaks
            if pii_out and profile.output_pii_action == "block":
                findings += [
                    Finding(
                        check="leak",
                        severity="high",
                        entity=f.entity,
                        span=f.span,
                        message=f"personal data in output: {f.entity}",
                    )
                    for f in pii_out
                ]
            elif pii_out:
                findings += pii_out

            out_text = text or ""
            if pii_out and profile.output_pii_action == "mask":
                out_text, mapping = mask_text(out_text, pii_out)
                if parsed is not None:
                    det = PiiDetector()
                    parsed = mask_object(
                        parsed,
                        lambda s: mask_text(
                            s,
                            [
                                f
                                for f in det.find(s)
                                if f.check == "pii"
                                and f.entity not in allowed_entities
                                and s[f.span[0] : f.span[1]] not in known
                            ],
                            mapping=mapping,
                        )[0],
                    )
                    out_text = json.dumps(parsed, ensure_ascii=False)
            elif parsed is not None and repairs:
                out_text = json.dumps(parsed, ensure_ascii=False)

            blocking = [f for f in findings if f.check in ("schema", "leak", "banned", "length")]
            blocking += self._blocking(
                [f for f in findings if f.check not in ("schema", "leak", "banned", "length")], profile
            )
            valid = not blocking
            self._count("output", findings, "blocked" if not valid else ("masked" if pii_out else "flagged"))
            sp.set(valid=valid, checks=sorted({f.check for f in findings}), repairs=repairs)
            if not valid:
                self._audit(
                    "guardrails.output_blocked", principal, "blocked", blocking, context, repairs=repairs
                )
            return ValidationResult(valid=valid, text=out_text, parsed=parsed, findings=findings)

    async def mask(self, text: str) -> ScreenResult:
        with span("guardrails.mask"), metrics.time_stage(SERVICE, "mask"):
            findings = await self._run(["pii"], text or "", {})
            masked, mapping = mask_text(text or "", findings)
            self._count("mask", findings, "masked")
            return ScreenResult(
                allowed=True,
                text=masked,
                findings=findings,
                data_class=classify(findings),
                masked_entities=mapping,
            )

    # ------------------------------------------------------------------ extras
    def unmask(self, text: str, mapping: dict[str, str], principal: Principal, *, purpose: str = "") -> str:
        """Restore masked values for an authorised flow (role-gated and audited)."""
        if not principal.has_role(*self.cfg.unmask_roles):
            get_audit().for_principal(
                principal,
                "guardrails.unmask",
                action="guardrails.unmask",
                decision="deny",
                placeholders=len(mapping),
                purpose=purpose,
            )
            raise PermissionDenied(f"unmasking requires one of roles {self.cfg.unmask_roles}")
        get_audit().for_principal(
            principal,
            "guardrails.unmask",
            action="guardrails.unmask",
            decision="allow",
            placeholders=len(mapping),
            purpose=purpose,
            ts=time.time(),
        )
        return unmask(text, mapping)

    def detectors_info(self) -> dict[str, Any]:
        from aiready.core.plugins import available

        return {
            "configured": self.cfg.detectors,
            "available": available("detector"),
            "defaults": self.cfg.defaults.model_dump(),
            "apps": sorted(self.cfg.apps),
        }

    @staticmethod
    def injection_score(findings: list[Finding]) -> float:
        return max((f.score or 0.0 for f in findings if f.check == "injection"), default=0.0)
