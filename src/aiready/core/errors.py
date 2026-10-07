"""Error types shared by every service. Each maps to an HTTP status in core.app."""

from __future__ import annotations


class AIReadyError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class AuthenticationError(AIReadyError):
    status_code = 401
    code = "unauthenticated"


class PermissionDenied(AIReadyError):
    status_code = 403
    code = "permission_denied"


class PolicyViolation(PermissionDenied):
    code = "policy_violation"


class GuardrailBlocked(AIReadyError):
    status_code = 422
    code = "guardrail_blocked"


class RateLimited(AIReadyError):
    status_code = 429
    code = "rate_limited"


class BudgetExceeded(AIReadyError):
    status_code = 429
    code = "budget_exceeded"


class NotFound(AIReadyError):
    status_code = 404
    code = "not_found"


class ValidationFailed(AIReadyError):
    status_code = 422
    code = "validation_failed"


class UpstreamError(AIReadyError):
    status_code = 502
    code = "upstream_error"


class ApprovalRequired(AIReadyError):
    status_code = 202
    code = "approval_required"


class ConfigError(AIReadyError):
    status_code = 500
    code = "config_error"
