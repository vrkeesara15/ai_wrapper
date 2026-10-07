"""Fixtures for gateway tests: the shipped config/gateway.yaml + config/models.yaml (mock backends), with
policy and guardrails replaced by fakes through ``wiring.override``."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from aiready import testing
from aiready.core import config, wiring
from aiready.core.config import _substitute, deep_merge
from aiready.core.identity import Principal
from aiready.core.interfaces import Finding, ScreenResult, ValidationResult
from aiready.core.types import DataClass

REPO = Path(__file__).resolve().parents[2]


def shipped_tree() -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for name in ("models.yaml", "gateway.yaml"):
        tree = deep_merge(tree, yaml.safe_load((REPO / "config" / name).read_text()))
    tree = _substitute(tree)
    tree["state"] = {"url": "sqlite:///:memory:"}
    tree["gateway"]["admin"]["sync_seconds"] = 0
    return tree


class ScriptedGuardrails:
    """Flags injection, masks e-mail addresses (data class personal) and fails outputs containing UNSAFE."""

    EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

    def __init__(self) -> None:
        self.screened: list[str] = []
        self.validated: list[str] = []

    async def screen_input(self, text: str, *, principal: Principal | None = None,
                           context: dict[str, Any] | None = None) -> ScreenResult:
        self.screened.append(text)
        if "ignore previous instructions" in text.lower():
            return ScreenResult(allowed=False, text=text, findings=[
                Finding(check="injection", severity="high", message="instruction override")])
        masked: dict[str, str] = {}

        def repl(m: re.Match[str]) -> str:
            ph = f"<EMAIL_{len(masked) + 1}>"
            masked[ph] = m.group(0)
            return ph

        out = self.EMAIL.sub(repl, text)
        if masked:
            return ScreenResult(allowed=True, text=out, data_class=DataClass.PERSONAL, masked_entities=masked,
                                findings=[Finding(check="pii", severity="medium", entity="EMAIL")])
        return ScreenResult(allowed=True, text=text)

    async def validate_output(self, text: str, *, schema: dict[str, Any] | None = None,
                              context: dict[str, Any] | None = None) -> ValidationResult:
        self.validated.append(text)
        if "UNSAFE" in text:
            return ValidationResult(valid=False, text=text, findings=[Finding(check="safety", severity="high")])
        return await testing.PassThroughGuardrails().validate_output(text, schema=schema, context=context)

    async def mask(self, text: str) -> ScreenResult:
        return await self.screen_input(text)


@pytest.fixture
def guardrails() -> ScriptedGuardrails:
    return ScriptedGuardrails()


@pytest.fixture
def policy() -> testing.AllowAllPolicy:
    return testing.AllowAllPolicy()


@pytest.fixture
def make_gateway(guardrails, policy) -> Callable[..., Any]:
    """Build a GatewayService over the shipped config, optionally mutated by ``mutate(tree)``."""

    def _make(mutate: Callable[[dict[str, Any]], None] | None = None, *, policy_impl: Any = None):
        from aiready.gateway.service import GatewayService

        tree = shipped_tree()
        if mutate:
            mutate(tree)
        config.set_config(config.ConfigStore(tree=copy.deepcopy(tree)))
        wiring.override("policy", policy_impl or policy)
        wiring.override("guardrails", guardrails)
        return GatewayService()

    return _make


@pytest.fixture
def alice(principal) -> Principal:
    return principal  # alice / acme / claims, groups [claims-team]


@pytest.fixture
def admin_principal(token_service) -> Principal:
    return token_service.verify(token_service.issue(sub="root", tenant="acme", roles=["platform-admin"]))
