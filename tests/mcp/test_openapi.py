from __future__ import annotations

import pytest

from aiready.core.errors import ConfigError
from aiready.mcp.openapi import tools_from_openapi

from mcp_fixtures import FACADE_SPEC


def _by_name(tools):
    return {t.definition.name: t for t in tools}


def test_one_tool_per_operation_with_effects_from_method():
    tools = _by_name(tools_from_openapi(FACADE_SPEC))
    assert set(tools) == {"getClaim", "updateClaim", "deleteClaim", "searchClaims", "payClaim", "adminReindex"}
    assert tools["getClaim"].definition.effect == "read"
    assert tools["searchClaims"].definition.effect == "read"
    assert tools["updateClaim"].definition.effect == "write"
    assert tools["payClaim"].definition.effect == "write"
    assert tools["deleteClaim"].definition.effect == "delete"
    assert tools["getClaim"].binding.base_url == "https://facade.test/api"
    assert tools["getClaim"].definition.description == "Read one claim"


def test_input_schema_merges_params_and_body_with_refs_resolved():
    tools = _by_name(tools_from_openapi(FACADE_SPEC))
    schema = tools["updateClaim"].definition.input_schema
    assert set(schema["properties"]) == {"claim_id", "status", "note"}
    assert schema["required"] == ["claim_id", "status"]
    assert schema["properties"]["note"]["type"] == ["string", "null"]  # OpenAPI nullable -> JSON schema
    assert schema["additionalProperties"] is False
    locs = {p.name: p.location for p in tools["updateClaim"].binding.params}
    assert locs == {"claim_id": "path", "status": "body_field", "note": "body_field"}
    q = tools["searchClaims"].definition.input_schema
    assert set(q["properties"]) == {"q", "limit"} and "required" not in q


def test_include_exclude_and_overrides():
    tools = _by_name(tools_from_openapi(
        FACADE_SPEC, include=["tag:claims", "payClaim"], exclude=["DELETE /claims/*"], prefix="crm_",
        overrides={"crm_payClaim": {"effect": "payment", "required_roles": ["finance"]},
                   "searchClaims": {"enabled": False}}))
    assert set(tools) == {"crm_getClaim", "crm_updateClaim", "crm_payClaim"}
    assert tools["crm_payClaim"].definition.effect == "payment"
    assert tools["crm_payClaim"].definition.required_roles == ["finance"]


def test_vendor_extension_sets_effect():
    spec = {"openapi": "3.1.0", "paths": {"/refund": {"get": {"operationId": "refund", "x-aiready-effect": "payment"}}}}
    (tool,) = tools_from_openapi(spec, base_url="http://x")
    assert tool.definition.effect == "payment"


def test_rejects_non_openapi_and_bad_override():
    with pytest.raises(ConfigError):
        tools_from_openapi({"swagger": "2.0"})
    with pytest.raises(Exception):
        tools_from_openapi(FACADE_SPEC, overrides={"getClaim": {"effect": "explode"}})
