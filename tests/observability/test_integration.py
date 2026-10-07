"""Contract check against the real gateway's spans (skipped when the gateway package is not installed): what
the gateway emits must show up as cost and latency in the observability reports."""

import copy
from pathlib import Path

import pytest
import yaml

from aiready import testing
from aiready.core import config, wiring
from aiready.core.config import _substitute, deep_merge
from aiready.core.telemetry import recorder
from aiready.core.types import ChatMessage, ChatRequest
from aiready.evals.golden import EvalCase
from aiready.evals.targets import AgentsTarget
from aiready.observability.config import ObservabilityConfig
from aiready.observability.scoring import AnswerSample
from aiready.observability.service import ObservabilityService

REPO = Path(__file__).resolve().parents[2]


def gateway_tree():
    tree = {}
    for name in ("models.yaml", "gateway.yaml"):
        p = REPO / "config" / name
        if not p.exists():
            pytest.skip(f"{name} not shipped")
        tree = deep_merge(tree, yaml.safe_load(p.read_text()))
    tree = _substitute(tree)
    tree["state"] = {"url": "sqlite:///:memory:"}
    tree.setdefault("gateway", {}).setdefault("admin", {})["sync_seconds"] = 0
    return tree


async def test_real_gateway_spans_feed_cost_and_latency(principal):
    gw_mod = pytest.importorskip("aiready.gateway.service")
    config.set_config(config.ConfigStore(tree=copy.deepcopy(gateway_tree())))
    wiring.override("policy", testing.AllowAllPolicy())
    wiring.override("guardrails", testing.PassThroughGuardrails())
    gw = gw_mod.GatewayService()
    recorder.buffer.clear()
    resp = await gw.chat(ChatRequest(messages=[ChatMessage(role="user", content="Summarise the claims process")],
                                     metadata={"feature": "claims-qa"}), principal)
    spans = recorder.trace(resp.aiready["trace_id"])
    assert spans, "gateway emitted no spans"
    svc = ObservabilityService(ObservabilityConfig.model_validate({"store": {"type": "memory"},
                                                                   "masking": {"use_guardrails": False}}))
    await svc.ingest(spans)
    admin = testing.make_principal("root", roles=["admin"])
    rows = await svc.usage(admin, by=["feature", "model"])
    assert rows and rows[0].requests >= 1 and rows[0].key["model"] != "-"
    tree = await svc.get_trace(resp.aiready["trace_id"], principal)  # alice sees her own request
    assert tree.summary.user == "alice"


async def test_submit_answer_path_scores_and_masks():
    svc = ObservabilityService(ObservabilityConfig.model_validate({"store": {"type": "memory"},
                                                                   "masking": {"use_guardrails": False},
                                                                   "scoring": {"sample_rate": 0.0}}))
    sample = AnswerSample(trace_id="t-ans", span="rag.answer", ts=0, question="Mail me at a@b.com",
                          answer="Sure, a@b.com noted.", app="claims", feature="qa")
    assert await svc.submit_answer(sample) is None  # not sampled
    rec = await svc.submit_answer(sample, force=True)
    assert rec is not None and "pii_leak" in rec.failed and rec.enqueued
    item = svc.review.list()[0]
    assert "a@b.com" not in item.answer and "a@b.com" not in item.question


async def test_agents_target_reads_run_view(principal):
    class Agents:
        async def start_run(self, workflow, input, principal):
            return {"id": "r1", "status": "completed", "result": {"answer": f"done: {input['question']}"},
                    "trace_id": "tr1"}

    wiring.override("agents", Agents())
    out = await AgentsTarget(workflow="claims").run(EvalCase(id="a", question="q?", must_include=["done"]), principal)
    assert out.answer == "done: q?" and out.trace_id == "tr1"
