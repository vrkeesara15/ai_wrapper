"""Release-gate verdicts become registry evidence (evals.gate.record_gate -> registry.record_evaluation)."""

from aiready.core.wiring import connect
from aiready.evals.gate import GateCheck, GateResult, record_gate
from aiready.testing import make_principal


async def test_gate_verdict_lands_on_model_record():
    gov = connect("policy")
    admin = make_principal("rita", roles=["model-approver", "ai-engineer", "admin"])
    gov.registry.register_model({"name": "chat-x", "version": "2", "licence": "apache-2.0",
                                 "context_length": 32768, "weights_checksum": "sha256:ab"}, admin)
    result = GateResult(release="1.4.0", passed=False, checks=[
        GateCheck(suite="claims", name="grounding", passed=False, actual=0.71, expected=0.8)])
    out = await record_gate(result, principal=admin, artifact={"kind": "model", "name": "chat-x", "version": "2"})
    assert out.registry_recorded
    rec = gov.registry.get_model("chat-x", "2")
    assert rec.eval_scores["claims.grounding"] == 0.71
    assert rec.history[-1]["event"] == "evaluated" and rec.history[-1]["passed"] is False
    assert gov.registry.evaluations("1.4.0")[0]["passed"] is False
