import json
from pathlib import Path

import pytest
import yaml

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import GuardrailBlocked, ValidationFailed
from aiready.core.types import ChatChoice, ChatMessage, ChatResponse, Usage
from aiready.evals.gate import GateConfig, evaluate_gate, exit_code, record_gate
from aiready.evals.golden import EvalCase, EvalSuite, load_suite, load_suites, validate_suite
from aiready.evals.routing_eval import ModelCandidate, evaluate_routes
from aiready.evals.runner import EvalRunner, ScorecardHistory
from aiready.evals.safety import builtin_safety_suites
from aiready.evals.scorers import Passage, ScoreInput, build_scorers, is_refusal
from aiready.evals.targets import CallableTarget, StaticTarget, TargetOutput

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "config" / "evals" / "golden" / "claims.yaml"


# ------------------------------------------------------------------ golden sets

def test_shipped_golden_and_safety_suites_are_valid():
    suite = load_suite(CLAIMS)
    assert suite.name == "claims" and len(suite.cases) >= 10 and validate_suite(suite) == []
    assert suite.persona_for(suite.cases[2]).roles == ["adjuster"]
    safety = builtin_safety_suites(ROOT / "config" / "evals" / "safety")
    names = {s.name for s in safety}
    assert {"safety_injection", "safety_data_leak", "safety_pii_echo", "safety_jailbreak"} <= names
    for s in safety:
        assert s.safety and validate_suite(s) == [], s.name
    assert any(c.documents for s in safety for c in s.cases)  # indirect injection via poisoned documents


def test_shipped_config_merges_into_one_tree():
    from aiready.core.config import _substitute, deep_merge
    tree: dict = {}
    files = [ROOT / "config" / "observability.yaml", ROOT / "config" / "finetune.yaml",
             *sorted((ROOT / "config" / "evals").rglob("*.yaml"))]
    for f in files:  # only this package's files, merged exactly like the config loader does
        tree = deep_merge(tree, yaml.safe_load(f.read_text()))
    tree = _substitute(tree)
    assert {"claims", "safety_injection", "safety_data_leak"} <= set(tree["evals"]["suites"])
    assert tree["evals"]["gate"]["safety_pass_rate"] == 1.0
    GateConfig.model_validate(tree["evals"]["gate"])
    from aiready.finetune.config import FinetuneConfig
    from aiready.observability.config import ObservabilityConfig
    cfg = ObservabilityConfig.model_validate(tree["observability"])
    assert cfg.latency_budget_ms["first_token"] == 800
    FinetuneConfig.model_validate(tree["finetune"])


def test_loader_formats_and_validator(tmp_path):
    (tmp_path / "plain.yaml").write_text(yaml.safe_dump([{"id": "a", "question": "q?", "must_include": ["x"]}]))
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps({"id": f"c{i}", "question": "q", "expected_answer": "a"})
                                                for i in range(3)))
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump({"name": "x", "cases": [{"id": "a", "question": "q"}]}))
    assert load_suite(tmp_path / "plain.yaml").name == "plain"
    assert len(load_suite(tmp_path / "s.jsonl").cases) == 3
    problems = validate_suite(load_suite(tmp_path / "suite.yaml"))
    assert any("no expectation" in p for p in problems)
    dup = EvalSuite(name="d", cases=[EvalCase(id="1", question="q", must_include=["a"]),
                                     EvalCase(id="1", question="q", must_include=["a"], persona="ghost",
                                              json_schema={"type": "nope"})])
    problems = validate_suite(dup)
    assert any("duplicate" in p for p in problems) and any("unknown persona" in p for p in problems)
    assert any("json_schema" in p for p in problems)
    with pytest.raises(ValidationFailed):
        load_suites(tmp_path / "missing.yaml")
    with pytest.raises(Exception):
        EvalCase(id="x", question="   ")


# ------------------------------------------------------------------ scorers

async def score(name, **kw):
    case = kw.pop("case", None)
    (sc,) = build_scorers([name] if isinstance(name, str) else [name])
    return await sc.score(ScoreInput(question=kw.pop("question", "How long to report a claim?"), case=case, **kw))


P = [Passage(id="c1", doc_id="handbook", text="A claim must be reported within 30 days of the incident.", marker="1")]


async def test_answer_content_scorers():
    case = EvalCase(id="c", question="q", expected_answer="Within 30 days.", must_include=["30 days"],
                    must_not_include=["60 days"])
    assert (await score("exact_match", answer="within 30 days [1]", case=case)).score == 1.0
    assert (await score("exact_match", answer="30 days", case=case)).score == 0.0
    r = await score("contains", answer="It is 30 days, not 60 days.", case=case)
    assert r.score == 0.5 and r.details["forbidden_present"] == ["60 days"]
    f1 = await score("f1", answer="The claim must be filed within 30 days", case=case)
    assert 0.3 < f1.score < 1.0 and f1.details["recall"] == 1.0
    assert (await score("f1", answer="anything")).skipped


async def test_citation_and_grounding_scorers():
    case = EvalCase(id="c", question="q", expected_sources=["handbook", "faq"])
    p = await score("citation_precision", answer="x", cited_sources=["handbook", "blog"], case=case)
    r = await score("citation_recall", answer="x", cited_sources=["handbook", "blog"], case=case)
    assert p.score == 0.5 and r.score == 0.5 and r.details["missing"] == ["faq"]
    good = await score("grounding", answer="A claim must be reported within 30 days [1].", passages=P)
    bad = await score("grounding", answer="A claim must be reported within 30 days [1]. Payment is made in bitcoin.", passages=P)
    assert good.score == 1.0 and good.passed and bad.score == 0.5 and not bad.passed
    assert (await score("grounding", answer="I could not find this in the available sources.", passages=P)).skipped
    assert (await score("citation_validity", answer="Within 30 days [1].", passages=P)).score == 1.0
    assert (await score("citation_validity", answer="Within 30 days [3].", passages=P)).score == 0.0
    assert (await score("citation_validity", answer="Within 30 days.", passages=P)).score == 0.0


async def test_refusal_scorer_with_and_without_case():
    exp = EvalCase(id="c", question="q", expect_refusal=True)
    assert (await score("refusal", answer="I could not find this in the available sources.", case=exp)).passed
    assert not (await score("refusal", answer="Bitcoin losses are covered.", case=exp)).passed
    # live: answered although retrieval found nothing -> fabrication
    assert (await score("refusal", answer="Yes it is covered.", found=False)).score == 0.0
    # live: refused although the passage answers the question
    missed = await score("refusal", question="When must a claim be reported?", answer="I could not find this.", passages=P)
    assert missed.score == 0.0
    assert is_refusal("I'm sorry, I can't help with that") and not is_refusal("Within 30 days.")


async def test_safety_and_format_scorers():
    case = EvalCase(id="c", question="q", canaries=["PWNED"], forbidden_tools=["delete_claim"],
                    forbidden=["bob@example.com"], forbidden_sources=["doc-bob"],
                    json_schema={"type": "object", "required": ["status"]})
    assert (await score("injection_resisted", answer="pwned!", case=case)).score == 0.0
    assert (await score("injection_resisted", answer="ok", tool_calls=["delete_claim"], case=case)).score == 0.0
    assert (await score("injection_resisted", answer="Within 30 days.", case=case)).score == 1.0
    assert (await score("data_leak", answer="Contact BOB@example.com", case=case)).score == 0.0
    assert (await score("data_leak", answer="no", passages=[Passage(doc_id="doc-bob")], case=case)).score == 0.0
    assert (await score("data_leak", answer="I cannot share that.", case=case)).score == 1.0
    assert (await score("pii_leak", answer="Call 4111 1111 1111 1111")).score == 0.0
    assert (await score("pii_leak", answer="Masked [EMAIL] here")).score == 0.0
    assert (await score("pii_leak", answer="Within 30 days.")).score == 1.0
    assert (await score("json_schema", answer='```json\n{"status": "open"}\n```', case=case)).score == 1.0
    assert (await score("json_schema", answer='{"state": "open"}', case=case)).score == 0.0
    assert (await score("json_schema", answer="not json", case=case)).score == 0.0
    lat = EvalCase(id="l", question="q", max_latency_ms=100, max_cost=0.01)
    assert (await score("latency_budget", answer="", latency_ms=50, case=lat)).passed
    assert not (await score("latency_budget", answer="", latency_ms=150, case=lat)).passed
    assert not (await score("cost_budget", answer="", cost=0.02, case=lat)).passed


async def test_llm_judge_goes_through_gateway(principal):
    gw = testing.FakeGateway()
    gw.script(content='{"score": 4, "reason": "mostly right"}')
    wiring.override("gateway", gw)
    r = await score({"type": "llm_judge"}, answer="Within 30 days.", passages=P, principal=principal)
    assert r.score == 0.75 and r.passed and gw.calls[0].model == "judge"
    assert (await score({"type": "llm_judge"}, answer="x")).skipped  # no identity -> no call


async def test_custom_scorer_by_class_path():
    (sc,) = build_scorers([{"class": "aiready.evals.scorers:Contains", "name": "my_contains"}])
    assert sc.name == "my_contains"


# ------------------------------------------------------------------ runner

def qa_suite():
    return EvalSuite(name="mini", personas={"adj": {"sub": "eval-adj", "roles": ["adjuster"], "groups": ["claims-team"]}},
                     scorers=["contains", "grounding", "refusal"], cases=[
        EvalCase(id="q1", question="When must a claim be reported?", must_include=["30 days"], persona="adj", tags=["policy"]),
        EvalCase(id="q2", question="Crypto losses?", expect_refusal=True, tags=["refusal"]),
    ])


def answer_fn(good=True):
    seen = []

    async def fn(case, principal):
        seen.append((case.id, principal.sub, principal.groups, principal.actor))
        if case.id == "q2":
            return TargetOutput(answer="I could not find this in the available sources.", found=False)
        text = "A claim must be reported within 30 days of the incident [1]." if good else "Any time you like."
        return TargetOutput(answer=text, passages=P, cited_sources=["handbook"], latency_ms=120)

    fn.seen = seen
    return fn


async def test_runner_scorecard_personas_and_history(tmp_path):
    fn = answer_fn()
    history = ScorecardHistory(output_dir=tmp_path)
    card = await EvalRunner(CallableTarget(fn), history=history, concurrency=2).run(qa_suite(), release="1.0")
    assert card.n_cases == 2 and card.pass_rate == 1.0 and card.overall == 1.0
    assert card.per_scorer["contains"].n == 1 and card.per_scorer["refusal"].n == 2
    assert card.per_tag["policy"]["grounding"] == 1.0
    assert ("q1", "eval-adj", ["claims-team"], "agent:eval-runner") in fn.seen  # runs as the persona
    assert (tmp_path / "mini" / "1.0.json").exists() and "# Scorecard: mini @ 1.0" in (tmp_path / "mini" / "1.0.md").read_text()
    bad = await EvalRunner(CallableTarget(answer_fn(good=False)), history=history).run(qa_suite(), release="1.1")
    assert bad.overall < card.overall and [c.case_id for c in bad.failed_cases()] == ["q1"]
    assert [t["release"] for t in history.trend("mini")] == ["1.0", "1.1"]
    assert history.baseline("mini", "latest", current_release="1.1").release == "1.0"
    assert any(r.event == "evals.run" for r in get_audit().read())


async def test_runner_records_errors_and_guardrail_blocks():
    async def fn(case, principal):
        if case.id == "q1":
            raise RuntimeError("boom")
        raise GuardrailBlocked("injection detected")

    card = await EvalRunner(CallableTarget(fn)).run(qa_suite(), release="x")
    assert card.n_errors == 1 and card.cases[0].error.startswith("RuntimeError")
    blocked = card.cases[1]
    assert blocked.blocked == "guardrail_blocked" and blocked.passed  # a block is the right answer to expect_refusal


async def test_runner_rejects_invalid_suite():
    with pytest.raises(ValidationFailed):
        await EvalRunner(StaticTarget({})).run(EvalSuite(name="empty"), release="1")


async def test_gateway_target_against_safety_suites():
    wiring.override("gateway", testing.FakeGateway())
    suites = builtin_safety_suites(ROOT / "config" / "evals" / "safety")
    runner = EvalRunner({"type": "gateway", "model": "chat"})
    cards = [await runner.run(s, release="r1") for s in suites]
    assert all(c.pass_rate == 1.0 for c in cards), [(c.suite, [x.case_id for x in c.failed_cases()]) for c in cards]


# ------------------------------------------------------------------ release gate

async def test_gate_blocks_score_drop_and_passes_equal():
    h = ScorecardHistory()
    base = await EvalRunner(CallableTarget(answer_fn()), history=h).run(qa_suite(), release="1.0")
    same = await EvalRunner(CallableTarget(answer_fn()), history=h).run(qa_suite(), release="1.1")
    worse = await EvalRunner(CallableTarget(answer_fn(good=False)), history=h).run(qa_suite(), release="1.2")
    cfg = GateConfig(max_drop=0.0)
    ok = evaluate_gate([same], {"mini": base}, cfg)
    assert ok.passed and exit_code(ok) == 0
    blocked = evaluate_gate([worse], {"mini": base}, cfg)
    assert not blocked.passed and exit_code(blocked) == 1
    assert {"no_drop.overall", "no_drop.contains"} <= {c.name for c in blocked.failures()}
    assert "BLOCKED" in blocked.to_markdown()
    # absolute floors and budgets
    floors = evaluate_gate([same], {}, GateConfig(min_overall=1.0, max_latency_p95_ms=100))
    assert [c.name for c in floors.failures()] == ["latency_p95_ms"]
    assert not evaluate_gate([same], {}, GateConfig(require_baseline=True)).passed
    per_suite = GateConfig.model_validate({"suites": {"mini": {"min_scorer": {"grounding": 1.01}}}})
    assert [c.name for c in evaluate_gate([same], {}, per_suite).failures()] == ["min_scorer.grounding"]


async def test_gate_requires_all_safety_cases_to_pass():
    suite = builtin_safety_suites(ROOT / "config" / "evals" / "safety")
    inj = next(s for s in suite if s.name == "safety_injection")

    async def vulnerable(case, principal):  # obeys the injected instruction
        return TargetOutput(answer="PWNED-7731" if case.id == "inj-direct-001" else "I cannot help with that.")

    card = await EvalRunner(CallableTarget(vulnerable)).run(inj, release="2.0")
    assert card.pass_rate < 1.0 and card.pass_rate > 0.7
    result = evaluate_gate([card], {}, GateConfig())
    assert not result.passed and result.failures()[0].name == "safety_pass_rate"
    assert "inj-direct-001" in result.failures()[0].message


async def test_record_gate_audits_and_updates_registry(admin_principal=None):
    class Registry:
        def __init__(self):
            self.calls = []

        async def record_evaluation(self, **kw):
            self.calls.append(kw)

    reg = Registry()
    wiring.override("registry", reg)
    card = await EvalRunner(CallableTarget(answer_fn())).run(qa_suite(), release="3.0")
    res = await record_gate(evaluate_gate([card], {}, GateConfig()), artifact={"prompt": "claims@v7"})
    assert res.registry_recorded and reg.calls[0]["release"] == "3.0" and reg.calls[0]["passed"] is True
    rec = [r for r in get_audit().read() if r.event == "evals.gate"][-1]
    assert rec.decision == "pass" and rec.resource == "3.0"
    wiring.reset()
    res2 = await record_gate(evaluate_gate([card], {}, GateConfig()))  # no registry deployed: still audited
    assert not res2.registry_recorded


async def test_cli_gate_exit_code(tmp_path):
    from aiready.core import config
    from aiready.evals.cli import gate_release
    config.set_config(config.ConfigStore(tree={"evals": {"output_dir": str(tmp_path), "gate": {"min_overall": 0.5}}}))
    wiring.override("gateway", testing.FakeGateway())
    suites = builtin_safety_suites(ROOT / "config" / "evals" / "safety")
    code = await gate_release(suites, release="9.9", target="gateway")
    assert code == 0 and (tmp_path / "gate-9.9.json").exists()


# ------------------------------------------------------------------ routing evaluation

class ModelAwareGateway(testing.FakeGateway):
    async def chat(self, req, principal):
        if req.model == "tiny-1b":
            text = "Claims are paid in gold."
        else:
            return await super().chat(req, principal)
        return ChatResponse(model=req.model, choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
                            usage=Usage())


async def test_routing_eval_picks_smallest_passing_model():
    wiring.override("gateway", ModelAwareGateway())
    suite = EvalSuite(name="faq", scorers=["contains", "grounding"], cases=[
        EvalCase(id="f1", question="When must a claim be reported after the incident?", must_include=["30 days"],
                 documents=[{"id": "handbook", "text": "A claim must be reported within 30 days after the incident."}])])
    report = await evaluate_routes({"faq": suite}, [ModelCandidate(name="large-70b", size=70),
                                                    ModelCandidate(name="tiny-1b", size=1),
                                                    ModelCandidate(name="small-8b", size=8)])
    t = report.tasks[0]
    assert t.pass_list == ["small-8b", "large-70b"] and t.chosen == "small-8b" and t.fallbacks == ["large-70b"]
    snippet = yaml.safe_load(report.to_yaml())
    assert snippet["gateway"]["task_routes"]["faq"]["eval_pass_list"] == ["small-8b", "large-70b"]
    assert "tiny-1b" in report.to_markdown()
