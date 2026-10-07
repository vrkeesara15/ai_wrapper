import time

import pytest
from fastapi.testclient import TestClient

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import NotFound, PermissionDenied, PolicyViolation
from aiready.core.telemetry import SpanRecord
from aiready.observability.alerts import MemoryNotifier
from aiready.observability.api import create_app
from aiready.observability.config import AlertRuleConfig, ModelPrice, ObservabilityConfig
from aiready.observability.feedback import FeedbackIn
from aiready.observability.masking import RegexMasker
from aiready.observability.review import ReviewLabel
from aiready.observability.scoring import sampled
from aiready.observability.service import ObservabilityService
from aiready.observability.usage import percentile


def mk(trace="t1", sid="s1", name="gateway.chat", parent=None, start=None, dur=0.1, status="ok", **attrs):
    start = start if start is not None else time.time() - 60
    return SpanRecord(trace_id=trace, span_id=sid, parent_id=parent, name=name, service="test", start=start,
                      end=start + dur, status=status, attributes=attrs)


def request_trace(trace="t1", user="alice", feature="claims-qa", team=None, cost=None, model="chat-small",
                  pt=100, ct=50, start=None, answer=None, passages=None, status="ok"):
    start = start if start is not None else time.time() - 60
    root_attrs = {"user": user, "app": "claims", "feature": feature, "tenant": "acme"}
    if answer is not None:
        root_attrs.update(answer=answer, question="How long to report a claim?", passages=passages or [], found=True)
    spans = [mk(trace, f"{trace}-root", "rag.answer", start=start, dur=1.2, **root_attrs),
             mk(trace, f"{trace}-scr", "rag.stage.screen", parent=f"{trace}-root", start=start, dur=0.04),
             mk(trace, f"{trace}-gw", "gateway.chat", parent=f"{trace}-root", start=start + 0.3, dur=0.8, status=status,
                model=model, prompt_tokens=pt, completion_tokens=ct, ttft_ms=650,
                request={"model": "auto", "messages": [{"role": "user", "content": "How long to report a claim?"}]},
                **({"cost": cost} if cost is not None else {}), **({"team": team} if team else {}))]
    return spans


@pytest.fixture
def cfg():
    return ObservabilityConfig.model_validate({
        "store": {"type": "memory"},
        "masking": {"use_guardrails": False},
        "pricing": {"chat-small": {"input_per_1k": 1.0, "output_per_1k": 2.0}},
        "teams": {"claims": "claims-team"},
        "budgets": {"team:claims-team": 10.0},
        "scoring": {"sample_rate": 1.0, "low_score_threshold": 0.6},
    })


@pytest.fixture
def svc(cfg):
    wiring.override("policy", testing.AllowAllPolicy())
    return ObservabilityService(cfg)


@pytest.fixture
def admin():
    return testing.make_principal("root", roles=["admin"])


# ------------------------------------------------------------------ masking and ingest

def test_regex_masker_detects_common_pii():
    m = RegexMasker()
    text = "Mail jane.doe@example.com, card 4111 1111 1111 1111, phone +44 20 7946 0958, claim CLM-2024-0012"
    out = m.mask(text)
    assert "jane.doe" not in out and "4111" not in out and "7946" not in out
    assert "[EMAIL]" in out and "[CREDIT_CARD]" in out and "[PHONE]" in out
    assert "CLM-2024-0012" in out  # business identifiers are not personal data
    assert "[CREDIT_CARD]" not in m.mask("ref 1234 5678 9012 3456")  # fails Luhn: not a card


async def test_ingest_masks_pii_before_storage(svc):
    s = mk(question="my email is bob@corp.example and card 4111111111111111",
           request={"messages": [{"role": "user", "content": "call me on 020 7946 0958"}]}, model="chat-small")
    s.events.append({"name": "retrieved", "text": "SSN 123-45-6789"})
    res = await svc.ingest([s])
    assert res["accepted"] == 1
    stored = svc.store.trace("t1")[0]
    blob = stored.model_dump_json()
    for raw in ("bob@corp.example", "4111111111111111", "7946 0958", "123-45-6789"):
        assert raw not in blob
    assert stored.attributes["model"] == "chat-small"  # safe keys untouched


async def test_ingest_uses_guardrails_mask_when_available(cfg):
    cfg.masking.use_guardrails = True

    class Guard(testing.PassThroughGuardrails):
        async def mask(self, text):
            from aiready.core.interfaces import ScreenResult
            return ScreenResult(allowed=True, text=text.replace("Jane Doe", "[PERSON]"))

    wiring.override("guardrails", Guard())
    svc = ObservabilityService(cfg)
    await svc.ingest([mk(question="Jane Doe lives at jane@x.org")])
    q = svc.store.trace("t1")[0].attributes["question"]
    assert q == "[PERSON] lives at [EMAIL]"


def test_span_endpoint_requires_service_identity(cfg, token_service):
    svc = ObservabilityService(cfg)
    client = TestClient(create_app(svc))
    body = [mk().model_dump()]
    assert client.post("/v1/spans", json=body).status_code == 401
    tok = token_service.issue_service_token("gateway")
    r = client.post("/v1/spans", json=body, headers={"X-Service-Token": tok})
    assert r.status_code == 200 and r.json()["accepted"] == 1
    cfg.ingest.allow_anonymous = True
    assert client.post("/v1/spans", json=body).status_code == 200


# ------------------------------------------------------------------ trace viewer and replay

async def test_trace_tree_and_access(svc, admin):
    await svc.ingest(request_trace("t1", user="alice"))
    alice = testing.make_principal("alice")
    tree = await svc.get_trace("t1", alice)
    assert len(tree.roots) == 1 and tree.roots[0].span.name == "rag.answer"
    assert {c.span.name for c in tree.roots[0].children} == {"rag.stage.screen", "gateway.chat"}
    assert tree.summary.user == "alice" and tree.summary.feature == "claims-qa"
    with pytest.raises(NotFound):
        await svc.get_trace("t1", testing.make_principal("mallory"))
    assert (await svc.get_trace("t1", admin)).trace_id == "t1"
    other_tenant_admin = testing.make_principal("root2", roles=["admin"], tenant="globex")
    with pytest.raises(NotFound):
        await svc.get_trace("t1", other_tenant_admin)
    # search: users only see their own
    assert [t.trace_id for t in await svc.search_traces(alice, feature="claims-qa")] == ["t1"]
    assert await svc.search_traces(testing.make_principal("bob")) == []
    with pytest.raises(PermissionDenied):
        await svc.search_traces(alice, user="bob")


async def test_replay_reissues_through_gateway_as_admin(svc, admin):
    gw = testing.FakeGateway()
    wiring.override("gateway", gw)
    await svc.ingest(request_trace("t1"))
    out = await svc.replay("t1", admin)
    assert out["original_trace_id"] == "t1" and out["replay_trace_id"]
    assert gw.calls[0].metadata["replay_of"] == "t1" and gw.calls[0].metadata["replayed_by"] == "root"
    rec = [r for r in get_audit().read() if r.event == "observability.replay"][-1]
    assert rec.actor == "root" and rec.resource == "t1" and rec.decision == "replayed"
    assert (await svc.get_trace("t1", admin)).replays == [out["replay_trace_id"]]
    with pytest.raises(PermissionDenied):
        await svc.replay("t1", testing.make_principal("alice"))


async def test_replay_denied_by_policy(svc, admin):
    wiring.override("policy", testing.DenyPolicy({"trace.replay"}))
    wiring.override("gateway", testing.FakeGateway())
    await svc.ingest(request_trace("t1"))
    with pytest.raises(PolicyViolation):
        await svc.replay("t1", admin)
    assert any(r.event == "observability.policy_denied" for r in get_audit().read())


# ------------------------------------------------------------------ cost, chargeback, latency

async def test_cost_aggregation_and_chargeback(svc, admin):
    await svc.ingest(request_trace("t1", pt=1000, ct=500))  # 1.0 + 1.0 = 2.0 from pricing, team from mapping
    await svc.ingest(request_trace("t2", pt=1000, ct=0, feature="claims-summary"))  # 1.0
    await svc.ingest(request_trace("t3", cost=5.0, team="finance-team", user="bob"))  # recorded cost wins
    rows = await svc.usage(admin, by=["team"])
    by_team = {r.key["team"]: r for r in rows}
    assert by_team["claims-team"].cost == pytest.approx(3.0) and by_team["claims-team"].requests == 2
    assert by_team["finance-team"].cost == pytest.approx(5.0)
    feat = {r.key["feature"]: r.cost for r in await svc.usage(admin, by=["feature"])}
    assert feat["claims-summary"] == pytest.approx(1.0)
    per_user_day = await svc.usage(admin, by=["user", "day"])
    assert {r.key["user"] for r in per_user_day} == {"alice", "bob"}

    report = await svc.chargeback(admin)
    assert report.total_cost == pytest.approx(8.0)
    line = next(ln for ln in report.lines if ln.team == "claims-team")
    assert line.share == pytest.approx(3 / 8) and line.budget_used == pytest.approx(0.3)
    csv = report.to_csv().splitlines()
    assert csv[0].startswith("period,team,requests") and len(csv) == 3
    with pytest.raises(PermissionDenied):
        await svc.usage(testing.make_principal("alice"), by=["team"])


async def test_latency_against_budget(svc, admin):
    await svc.ingest(request_trace("t1"))
    rows = {r.stage: r for r in await svc.latency(admin)}
    assert rows["screen"].budget_ms == 50 and rows["screen"].within_budget is True
    assert rows["first_token"].p95_ms == 650 and rows["first_token"].within_budget is True
    await svc.ingest([mk("t9", "x", "rag.stage.screen", dur=0.2)])
    rows = {r.stage: r for r in await svc.latency(admin)}
    assert rows["screen"].within_budget is False and rows["screen"].over_budget_share == 0.5


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 95) == 0.0


# ------------------------------------------------------------------ live scoring, review, feedback

PASSAGES = [{"id": "c1", "doc_id": "policy", "text": "Customers must report a claim within 30 days of the incident."}]


async def test_live_scorer_sampling_and_low_score_enqueues_review(svc, admin):
    await svc.ingest(request_trace("good", answer="Customers must report a claim within 30 days [1].", passages=PASSAGES))
    await svc.ingest(request_trace("bad", answer="Claims can be reported any time within five years, call 555-123-4567.",
                                   passages=PASSAGES))
    recs = {r.trace_id: r for r in svc.scorer.records()}
    assert recs["good"].overall >= 0.9 and not recs["good"].enqueued
    assert recs["bad"].enqueued and "pii_leak" in recs["bad"].failed and "grounding" in recs["bad"].failed
    items = await svc.review_list(admin)
    assert [i.trace_id for i in items] == ["bad"] and items[0].source == "live_score"
    assert "555-123-4567" not in (items[0].answer or "")  # masked before it reached the queue
    trend = await svc.quality(admin)
    assert trend and trend[0]["samples"] == 2


def test_sampling_is_deterministic_and_proportional():
    ids = [f"trace-{i}" for i in range(4000)]
    picked = [t for t in ids if sampled(t, 0.1)]
    assert 300 < len(picked) < 500
    assert picked == [t for t in ids if sampled(t, 0.1)]
    assert not sampled("x", 0.0) and sampled("x", 1.0)


async def test_unsampled_traces_are_not_scored(cfg):
    cfg.scoring.sample_rate = 0.0
    svc = ObservabilityService(cfg)
    await svc.ingest(request_trace("t1", answer="whatever", passages=PASSAGES))
    assert svc.scorer.records() == []


async def test_feedback_review_label_and_golden_export(svc, admin):
    await svc.ingest(request_trace("t1", answer="Within 60 days [1].", passages=PASSAGES))
    alice = testing.make_principal("alice")
    fb = await svc.record_feedback(FeedbackIn(trace_id="t1", thumbs="down", edit="Within 30 days.",
                                              comment="wrong, mail me at a@b.com"), alice)
    assert fb.negative and "[EMAIL]" in fb.comment
    with pytest.raises(PermissionDenied):
        await svc.record_feedback(FeedbackIn(trace_id="t1", rating=1), testing.make_principal("bob"))
    items = await svc.review_list(admin, source="feedback")
    assert len(items) == 1 and items[0].expected_answer == "Within 30 days."
    reviewer = testing.make_principal("rita", roles=["reviewer"])
    from aiready.core.errors import ValidationFailed
    with pytest.raises(ValidationFailed):
        await svc.review_label(items[0].id, ReviewLabel(cause="gremlins"), reviewer)
    await svc.review_label(items[0].id, ReviewLabel(cause="retrieval", must_include=["30 days"],
                                                    expected_sources=["policy"]), reviewer)
    cases = await svc.golden_candidates(reviewer, mark_exported=True)
    assert len(cases) == 1 and cases[0]["must_include"] == ["30 days"] and "cause:retrieval" in cases[0]["tags"]
    from aiready.evals.golden import EvalCase
    EvalCase.model_validate(cases[0])  # valid golden-set case
    assert await svc.golden_candidates(reviewer) == []  # exported once
    assert svc.review.cause_summary() == {"retrieval": 1}
    with pytest.raises(PermissionDenied):
        await svc.review_list(alice)


# ------------------------------------------------------------------ alerts

async def test_alert_fires_once_and_resolves(cfg):
    cfg.alerts.rules = [AlertRuleConfig(name="quality-low", kind="quality", threshold=0.8, window_minutes=60),
                        AlertRuleConfig(name="spend", kind="spend", scope={"team": "claims-team"}),
                        AlertRuleConfig(name="ttft", kind="latency", scope={"stage": "first_token"}, threshold=500)]
    notifier = MemoryNotifier()
    svc = ObservabilityService(cfg, notifiers={"mem": notifier})
    await svc.ingest(request_trace("bad", answer="Something unrelated entirely here.", passages=PASSAGES, pt=6000, ct=3000))
    events = await svc.evaluate_alerts()
    assert {e.rule for e in events} == {"quality-low", "spend", "ttft"}  # spend 12 > budget 10, ttft 650 > 500
    assert all(e.state == "firing" for e in events) and len(notifier.events) == 3
    assert await svc.evaluate_alerts() == []  # no re-page while still firing
    for i in range(10):
        await svc.ingest(request_trace(f"g{i}", answer="Customers must report a claim within 30 days [1].", passages=PASSAGES, pt=0, ct=0))
    resolved = await svc.evaluate_alerts()
    assert [e.rule for e in resolved if e.state == "resolved"] == ["quality-low"]
    assert any(r.event == "observability.alert" for r in get_audit().read())


async def test_webhook_notifier_posts(monkeypatch):
    import httpx

    from aiready.observability.alerts import AlertEvent, SlackNotifier, WebhookNotifier
    sent = []
    real = httpx.AsyncClient

    def fake_client(*a, **kw):
        return real(transport=httpx.MockTransport(lambda req: sent.append(req) or httpx.Response(200)))

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    ev = AlertEvent(rule="r", kind="quality", state="firing", severity="warning", value=0.5, threshold=0.7, message="m")
    await WebhookNotifier("http://hook.local/x").send(ev)
    await SlackNotifier("http://slack.local/y").send(ev)
    assert [str(r.url) for r in sent] == ["http://hook.local/x", "http://slack.local/y"]


# ------------------------------------------------------------------ dashboard and exports

async def test_dashboard_is_self_contained(svc, admin, user_token):
    await svc.ingest(request_trace("t1", answer="Customers must report a claim within 30 days [1].", passages=PASSAGES))
    data = await svc.dashboard_data(admin)
    assert data["requests"] == 1 and data["spend_by_feature"][0]["key"]["feature"] == "claims-qa"
    from aiready.core.identity import get_token_service
    tok = get_token_service().issue(sub="root", tenant="acme", roles=["admin"])
    client = TestClient(create_app(svc))
    r = client.get("/dashboard", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and "<svg" in r.text
    assert "src=\"http" not in r.text and "href=\"http" not in r.text and "cdn" not in r.text.lower()
    assert client.get("/dashboard").status_code == 401
    assert client.get("/dashboard", headers={"Authorization": f"Bearer {user_token}"}).status_code == 403
    csv = client.get("/v1/chargeback?format=csv", headers={"Authorization": f"Bearer {tok}"})
    assert csv.status_code == 200 and csv.text.startswith("period,team")


def test_grafana_and_prometheus_exports(cfg):
    from aiready.observability.exports import grafana_dashboard, prometheus_rules
    g = grafana_dashboard(cfg)
    assert g["uid"] == "aiready-overview" and len(g["panels"]) >= 8
    titles = " ".join(p["title"] for p in g["panels"])
    assert "first_token p95 vs budget 800 ms" in titles and "Spend per team" in titles
    rules = prometheus_rules(cfg)["groups"][0]["rules"]
    names = {r["alert"] for r in rules}
    assert "AIReadyLatencyOverBudget_screen" in names and "AIReadySpendOverBudget_team_claims_team" in names
    screen = next(r for r in rules if r["alert"] == "AIReadyLatencyOverBudget_screen")
    assert screen["expr"].endswith("> 0.05")


def test_wiring_local_provider(monkeypatch):
    svc = wiring.connect("observability")
    assert isinstance(svc, ObservabilityService)


def test_model_price_default():
    assert ModelPrice().input_per_1k == 0
