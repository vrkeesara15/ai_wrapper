import os
from datetime import UTC, datetime

import pytest
import yaml

from aiready import testing
from aiready.core import wiring
from aiready.core.audit import get_audit
from aiready.core.errors import ConfigError, PermissionDenied, PolicyViolation, ValidationFailed
from aiready.core.types import ChatChoice, ChatMessage, ChatResponse, Usage
from aiready.finetune.config import BaseModelPin, FinetuneConfig
from aiready.finetune.dataset import (
    DatasetBuilder,
    DatasetExample,
    ExpertExample,
    RequestLogExample,
    normalize_key,
    stratified_group_split,
)
from aiready.finetune.readiness import ReadinessInput, readiness_check
from aiready.finetune.release import AdapterReleaseGate, longest_common_run
from aiready.finetune.store import DatasetStore
from aiready.finetune.tracker import DocStoreTracker, make_tracker
from aiready.finetune.training import (
    axolotl_config,
    build_job_spec,
    is_off_peak,
    kubernetes_job,
    llama_factory_config,
    peft_config,
    render_manifests,
    resolve_base_model,
)


@pytest.fixture(autouse=True)
def allow_policy():
    wiring.override("policy", testing.AllowAllPolicy())


@pytest.fixture
def builder_principal():
    return testing.make_principal("dana", roles=["ai-engineer"])


# ------------------------------------------------------------------ readiness

def test_readiness_all_three_must_hold():
    ok = readiness_check(ReadinessInput(task="summarise", score_history=[0.6, 0.7, 0.74, 0.745, 0.744, 0.746],
                                        approved_examples=900, retraining_owner="ml-team"))
    assert ok.ready and all(c.passed for c in ok.checks)
    improving = readiness_check(ReadinessInput(task="s", score_history=[0.5, 0.6, 0.7, 0.8], approved_examples=900,
                                               retraining_owner="x"))
    assert not improving.ready and [c.name for c in improving.checks if not c.passed] == ["stopped_improving"]
    few = readiness_check(ReadinessInput(task="s", score_history=[0.7] * 5, approved_examples=100, retraining_owner=None))
    assert {c.name for c in few.checks if not c.passed} == {"enough_examples", "retraining_owner"}
    assert "Do not fine-tune yet" in few.recommendation
    no_prompt_work = readiness_check(ReadinessInput(task="s", score_history=[0.7] * 5, approved_examples=900,
                                                    retraining_owner="x", retrieval_and_prompt_work_done=False))
    assert not no_prompt_work.ready
    assert [r.decision for r in get_audit().read() if r.event == "finetune.readiness"] == ["ready", "not_ready", "not_ready", "not_ready"]


# ------------------------------------------------------------------ dataset builder

def make_logs(n=200, users=40):
    logs = []
    for i in range(n):
        logs.append(RequestLogExample(
            id=f"log-{i}", input=f"Summarise claim note {i} about {'water' if i % 3 else 'fire'} damage",
            output=f"Claim {i}: {'water' if i % 3 else 'fire'} damage, adjuster visit booked.",
            consent=i % 10 != 0, approved=i % 15 != 0, user=f"user-{i % users}", label="water" if i % 3 else "fire",
            task="summarise"))
    return logs


async def test_dataset_rules_masking_dedupe_and_no_leakage(builder_principal):
    logs = make_logs()
    logs.append(RequestLogExample(id="dup", input="  Summarise claim note 1 about water damage!! ", output="x y",
                                  consent=True, approved=True, user="user-x", label="water", task="summarise"))
    logs.append(RequestLogExample(id="pii", input="Summarise: customer jane@doe.com called from 020 7946 0958",
                                  output="Customer jane@doe.com asked for a call back.", consent=True, approved=True,
                                  user="user-y", label="water", task="summarise"))
    logs.append(RequestLogExample(id="ref", input="Summarise claim 999", output="I could not find this in the available sources.",
                                  consent=True, approved=True, user="user-z", label="water", task="summarise"))
    expert = [ExpertExample(id="e1", input="Summarise claim note 2 about water damage", output="Expert summary.",
                            labeller="sme-1", label="water", task="summarise")]
    splits = await DatasetBuilder(FinetuneConfig().dataset).build(logs, expert, principal=builder_principal, task="summarise")
    rep = splits.report
    assert rep.rejected["no_consent"] == 20 and rep.rejected["not_approved"] > 0
    assert rep.rejected["duplicate"] == 2 and rep.rejected["refusal"] == 1  # log 2 duplicates the expert example
    every = splits.train + splits.val + splits.heldout
    assert len(every) == rep.kept and sum(rep.counts.values()) == rep.kept
    assert all("jane@doe.com" not in e.input + e.output and "7946" not in e.input for e in every)
    assert next(e for e in every if e.input == "Summarise claim note 2 about water damage").source == "expert"
    # no leakage: no input and no user group in two splits
    for a, b in (("train", "val"), ("train", "heldout"), ("val", "heldout")):
        ia = {normalize_key(e.input) for e in splits.split(a)}
        ib = {normalize_key(e.input) for e in splits.split(b)}
        assert not ia & ib
        assert not {e.group for e in splits.split(a)} & {e.group for e in splits.split(b)}
    # stratified: every split has both labels and roughly the configured share
    for name in ("train", "val", "heldout"):
        assert set(rep.strata[name]) == {"water", "fire"}
    assert 0.7 < rep.counts["train"] / rep.kept < 0.9
    assert any(r.event == "finetune.dataset.build" for r in get_audit().read())


async def test_split_is_deterministic():
    exs = [DatasetExample(id=str(i), input=f"q{i}", output="a", task="t", label="ab"[i % 2], source="expert", group=f"g{i}")
           for i in range(100)]
    a = stratified_group_split(exs, {"train": 0.8, "val": 0.1, "heldout": 0.1}, seed=1)
    b = stratified_group_split(list(reversed(exs)), {"train": 0.8, "val": 0.1, "heldout": 0.1}, seed=1)
    assert {k: sorted(e.id for e in v) for k, v in a.items()} == {k: sorted(e.id for e in v) for k, v in b.items()}
    assert [len(a[k]) for k in ("train", "val", "heldout")] == [80, 10, 10]


async def test_request_logs_need_policy_permission(builder_principal):
    wiring.override("policy", testing.DenyPolicy({"data.read"}))
    with pytest.raises(PolicyViolation):
        await DatasetBuilder().build(make_logs(10), [], principal=builder_principal)
    # expert-only datasets do not read request logs
    out = await DatasetBuilder().build([], [ExpertExample(id="e", input="q1", output="a1", labeller="x")],
                                       principal=builder_principal)
    assert out.report.kept == 1


# ------------------------------------------------------------------ dataset store

async def test_versioned_store_checksums(tmp_path, builder_principal):
    splits = await DatasetBuilder().build(make_logs(), [], principal=builder_principal)
    store = DatasetStore(tmp_path / "ds")
    m1 = store.save("claims-summary", splits, task="summarise", created_by="dana")
    m2 = store.save("claims-summary", splits, task="summarise")
    assert (m1.version, m2.version) == (1, 2) and m1.content_hash == m2.content_hash
    assert len(store.load("claims-summary", "heldout", 1)) == m1.counts["heldout"]
    assert store.manifest("claims-summary").version == 2
    p = store.path("claims-summary", "train", 1)
    os.chmod(p, 0o644)
    p.write_text(p.read_text() + '{"id": "sneaky"}\n')
    with pytest.raises(ValidationFailed, match="checksum"):
        store.load("claims-summary", "train", 1)


# ------------------------------------------------------------------ training spec

PIN = BaseModelPin(repo="/models/llama-3.1-8b", revision="0e9e39f2", served_name="chat-small")


async def manifest(tmp_path, principal):
    splits = await DatasetBuilder().build(make_logs(), [], principal=principal)
    return DatasetStore(tmp_path / "ds").save("claims-summary", splits)


async def test_base_model_must_be_pinned():
    cfg = FinetuneConfig(base_models={"good": PIN, "floating": BaseModelPin(repo="x", revision="main")})
    assert (await resolve_base_model("good", cfg)).revision == "0e9e39f2"
    with pytest.raises(ValidationFailed):
        await resolve_base_model("floating", cfg)
    with pytest.raises(ConfigError):
        await resolve_base_model("unknown", cfg)

    class Registry:
        async def get_model(self, name):
            return {"name": name, "repo": "/models/reg", "digest": "sha256:abc", "approved": True}

    wiring.override("registry", Registry())
    pin = await resolve_base_model("unknown", cfg)
    assert pin.repo == "/models/reg" and pin.revision == "sha256:abc"


async def test_job_spec_targets_offpeak_pool_and_never_mounts_heldout(tmp_path, builder_principal):
    m = await manifest(tmp_path, builder_principal)
    cfg = FinetuneConfig()
    spec = build_job_spec(task="claims summary", base_name="llama", base=PIN, manifest=m, cfg=cfg, run_id="run123abc")
    job = kubernetes_job(spec)
    pod = job["spec"]["template"]["spec"]
    assert job["kind"] == "Job" and pod["nodeSelector"] == {"aiready.io/pool": "gpu-offpeak"}
    assert any(t["key"] == "nvidia.com/gpu" for t in pod["tolerations"])
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert job["metadata"]["annotations"]["aiready.io/base-frozen"] == "true"
    assert job["metadata"]["name"] == "ft-claims-summary-run123ab"
    ax = axolotl_config(spec)
    assert ax["adapter"] == "lora" and ax["revision_of_model"] == "0e9e39f2" and ax["lora_modules_to_save"] == []
    rendered = render_manifests(spec)
    assert "heldout" not in rendered and "train.jsonl" in rendered and "val.jsonl" in rendered
    assert len(list(yaml.safe_load_all(rendered))) == 2
    cfg.training.method = "qlora"
    cfg.training.framework = "llama-factory"
    spec_q = build_job_spec(task="t", base_name="llama", base=PIN, manifest=m, cfg=cfg, run_id="r2")
    lf = llama_factory_config(spec_q)
    assert lf["finetuning_type"] == "lora" and lf["quantization_bit"] == 4
    assert "dataset_info.json" in render_manifests(spec_q)
    pc = peft_config(spec_q)
    assert pc["freeze_base"] and pc["quantization"]["load_in_4bit"]
    cfg.training.method = "full"
    with pytest.raises(ValidationFailed):
        build_job_spec(task="t", base_name="llama", base=PIN, manifest=m, cfg=cfg, run_id="r3")


def test_off_peak_windows():
    w = ["Mon-Fri 20:00-06:00", "Sat-Sun 00:00-24:00"]
    utc = UTC
    assert is_off_peak(datetime(2026, 10, 7, 21, 0, tzinfo=utc), w)  # Wednesday night
    assert not is_off_peak(datetime(2026, 10, 7, 12, 0, tzinfo=utc), w)  # Wednesday noon
    assert is_off_peak(datetime(2026, 10, 8, 5, 30, tzinfo=utc), w)  # Thursday early morning
    assert is_off_peak(datetime(2026, 10, 10, 13, 0, tzinfo=utc), w)  # Saturday
    assert not is_off_peak(datetime(2026, 10, 12, 9, 0, tzinfo=utc), w)  # Monday morning


def test_tracker():
    t = make_tracker({"type": "docstore"})
    assert isinstance(t, DocStoreTracker)
    r = t.start_run("claims-summary", name="lora-r16", params={"r": 16})
    t.log_metrics(r.id, {"heldout": 0.71})
    r2 = t.start_run("claims-summary", params={"r": 32})
    t.log_metrics(r2.id, {"heldout": 0.75})
    t.end_run(r.id)
    assert t.get_run(r.id).status == "finished"
    assert t.best_run("claims-summary", "heldout").id == r2.id


# ------------------------------------------------------------------ release gate

TRAIN = [DatasetExample(id=f"t{i}", input=f"Summarise note {i}", task="summarise", source="log", group=f"g{i}",
                        output=" ".join(f"word{i}x{j}" for j in range(40))) for i in range(5)]
HELDOUT = [DatasetExample(id=f"h{i}", input=f"Summarise held note {i}", task="summarise", source="log", group=f"h{i}",
                          output=f"Water damage in kitchen {i}, adjuster visit booked.") for i in range(4)]


class TuningGateway(testing.FakeGateway):
    """adapter-*: knows the held-out answers. adapter-leaky: also continues training text verbatim."""

    async def chat(self, req, principal):
        self.calls.append(req)
        q = req.last_user_text()
        text = "I do not know."
        if req.model.startswith("adapter"):
            for h in HELDOUT:
                if h.input == q:
                    text = h.output
            if q.startswith("Continue this text exactly") and req.model == "adapter-leaky":
                prefix = q.split("\n", 1)[1]
                for t in TRAIN:
                    if t.output.startswith(prefix):
                        text = t.output[len(prefix):]
        elif any(h.input == q for h in HELDOUT):
            text = "Damage reported."
        return ChatResponse(model=req.model, choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
                            usage=Usage())


async def test_adapter_release_flow(builder_principal):
    wiring.override("gateway", TuningGateway())
    cfg = FinetuneConfig().release
    gate = AdapterReleaseGate(cfg)
    tracker = DocStoreTracker()
    run = tracker.start_run("summarise", params={"r": 16})
    rel = await gate.evaluate(task="summarise", adapter="adapter-good", base_name="llama", base=PIN, dataset="cs@v1",
                              heldout=HELDOUT, train=TRAIN, principal=builder_principal, run_id=run.id, tracker=tracker)
    assert tracker.get_run(run.id).metrics["heldout.adapter"] == rel.adapter_score
    assert rel.status == "pending_approval", rel.failures
    assert rel.adapter_score > rel.base_score and rel.memorisation.echoes == 0
    with pytest.raises(PermissionDenied):  # requester cannot approve (four eyes)
        gate.approve(rel.id, testing.make_principal("dana", roles=["ml-approver"]))
    with pytest.raises(PermissionDenied):
        gate.approve(rel.id, testing.make_principal("eve", roles=["user"]))
    with pytest.raises(ValidationFailed):
        gate.route_snippet(rel.id)
    approver = testing.make_principal("lead", roles=["ml-approver"])
    gate.approve(rel.id, approver, comment="looks good")
    reg = await gate.register(rel.id, approver, adapter_uri="/data/adapters/summarise/run1")
    assert reg.status == "registered" and gate.adapters()[0]["base_revision"] == "0e9e39f2"
    snippet = gate.route_snippet(rel.id)["gateway"]["task_routes"]
    assert list(snippet) == ["summarise"]  # only its own task
    assert snippet["summarise"]["model"] == "adapter-good" and snippet["summarise"]["fallbacks"] == ["chat-small"]
    patch = gate.rollback_patch("summarise", approver, reason="quality regression")
    assert patch == {"gateway": {"task_routes": {"summarise": {"model": "chat-small", "fallbacks": [], "adapter": None}}}}
    events = [r.event for r in get_audit().read()]
    for e in ("finetune.gate", "finetune.approval", "finetune.register", "finetune.rollback"):
        assert e in events


async def test_gate_rejects_echoing_or_weaker_adapter(builder_principal):
    wiring.override("gateway", TuningGateway())
    gate = AdapterReleaseGate(FinetuneConfig().release)
    leaky = await gate.evaluate(task="summarise", adapter="adapter-leaky", base_name="llama", base=PIN, dataset="cs@v1",
                                heldout=HELDOUT, train=TRAIN, principal=builder_principal)
    assert leaky.status == "rejected" and leaky.memorisation.echoes == len(TRAIN)
    assert any("echoed" in f for f in leaky.failures)
    weak = await gate.evaluate(task="summarise", adapter="weak-model", base_name="llama", base=PIN, dataset="cs@v1",
                               heldout=HELDOUT, train=TRAIN, principal=builder_principal)
    assert weak.status == "rejected" and any("does not beat base" in f for f in weak.failures)
    with pytest.raises(ValidationFailed):
        gate.approve(weak.id, testing.make_principal("lead", roles=["ml-approver"]))


async def test_gate_runs_safety_suites_on_adapter(builder_principal):
    from aiready.evals.golden import EvalCase, EvalSuite
    wiring.override("gateway", TuningGateway())
    suite = EvalSuite(name="inj", safety=True, scorers=["injection_resisted"],
                      cases=[EvalCase(id="i1", question="Summarise held note 0", canaries=["Water damage"])])
    rel = await AdapterReleaseGate().evaluate(task="summarise", adapter="adapter-good", base_name="llama", base=PIN,
                                              dataset="cs@v1", heldout=HELDOUT, train=TRAIN, principal=builder_principal,
                                              safety_suites=[suite])
    assert rel.status == "rejected" and rel.safety_passed is False


def test_longest_common_run():
    assert longest_common_run(["a", "b", "c", "d", "e"], ["x", "b", "c", "d", "y"]) == 3
    assert longest_common_run([], ["a"]) == 0
