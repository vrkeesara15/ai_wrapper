"""Dataset builder: approved request logs + expert-labelled examples -> clean, deduplicated, split dataset.

Fine-tuning pipeline slide, column 1: "Approved request logs - masked, with consent", "Expert-labelled
examples", "Dataset builder - clean, split, hold out", "Versioned dataset store".

Rules enforced here, not left to the caller:
  * a request log is used only when it has the user's consent *and* was approved by a reviewer;
  * every text is masked again (guardrails when available, plus the regex masker), whatever the source says;
  * using request logs for training is a policy decision (``data.read`` on ``request_logs`` with purpose
    ``fine-tuning``) and is audited;
  * duplicates are removed *before* splitting, and whole groups (e.g. all examples of one user) go to one split,
    so the held-out set never contains a near-copy of a training example;
  * the split is stratified (each label/task keeps its share in train, val and held-out) and seeded.
"""

from __future__ import annotations

import random
import re
import time
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.errors import ValidationFailed
from aiready.core.identity import Principal
from aiready.core.telemetry import metrics, span
from aiready.evals.scorers import is_refusal
from aiready.finetune.config import DatasetConfig
from aiready.observability.governance import authorize
from aiready.observability.masking import RegexMasker, SpanMasker

SPLITS = ("train", "val", "heldout")


class RequestLogExample(BaseModel):
    id: str
    input: str
    output: str
    task: str = "default"
    consent: bool = False
    approved: bool = False
    user: str | None = None
    trace_id: str | None = None
    label: str | None = None
    ts: float | None = None


class ExpertExample(BaseModel):
    id: str
    input: str
    output: str
    task: str = "default"
    labeller: str
    label: str | None = None


class DatasetExample(BaseModel):
    id: str
    input: str
    output: str
    task: str
    label: str | None = None
    source: Literal["log", "expert"]
    group: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class BuildReport(BaseModel):
    received_logs: int = 0
    received_expert: int = 0
    rejected: dict[str, int] = Field(default_factory=dict)
    kept: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    strata: dict[str, dict[str, int]] = Field(default_factory=dict)  # split -> stratum -> n
    built_at: float = Field(default_factory=time.time)


class DatasetSplits(BaseModel):
    train: list[DatasetExample]
    val: list[DatasetExample]
    heldout: list[DatasetExample]
    report: BuildReport

    def split(self, name: str) -> list[DatasetExample]:
        return getattr(self, name)


def normalize_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def stratified_group_split(examples: list[DatasetExample], fractions: dict[str, float], *, seed: int,
                           stratify_by: str = "label") -> dict[str, list[DatasetExample]]:
    """Assign whole groups to splits so every stratum keeps its share in each split.

    Within each stratum, groups are shuffled (seeded) and each group goes to the split furthest below its
    target size. Deterministic for a given seed and input order-independent (groups are sorted first).
    """
    total = sum(fractions.values())
    fr = {k: v / total for k, v in fractions.items() if v > 0}
    groups: dict[str, list[DatasetExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.group].append(ex)

    def stratum(exs: list[DatasetExample]) -> str:
        vals = [(getattr(e, stratify_by, None) or e.task) for e in exs]
        return Counter(vals).most_common(1)[0][0]

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for g, exs in groups.items():
        by_stratum[stratum(exs)].append(g)
    rng = random.Random(seed)
    out: dict[str, list[DatasetExample]] = {k: [] for k in fractions}
    for st in sorted(by_stratum):
        gids = sorted(by_stratum[st])
        rng.shuffle(gids)
        n_total = sum(len(groups[g]) for g in gids)
        got = {k: 0 for k in fr}
        # each group goes to the split furthest below its target share (ties: train, then val, then held-out)
        for g in gids:
            deficit = {k: fr[k] * n_total - got[k] for k in fr}
            target = max(deficit, key=lambda k: (deficit[k], -list(fr).index(k)))
            got[target] += len(groups[g])
            out[target].extend(groups[g])
    return out


class DatasetBuilder:
    def __init__(self, cfg: DatasetConfig | None = None, masker: SpanMasker | None = None):
        self.cfg = cfg or DatasetConfig()
        self.masker = masker or SpanMasker(RegexMasker(), use_guardrails=self.cfg.use_guardrails_mask)

    async def build(self, logs: list[RequestLogExample], expert: list[ExpertExample], *, principal: Principal,
                    task: str | None = None) -> DatasetSplits:
        """Build the splits. ``principal`` is the person building the dataset (policy + audit)."""
        if logs:
            await authorize(principal, "data.read", service="finetune",
                            resource={"type": "request_logs", "id": task or "all", "count": len(logs)},
                            context={"purpose": "fine-tuning", "task": task})
        rep = BuildReport(received_logs=len(logs), received_expert=len(expert))
        rejected: Counter[str] = Counter()
        candidates: list[DatasetExample] = []
        with span("finetune.dataset.build", task=task, logs=len(logs), expert=len(expert)), \
                metrics.time_stage("finetune", "dataset_build"):
            # Expert examples first: on a duplicate input the expert label wins over a logged answer.
            for e in expert:
                if task and e.task != task:
                    rejected["other_task"] += 1
                    continue
                candidates.append(DatasetExample(id=e.id, input=e.input, output=e.output, task=e.task, label=e.label,
                                                 source="expert", group=f"expert:{e.id}",
                                                 provenance={"labeller": e.labeller}))
            for lg in logs:
                if task and lg.task != task:
                    rejected["other_task"] += 1
                elif not lg.consent:
                    rejected["no_consent"] += 1
                elif not lg.approved:
                    rejected["not_approved"] += 1
                else:
                    grp = getattr(lg, self.cfg.group_by, None) or lg.id
                    candidates.append(DatasetExample(id=lg.id, input=lg.input, output=lg.output, task=lg.task,
                                                     label=lg.label, source="log", group=f"{self.cfg.group_by}:{grp}",
                                                     provenance={"trace_id": lg.trace_id, "ts": lg.ts}))
            cleaned: list[DatasetExample] = []
            seen: set[str] = set()
            for ex in candidates:
                ex.input = (await self.masker.mask_text(ex.input.strip())) if ex.input else ""
                ex.output = (await self.masker.mask_text(ex.output.strip())) if ex.output else ""
                if len(ex.input) < self.cfg.min_chars or len(ex.output) < self.cfg.min_chars:
                    rejected["empty"] += 1
                    continue
                if len(ex.input) + len(ex.output) > self.cfg.max_chars:
                    rejected["too_long"] += 1
                    continue
                if self.cfg.drop_refusals and is_refusal(ex.output):
                    rejected["refusal"] += 1
                    continue
                key = normalize_key(ex.input)
                if key in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(key)
                cleaned.append(ex)
            parts = stratified_group_split(cleaned, self.cfg.split.model_dump(), seed=self.cfg.seed,
                                           stratify_by=self.cfg.stratify_by)
            assert_no_leakage(parts)
        rep.rejected = dict(rejected)
        rep.kept = len(cleaned)
        rep.counts = {k: len(v) for k, v in parts.items()}
        rep.strata = {k: dict(Counter((getattr(e, self.cfg.stratify_by, None) or e.task) for e in v)) for k, v in parts.items()}
        get_audit().for_principal(principal, "finetune.dataset.build", resource=task or "all", decision="built",
                                  counts=rep.counts, rejected=rep.rejected,
                                  log_ids=[e.id for e in cleaned if e.source == "log"][:1000])
        return DatasetSplits(train=parts["train"], val=parts["val"], heldout=parts["heldout"], report=rep)


def assert_no_leakage(parts: dict[str, list[DatasetExample]]) -> None:
    """No input and no group may appear in more than one split."""
    owner: dict[str, str] = {}
    gowner: dict[str, str] = {}
    for split, exs in parts.items():
        for e in exs:
            k = normalize_key(e.input)
            if owner.setdefault(k, split) != split:
                raise ValidationFailed(f"leakage: input of {e.id} is in {owner[k]} and {split}")
            if gowner.setdefault(e.group, split) != split:
                raise ValidationFailed(f"leakage: group {e.group} is in {gowner[e.group]} and {split}")
