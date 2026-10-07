""""Tune only when all three hold" (fine-tuning pipeline slide):

1. A measured task has stopped improving after retrieval and prompt work.
2. There are enough approved examples to train on and to test with.
3. Someone owns retraining every time the base model changes.

Fine-tuning adds a model to own forever; this check makes the decision explicit and auditable instead of a
reflex when quality disappoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.identity import Principal
from aiready.finetune.config import ReadinessConfig
from aiready.observability.governance import service_principal


class ReadinessCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class ReadinessReport(BaseModel):
    task: str
    ready: bool
    checks: list[ReadinessCheck]
    recommendation: str = ""


class ReadinessInput(BaseModel):
    task: str
    score_history: list[float] = Field(default_factory=list)  # one per release, oldest first, after retrieval/prompt work
    retrieval_and_prompt_work_done: bool = True
    approved_examples: int = 0
    retraining_owner: str | None = None


def plateaued(history: list[float], window: int, min_improvement: float) -> tuple[bool, str]:
    if len(history) < window + 1:
        return False, f"need at least {window + 1} measured releases, have {len(history)}"
    before = max(history[: -window])
    recent = max(history[-window:])
    gain = recent - before
    return gain < min_improvement, f"best of last {window} releases {recent:.3f} vs earlier best {before:.3f} (gain {gain:+.3f})"


def readiness_check(inp: ReadinessInput, cfg: ReadinessConfig | None = None, *,
                    principal: Principal | None = None) -> ReadinessReport:
    cfg = cfg or ReadinessConfig()
    stalled, why = plateaued(inp.score_history, cfg.plateau_window, cfg.min_improvement)
    c1 = ReadinessCheck(
        name="stopped_improving", passed=stalled and inp.retrieval_and_prompt_work_done,
        detail=why if inp.retrieval_and_prompt_work_done else "retrieval and prompt work not done yet: do that first")
    need = cfg.min_train_examples + cfg.min_test_examples
    c2 = ReadinessCheck(name="enough_examples", passed=inp.approved_examples >= need,
                        detail=f"{inp.approved_examples} approved examples; need {cfg.min_train_examples} to train "
                               f"+ {cfg.min_test_examples} to test")
    c3 = ReadinessCheck(name="retraining_owner", passed=bool(inp.retraining_owner and inp.retraining_owner.strip()),
                        detail=f"owner: {inp.retraining_owner}" if inp.retraining_owner else
                        "nobody owns retraining when the base model changes")
    checks = [c1, c2, c3]
    ready = all(c.passed for c in checks)
    rec = ("All three hold: build the dataset and train an adapter." if ready else
           "Do not fine-tune yet: " + "; ".join(c.detail for c in checks if not c.passed))
    report = ReadinessReport(task=inp.task, ready=ready, checks=checks, recommendation=rec)
    get_audit().for_principal(principal or service_principal("finetune"), "finetune.readiness", resource=inp.task,
                              decision="ready" if ready else "not_ready", failed=[c.name for c in checks if not c.passed])
    return report


def history_from_scorecards(suite: str, metric: str = "overall") -> list[float]:
    """Score history of a task's evaluation suite, one value per release (from the evals scorecard history)."""
    from aiready.evals.runner import ScorecardHistory

    out = []
    for card in ScorecardHistory().list(suite):
        if metric == "overall":
            out.append(card.overall)
        elif metric in card.per_scorer:
            out.append(card.per_scorer[metric].mean)
    return out
