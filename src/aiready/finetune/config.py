"""Configuration of the fine-tuning pipeline (top-level YAML section ``finetune``)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReadinessConfig(BaseModel):
    plateau_window: int = 3  # last N measured releases after retrieval/prompt work
    min_improvement: float = 0.01  # less than this over the window = "stopped improving"
    min_train_examples: int = 500
    min_test_examples: int = 100


class SplitConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    heldout: float = 0.1


class DatasetConfig(BaseModel):
    store_dir: str = "data/finetune/datasets"
    split: SplitConfig = Field(default_factory=SplitConfig)
    seed: int = 42
    min_chars: int = 2
    max_chars: int = 16000
    drop_refusals: bool = True  # "I could not find this" is not a behaviour to train in
    group_by: str = "user"  # examples of one group never straddle splits (no leakage via near-duplicates)
    stratify_by: str = "label"  # label | task
    use_guardrails_mask: bool = True


class BaseModelPin(BaseModel):
    repo: str  # e.g. meta-llama/Llama-3.1-8B-Instruct, or a path in the client's model store
    revision: str  # commit sha / digest - never "main" or "latest"
    served_name: str | None = None  # name of the base route in the gateway
    license: str | None = None


class GpuPoolConfig(BaseModel):
    node_selector: dict[str, str] = Field(default_factory=lambda: {"aiready.io/pool": "gpu-offpeak"})
    tolerations: list[dict[str, Any]] = Field(default_factory=lambda: [
        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
        {"key": "aiready.io/offpeak", "operator": "Equal", "value": "true", "effect": "NoSchedule"}])
    gpus: int = 1
    gpu_resource: str = "nvidia.com/gpu"
    cpu: str = "8"
    memory: str = "64Gi"
    priority_class: str | None = "aiready-batch-low"
    # Off-peak windows ("nights and weekends"), UTC: "Mon-Fri 20:00-06:00", "Sat-Sun 00:00-24:00"
    off_peak_windows: list[str] = Field(default_factory=lambda: ["Mon-Fri 20:00-06:00", "Sat-Sun 00:00-24:00"])


class TrainingConfig(BaseModel):
    framework: str = "axolotl"  # axolotl | llama-factory | peft
    method: str = "lora"  # lora | qlora
    image: str = "registry.local/aiready/trainer:0.1.0"
    namespace: str = "aiready-training"
    service_account: str = "aiready-trainer"
    data_volume_claim: str = "aiready-finetune-data"
    output_uri: str = "/data/adapters"
    lora: dict[str, Any] = Field(default_factory=lambda: {
        "r": 16, "alpha": 32, "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]})
    hyperparameters: dict[str, Any] = Field(default_factory=lambda: {
        "learning_rate": 2e-4, "epochs": 3, "micro_batch_size": 4, "gradient_accumulation_steps": 4,
        "max_seq_len": 4096, "warmup_ratio": 0.03, "lr_scheduler": "cosine", "bf16": True})
    gpu_pool: GpuPoolConfig = Field(default_factory=GpuPoolConfig)
    active_deadline_seconds: int = 6 * 3600


class ReleaseConfig(BaseModel):
    min_margin: float = 0.0  # adapter must beat the base by more than this on held-out
    scorers: list[str] = Field(default_factory=lambda: ["f1", "contains", "pii_leak"])
    max_echo_rate: float = 0.0  # share of memorisation probes that may reproduce training text
    echo_ngram: int = 12  # verbatim tokens that count as an echo
    prefix_tokens: int = 24
    probes: int = 50  # training examples probed for memorisation
    approver_roles: list[str] = Field(default_factory=lambda: ["admin", "ml-approver"])
    require_distinct_approver: bool = True  # four eyes: the requester cannot approve


class FinetuneConfig(BaseModel):
    readiness: ReadinessConfig = Field(default_factory=ReadinessConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    base_models: dict[str, BaseModelPin] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracker: dict[str, Any] = Field(default_factory=lambda: {"type": "docstore"})
    release: ReleaseConfig = Field(default_factory=ReleaseConfig)


def load_config() -> FinetuneConfig:
    from aiready.core.config import get_config

    return get_config().section("finetune", FinetuneConfig)
