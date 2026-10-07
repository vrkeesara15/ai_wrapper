"""Training job specs: LoRA / QLoRA adapter on a pinned, frozen base model, run on the off-peak GPU pool.

Fine-tuning pipeline slide, column 2: "Pinned base model - from the model registry", "Adapter training job -
base weights stay frozen", "Off-peak GPU pool - nights and weekends", "Experiment tracker".

This module *generates* configuration; it does not run training. The output is a framework config (axolotl,
LLaMA-Factory or a plain peft/transformers config) and a Kubernetes Job manifest (plus the ConfigMap holding
the config) that targets the off-peak pool through nodeSelector + tolerations. Only the train and validation
files are mounted: the held-out split never reaches the training job.
"""

from __future__ import annotations

import inspect
import logging
import re
from datetime import UTC, datetime
from typing import Any

import yaml
from pydantic import BaseModel, Field

from aiready.core.errors import ConfigError, ValidationFailed
from aiready.finetune.config import BaseModelPin, FinetuneConfig, GpuPoolConfig
from aiready.finetune.store import DatasetManifest

log = logging.getLogger(__name__)
FLOATING_REVISIONS = {"", "main", "master", "latest", "head", "HEAD"}


class TrainingJobSpec(BaseModel):
    name: str
    task: str
    run_id: str
    base_name: str
    base: BaseModelPin
    dataset: str  # name@vN
    dataset_hash: str
    train_path: str
    val_path: str
    method: str = "lora"
    framework: str = "axolotl"
    lora: dict[str, Any]
    hyperparameters: dict[str, Any]
    seed: int = 42
    output_dir: str
    image: str
    namespace: str
    service_account: str
    data_volume_claim: str
    gpu_pool: GpuPoolConfig
    active_deadline_seconds: int
    freeze_base: bool = True  # fixed: only adapter weights are trained
    labels: dict[str, str] = Field(default_factory=dict)


def k8s_name(*parts: str, max_len: int = 63) -> str:
    raw = "-".join(p for p in parts if p)
    name = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    return name[:max_len].rstrip("-") or "job"


async def resolve_base_model(name: str, cfg: FinetuneConfig) -> BaseModelPin:
    """The base model pin: from the model registry when deployed (duck-typed ``get_model``), else from
    ``finetune.base_models``. A floating revision (main/latest) is refused: an adapter is only valid for the
    exact base weights it was trained on."""
    pin: BaseModelPin | None = None
    try:
        from aiready.core.wiring import connect

        reg = connect("registry")
        fn = getattr(reg, "get_model", None)
        if fn is not None:
            res = fn(name)
            if inspect.isawaitable(res):
                res = await res
            data = res.model_dump() if hasattr(res, "model_dump") else res
            if isinstance(data, dict):
                if data.get("disabled") or data.get("status") in ("draft", "deprecated", "rejected", "retired") \
                        or data.get("approved") is False:
                    raise ValidationFailed(f"base model {name} is not approved in the registry")
                # The registry record itself is immutable per version; its weights checksum pins the bytes.
                revision = (data.get("revision") or data.get("digest") or data.get("weights_checksum")
                            or (f"{data.get('name', name)}@{data['version']}" if data.get("version") else None))
                if revision:
                    pin = BaseModelPin(repo=data.get("repo") or data.get("source") or data.get("endpoint") or name,
                                       revision=str(revision), served_name=data.get("served_name") or data.get("name") or name,
                                       license=data.get("license") or data.get("licence"),
                                       registry_version=str(data["version"]) if data.get("version") else None)
    except ValidationFailed:
        raise
    except Exception as exc:
        log.info("registry lookup for %s unavailable (%s); using finetune.base_models", name, exc)
    if pin is None:
        pin = cfg.base_models.get(name)
    if pin is None:
        raise ConfigError(f"base model '{name}' is not pinned (registry or finetune.base_models)")
    if pin.revision.strip() in FLOATING_REVISIONS:
        raise ValidationFailed(f"base model '{name}' revision '{pin.revision}' is floating; pin a commit or digest")
    return pin.model_copy(update={"served_name": pin.served_name or name})


def build_job_spec(*, task: str, base_name: str, base: BaseModelPin, manifest: DatasetManifest, cfg: FinetuneConfig,
                   run_id: str, data_root: str = "/data/datasets", overrides: dict[str, Any] | None = None) -> TrainingJobSpec:
    t = cfg.training
    if t.method not in ("lora", "qlora"):
        raise ValidationFailed(f"method must be lora or qlora (base weights stay frozen), not '{t.method}'")
    if t.framework not in ("axolotl", "llama-factory", "peft"):
        raise ValidationFailed(f"unknown framework '{t.framework}'")
    hyper = {**t.hyperparameters, **(overrides or {}).get("hyperparameters", {})}
    lora = {**t.lora, **(overrides or {}).get("lora", {})}
    return TrainingJobSpec(
        name=k8s_name("ft", task, run_id[:8]), task=task, run_id=run_id, base_name=base_name, base=base,
        dataset=manifest.ref, dataset_hash=manifest.content_hash,
        train_path=f"{data_root}/{manifest.files['train']}", val_path=f"{data_root}/{manifest.files['val']}",
        method=t.method, framework=t.framework, lora=lora, hyperparameters=hyper, seed=cfg.dataset.seed,
        output_dir=f"{t.output_uri.rstrip('/')}/{k8s_name(task)}/{run_id}", image=t.image, namespace=t.namespace,
        service_account=t.service_account, data_volume_claim=t.data_volume_claim, gpu_pool=t.gpu_pool,
        active_deadline_seconds=t.active_deadline_seconds,
        labels={"aiready.io/component": "finetune", "aiready.io/task": k8s_name(task), "aiready.io/run": run_id[:63]},
    )


# --------------------------------------------------------------------------- framework configs

def axolotl_config(spec: TrainingJobSpec) -> dict[str, Any]:
    h, lo = spec.hyperparameters, spec.lora
    ds_type = {"field_instruction": "input", "field_output": "output", "format": "{instruction}",
               "no_input_format": "{instruction}"}
    return {
        "base_model": spec.base.repo, "revision_of_model": spec.base.revision, "trust_remote_code": False,
        "adapter": spec.method, "load_in_4bit": spec.method == "qlora", "load_in_8bit": False,
        "lora_r": lo["r"], "lora_alpha": lo["alpha"], "lora_dropout": lo["dropout"],
        "lora_target_modules": lo["target_modules"], "lora_modules_to_save": [],  # nothing of the base is trained
        "datasets": [{"path": spec.train_path, "ds_type": "json", "type": ds_type}],
        "test_datasets": [{"path": spec.val_path, "ds_type": "json", "split": "train", "type": ds_type}],
        "dataset_prepared_path": f"{spec.output_dir}/prepared", "output_dir": spec.output_dir,
        "sequence_len": h.get("max_seq_len", 4096), "sample_packing": True,
        "micro_batch_size": h.get("micro_batch_size", 4), "gradient_accumulation_steps": h.get("gradient_accumulation_steps", 4),
        "num_epochs": h.get("epochs", 3), "learning_rate": h.get("learning_rate", 2e-4),
        "lr_scheduler": h.get("lr_scheduler", "cosine"), "warmup_ratio": h.get("warmup_ratio", 0.03),
        "optimizer": "paged_adamw_8bit" if spec.method == "qlora" else "adamw_torch",
        "bf16": h.get("bf16", True), "gradient_checkpointing": True, "save_safetensors": True,
        "seed": spec.seed, "eval_steps": 0.1, "saves_per_epoch": 1, "logging_steps": 10,
    }


def llama_factory_config(spec: TrainingJobSpec) -> dict[str, Any]:
    h, lo = spec.hyperparameters, spec.lora
    cfg: dict[str, Any] = {
        "model_name_or_path": spec.base.repo, "model_revision": spec.base.revision, "stage": "sft", "do_train": True,
        "finetuning_type": "lora", "lora_rank": lo["r"], "lora_alpha": lo["alpha"], "lora_dropout": lo["dropout"],
        "lora_target": ",".join(lo["target_modules"]), "dataset": f"{k8s_name(spec.task)}_train",
        "eval_dataset": f"{k8s_name(spec.task)}_val", "dataset_dir": "/config", "template": "default",
        "cutoff_len": h.get("max_seq_len", 4096), "output_dir": spec.output_dir,
        "per_device_train_batch_size": h.get("micro_batch_size", 4),
        "gradient_accumulation_steps": h.get("gradient_accumulation_steps", 4), "learning_rate": h.get("learning_rate", 2e-4),
        "num_train_epochs": float(h.get("epochs", 3)), "lr_scheduler_type": h.get("lr_scheduler", "cosine"),
        "warmup_ratio": h.get("warmup_ratio", 0.03), "bf16": h.get("bf16", True), "seed": spec.seed,
        "logging_steps": 10, "save_strategy": "epoch", "eval_strategy": "epoch", "overwrite_output_dir": False,
    }
    if spec.method == "qlora":
        cfg.update({"quantization_bit": 4, "quantization_method": "bitsandbytes"})
    return cfg


def llama_factory_dataset_info(spec: TrainingJobSpec) -> dict[str, Any]:
    cols = {"prompt": "input", "response": "output"}
    return {f"{k8s_name(spec.task)}_train": {"file_name": spec.train_path, "columns": cols},
            f"{k8s_name(spec.task)}_val": {"file_name": spec.val_path, "columns": cols}}


def peft_config(spec: TrainingJobSpec) -> dict[str, Any]:
    """Config for a plain transformers + peft training script (``peft.LoraConfig`` + ``TrainingArguments``)."""
    h, lo = spec.hyperparameters, spec.lora
    out: dict[str, Any] = {
        "base_model": {"pretrained_model_name_or_path": spec.base.repo, "revision": spec.base.revision},
        "freeze_base": True,
        "lora_config": {"r": lo["r"], "lora_alpha": lo["alpha"], "lora_dropout": lo["dropout"],
                        "target_modules": lo["target_modules"], "bias": "none", "task_type": "CAUSAL_LM",
                        "modules_to_save": None},
        "training_arguments": {
            "output_dir": spec.output_dir, "num_train_epochs": h.get("epochs", 3),
            "per_device_train_batch_size": h.get("micro_batch_size", 4),
            "gradient_accumulation_steps": h.get("gradient_accumulation_steps", 4),
            "learning_rate": h.get("learning_rate", 2e-4), "lr_scheduler_type": h.get("lr_scheduler", "cosine"),
            "warmup_ratio": h.get("warmup_ratio", 0.03), "bf16": h.get("bf16", True), "seed": spec.seed,
            "save_strategy": "epoch", "eval_strategy": "epoch", "logging_steps": 10, "report_to": [],
            "gradient_checkpointing": True},
        "data": {"train_file": spec.train_path, "validation_file": spec.val_path, "input_field": "input",
                 "output_field": "output", "max_seq_len": h.get("max_seq_len", 4096)},
    }
    if spec.method == "qlora":
        out["quantization"] = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
                               "bnb_4bit_compute_dtype": "bfloat16"}
    return out


def framework_config(spec: TrainingJobSpec) -> dict[str, Any]:
    return {"axolotl": axolotl_config, "llama-factory": llama_factory_config, "peft": peft_config}[spec.framework](spec)


COMMANDS = {
    "axolotl": ["accelerate", "launch", "-m", "axolotl.cli.train", "/config/train.yaml"],
    "llama-factory": ["llamafactory-cli", "train", "/config/train.yaml"],
    "peft": ["python", "-m", "aiready_trainer.peft_train", "/config/train.yaml"],
}


# --------------------------------------------------------------------------- kubernetes

def kubernetes_configmap(spec: TrainingJobSpec) -> dict[str, Any]:
    data = {"train.yaml": yaml.safe_dump(framework_config(spec), sort_keys=False)}
    if spec.framework == "llama-factory":
        import json

        data["dataset_info.json"] = json.dumps(llama_factory_dataset_info(spec), indent=2)
    return {"apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": f"{spec.name}-config", "namespace": spec.namespace, "labels": spec.labels}, "data": data}


def kubernetes_job(spec: TrainingJobSpec) -> dict[str, Any]:
    """batch/v1 Job on the off-peak GPU pool. Offline by design: no hub downloads, weights come from the
    client's model store; the dataset volume is mounted read-only."""
    pool = spec.gpu_pool
    pod: dict[str, Any] = {
        "restartPolicy": "Never", "serviceAccountName": spec.service_account,
        "nodeSelector": dict(pool.node_selector), "tolerations": [dict(t) for t in pool.tolerations],
        "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "fsGroup": 1000},
        "containers": [{
            "name": "trainer", "image": spec.image, "command": COMMANDS[spec.framework],
            "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                    {"name": "AIREADY_RUN_ID", "value": spec.run_id}, {"name": "AIREADY_DATASET", "value": spec.dataset},
                    {"name": "AIREADY_DATASET_HASH", "value": spec.dataset_hash},
                    {"name": "AIREADY_BASE_REVISION", "value": spec.base.revision}],
            "resources": {"limits": {pool.gpu_resource: pool.gpus, "cpu": pool.cpu, "memory": pool.memory},
                          "requests": {pool.gpu_resource: pool.gpus, "cpu": pool.cpu, "memory": pool.memory}},
            "volumeMounts": [{"name": "config", "mountPath": "/config", "readOnly": True},
                             {"name": "datasets", "mountPath": "/data/datasets", "readOnly": True},
                             {"name": "adapters", "mountPath": "/data/adapters"},
                             {"name": "shm", "mountPath": "/dev/shm"}],
            "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        }],
        "volumes": [{"name": "config", "configMap": {"name": f"{spec.name}-config"}},
                    {"name": "datasets", "persistentVolumeClaim": {"claimName": spec.data_volume_claim, "readOnly": True}},
                    {"name": "adapters", "persistentVolumeClaim": {"claimName": f"{spec.data_volume_claim}-adapters"}},
                    {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}}],
    }
    if pool.priority_class:
        pod["priorityClassName"] = pool.priority_class
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": spec.name, "namespace": spec.namespace, "labels": spec.labels,
                     "annotations": {"aiready.io/base-model": f"{spec.base.repo}@{spec.base.revision}",
                                     "aiready.io/dataset": spec.dataset, "aiready.io/dataset-hash": spec.dataset_hash,
                                     "aiready.io/method": spec.method, "aiready.io/base-frozen": "true",
                                     "aiready.io/off-peak-windows": "; ".join(pool.off_peak_windows)}},
        "spec": {"backoffLimit": 1, "activeDeadlineSeconds": spec.active_deadline_seconds,
                 "ttlSecondsAfterFinished": 7 * 86400,
                 "template": {"metadata": {"labels": spec.labels}, "spec": pod}},
    }


def render_manifests(spec: TrainingJobSpec) -> str:
    return yaml.safe_dump_all([kubernetes_configmap(spec), kubernetes_job(spec)], sort_keys=False)


# --------------------------------------------------------------------------- off-peak windows

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _days(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.lower().split(","):
        a, _, b = part.strip().partition("-")
        i = DAYS.index(a[:3])
        j = DAYS.index(b[:3]) if b else i
        k = i
        while True:
            out.add(k)
            if k == j:
                break
            k = (k + 1) % 7
    return out


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_off_peak(when: datetime, windows: list[str]) -> bool:
    """Whether ``when`` (UTC) falls in one of the windows, e.g. ``Mon-Fri 20:00-06:00`` (crosses midnight:
    Friday 20:00 to Saturday 06:00 is included) or ``Sat-Sun 00:00-24:00``."""
    when = when.astimezone(UTC) if when.tzinfo else when.replace(tzinfo=UTC)
    day, t = when.weekday(), when.hour * 60 + when.minute
    for w in windows:
        days_s, _, times = w.strip().partition(" ")
        start_s, _, end_s = times.partition("-")
        days, start, end = _days(days_s), _minutes(start_s), _minutes(end_s)
        if start < end:
            if day in days and start <= t < end:
                return True
        else:  # crosses midnight
            if day in days and t >= start or (day - 1) % 7 in days and t < end:
                return True
    return False
