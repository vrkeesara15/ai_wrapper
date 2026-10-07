"""Model catalog: the logical models the gateway may route to (configuration section ``models:``).

A *logical model* is what routes, budgets and policies talk about ("chat-mid", "domain-claims",
"escalation"). It points at a *pool* of identical replicas and carries what governance needs to decide whether
it may be used: size tier, category (private / public), the data classes allowed to reach it, context length,
licence, cost and capabilities. Changing which server answers is a configuration change, never a code change.

    models:
      pools:
        mid-llm:
          model: Qwen/Qwen2.5-32B-Instruct          # upstream model id served by every replica
          replicas: [{type: openai, base_url: http://vllm-mid-0:8000/v1}, ...]
      catalog:
        chat-mid: {pool: mid-llm, tier: mid, category: private-original, capabilities: [chat, tools, json]}

The catalog reloads whenever the configuration changes (file watcher or the admin API).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from aiready.core.config import ConfigStore, get_config
from aiready.core.errors import ConfigError, NotFound
from aiready.core.types import DataClass

log = logging.getLogger(__name__)


class Tier(str, Enum):
    """Working size tiers: small < 10B parameters, mid 10B-40B, large > 40B (or unpublished)."""

    SMALL = "small"
    MID = "mid"
    LARGE = "large"

    @property
    def rank(self) -> int:
        return {"small": 0, "mid": 1, "large": 2}[self.value]


class Category(str, Enum):
    """Private models run inside the client's boundary; public ones send prompts outside it."""

    PRIVATE_ORIGINAL = "private-original"   # published open weights, run as released on the client's GPUs
    PRIVATE_TUNED = "private-tuned"         # adapter / fine-tune trained on the client's data
    PRIVATE_ENDPOINT = "private-endpoint"   # vendor model served inside the client's own cloud tenancy
    PUBLIC_PROPRIETARY = "public-proprietary"  # vendor API, weights not published
    PUBLIC_HOSTED_OPEN = "public-hosted-open"  # open model run by a hosting provider

    @property
    def is_public(self) -> bool:
        return self.value.startswith("public")


Capability = Literal["chat", "tools", "json", "embedding", "rerank", "guard", "vision"]


class ModelSpec(BaseModel):
    """One logical model. Mirrors a row of the model registry slide."""

    name: str = ""
    pool: str
    tier: Tier = Tier.MID
    category: Category = Category.PRIVATE_ORIGINAL
    # Data classes allowed to reach this model. Default: everything for private models, nothing for public
    # ones - a public route needs an explicit list here *and* an egress policy allowing the request's class.
    data_classes: list[DataClass] | None = None
    context_length: int = Field(8192, ge=256)
    max_output_tokens: int = Field(2048, ge=1)
    licence: str = ""
    cost_per_1k_input: float = Field(0.0, ge=0)
    cost_per_1k_output: float = Field(0.0, ge=0)
    base_model: str | None = None      # e.g. the open-weight original a tuned adapter sits on
    adapter: str | None = None         # LoRA adapter name served by the pool (vLLM --lora-modules)
    upstream_model: str | None = None  # model id sent upstream; default: adapter, pool model, then name
    capabilities: list[Capability] = Field(default_factory=lambda: ["chat"])
    version: str = "1"
    owner: str | None = None
    approved: bool = True              # registry approval; unapproved models are never routed to
    enabled: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _defaults(self) -> ModelSpec:
        if self.data_classes is None:
            self.data_classes = [] if self.category.is_public else list(DataClass)
        return self

    @property
    def is_public(self) -> bool:
        return self.category.is_public

    def allows(self, data_class: DataClass) -> bool:
        return data_class in (self.data_classes or [])

    def has(self, *caps: str) -> bool:
        return all(c in self.capabilities for c in caps)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1000 * self.cost_per_1k_input + completion_tokens / 1000 * self.cost_per_1k_output

    def public_view(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"upstream_model", "adapter"})


class PoolSpec(BaseModel):
    """A named pool of interchangeable replicas serving the same model."""

    model: str | None = None                       # upstream model id served by the replicas
    replicas: list[dict[str, Any]] = Field(default_factory=lambda: [{"type": "mock"}])
    eject_after_failures: int = Field(3, ge=1)    # circuit breaker: consecutive failures before ejection
    cooldown_seconds: float = Field(30.0, ge=0)   # how long an ejected replica rests before a trial request
    max_in_flight_per_replica: int = Field(0, ge=0)  # 0 = unlimited; above it requests queue (queue depth)
    queue_timeout_seconds: float = Field(30.0, ge=0)
    ewma_alpha: float = Field(0.3, gt=0, le=1)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str).encode()).hexdigest()


class ModelsConfig(BaseModel):
    pools: dict[str, PoolSpec] = Field(default_factory=dict)
    catalog: dict[str, ModelSpec] = Field(default_factory=dict)


class ModelCatalog:
    """Immutable snapshot of the catalog for one configuration version."""

    def __init__(self, cfg: ModelsConfig, version: str = ""):
        self.version = version
        self.pools = dict(cfg.pools)
        self.models: dict[str, ModelSpec] = {}
        for name, spec in cfg.catalog.items():
            spec = spec.model_copy(update={"name": name})
            if spec.pool not in self.pools:
                raise ConfigError(f"model '{name}' refers to unknown pool '{spec.pool}'")
            self.models[name] = spec

    @classmethod
    def from_tree(cls, tree: dict[str, Any], version: str = "") -> ModelCatalog:
        try:
            cfg = ModelsConfig.model_validate(tree.get("models") or {})
        except Exception as exc:
            raise ConfigError(f"invalid 'models' configuration: {exc}") from exc
        return cls(cfg, version)

    def __contains__(self, name: str) -> bool:
        return name in self.models

    def get(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError:
            raise NotFound(f"unknown model '{name}'") from None

    def find(self, name: str) -> ModelSpec | None:
        return self.models.get(name)

    def list(self, capability: str | None = None, *, usable_only: bool = True) -> list[ModelSpec]:
        out = [m for m in self.models.values()
               if (capability is None or capability in m.capabilities)
               and (not usable_only or (m.enabled and m.approved))]
        return sorted(out, key=lambda m: (m.tier.rank, m.cost_per_1k_input + m.cost_per_1k_output, m.name))

    def upstream_model(self, spec: ModelSpec) -> str:
        pool = self.pools.get(spec.pool)
        return spec.upstream_model or spec.adapter or (pool.model if pool else None) or spec.name


class LiveCatalog:
    """Keeps a ``ModelCatalog`` in step with the configuration store.

    Subscribes to the store for push updates and also re-checks the store version on access, so a store
    replaced with ``set_config`` (tests, embedding) or changed by the admin API is picked up without restart.
    An invalid new configuration is rejected and the previous catalog stays in service.
    """

    def __init__(self, store: ConfigStore | None = None):
        self._fixed_store = store
        self._lock = threading.RLock()
        self._catalog: ModelCatalog | None = None
        self._listeners: list[Callable[[ModelCatalog], None]] = []
        self._subscribed: set[int] = set()
        self._rejected_version: str | None = None

    @property
    def store(self) -> ConfigStore:
        return self._fixed_store or get_config()

    def on_change(self, cb: Callable[[ModelCatalog], None]) -> None:
        self._listeners.append(cb)

    def _subscribe(self, store: ConfigStore) -> None:
        if id(store) in self._subscribed:
            return
        self._subscribed.add(id(store))
        store.subscribe(lambda tree: self._load(tree, store.version))

    def _load(self, tree: dict[str, Any], version: str) -> ModelCatalog:
        with self._lock:
            if self._catalog is not None and self._catalog.version == version:
                return self._catalog
            try:
                cat = ModelCatalog.from_tree(tree, version)
            except ConfigError:
                if self._catalog is None:
                    raise
                log.exception("model catalog reload rejected; keeping version %s", self._catalog.version[:12])
                self._rejected_version = version
                return self._catalog
            self._catalog = cat
        for cb in list(self._listeners):
            try:
                cb(cat)
            except Exception:
                log.exception("catalog listener failed")
        return cat

    def current(self) -> ModelCatalog:
        store = self.store
        self._subscribe(store)
        cat = self._catalog
        if cat is None or (cat.version != store.version and store.version != self._rejected_version):
            cat = self._load(store.tree, store.version)
        return cat
