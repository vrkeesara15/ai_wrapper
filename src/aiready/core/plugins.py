"""Plugin registry: the single customisation mechanism of the platform.

Every swappable component (model backend, vector store, connector, parser, chunker, guardrail detector,
scorer, tool, policy source...) is created from a *spec* in configuration:

    vector_store:
      type: qdrant            # a registered name
      url: http://qdrant:6333

    vector_store:
      class: acme_ai.stores:MyStore   # or any importable class - no fork of this repo needed
      option: value

Built-ins register themselves with ``@register("vector_store", "qdrant")``. Third-party packages can also
register through the ``aiready.plugins`` entry-point group, so a client-specific package is installed
next to the platform and switched on purely by configuration.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections import defaultdict
from importlib.metadata import entry_points
from typing import Any, Callable, TypeVar

from aiready.core.errors import ConfigError

log = logging.getLogger(__name__)
T = TypeVar("T")

_REGISTRY: dict[str, dict[str, Any]] = defaultdict(dict)
_ENTRYPOINTS_LOADED = False

# Modules imported lazily the first time a kind is requested, so that built-ins self-register
# without the caller importing them explicitly.
_BUILTIN_MODULES: dict[str, list[str]] = {
    "model_backend": ["aiready.models.backends"],
    "vector_store": ["aiready.search.stores"],
    "keyword_index": ["aiready.search.keyword"],
    "kv_store": ["aiready.core.stores"],
    "connector": ["aiready.ingestion.connectors"],
    "parser": ["aiready.ingestion.parsers"],
    "chunker": ["aiready.ingestion.chunking"],
    "detector": ["aiready.guardrails.detectors"],
    "scorer": ["aiready.evals.scorers"],
    "audit_sink": ["aiready.core.audit"],
    "embedder": ["aiready.embedding.encoders"],
    "reranker": ["aiready.rerank.scorers"],
    "queue": ["aiready.core.queues"],
    "session_validator": ["aiready.integration.identity_bridge"],
    "acl_resolver": ["aiready.integration.file_crawler"],
    "change_capture": ["aiready.integration.change_capture"],
    "landing_store": ["aiready.ingestion.landing"],
    "ocr": ["aiready.ingestion.parsers"],
    "policy_engine": ["aiready.governance.policy"],
    "policy_source": ["aiready.governance.policy"],
    "rag_planner": ["aiready.rag.planner"],
    "rag_retriever": ["aiready.rag.retriever"],
    "rag_reranker": ["aiready.rag.rerank_client"],
    "rag_context_builder": ["aiready.rag.context"],
    "rag_citation_mapper": ["aiready.rag.citations"],
    "rag_grounding": ["aiready.rag.grounding"],
    "tool": ["aiready.mcp.registry"],
    "agent_step": ["aiready.agents.steps"],
}


def register(kind: str, name: str) -> Callable[[T], T]:
    def deco(obj: T) -> T:
        _REGISTRY[kind][name] = obj
        return obj

    return deco


def _load_builtins(kind: str) -> None:
    for mod in _BUILTIN_MODULES.get(kind, []):
        try:
            importlib.import_module(mod)
        except ImportError as exc:  # optional dependency missing - built-ins that need it are skipped
            log.debug("builtin module %s not importable: %s", mod, exc)


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    try:
        for ep in entry_points(group="aiready.plugins"):
            try:
                ep.load()  # importing the module runs its @register decorators
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("plugin %s failed to load: %s", ep.name, exc)
    except Exception:  # pragma: no cover
        pass


def import_object(path: str) -> Any:
    """Import ``package.module:Name`` (or ``package.module.Name``)."""
    if ":" in path:
        mod_name, attr = path.split(":", 1)
    else:
        mod_name, _, attr = path.rpartition(".")
    try:
        obj: Any = importlib.import_module(mod_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot import '{path}': {exc}") from exc


def resolve(kind: str, name: str) -> Any:
    if name not in _REGISTRY[kind]:
        _load_builtins(kind)
    if name not in _REGISTRY[kind]:
        _load_entrypoints()
    if name not in _REGISTRY[kind]:
        raise ConfigError(
            f"unknown {kind} '{name}'. Registered: {sorted(_REGISTRY[kind])}. "
            f"Use 'class: module:Class' for a custom implementation."
        )
    return _REGISTRY[kind][name]


def available(kind: str) -> list[str]:
    _load_builtins(kind)
    _load_entrypoints()
    return sorted(_REGISTRY[kind])


def create(kind: str, spec: dict[str, Any] | str | None, /, **extra: Any) -> Any:
    """Build a component from a spec dict. ``extra`` supplies runtime dependencies (clients, stores)."""
    if spec is None:
        raise ConfigError(f"missing spec for {kind}")
    if isinstance(spec, str):
        spec = {"type": spec}
    params = dict(spec)
    if "class" in params:
        cls = import_object(params.pop("class"))
        params.pop("type", None)
    elif "type" in params:
        cls = resolve(kind, params.pop("type"))
    else:
        raise ConfigError(f"{kind} spec needs 'type' or 'class': {spec}")
    params.update(extra)
    return _instantiate(cls, params)


def _instantiate(cls: Any, params: dict[str, Any]) -> Any:
    """Pass only the keyword arguments the constructor accepts unless it takes **kwargs."""
    target = cls.__init__ if inspect.isclass(cls) else cls
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return cls(**params)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**params)
    accepted = {k: v for k, v in params.items() if k in sig.parameters}
    unknown = set(params) - set(accepted)
    if unknown:
        log.debug("ignoring unknown params for %s: %s", getattr(cls, "__name__", cls), sorted(unknown))
    return cls(**accepted)
