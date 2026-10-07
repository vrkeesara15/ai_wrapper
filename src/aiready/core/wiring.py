"""Service wiring: in-process or over HTTP, chosen by configuration.

    services:
      gateway:   {url: http://gateway:8080}     # -> HTTP client
      rag:       {}                             # -> in-process service (pilot / tests)

``connect("gateway")`` returns an object implementing the matching Protocol in ``core.interfaces``.
Factories register themselves with ``@provider(name, "local"|"http")`` in each package's ``__init__``.
Instances are cached per process; ``reset()`` clears them (tests, config change).
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable

from aiready.core.config import get_config
from aiready.core.errors import ConfigError

_PROVIDERS: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = {}
_INSTANCES: dict[str, Any] = {}
_OVERRIDES: dict[str, Any] = {}
_LOCK = threading.RLock()

# where each service's providers live
_PACKAGES = {
    "gateway": "aiready.gateway",
    "guardrails": "aiready.guardrails",
    "policy": "aiready.governance",
    "registry": "aiready.governance",
    "search": "aiready.search",
    "embedding": "aiready.embedding",
    "rerank": "aiready.rerank",
    "rag": "aiready.rag",
    "memory": "aiready.memory",
    "query": "aiready.query",
    "tools": "aiready.mcp",
    "agents": "aiready.agents",
    "observability": "aiready.observability",
}


def provider(name: str, kind: str) -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
    def deco(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        _PROVIDERS[(name, kind)] = fn
        return fn

    return deco


def override(name: str, instance: Any) -> None:
    """Inject an implementation (tests, embedding the platform in another process)."""
    with _LOCK:
        _OVERRIDES[name] = instance


def reset() -> None:
    with _LOCK:
        _INSTANCES.clear()
        _OVERRIDES.clear()


def connect(name: str) -> Any:
    with _LOCK:
        if name in _OVERRIDES:
            return _OVERRIDES[name]
        if name in _INSTANCES:
            return _INSTANCES[name]
        cfg = get_config().get(f"services.{name}", {}) or {}
        kind = "http" if cfg.get("url") else "local"
        if (name, kind) not in _PROVIDERS and name in _PACKAGES:
            importlib.import_module(_PACKAGES[name])
        factory = _PROVIDERS.get((name, kind))
        if factory is None:
            raise ConfigError(f"no '{kind}' provider for service '{name}'")
        inst = factory(cfg)
        _INSTANCES[name] = inst
        return inst


def local(name: str) -> Any:
    """The in-process implementation of ``name``, shared with ``connect(name)`` when that resolves locally.

    Each service's ``api.create_app()`` serves ``local(<own name>)``. This keeps one instance per process
    (in the all-in-one pilot, the search API and the RAG service see the same index) and never resolves to an
    HTTP client pointing back at the pod itself when ``services.<own name>.url`` is configured.
    """
    with _LOCK:
        if name in _OVERRIDES:
            return _OVERRIDES[name]
        cfg = get_config().get(f"services.{name}", {}) or {}
        if not cfg.get("url"):
            return connect(name)
        key = f"{name}:local"
        if key not in _INSTANCES:
            if (name, "local") not in _PROVIDERS and name in _PACKAGES:
                importlib.import_module(_PACKAGES[name])
            factory = _PROVIDERS.get((name, "local"))
            if factory is None:
                raise ConfigError(f"no 'local' provider for service '{name}'")
            _INSTANCES[key] = factory(cfg)
        return _INSTANCES[key]


def service_token(service: str) -> str | None:
    """This service's own token for X-Service-Token, when a signing key is available."""
    try:
        from aiready.core.identity import get_token_service

        return get_token_service().issue_service_token(service)
    except Exception:
        return None
