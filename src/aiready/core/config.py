"""Configuration store with hot reload.

Everything a client customises lives in a configuration directory (``AIREADY_CONFIG_DIR``, default
``./config``). All ``*.yaml`` files are deep-merged into one tree (dicts merge, lists concatenate). An
optional overlay directory (``AIREADY_CONFIG_OVERLAY``, e.g. ``config/overlays/prod``) is merged last;
in an overlay a key written as ``=key`` replaces instead of merging.

Values may reference the environment and secrets:
    url: ${QDRANT_URL:-http://localhost:6333}
    api_key: ${secret:llm_api_key}     # env AIREADY_SECRET_LLM_API_KEY or file /run/secrets/llm_api_key

The control plane changes routes, models, budgets and policies without a restart: the store polls file
modification times (or is told to reload by the admin API) and notifies subscribers with the new tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml
from pydantic import BaseModel

from aiready.core.errors import ConfigError

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)

_VAR = re.compile(r"\$\{([^}:]+?)(?::-(.*?))?\}")
SECRET_DIRS = [Path(p) for p in os.environ.get("AIREADY_SECRET_DIRS", "/run/secrets:/var/run/secrets/aiready").split(":")]


def resolve_secret(name: str) -> str | None:
    env_name = "AIREADY_SECRET_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
    if env_name in os.environ:
        return os.environ[env_name]
    for d in SECRET_DIRS:
        p = d / name
        if p.is_file():
            return p.read_text().strip()
    return None


def _substitute(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            key, default = m.group(1).strip(), m.group(2)
            if key.startswith("secret:"):
                found = resolve_secret(key[len("secret:"):])
            else:
                found = os.environ.get(key)
            if found is None:
                if default is None:
                    return ""
                return default
            return found

        full = _VAR.fullmatch(value)
        out = _VAR.sub(repl, value)
        if full:  # a value that is only a variable keeps YAML typing (numbers, booleans)
            try:
                return yaml.safe_load(out) if out != "" else None
            except yaml.YAMLError:
                return out
        return out
    if isinstance(value, dict):
        return {k: _substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v) for v in value]
    return value


def deep_merge(base: dict, other: dict, *, overlay: bool = False) -> dict:
    out = dict(base)
    for key, val in other.items():
        replace = overlay and isinstance(key, str) and key.startswith("=")
        k = key[1:] if replace else key
        if replace or k not in out:
            out[k] = copy.deepcopy(val)
        elif isinstance(out[k], dict) and isinstance(val, dict):
            out[k] = deep_merge(out[k], val, overlay=overlay)
        elif isinstance(out[k], list) and isinstance(val, list):
            out[k] = out[k] + copy.deepcopy(val)
        else:
            out[k] = copy.deepcopy(val)
    return out


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*.y*ml") if "overlays" not in p.relative_to(directory).parts)


def load_tree(config_dir: Path, overlay_dir: Path | None = None) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for f in _yaml_files(config_dir):
        try:
            data = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {f}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{f} must contain a mapping at the top level")
        tree = deep_merge(tree, data)
    if overlay_dir and overlay_dir.exists():
        for f in sorted(overlay_dir.rglob("*.y*ml")):
            tree = deep_merge(tree, yaml.safe_load(f.read_text()) or {}, overlay=True)
    return _substitute(tree)


class ConfigStore:
    """Thread-safe holder of the merged configuration tree with change notification."""

    def __init__(self, config_dir: str | Path | None = None, overlay_dir: str | Path | None = None,
                 tree: dict[str, Any] | None = None):
        self.config_dir = Path(config_dir or os.environ.get("AIREADY_CONFIG_DIR", "config")).resolve()
        ov = overlay_dir or os.environ.get("AIREADY_CONFIG_OVERLAY")
        self.overlay_dir = Path(ov).resolve() if ov else None
        self._lock = threading.RLock()
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._mtimes: dict[str, float] = {}
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._static = tree is not None
        self._tree: dict[str, Any] = tree if tree is not None else {}
        self._version = ""
        if tree is None:
            self.reload()
        else:
            self._version = self._hash(self._tree)

    # ------------------------------------------------------------------ access
    @property
    def tree(self) -> dict[str, Any]:
        with self._lock:
            return self._tree

    @property
    def version(self) -> str:
        return self._version

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.tree
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, path: str, model: type[M]) -> M:
        try:
            return model.model_validate(self.get(path, {}) or {})
        except Exception as exc:
            raise ConfigError(f"invalid configuration at '{path}': {exc}") from exc

    # ------------------------------------------------------------------ mutation
    def reload(self) -> bool:
        if self._static:
            return False
        tree = load_tree(self.config_dir, self.overlay_dir)
        return self._swap(tree)

    def replace(self, tree: dict[str, Any]) -> bool:
        """Install a tree directly (admin API / tests)."""
        return self._swap(tree)

    def _swap(self, tree: dict[str, Any]) -> bool:
        version = self._hash(tree)
        with self._lock:
            if version == self._version:
                return False
            self._tree, self._version = tree, version
            subs = list(self._subscribers)
        log.info("configuration version %s loaded", version[:12])
        for cb in subs:
            try:
                cb(tree)
            except Exception:  # a broken subscriber must not block the others
                log.exception("config subscriber failed")
        return True

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    # ------------------------------------------------------------------ watching
    def _snapshot_mtimes(self) -> dict[str, float]:
        files = _yaml_files(self.config_dir)
        if self.overlay_dir:
            files += sorted(self.overlay_dir.rglob("*.y*ml"))
        return {str(f): f.stat().st_mtime for f in files if f.exists()}

    def watch(self, interval: float = 2.0) -> None:
        if self._static or self._watcher:
            return
        self._mtimes = self._snapshot_mtimes()

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    now = self._snapshot_mtimes()
                    if now != self._mtimes:
                        self._mtimes = now
                        self.reload()
                except ConfigError as exc:
                    log.error("config reload rejected, keeping previous version: %s", exc)
                except Exception:
                    log.exception("config watcher error")

        self._watcher = threading.Thread(target=loop, name="config-watcher", daemon=True)
        self._watcher.start()

    def stop(self) -> None:
        self._stop.set()

    @staticmethod
    def _hash(tree: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(tree, sort_keys=True, default=str).encode()).hexdigest()


_default: ConfigStore | None = None


def get_config() -> ConfigStore:
    global _default
    if _default is None:
        _default = ConfigStore()
    return _default


def set_config(store: ConfigStore) -> None:
    global _default
    _default = store
