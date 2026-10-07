"""Versioned prompt templates.

Prompts are configuration, not code: a client tunes wording per application without a release, and every
answer records which prompt version produced it (the version is part of the semantic-cache scope, so changing
a prompt never serves answers written under the old one).

Lookup order (``source: auto``):

1. the governance prompt registry (``connect("registry").get_prompt(name, version)``) when it is deployed;
2. the ``prompts:`` section of the merged configuration tree - that is ``config/prompts/*.yaml``, hot-reloaded
   with the rest of the configuration;
3. ``*.yaml`` files in an explicit ``prompts_dir`` and, last, the defaults shipped in the repository.

File layout (each file holds a top-level ``prompts:`` mapping so it merges cleanly with other config)::

    prompts:
      rag.answer:
        active: v2
        versions:
          v1: {system: "...", user: "{question}"}
          v2: {system: "...", user: "{question}"}

Templates use ``{name}`` placeholders; unknown placeholders are left untouched so braces in examples
(JSON, SQL) need no escaping.
"""

from __future__ import annotations

import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from aiready.core.config import get_config
from aiready.core.errors import ConfigError

log = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# repo/config/prompts - the shipped defaults (src/aiready/rag/prompts.py -> parents[3] is the repo root)
_REPO_PROMPTS = Path(__file__).resolve().parents[3] / "config" / "prompts"


def render(template: str, **values: Any) -> str:
    """Replace ``{name}`` for the given names only."""

    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    return _PLACEHOLDER.sub(repl, template or "")


class PromptTemplate(BaseModel):
    name: str
    version: str
    system: str = ""
    user: str = "{question}"
    params: dict[str, Any] = {}  # optional model parameters suggested by the prompt author (temperature...)

    @property
    def ref(self) -> str:
        """``name@version`` - recorded in audit records and used in cache scopes."""
        return f"{self.name}@{self.version}"

    def render_system(self, **values: Any) -> str:
        return render(self.system, **values)

    def render_user(self, **values: Any) -> str:
        return render(self.user, **values)


def _from_entry(name: str, entry: Any, version: str | None) -> PromptTemplate | None:
    """Accept both the versioned layout and a flat ``{version, system, user}`` mapping."""
    if not isinstance(entry, dict):
        return None
    versions = entry.get("versions")
    if isinstance(versions, dict) and versions:
        v = version or entry.get("active") or sorted(versions)[-1]
        body = versions.get(v)
        if not isinstance(body, dict):
            return None
        return PromptTemplate(name=name, version=str(v), **{k: body[k] for k in ("system", "user", "params")
                                                             if k in body})
    if "system" in entry or "user" in entry:
        v = str(entry.get("version", "v1"))
        if version and version != v:
            return None
        return PromptTemplate(name=name, version=v, **{k: entry[k] for k in ("system", "user", "params")
                                                        if k in entry})
    return None


class PromptStore:
    """Resolves prompt templates by name and optional version."""

    def __init__(self, source: Literal["auto", "registry", "files"] = "auto", prompts_dir: str | None = None,
                 registry_retry_seconds: float = 60.0):
        self.source = source
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.registry_retry_seconds = registry_retry_seconds
        self._registry_down_until = 0.0
        self._file_cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get(self, name: str, version: str | None = None) -> PromptTemplate:
        if self.source in ("auto", "registry"):
            tpl = await self._from_registry(name, version)
            if tpl is not None:
                return tpl
            if self.source == "registry":
                raise ConfigError(f"prompt '{name}' (version {version or 'active'}) not found in the prompt registry")
        tpl = self._from_config(name, version)
        if tpl is not None:
            return tpl
        raise ConfigError(
            f"prompt '{name}' (version {version or 'active'}) not found: add it under 'prompts:' in "
            f"config/prompts/*.yaml or register it in the prompt registry"
        )

    # ------------------------------------------------------------------ sources
    async def _from_registry(self, name: str, version: str | None) -> PromptTemplate | None:
        """Governance prompt registry: ``get_prompt(name, version: int | None, environment)``.

        The registry stores one ``template`` per version; it is used as the system message and the user message
        is the question, unless the version's metadata carries ``system``/``user`` explicitly. A prompt that is
        not registered (NotFound) falls through to the files; an unreachable registry is retried later.
        """
        if time.monotonic() < self._registry_down_until:
            return None
        from aiready.core.errors import NotFound
        from aiready.core.wiring import connect

        num = int(version[1:]) if version and version[1:].isdigit() and version[0] in "vV" else (
            int(version) if version and version.isdigit() else None)
        env = get_config().get("platform.environment")
        try:
            reg = connect("registry")
            fn = getattr(reg, "get_prompt", None)
            if fn is None:
                raise AttributeError("registry has no get_prompt")
            try:
                res = fn(name, num, env)
            except TypeError:
                res = fn(name, num)
            if inspect.isawaitable(res):
                res = await res
        except NotFound:
            return None
        except Exception as exc:  # the registry is optional; files are the fallback
            log.debug("prompt registry unavailable (%s); using configuration files", exc)
            self._registry_down_until = time.monotonic() + self.registry_retry_seconds
            return None
        if res is None:
            return None
        data = res.model_dump() if hasattr(res, "model_dump") else res
        if not isinstance(data, dict):
            return None
        if "versions" in data:
            return _from_entry(name, data, version)
        meta = data.get("metadata") or {}
        try:
            return PromptTemplate(name=name, version=f"r{data.get('version', num or 1)}",
                                  system=meta.get("system") or data.get("system") or data.get("template", ""),
                                  user=meta.get("user") or data.get("user") or "{question}",
                                  params=meta.get("params") or data.get("params") or {})
        except Exception:
            return None

    def _from_config(self, name: str, version: str | None) -> PromptTemplate | None:
        tree = get_config().get("prompts", {}) or {}
        tpl = _from_entry(name, tree.get(name), version)
        if tpl is not None:
            return tpl
        for directory in [d for d in (self.prompts_dir, _REPO_PROMPTS) if d is not None]:
            tpl = _from_entry(name, self._dir_prompts(directory).get(name), version)
            if tpl is not None:
                return tpl
        return None

    def _dir_prompts(self, directory: Path) -> dict[str, Any]:
        """All prompts in a directory, re-read only when a file changes."""
        if not directory.is_dir():
            return {}
        files = sorted(directory.glob("*.y*ml"))
        stamp = sum(f.stat().st_mtime for f in files) + len(files)
        cached = self._file_cache.get(str(directory))
        if cached and cached[0] == stamp:
            return cached[1]
        merged: dict[str, Any] = {}
        for f in files:
            try:
                data = yaml.safe_load(f.read_text()) or {}
            except yaml.YAMLError as exc:
                log.error("invalid prompt file %s: %s", f, exc)
                continue
            merged.update((data.get("prompts") or {}) if isinstance(data, dict) else {})
        self._file_cache[str(directory)] = (stamp, merged)
        return merged
