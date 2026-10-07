"""Gateway control plane: routes, models, pools, budgets and rate limits changed live, without a restart.

Changes made through the admin API are:

1. validated as a whole (the new gateway section must parse and every model must point at an existing pool),
2. installed in the configuration store with ``ConfigStore.replace`` - subscribers (catalog, router, gateway)
   pick them up on the next request,
3. persisted in the state store (``gateway.admin`` namespace) with a revision number, so every data-plane
   replica pulls them (``sync``) and a reload from files does not lose them; optionally also written to a YAML
   overlay file for GitOps-style review,
4. audited as ``config.change`` with the before and after values.

Only the entries changed through the admin API are persisted, in their raw form, so ``${secret:...}``
references stay references and resolved secrets never reach the state store or the audit log.
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from aiready.core.audit import get_audit
from aiready.core.config import ConfigStore, _substitute, get_config
from aiready.core.errors import NotFound, ValidationFailed
from aiready.core.identity import Principal
from aiready.core.stores import DocStore, get_docstore
from aiready.gateway.config import BudgetRule, GatewayConfig, RateLimitRule, RouteConfig
from aiready.models.catalog import ModelCatalog, ModelSpec, PoolSpec

NAMESPACE = "gateway.admin"
HISTORY = "gateway.admin.history"

# kind -> (top-level section, key, model)
KINDS: dict[str, tuple[str, str, type[BaseModel]]] = {
    "routes": ("gateway", "routes", RouteConfig),
    "budgets": ("gateway", "budgets", BudgetRule),
    "rate_limits": ("gateway", "rate_limits", RateLimitRule),
    "models": ("models", "catalog", ModelSpec),
    "pools": ("models", "pools", PoolSpec),
}
_SECRET_KEYS = {"api_key", "apikey", "authorization", "x-api-key", "proxy-authorization", "password", "token",
                "access_token", "secret", "client_secret"}
_lock = threading.RLock()


def redact(value: Any) -> Any:
    """Hide credentials in pool specs before returning or auditing them."""
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in _SECRET_KEYS and isinstance(v, str) and v
                    and not v.startswith("${secret:") else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class ControlPlane:
    def __init__(self, store: ConfigStore | None = None, docstore: DocStore | None = None):
        self._store = store
        self._docstore = docstore

    @property
    def store(self) -> ConfigStore:
        return self._store or get_config()

    @property
    def docstore(self) -> DocStore:
        return self._docstore or get_docstore()

    # ------------------------------------------------------------------ read
    @staticmethod
    def _kind(kind: str) -> tuple[str, str, type[BaseModel]]:
        if kind not in KINDS:
            raise NotFound(f"unknown configuration kind '{kind}'; use one of {sorted(KINDS)}")
        return KINDS[kind]

    def list(self, kind: str) -> dict[str, Any]:
        top, key, _ = self._kind(kind)
        return redact(copy.deepcopy((self.store.get(top) or {}).get(key) or {}))

    def get(self, kind: str, name: str) -> dict[str, Any]:
        items = self.list(kind)
        if name not in items:
            raise NotFound(f"{kind[:-1]} '{name}' not found")
        return items[name]

    def _snapshot(self) -> dict[str, Any]:
        return self.docstore.get(NAMESPACE, "overrides", {"revision": 0, "kinds": {}})

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.docstore.values(HISTORY)
        return sorted(rows, key=lambda r: r["ts"], reverse=True)[:limit]

    # ------------------------------------------------------------------ write
    def upsert(self, kind: str, name: str, body: dict[str, Any], principal: Principal) -> dict[str, Any]:
        _, _, model = self._kind(kind)
        try:
            model.model_validate(_substitute(copy.deepcopy(body)))
        except Exception as exc:
            raise ValidationFailed(f"invalid {kind[:-1]} '{name}': {exc}") from exc
        return self._change(kind, name, body, principal)

    def delete(self, kind: str, name: str, principal: Principal) -> dict[str, Any]:
        top, key, _ = self._kind(kind)
        if name not in ((self.store.get(top) or {}).get(key) or {}):
            raise NotFound(f"{kind[:-1]} '{name}' not found")
        return self._change(kind, name, None, principal)

    def _change(self, kind: str, name: str, body: dict[str, Any] | None, principal: Principal) -> dict[str, Any]:
        top, key, _ = self._kind(kind)
        with _lock:
            snap = self._snapshot()
            entry = snap["kinds"].setdefault(kind, {"set": {}, "deleted": []})
            if body is None:
                entry["set"].pop(name, None)
                if name not in entry["deleted"]:
                    entry["deleted"].append(name)
            else:
                entry["set"][name] = body
                entry["deleted"] = [n for n in entry["deleted"] if n != name]
            before = copy.deepcopy(((self.store.get(top) or {}).get(key) or {}).get(name))
            tree = self._apply_snapshot(self.store.tree, snap)
            self._validate_tree(tree)
            snap["revision"] = int(snap.get("revision", 0)) + 1
            snap["updated_at"] = time.time()
            snap["updated_by"] = principal.sub
            self.docstore.put(NAMESPACE, "overrides", snap)
            self.store.replace(tree)
            self.store._aiready_admin_revision = snap["revision"]  # type: ignore[attr-defined]
            self._write_overlay(snap)
            after = copy.deepcopy(((tree.get(top) or {}).get(key) or {}).get(name))
            change = {"id": uuid.uuid4().hex, "ts": time.time(), "kind": kind, "name": name,
                      "action": "delete" if body is None else ("create" if before is None else "update"),
                      "revision": snap["revision"], "config_version": self.store.version, "by": principal.sub,
                      "before": redact(before), "after": redact(after)}
            self.docstore.put(HISTORY, f"{change['ts']:.6f}-{change['id']}", change)
        get_audit().for_principal(principal, "config.change", action=f"gateway.{kind}.{change['action']}",
                                  resource=f"{kind}/{name}", decision="applied", revision=snap["revision"],
                                  config_version=self.store.version, before=change["before"], after=change["after"])
        return change

    @staticmethod
    def _apply_snapshot(tree: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(tree)
        for kind, entry in (snap.get("kinds") or {}).items():
            if kind not in KINDS:
                continue
            top, key, _ = KINDS[kind]
            section = out.setdefault(top, {}) or {}
            out[top] = section
            items = dict(section.get(key) or {})
            for n in entry.get("deleted", []):
                items.pop(n, None)
            for n, body in (entry.get("set") or {}).items():
                items[n] = _substitute(copy.deepcopy(body))
            section[key] = items
        return out

    @staticmethod
    def _validate_tree(tree: dict[str, Any]) -> None:
        try:
            GatewayConfig.model_validate(tree.get("gateway") or {})
            ModelCatalog.from_tree(tree)
        except Exception as exc:
            raise ValidationFailed(f"change rejected: {exc}") from exc

    def _write_overlay(self, snap: dict[str, Any]) -> None:
        path = (self.store.get("gateway.admin") or {}).get("overlay_file")
        if not path:
            return
        doc: dict[str, Any] = {}
        for kind, entry in (snap.get("kinds") or {}).items():
            top, key, _ = KINDS[kind]
            sets = {f"={n}": body for n, body in (entry.get("set") or {}).items()}
            # an overlay cannot delete a key: deleted entries are merged as disabled (pools have no switch;
            # their deletion lives in the state store only)
            disabled = {n: {"enabled": False} for n in entry.get("deleted", []) if kind != "pools"}
            if sets or disabled:
                doc.setdefault(top, {})[key] = {**disabled, **sets}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text("# Written by the AI-Ready gateway admin API. Review changes in the audit log.\n"
                       + yaml.safe_dump(doc, sort_keys=True))
        tmp.replace(p)

    # ------------------------------------------------------------------ replicas
    def sync(self) -> bool:
        """Install the persisted admin changes into this process's configuration if they are missing or newer."""
        snap = self.docstore.get(NAMESPACE, "overrides")
        if not snap or not snap.get("kinds"):
            return False
        store = self.store
        applied = getattr(store, "_aiready_admin_revision", None)
        tree = self._apply_snapshot(store.tree, snap)
        if applied == snap.get("revision") and tree == store.tree:
            return False
        with _lock:
            changed = store.replace(tree)
            store._aiready_admin_revision = snap.get("revision")  # type: ignore[attr-defined]
        return changed
