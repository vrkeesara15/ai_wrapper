"""Structured-data query service (slide "Structured data access"), called as an MCP tool.

    1 Route       a defined metric, or a new question? (an approved metric is always preferred)
    2a Writer     the model drafts SQL from the schema catalog the caller may see; retried once with the
                  validator's errors
    2b Metric     fixed, approved SQL with bound parameters
    3 Validate    read-only, allowed tables/columns, row filters, row and cost limits
    4 Execute     on the read replica or warehouse - never the live database - as the user, with a timeout
    5 Check       empty, too large, out of the expected range
    6 Format      rows plus a short summary
    7 Answer      with the SQL shown for audit

Governance: policy ``sql.execute`` (its obligations may add row filters or lower the row cap) and an audit
record ``query.execute`` with the SQL and the row count; rejected SQL is audited as ``query.rejected``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from aiready.core.audit import get_audit
from aiready.core.config import get_config
from aiready.core.errors import (
    ApprovalRequired,
    NotFound,
    PermissionDenied,
    PolicyViolation,
    ValidationFailed,
)
from aiready.core.identity import Principal
from aiready.core.interfaces import PolicyInput, QueryRequest, QueryResult
from aiready.core.telemetry import metrics, span
from aiready.core.types import ChatMessage, ChatRequest
from aiready.core.wiring import connect
from aiready.query.catalog import SchemaCatalog
from aiready.query.config import Metric, QueryConfig, SemanticLayerConfig
from aiready.query.executor import ReplicaExecutor
from aiready.query.semantic_layer import SemanticLayer
from aiready.query.validator import Analysis, SqlRejected, SqlValidator, dialect_for_url, to_jsonable
from aiready.rag.prompts import PromptStore

log = logging.getLogger(__name__)
SERVICE = "query"

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_START = re.compile(r"\b(with|select)\b", re.IGNORECASE)


def extract_sql(text: str) -> str:
    """The SQL in a model reply: a fenced block, else everything from the first SELECT/WITH."""
    m = _FENCE.search(text or "")
    if m:
        return m.group(1).strip()
    s = _START.search(text or "")
    return (text[s.start():] if s else text or "").strip()


def _principal_facts(p: Principal) -> dict[str, Any]:
    return p.model_dump(include={"sub", "tenant", "app", "roles", "groups", "scopes", "service", "actor"})


class QueryServiceImpl:
    """In-process implementation of ``core.interfaces.QueryService``."""

    def __init__(self, config: QueryConfig | None = None, semantic: SemanticLayerConfig | None = None, *,
                 executor: ReplicaExecutor | None = None, prompts: PromptStore | None = None):
        store = get_config()
        self.cfg = config or store.section("query", QueryConfig)
        self.semantic = SemanticLayer(semantic or store.section("semantic_layer", SemanticLayerConfig))
        self.dialect = self.cfg.dialect or dialect_for_url(self.cfg.executor.replica_url)
        self.validator = SqlValidator(self.cfg.validator, self.dialect)
        self.executor = executor or ReplicaExecutor(self.cfg.executor, self.dialect)
        self.catalog = SchemaCatalog(self.cfg.catalog)
        self.prompts = prompts or PromptStore(self.cfg.prompts_source)

    @classmethod
    def from_config(cls) -> QueryServiceImpl:
        return cls()

    # ================================================================== public
    async def query(self, req: QueryRequest, principal: Principal) -> QueryResult:
        t0 = time.perf_counter()
        identity = {"tenant": principal.tenant, "user": principal.sub, "app": principal.app}
        reserved = set(self.cfg.validator.reserved_params)
        warnings: list[str] = []
        with span("query.query", has_metric=bool(req.metric)) as sp:
            # 1 route
            with metrics.time_stage(SERVICE, "route"):
                metric = self._route(req, principal)
            sp.set(route="metric" if metric else "generated", metric=metric.name if metric else None)
            access, role_filters = self._access(principal)

            if metric is not None:
                # 2b metric tool: fixed approved SQL, parameters validated and bound
                params = self.semantic.bind_params(metric, req.params, reserved)
                sql_text, source, attempts = metric.sql, "metric", 0
                with metrics.time_stage(SERVICE, "validate"):
                    tree, analysis = self._analyze(sql_text, None, set(metric.params), principal, source)
            else:
                # 2a query writer
                if req.params:
                    warnings.append("parameters are only used by approved metrics and were ignored")
                params, source = {}, "generated"
                if not access:
                    raise PermissionDenied("no tables are available to your roles for generated queries")
                tree, analysis, sql_text, attempts = await self._write_valid(req.question or "", access, principal)

            # policy sql.execute (sees the tables the statement touches)
            obligations = await self._policy(principal, analysis, source, metric)
            filters = self._row_filters(analysis, role_filters, obligations)
            max_rows = min(req.max_rows, self.cfg.validator.max_rows)
            cap = obligations.get("max_rows")
            if isinstance(cap, int) and cap > 0:
                max_rows = min(max_rows, cap)

            # 3 validate: row filters + limit
            with metrics.time_stage(SERVICE, "validate"):
                validated = self.validator.enforce(tree, analysis, row_filters=filters, max_rows=max_rows)
            warnings += validated.warnings
            await self._check_cost(validated.sql, params, identity, principal)

            # 4 execute on the replica, as the user, with a timeout
            with span("query.execute", tables=validated.tables), metrics.time_stage(SERVICE, "execute"):
                cols, rows = await self.executor.run(validated.sql, params, identity, max_fetch=validated.limit)

            # 5 check the result
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
            if not rows:
                warnings.append("the query returned no rows")
            if truncated:
                warnings.append(f"result truncated to {max_rows} rows")
            if metric is not None:
                warnings += self._range_warnings(metric, cols, rows)

            # 6 format
            data = [[to_jsonable(v) for v in r] for r in rows]
            summary = self._summary(cols, data, truncated, metric)
            ms = round((time.perf_counter() - t0) * 1000, 2)
            sp.set(rows=len(data), truncated=truncated, ms=ms)

            # 7 answer, with the SQL shown for audit
            get_audit().for_principal(
                principal, "query.execute", action="sql.execute", resource=",".join(validated.tables),
                decision="executed", sql=validated.sql, row_count=len(data), truncated=truncated, source=source,
                metric=metric.name if metric else None, question=(req.question or "")[:2000],
                params=sorted(params), row_filters=validated.row_filters, attempts=attempts, warnings=warnings,
                duration_ms=ms)
            return QueryResult(sql=validated.sql, columns=cols, rows=data, row_count=len(data), truncated=truncated,
                               summary=summary, source=source, warnings=warnings)  # type: ignore[arg-type]

    def list_metrics(self, principal: Principal) -> list[dict[str, Any]]:
        """Approved metrics this caller may use (for tool discovery and UIs)."""
        return [{"name": m.name, "description": m.description, "synonyms": m.synonyms,
                 "params": {k: v.model_dump() for k, v in m.params.items()}} for m in self.semantic.visible(principal)]

    async def describe_catalog(self, principal: Principal) -> str:
        access, _ = self._access(principal)
        await self._ensure_catalog()
        return self.catalog.describe(access)

    # ================================================================== routing
    def _route(self, req: QueryRequest, principal: Principal) -> Metric | None:
        if req.metric:
            m = self.semantic.get(req.metric)
            if m is None:
                raise NotFound(f"unknown metric '{req.metric}'")
            if not self.semantic.allowed(m, principal):
                get_audit().for_principal(principal, "query.rejected", action="sql.execute", resource=m.name,
                                          decision="deny", reason="metric role")
                raise PermissionDenied(f"metric '{m.name}' is not available to your roles")
            return m
        if not (req.question or "").strip():
            raise ValidationFailed("a question or an approved metric name is required")
        match = self.semantic.match(req.question or "", principal)
        if match is not None:
            m, score = match
            if score >= self.cfg.routing.metric_match_threshold and not self.semantic.missing_required(m, req.params):
                return m
        if not self.cfg.allow_generated_sql:
            raise ValidationFailed("only approved metrics can be queried; no metric matches this question",
                                   details={"metrics": [m.name for m in self.semantic.visible(principal)]})
        return None

    # ================================================================== access
    def _access(self, principal: Principal) -> tuple[dict[str, set[str] | None], dict[str, list[str]]]:
        """Union of the caller's role grants -> allowed columns per table (None = all) and role row filters."""
        access: dict[str, set[str] | None] = {}
        role_filters: dict[str, list[str | None]] = {}
        for role in principal.roles:
            ra = self.cfg.access.roles.get(role)
            if ra is None:
                continue
            for table, ta in ra.tables.items():
                t = table.lower()
                cols = None if ta.columns == "*" else {c.lower() for c in ta.columns}
                if t not in access:
                    access[t] = cols
                else:
                    prev = access[t]
                    access[t] = None if prev is None or cols is None else prev | cols
                role_filters.setdefault(t, []).append(ta.row_filter)
        filters: dict[str, list[str]] = {}
        for t, fl in role_filters.items():
            if any(f is None for f in fl):  # one of the roles reads the table unfiltered
                continue
            uniq = list(dict.fromkeys(f for f in fl if f))
            filters[t] = [uniq[0]] if len(uniq) == 1 else ["(" + ") OR (".join(uniq) + ")"]
        return access, filters

    def _row_filters(self, analysis: Analysis, role_filters: dict[str, list[str]],
                     obligations: dict[str, Any]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        ob = obligations.get("row_filters") or {}
        for table in analysis.tables:
            bare = table.split(".")[-1]
            conds: list[str] = []
            for source in (self.cfg.access.row_filters, role_filters, ob):
                val = source.get(table) or source.get(bare)
                if val:
                    conds += val if isinstance(val, list) else [val]
            if conds:
                out[table] = list(dict.fromkeys(conds))
        return out

    # ================================================================== validation + writer
    def _analyze(self, sql: str, access: dict[str, set[str] | None] | None, extra: set[str], principal: Principal,
                 source: str) -> tuple[Any, Analysis]:
        try:
            return self.validator.analyze(sql, access=access, extra_params=extra)
        except SqlRejected as exc:
            get_audit().for_principal(principal, "query.rejected", action="sql.execute", decision="rejected",
                                      sql=sql[:4000], errors=exc.errors, source=source)
            metrics.guardrail.labels(SERVICE, "sql_validation", "blocked").inc()
            raise

    async def _write_valid(self, question: str, access: dict[str, set[str] | None],
                           principal: Principal) -> tuple[Any, Analysis, str, int]:
        await self._ensure_catalog()
        errors: list[str] = []
        last: SqlRejected | None = None
        for attempt in (1, 2):  # retry once with the validator's errors
            sql = await self._write(question, access, principal, errors)
            try:
                tree, analysis = self._analyze(sql, access, set(), principal, "generated")
                return tree, analysis, sql, attempt
            except SqlRejected as exc:
                last, errors = exc, exc.errors
                log.info("generated SQL rejected (attempt %s): %s", attempt, exc.errors)
        assert last is not None
        raise last

    async def _write(self, question: str, access: dict[str, set[str] | None], principal: Principal,
                     errors: list[str]) -> str:
        w = self.cfg.writer
        tpl = await self.prompts.get(w.prompt, w.prompt_version)
        feedback = ""
        if errors:
            feedback = ("\n\nYour previous SQL was rejected for these reasons:\n- " + "\n- ".join(errors)
                        + "\nWrite a corrected statement.")
        system = tpl.render_system(dialect=self.dialect or "ANSI", catalog=self.catalog.describe(access),
                                   terms=self.semantic.describe_terms(),
                                   metrics=", ".join(m.name for m in self.semantic.visible(principal)) or "(none)",
                                   max_rows=self.cfg.validator.default_limit)
        req = ChatRequest(model=w.route, messages=[ChatMessage(role="system", content=system),
                                                   ChatMessage(role="user", content=tpl.render_user(
                                                       question=question, feedback=feedback))],
                          temperature=0.0, max_tokens=w.max_tokens,
                          metadata={"task": w.task, "feature": "query.writer", "prompt": tpl.ref, "service": SERVICE})
        with span("query.write", attempt=2 if errors else 1), metrics.time_stage(SERVICE, "write"):
            resp = await asyncio.wait_for(connect("gateway").chat(req, principal), w.timeout_seconds)
        return extract_sql(resp.text)

    async def _ensure_catalog(self) -> None:
        if self.cfg.catalog.reflect and not self.catalog.reflected and self.cfg.executor.replica_url:
            try:
                engine = self.executor.engine
            except Exception as exc:
                log.warning("cannot reflect the catalog: %s", exc)
                return
            await asyncio.to_thread(self.catalog.reflect, engine)

    # ================================================================== governance
    async def _policy(self, principal: Principal, analysis: Analysis, source: str, metric: Metric | None) -> dict[str, Any]:
        inp = PolicyInput(action="sql.execute", principal=_principal_facts(principal),
                          resource={"type": "sql", "tables": analysis.tables, "columns": analysis.columns,
                                    "source": source, "metric": metric.name if metric else None},
                          context={"feature": "query", "app": principal.app, "service": SERVICE,
                                   "target": "replica"})
        with span("query.policy") as sp:
            decision = await connect("policy").decide(inp)
            sp.set(allow=decision.allow, effect=decision.effect)
        metrics.policy.labels(SERVICE, "sql.execute", decision.effect if decision.allow else "deny").inc()
        if not decision.allow or decision.effect != "allow":
            get_audit().for_principal(principal, "query.rejected", action="sql.execute",
                                      resource=",".join(analysis.tables), decision=decision.effect,
                                      reasons=decision.reasons, rules=decision.matched_rules)
            if decision.allow and decision.effect == "require_approval":
                raise ApprovalRequired("this query requires approval", details={"reasons": decision.reasons})
            raise PolicyViolation("policy denied sql.execute", details={"reasons": decision.reasons})
        return decision.obligations or {}

    async def _check_cost(self, sql: str, params: dict[str, Any], identity: dict[str, Any], principal: Principal) -> None:
        limit = self.cfg.executor.max_estimated_cost
        if limit is None:
            return
        cost = await asyncio.to_thread(self.executor.estimated_cost, sql, params, identity)
        if cost is not None and cost > limit:
            get_audit().for_principal(principal, "query.rejected", action="sql.execute", decision="rejected",
                                      sql=sql, errors=[f"estimated cost {cost:.0f} > {limit:.0f}"])
            raise SqlRejected([f"estimated cost {cost:.0f} exceeds the limit {limit:.0f}"], sql)

    # ================================================================== result
    @staticmethod
    def _range_warnings(metric: Metric, cols: list[str], rows: list[tuple[Any, ...]]) -> list[str]:
        out: list[str] = []
        idx = {c.lower(): i for i, c in enumerate(cols)}
        for col, rng in metric.expected_range.items():
            i = idx.get(col.lower())
            if i is None:
                continue
            bad = 0
            for r in rows:
                try:
                    v = float(r[i])
                except (TypeError, ValueError):
                    continue
                if (rng.min is not None and v < rng.min) or (rng.max is not None and v > rng.max):
                    bad += 1
            if bad:
                out.append(f"{bad} value(s) of {col} outside the expected range [{rng.min}, {rng.max}]")
        return out

    @staticmethod
    def _summary(cols: list[str], rows: list[list[Any]], truncated: bool, metric: Metric | None) -> str:
        if not rows:
            text = "No rows matched."
        elif len(rows) == 1 and len(cols) == 1:
            text = f"{cols[0]} = {rows[0][0]}."
        else:
            text = f"{len(rows)} row{'s' if len(rows) != 1 else ''}{' (truncated)' if truncated else ''} " \
                   f"with columns {', '.join(cols)}."
            preview = "; ".join(", ".join(f"{c}={v}" for c, v in zip(cols, r)) for r in rows[:3])
            text += f" First rows: {preview}{' ...' if len(rows) > 3 else ''}."
        if metric is not None:
            text = f"Approved metric '{metric.name}'" + (f" ({metric.description})" if metric.description else "") + \
                   f": {text}"
        return text
