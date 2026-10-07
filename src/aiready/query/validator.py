"""SQL validator: generated SQL is read-only, limited, checked and only then executed.

Checks (sqlglot AST, never regular expressions on the text):

* exactly one statement; a SELECT (or UNION/INTERSECT/EXCEPT of SELECTs, CTEs allowed);
* no DDL/DML/commands anywhere in the tree, no ``SELECT ... INTO``, no ``FOR UPDATE`` locks;
* no function on the deny list (sleep, file access, dblink, set_config...);
* only allowed tables and columns for the caller's roles - ``SELECT *`` only on fully readable tables;
* no cartesian products (joins need ON/USING or an equality between the joined tables in WHERE);
  a join-count limit bounds cost;
* only known bind parameters (``:tenant``, ``:user``, ``:app`` and the metric's own);

Then it rewrites the statement:

* each filtered table reference becomes ``(SELECT * FROM t WHERE <row filter>) AS t`` so row-level filters
  (configured, per role, or policy obligations such as ``tenant_id = :tenant``) apply in every join and
  subquery;
* a ``LIMIT`` is injected (or lowered) to the row cap, plus one row to detect truncation.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from aiready.core.errors import ValidationFailed
from aiready.query.config import ValidatorConfig

ALL = None


class SqlRejected(ValidationFailed):
    code = "sql_rejected"

    def __init__(self, errors: list[str], sql: str = ""):
        super().__init__("SQL rejected: " + "; ".join(errors), details={"errors": errors, "sql": sql[:2000]})
        self.errors = errors


class Analysis(BaseModel):
    tables: list[str] = Field(default_factory=list)
    columns: dict[str, list[str]] = Field(default_factory=dict)
    params: list[str] = Field(default_factory=list)
    joins: int = 0


class ValidatedSQL(BaseModel):
    sql: str
    tables: list[str]
    params: list[str]
    limit: int
    row_filters: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _types(*names: str) -> tuple[type, ...]:
    return tuple(t for n in names if isinstance(t := getattr(exp, n, None), type))


_FORBIDDEN = _types("Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable", "TruncateTable",
                    "Command", "Pragma", "Set", "Use", "Transaction", "Commit", "Rollback", "Grant", "Revoke",
                    "Copy", "LoadData", "Attach", "Detach", "Analyze", "Kill", "Into", "Lock", "Cache", "Uncache",
                    "Refresh", "Install", "Export")
_SET_OPS = _types("SetOperation", "Union", "Intersect", "Except")


class SqlValidator:
    def __init__(self, config: ValidatorConfig | None = None, dialect: str | None = None):
        self.cfg = config or ValidatorConfig()
        self.dialect = dialect
        self.denied = {f.lower() for f in self.cfg.denied_functions}

    # ------------------------------------------------------------------ public
    def validate(self, sql: str, *, access: dict[str, set[str] | None] | None, row_filters: dict[str, list[str]] | None,
                 max_rows: int, extra_params: set[str] | None = None) -> ValidatedSQL:
        tree, analysis = self.analyze(sql, access=access, extra_params=extra_params)
        return self.enforce(tree, analysis, row_filters=row_filters or {}, max_rows=max_rows)

    def analyze(self, sql: str, *, access: dict[str, set[str] | None] | None,
                extra_params: set[str] | None = None) -> tuple[exp.Expression, Analysis]:
        """Static checks. ``access=None`` skips table/column permissions (approved metrics)."""
        sql = (sql or "").strip().rstrip(";").strip()
        if not sql:
            raise SqlRejected(["empty statement"])
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError as exc:
            raise SqlRejected([f"cannot parse SQL: {str(exc).splitlines()[0]}"], sql) from exc
        if len(statements) != 1:
            raise SqlRejected([f"exactly one statement is allowed, got {len(statements)}"], sql)
        tree = statements[0]
        errors: list[str] = []
        if not isinstance(tree, (exp.Select, *_SET_OPS)):
            raise SqlRejected([f"only SELECT statements are allowed, got {tree.key.upper()}"], sql)

        for node in tree.walk():
            if isinstance(node, _FORBIDDEN):
                errors.append(f"{node.key.upper()} is not allowed (read-only queries only)")
            elif isinstance(node, exp.Select) and (node.args.get("into") or node.args.get("locks")):
                errors.append("SELECT INTO / FOR UPDATE is not allowed")
        for fn in tree.find_all(exp.Func):
            name = (fn.name if isinstance(fn, exp.Anonymous) else fn.sql_name()).lower()
            if name in self.denied:
                errors.append(f"function {name}() is not allowed")

        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        tables: list[str] = []
        for t in tree.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier):
                errors.append("table functions are not allowed in FROM")
                continue
            if t.name.lower() in cte_names and not t.db:
                continue
            key = self._table_key(t)
            tables.append(key)
            if access is not None and self._resolve(key, t, access) is None:
                errors.append(f"table {key} is not allowed")

        analysis = Analysis(tables=sorted(set(tables)))
        if access is not None:
            errors += self._check_columns(tree, access, cte_names, analysis)
        errors += self._check_joins(tree, analysis)

        allowed_params = set(self.cfg.reserved_params) | set(extra_params or ())
        for ph in tree.find_all(exp.Placeholder):
            name = ph.name
            if not name:
                errors.append("positional parameters are not allowed; use :name")
            elif name not in allowed_params:
                errors.append(f"unknown parameter :{name}")
            else:
                analysis.params.append(name)
        analysis.params = sorted(set(analysis.params))
        if errors:
            raise SqlRejected(list(dict.fromkeys(errors)), sql)
        return tree, analysis

    def enforce(self, tree: exp.Expression, analysis: Analysis, *, row_filters: dict[str, list[str]],
                max_rows: int) -> ValidatedSQL:
        """Apply row filters and the row limit; returns the SQL that will run (and is shown for audit)."""
        tree = tree.copy()
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        applied: dict[str, list[str]] = {}
        params = set(analysis.params)
        targets = [t for t in tree.find_all(exp.Table)
                   if isinstance(t.this, exp.Identifier) and not (t.name.lower() in cte_names and not t.db)]
        for t in targets:
            key = self._table_key(t)
            conds = row_filters.get(key) or row_filters.get(t.name.lower()) or []
            if not conds:
                continue
            parsed: list[exp.Expression] = []
            for c in conds:
                try:
                    cond = sqlglot.condition(c, dialect=self.dialect)
                except ParseError as exc:
                    raise SqlRejected([f"invalid row filter for {key}: {c}"]) from exc
                params.update(p.name for p in cond.find_all(exp.Placeholder) if p.name)
                parsed.append(cond)
            where = exp.and_(*parsed) if len(parsed) > 1 else parsed[0]
            inner = exp.select("*").from_(exp.Table(this=t.this.copy(), db=t.args.get("db"), catalog=t.args.get("catalog")))
            inner = inner.where(where)
            alias = t.alias_or_name
            t.replace(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias))))
            applied.setdefault(key, conds)

        cap = max(1, min(max_rows, self.cfg.max_rows))
        limit = cap + 1  # one extra row tells us the result was truncated
        existing = tree.args.get("limit")
        warnings: list[str] = []
        if existing is not None:
            val = existing.args.get("expression")
            if isinstance(val, exp.Literal) and not val.is_string:
                try:
                    requested = int(val.this)
                except ValueError:
                    requested = limit
                if requested <= cap:
                    limit = requested
                else:
                    warnings.append(f"LIMIT {requested} lowered to {cap}")
        tree = tree.limit(limit)
        return ValidatedSQL(sql=tree.sql(dialect=self.dialect), tables=analysis.tables, params=sorted(params),
                            limit=limit, row_filters=applied, warnings=warnings)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _table_key(t: exp.Table) -> str:
        return f"{t.db}.{t.name}".lower() if t.db else t.name.lower()

    def _resolve(self, key: str, t: exp.Table, access: dict[str, set[str] | None]) -> str | None:
        """The access entry for a table reference, or None when the table is not allowed."""
        if key in access:
            return key
        if t.db and t.db.lower() in self.cfg.default_schemas and t.name.lower() in access:
            return t.name.lower()
        return None

    def _check_columns(self, tree: exp.Expression, access: dict[str, set[str] | None], cte_names: set[str],
                       analysis: Analysis) -> list[str]:
        errors: list[str] = []
        try:
            scopes = traverse_scope(tree)
        except Exception as exc:  # pragma: no cover - sqlglot could not analyse it: refuse rather than guess
            return [f"query structure not supported: {exc}"]
        used: dict[str, set[str]] = {}

        def table_access(src: exp.Table) -> tuple[str | None, set[str] | None | bool]:
            entry = self._resolve(self._table_key(src), src, access)
            if entry is None:
                return None, False
            return entry, access[entry]

        for scope in scopes:
            sources = scope.sources
            real = {alias: s for alias, s in sources.items() if isinstance(s, exp.Table)
                    and not (s.name.lower() in cte_names and not s.db)}
            derived_cols: set[str] = set()
            for s in sources.values():
                if not isinstance(s, exp.Table) or (s.name.lower() in cte_names and not s.db):
                    expr = getattr(s, "expression", None)
                    derived_cols.update(c.lower() for c in getattr(expr, "named_selects", []) or [])
            select = scope.expression
            aliases: set[str] = set()
            in_projection: set[int] = set()
            if isinstance(select, exp.Select):
                aliases = {a.alias.lower() for a in select.expressions if isinstance(a, exp.Alias)}
                in_projection = {id(c) for p in select.expressions for c in p.find_all(exp.Column)}

            # SELECT * / t.* only on tables the caller may read in full
            if isinstance(select, exp.Select):
                for proj in select.expressions:
                    if isinstance(proj, exp.Star) or (isinstance(proj, exp.Column) and proj.is_star):
                        qual = proj.table if isinstance(proj, exp.Column) else ""
                        srcs = [real[qual]] if qual and qual in real else ([] if qual else list(real.values()))
                        for src in srcs:
                            name, cols = table_access(src)
                            if name is not None and cols is not ALL:
                                errors.append(f"SELECT * is not allowed on {name}: list the permitted columns")

            for col in scope.columns:
                cname = col.name.lower()
                if not cname or isinstance(col.this, exp.Star):
                    continue
                if col.table:
                    src = sources.get(col.table)
                    if isinstance(src, exp.Table) and col.table in real:
                        name, cols = table_access(src)
                        if name is None:
                            continue  # table error already reported
                        if cols is not ALL and cname not in cols:
                            errors.append(f"column {name}.{cname} is not allowed")
                        used.setdefault(name, set()).add(cname)
                    continue
                # GROUP BY / HAVING may name an output alias; a name inside the projection is always a column
                is_alias_ref = cname in aliases and id(col) not in in_projection
                candidates = []
                for src in real.values():
                    name, cols = table_access(src)
                    if name is not None:
                        candidates.append((name, cols))
                ok = [n for n, cols in candidates if cols is ALL or cname in cols]
                if ok:
                    used.setdefault(ok[0], set()).add(cname)
                elif cname in derived_cols or is_alias_ref:
                    continue
                elif candidates:
                    errors.append(f"column {cname} is not allowed on {', '.join(n for n, _ in candidates)}")
        analysis.columns = {k: sorted(v) for k, v in used.items()}
        return errors

    def _check_joins(self, tree: exp.Expression, analysis: Analysis) -> list[str]:
        errors: list[str] = []
        total = 0
        for sel in tree.find_all(exp.Select):
            joins = sel.args.get("joins") or []
            total += len(joins)
            if not self.cfg.forbid_cartesian:
                continue
            where = sel.args.get("where")
            for j in joins:
                if j.args.get("on") or j.args.get("using"):
                    continue
                if isinstance(j.this, exp.Unnest) or (j.args.get("kind") or "").upper() == "LATERAL":
                    continue
                alias = j.this.alias_or_name
                linked = False
                if where is not None:
                    for eq in where.find_all(exp.EQ):
                        left, right = eq.left, eq.right
                        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                            quals = {left.table, right.table}
                            if alias in quals and len(quals - {""}) == 2:
                                linked = True
                                break
                if not linked:
                    errors.append(f"cartesian join with {alias} is not allowed: join on a key")
        analysis.joins = total
        if total > self.cfg.max_joins:
            errors.append(f"too many joins ({total} > {self.cfg.max_joins})")
        return errors


def dialect_for_url(url: str | None) -> str | None:
    """sqlglot dialect name from an SQLAlchemy URL."""
    if not url:
        return None
    scheme = url.split(":", 1)[0].split("+", 1)[0].lower()
    return {"postgresql": "postgres", "postgres": "postgres", "sqlite": "sqlite", "mysql": "mysql",
            "mariadb": "mysql", "mssql": "tsql", "oracle": "oracle", "snowflake": "snowflake",
            "bigquery": "bigquery", "redshift": "redshift", "databricks": "databricks", "duckdb": "duckdb",
            "trino": "trino", "presto": "presto", "clickhouse": "clickhouse"}.get(scheme)


def to_jsonable(value: Any) -> Any:
    """Row values as JSON-safe types (numbers stay numbers)."""
    from datetime import date, datetime, time
    from decimal import Decimal
    from uuid import UUID

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    return str(value)
