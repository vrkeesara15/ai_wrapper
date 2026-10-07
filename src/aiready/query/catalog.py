"""Schema catalog: what the query writer may know about the data.

Tables, columns, keys, descriptions and sample queries come from configuration and/or are reflected from the
read replica with SQLAlchemy ``inspect``; configuration wins for descriptions so business wording is kept.
The writer only ever sees the tables and columns the caller's roles allow (``describe(access)``).
"""

from __future__ import annotations

import logging
from typing import Any

from aiready.query.config import CatalogConfig, ColumnConfig, ForeignKey, TableConfig

log = logging.getLogger(__name__)

ALL = None  # access marker: every column of the table


class SchemaCatalog:
    def __init__(self, config: CatalogConfig | None = None):
        self.cfg = config or CatalogConfig()
        self.tables: dict[str, TableConfig] = {k.lower(): v.model_copy(deep=True) for k, v in self.cfg.tables.items()}
        self.reflected = False

    def reflect(self, engine: Any) -> None:
        """Merge the replica's schema into the catalog (columns, primary and foreign keys)."""
        from sqlalchemy import inspect

        try:
            insp = inspect(engine)
            schema = self.cfg.reflect_schema
            for name in insp.get_table_names(schema=schema):
                key = name.lower()
                t = self.tables.setdefault(key, TableConfig())
                for col in insp.get_columns(name, schema=schema):
                    c = t.columns.setdefault(col["name"], ColumnConfig())
                    if not c.type:
                        c.type = str(col.get("type", ""))
                pk = insp.get_pk_constraint(name, schema=schema) or {}
                if not t.primary_key:
                    t.primary_key = list(pk.get("constrained_columns") or [])
                if not t.foreign_keys:
                    for fk in insp.get_foreign_keys(name, schema=schema):
                        for col, ref in zip(fk.get("constrained_columns", []), fk.get("referred_columns", [])):
                            t.foreign_keys.append(ForeignKey(column=col, references=f"{fk['referred_table']}.{ref}"))
            self.reflected = True
        except Exception as exc:  # the configured catalog still works
            log.warning("schema reflection from the replica failed: %s", exc)

    def columns(self, table: str) -> list[str]:
        t = self.tables.get(table.lower())
        return list(t.columns) if t else []

    def describe(self, access: dict[str, set[str] | None]) -> str:
        """Catalog text for the query writer, restricted to the allowed tables/columns."""
        lines: list[str] = []
        for name in sorted(access):
            t = self.tables.get(name)
            allowed = access[name]
            cols = self.columns(name) if t else []
            if allowed is not ALL:
                cols = [c for c in cols if c.lower() in allowed] or sorted(allowed)
            head = f"table {name}" + (f" -- {t.description}" if t and t.description else "")
            lines.append(head)
            for c in cols:
                info = t.columns.get(c) if t else None
                desc = f" {info.type}" if info and info.type else ""
                desc += f" -- {info.description}" if info and info.description else ""
                lines.append(f"  {c}{desc}")
            if t and t.primary_key:
                lines.append(f"  primary key: {', '.join(t.primary_key)}")
            if t:
                for fk in t.foreign_keys:
                    if fk.references.split(".")[0].lower() in access:
                        lines.append(f"  {fk.column} references {fk.references}")
                for s in t.sample_queries[:2]:
                    lines.append(f"  example: {s}")
        return "\n".join(lines) or "(no tables available to you)"
