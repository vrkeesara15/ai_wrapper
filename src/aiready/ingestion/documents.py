"""Work-queue message contract and row-to-document mapping shared by connectors and change capture.

Everything that produces source items (connector workers, the change-capture reader, the file crawler)
puts **one message per document** on the ingestion work queue. Keeping the encoding here means producers in
other processes (e.g. a change-capture pod) and the worker pool agree on one format, and raw bytes survive
any JSON-based broker (Redis lists, Kafka, SQS).
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from datetime import date, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.types import DataClass, Document

WORK_TOPIC = "ingestion.documents"


# --------------------------------------------------------------------------- encoding


def checksum_of(content: bytes | str) -> str:
    data = content.encode() if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def encode_document(doc: Document) -> dict[str, Any]:
    """JSON-safe dict; bytes are base64-encoded so binary files (PDF, DOCX) pass through any broker."""
    body = doc.model_dump(mode="python", exclude={"content"})
    if isinstance(doc.content, bytes):
        body["content"] = base64.b64encode(doc.content).decode("ascii")
        body["content_encoding"] = "base64"
    else:
        body["content"] = doc.content
        body["content_encoding"] = "text"
    return json.loads(json.dumps(body, default=_json_default))


def decode_document(data: dict[str, Any]) -> Document:
    body = dict(data)
    enc = body.pop("content_encoding", "text")
    if enc == "base64" and isinstance(body.get("content"), str):
        body["content"] = base64.b64decode(body["content"])
    return Document.model_validate(body)


def work_message(source: str, doc: Document, *, run_id: str | None = None, origin: str = "connector") -> dict[str, Any]:
    """Payload of one work-queue message. ``origin`` says who produced it (connector / change_capture / crawler)."""
    return {"source": source, "run_id": run_id, "origin": origin, "document": encode_document(doc),
            "enqueued_at": time.time()}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if hasattr(value, "value"):  # enums
        return value.value
    return str(value)


# --------------------------------------------------------------------------- sinks


class DocumentSink(Protocol):
    """Where producers send documents. The default puts them on the work queue."""

    async def emit(self, doc: Document, *, source: str, run_id: str | None = None) -> None: ...


class QueueSink:
    """Puts each document on the ingestion work queue (``core.queues``)."""

    def __init__(self, queue: Any, topic: str = WORK_TOPIC, origin: str = "connector"):
        self.queue, self.topic, self.origin = queue, topic, origin

    async def emit(self, doc: Document, *, source: str, run_id: str | None = None) -> None:
        await self.queue.put(self.topic, work_message(source, doc, run_id=run_id, origin=self.origin))


class ListSink:
    """Collects documents in memory (tests, dry runs)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, Document]] = []

    async def emit(self, doc: Document, *, source: str, run_id: str | None = None) -> None:
        self.items.append((source, doc))


# --------------------------------------------------------------------------- rows -> documents


class AccessTagColumn(BaseModel):
    """Derive an access tag from a column value, e.g. ``{column: region, template: "region:{value}"}``."""

    column: str
    template: str = "{value}"


class AccessRule(BaseModel):
    """Row-level rule: when every ``when`` column equals the given value, add ``tags``."""

    when: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class RowMapping(BaseModel):
    """How one table (or query) of the replica becomes documents.

    The defaults turn every row into ``key: value`` lines, which keeps all columns searchable. Access tags
    combine static tags, tags derived from columns and conditional rules, so row-level visibility in the
    application (e.g. "adjusters see their region") is reproduced in the index.
    """

    table: str
    key_columns: list[str] = Field(default_factory=lambda: ["id"])
    query: str | None = None  # optional custom SELECT (read-only) instead of the whole table
    title_template: str | None = None  # e.g. "Claim {claim_number}"
    content_columns: list[str] | None = None  # None -> every column
    content_template: str | None = None  # e.g. "Claim {claim_number} ({status}): {description}"
    exclude_columns: list[str] = Field(default_factory=list)  # never indexed (secrets, hashes)
    updated_at_column: str | None = "updated_at"
    owner_column: str | None = None
    access_tags: list[str] = Field(default_factory=list)
    access_tag_columns: list[AccessTagColumn] = Field(default_factory=list)
    access_rules: list[AccessRule] = Field(default_factory=list)
    data_class: DataClass = DataClass.INTERNAL
    uri_template: str | None = None  # e.g. "legacy://claims/{claim_number}"

    def doc_id(self, source: str, row: dict[str, Any]) -> str:
        key = ":".join(str(row.get(c)) for c in self.key_columns)
        return f"{source}:{self.table}:{key}"

    def doc_id_from_key(self, source: str, key: Any) -> str:
        parts = key if isinstance(key, (list, tuple)) else [key]
        return f"{source}:{self.table}:" + ":".join(str(p) for p in parts)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def map_row(row: dict[str, Any], mapping: RowMapping, source: str) -> Document:
    """Turn one database row into a Document with access tags, owner, updated_at and checksum."""
    visible = {k: v for k, v in row.items() if k not in set(mapping.exclude_columns)}
    fmt = _SafeDict({k: "" if v is None else v for k, v in visible.items()})
    if mapping.content_template:
        content = mapping.content_template.format_map(fmt)
    else:
        cols = mapping.content_columns or list(visible)
        content = "\n".join(f"{c}: {visible.get(c)}" for c in cols if c in visible and visible.get(c) is not None)
    title = mapping.title_template.format_map(fmt) if mapping.title_template else f"{mapping.table} {fmt.get(mapping.key_columns[0], '')}"
    tags = list(mapping.access_tags)
    for col in mapping.access_tag_columns:
        val = row.get(col.column)
        if val is not None and val != "":
            tags.append(col.template.format(value=val))
    for rule in mapping.access_rules:
        if all(str(row.get(k)) == str(v) for k, v in rule.when.items()):
            tags.extend(rule.tags)
    updated = _as_epoch(row.get(mapping.updated_at_column)) if mapping.updated_at_column else None
    owner = row.get(mapping.owner_column) if mapping.owner_column else None
    metadata: dict[str, Any] = {"table": mapping.table, "key": {c: row.get(c) for c in mapping.key_columns},
                                "data_class": mapping.data_class.value}
    return Document(
        id=mapping.doc_id(source, row), source=source,
        uri=mapping.uri_template.format_map(fmt) if mapping.uri_template else f"sql://{mapping.table}/{mapping.doc_id(source, row).rsplit(':', 1)[-1]}",
        title=title, content=content, mime_type="text/plain", access_tags=sorted(set(tags)),
        owner=str(owner) if owner is not None else None, updated_at=updated or time.time(),
        checksum=checksum_of(content + "|" + ",".join(sorted(set(tags)))), metadata=metadata,
    )


def _as_epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                return None
    return None
