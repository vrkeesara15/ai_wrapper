"""Connector workers (plugin kind ``connector``): one per source type.

A connector only *reads* a source and yields ``Document`` objects carrying access tags, owner,
``updated_at`` and a checksum. It never decides what changed (the document registry does) and never
parses (the parsers do), so adding a source type is one small class:

    @register("connector", "sharepoint")
    class SharePointConnector:
        full_listing = True
        def __init__(self, source: str, site: str, ...): ...
        async def documents(self) -> AsyncIterator[Document]: ...

``full_listing = True`` means every run yields every live item, so items missing from a run are deletions.
Connectors that only see changes (feeds, change capture) set it to False and emit ``deleted=True`` items.

Built-ins: ``filesystem`` (local / mounted SMB-NFS via the file crawler), ``s3``, ``sql`` (read replica
tables -> one document per row, row-level access tags), ``http_json`` (REST endpoints with pagination),
``confluence`` (wiki pages with space permissions -> access tags) and ``jira`` (tickets).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from aiready.core.errors import ConfigError, PermissionDenied
from aiready.core.plugins import register
from aiready.core.types import DataClass, Document
from aiready.ingestion.documents import RowMapping, checksum_of, map_row

log = logging.getLogger(__name__)


@runtime_checkable
class Connector(Protocol):
    full_listing: bool

    def documents(self) -> AsyncIterator[Document]: ...


# --------------------------------------------------------------------------- helpers


def get_path(obj: Any, path: str | None, default: Any = None) -> Any:
    """Dotted path lookup (``fields.status.name``, ``items.0.id``). Empty path returns the object."""
    if not path:
        return obj
    node = obj
    for part in path.split("."):
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        elif isinstance(node, list) and part.lstrip("-").isdigit():
            idx = int(part)
            if idx >= len(node) or idx < -len(node):
                return default
            node = node[idx]
        else:
            return default
    return node


class _Fmt(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render(template: str, item: dict[str, Any]) -> str:
    """``str.format`` over a (flattened) item where missing keys render empty and dotted keys work."""
    flat = _Fmt()
    _flatten(item, "", flat)
    safe = re.sub(r"\{([A-Za-z0-9_.]+)\}", lambda m: "{" + m.group(1).replace(".", "__") + "}", template)
    return safe.format_map(_Fmt({k.replace(".", "__"): v for k, v in flat.items()}))


def _flatten(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
        if prefix:
            out[prefix] = obj
    else:
        out[prefix] = "" if obj is None else obj


def parse_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value) / (1000.0 if value > 1e11 else 1.0)  # epoch millis from Java systems
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        v = value.replace("Z", "+00:00")
        v = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", v)  # Jira: 2024-01-01T10:00:00.000+0100
        try:
            return datetime.fromisoformat(v).timestamp()
        except ValueError:
            return None
    return None


def build_auth_headers(headers: dict[str, str] | None = None, bearer_token: str | None = None,
                       basic: dict[str, str] | None = None) -> dict[str, str]:
    """Integration credentials come from config secrets (``${secret:...}``) and are read-only accounts that
    only fetch content for indexing; the *user's* rights are applied at search time via access tags."""
    h = dict(headers or {})
    if bearer_token:
        h["Authorization"] = f"Bearer {bearer_token}"
    elif basic:
        raw = f"{basic.get('username', '')}:{basic.get('password', '')}".encode()
        h["Authorization"] = "Basic " + base64.b64encode(raw).decode()
    return h


class _HttpBase:
    def __init__(self, base_url: str, *, headers: dict[str, str] | None = None, bearer_token: str | None = None,
                 basic: dict[str, str] | None = None, cookies: dict[str, str] | None = None, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None, verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self._client_kw: dict[str, Any] = {
            "base_url": self.base_url, "timeout": timeout, "transport": transport, "cookies": cookies,
            "headers": build_auth_headers(headers, bearer_token, basic), "verify": verify_tls,
        }

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(**self._client_kw)

    @staticmethod
    async def get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(3):
            r = await client.get(url, params=params)
            if r.status_code in (429, 502, 503, 504) and attempt < 2:
                await asyncio.sleep(float(r.headers.get("Retry-After", 0.5 * 2 ** attempt)))
                continue
            if r.status_code >= 400:
                raise ConfigError(f"GET {url} returned {r.status_code}: {r.text[:200]}")
            return r.json()
        raise ConfigError(f"GET {url} kept failing")  # pragma: no cover - loop always returns or raises


# --------------------------------------------------------------------------- filesystem / s3


@register("connector", "filesystem")
class FilesystemConnector:
    """Local directory or mounted file share, read with each file's access list (see the file crawler)."""

    full_listing = True

    def __init__(self, source: str, root: str, include: list[str] | None = None, exclude: list[str] | None = None,
                 acl: list[dict[str, Any] | str] | None = None, default_tags: list[str] | None = None,
                 max_file_mb: float = 50.0, follow_symlinks: bool = False, data_class: str = "internal"):
        from aiready.integration.file_crawler import FileCrawler

        self.crawler = FileCrawler(root, source=source, include=include, exclude=exclude, acl=acl,
                                   default_tags=default_tags, max_file_mb=max_file_mb,
                                   follow_symlinks=follow_symlinks, default_data_class=data_class)

    async def documents(self) -> AsyncIterator[Document]:
        for doc in await asyncio.to_thread(lambda: list(self.crawler.crawl())):
            yield doc


@register("connector", "s3")
class S3Connector:
    """S3 bucket/prefix; ``boto3`` is imported only when this connector is used."""

    full_listing = True

    def __init__(self, source: str, bucket: str, prefix: str = "", include: list[str] | None = None,
                 exclude: list[str] | None = None, acl: list[dict[str, Any] | str] | None = None,
                 default_tags: list[str] | None = None, endpoint_url: str | None = None, region: str | None = None,
                 read_object_tags: bool = False, client: Any = None, max_file_mb: float = 50.0):
        from aiready.integration.file_crawler import FileCrawler

        self.crawler = FileCrawler(source=source, backend="s3", bucket=bucket, prefix=prefix, include=include,
                                   exclude=exclude, acl=acl, default_tags=default_tags, s3_client=client,
                                   endpoint_url=endpoint_url, region=region, read_object_tags=read_object_tags,
                                   max_file_mb=max_file_mb)

    async def documents(self) -> AsyncIterator[Document]:
        for doc in await asyncio.to_thread(lambda: list(self.crawler.crawl())):
            yield doc


# --------------------------------------------------------------------------- sql (replica)


def assert_read_only(sql: str, dialect: str | None = None) -> None:
    """Only a single SELECT may run against a replica - even a replica must not be written by mistake."""
    import sqlglot
    from sqlglot import exp

    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise PermissionDenied(f"unparseable SQL refused: {exc}") from exc
    if len(statements) != 1:
        raise PermissionDenied("exactly one SELECT statement is allowed")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise PermissionDenied(f"only SELECT is allowed, got {stmt.key.upper()}")
    for node in stmt.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)):
            raise PermissionDenied("write statements are not allowed")


@register("connector", "sql")
class SqlConnector:
    """Rows of the **read replica** become documents (one per row) with row-level access tags.

    ``url`` must point at a replica / reporting copy. If ``primary_url`` is configured and equal to
    ``url`` the connector refuses to run: ingestion must never add load to the live system of record.
    """

    full_listing = True

    def __init__(self, source: str, url: str, tables: list[dict[str, Any]], primary_url: str | None = None,
                 batch_size: int = 1000):
        if not url:
            raise ConfigError(f"sql connector '{source}': a read-replica 'url' is required")
        if primary_url and primary_url == url:
            raise ConfigError(f"sql connector '{source}': refusing to read the primary database; use a replica")
        self.source, self.url, self.batch_size = source, url, batch_size
        self.mappings = [RowMapping.model_validate(t) for t in tables]
        for m in self.mappings:
            if m.query:
                assert_read_only(m.query)
            elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", m.table):
                raise ConfigError(f"invalid table name '{m.table}'")

    def _read(self) -> list[Document]:
        from sqlalchemy import create_engine, text

        engine = create_engine(self.url, future=True)
        out: list[Document] = []
        try:
            with engine.connect() as conn:
                for m in self.mappings:
                    sql = m.query or f"SELECT * FROM {m.table}"
                    result = conn.execution_options(stream_results=True).execute(text(sql))
                    while True:
                        rows = result.mappings().fetchmany(self.batch_size)
                        if not rows:
                            break
                        out.extend(map_row(dict(r), m, self.source) for r in rows)
        finally:
            engine.dispose()
        return out

    async def documents(self) -> AsyncIterator[Document]:
        for doc in await asyncio.to_thread(self._read):
            yield doc


# --------------------------------------------------------------------------- generic REST


@register("connector", "http_json")
class HttpJsonConnector(_HttpBase):
    """Any JSON REST endpoint that lists items, with pagination.

    pagination.type:
      * ``none`` - one request;
      * ``page`` - ``?page=N`` (``page_param``, ``start``), stops on an empty page or when ``next_path`` is null;
      * ``offset`` - ``?offset=N&limit=M``;
      * ``cursor`` - the cursor at ``next_path`` is sent back as ``cursor_param``;
      * ``next_url`` - the absolute/relative URL at ``next_path`` is fetched next.
    """

    full_listing = True

    def __init__(self, source: str, base_url: str, path: str, *, params: dict[str, Any] | None = None,
                 items_path: str = "items", pagination: dict[str, Any] | None = None, id_field: str = "id",
                 title_template: str | None = None, content_template: str | None = None,
                 content_fields: list[str] | None = None, uri_template: str | None = None,
                 updated_at_field: str | None = "updated_at", owner_field: str | None = None,
                 access_tags: list[str] | None = None, access_tag_fields: list[dict[str, str]] | None = None,
                 data_class: str = "internal", max_pages: int = 1000, **http: Any):
        super().__init__(base_url, **http)
        self.source, self.path, self.params = source, path, dict(params or {})
        self.items_path, self.id_field = items_path, id_field
        self.pagination = {"type": "none", **(pagination or {})}
        self.title_template, self.content_template = title_template, content_template
        self.content_fields, self.uri_template = content_fields, uri_template
        self.updated_at_field, self.owner_field = updated_at_field, owner_field
        self.access_tags = list(access_tags or [])
        self.access_tag_fields = access_tag_fields or []
        self.data_class = DataClass(data_class)
        self.max_pages = max_pages

    def to_document(self, item: dict[str, Any]) -> Document:
        item_id = get_path(item, self.id_field)
        if item_id is None:
            raise ConfigError(f"item without '{self.id_field}' from {self.path}")
        if self.content_template:
            content = render(self.content_template, item)
        else:
            fields = self.content_fields or list(item)
            content = "\n".join(f"{f}: {get_path(item, f)}" for f in fields if get_path(item, f) not in (None, ""))
        tags = list(self.access_tags)
        for spec in self.access_tag_fields:
            val = get_path(item, spec["field"])
            for v in val if isinstance(val, list) else [val]:
                if v not in (None, ""):
                    tags.append(spec.get("template", "{value}").format(value=v))
        tags = sorted(set(tags))
        owner = get_path(item, self.owner_field) if self.owner_field else None
        return Document(
            id=f"{self.source}:{item_id}", source=self.source,
            uri=render(self.uri_template, item) if self.uri_template else f"{self.base_url}{self.path}#{item_id}",
            title=render(self.title_template, item) if self.title_template else str(item_id), content=content,
            mime_type="text/plain", access_tags=tags, owner=str(owner) if owner is not None else None,
            updated_at=parse_time(get_path(item, self.updated_at_field)) or 0.0 if self.updated_at_field else 0.0,
            checksum=checksum_of(content + "|" + ",".join(tags)),
            metadata={"data_class": self.data_class.value, "item_id": item_id},
        )

    async def documents(self) -> AsyncIterator[Document]:
        p = self.pagination
        kind = p.get("type", "none")
        params = dict(self.params)
        url: str | None = self.path
        page = int(p.get("start", 1))
        offset = 0
        if kind == "page" and p.get("page_size"):
            params[p.get("size_param", "page_size")] = p["page_size"]
        async with self.client() as client:
            for _ in range(self.max_pages):
                if url is None:
                    break
                if kind == "page":
                    params[p.get("page_param", "page")] = page
                elif kind == "offset":
                    params[p.get("offset_param", "offset")] = offset
                    params[p.get("limit_param", "limit")] = int(p.get("page_size", 100))
                body = await self.get_json(client, url, params=params)
                items = get_path(body, self.items_path, []) if self.items_path else body
                for item in items or []:
                    yield self.to_document(item)
                nxt = get_path(body, p["next_path"]) if p.get("next_path") else None
                if kind == "none" or not items:
                    break
                if kind == "page":
                    if p.get("next_path") and nxt is None:
                        break
                    page = int(nxt) if isinstance(nxt, int) else page + 1
                elif kind == "offset":
                    offset += len(items)
                elif kind == "cursor":
                    if not nxt:
                        break
                    params[p.get("cursor_param", "cursor")] = nxt
                elif kind == "next_url":
                    url, params = (str(nxt), {}) if nxt else (None, params)
                else:
                    raise ConfigError(f"unknown pagination type '{kind}'")


# --------------------------------------------------------------------------- wiki / tickets


@register("connector", "confluence")
class ConfluenceConnector(_HttpBase):
    """Confluence pages (storage-format HTML) per space; space read permissions become access tags.

    Permissions are read from the space API; when the integration account may not read them,
    ``space_access`` gives the tags per space key explicitly. ``group_template`` / ``user_template``
    turn Confluence principals into AI-Ready access tags so they line up with the identity bridge's groups.
    """

    full_listing = True

    def __init__(self, source: str, base_url: str, spaces: list[str], *, page_size: int = 50,
                 space_access: dict[str, list[str]] | None = None, read_space_permissions: bool = True,
                 group_template: str = "{name}", user_template: str = "user:{name}",
                 data_class: str = "internal", **http: Any):
        super().__init__(base_url, **http)
        self.source, self.spaces, self.page_size = source, spaces, page_size
        self.space_access = space_access or {}
        self.read_space_permissions = read_space_permissions
        self.group_template, self.user_template = group_template, user_template
        self.data_class = DataClass(data_class)

    async def _space_tags(self, client: httpx.AsyncClient, key: str) -> list[str]:
        tags = list(self.space_access.get(key, []))
        if not self.read_space_permissions:
            return tags
        try:
            space = await self.get_json(client, f"/rest/api/space/{key}", params={"expand": "permissions"})
        except ConfigError as exc:
            log.warning("cannot read permissions of space %s: %s", key, exc)
            return tags
        for perm in space.get("permissions", []):
            op = perm.get("operation", {})
            if op.get("operation") != "read" or op.get("targetType") not in (None, "space"):
                continue
            subjects = perm.get("subjects", {})
            for g in get_path(subjects, "group.results", []) or []:
                tags.append(self.group_template.format(name=g.get("name")))
            for u in get_path(subjects, "user.results", []) or []:
                tags.append(self.user_template.format(name=u.get("username") or u.get("accountId")))
            if perm.get("anonymousAccess"):
                tags.append("public")
        return sorted(set(tags))

    async def documents(self) -> AsyncIterator[Document]:
        async with self.client() as client:
            for key in self.spaces:
                tags = await self._space_tags(client, key)
                start = 0
                while True:
                    body = await self.get_json(client, "/rest/api/content", params={
                        "spaceKey": key, "type": "page", "start": start, "limit": self.page_size,
                        "expand": "body.storage,version,history,ancestors"})
                    results = body.get("results", [])
                    for page in results:
                        html = get_path(page, "body.storage.value", "") or ""
                        ancestors = " > ".join(a.get("title", "") for a in page.get("ancestors", []))
                        owner = get_path(page, "history.createdBy.username") or get_path(page, "history.createdBy.accountId")
                        yield Document(
                            id=f"{self.source}:{page['id']}", source=self.source,
                            uri=self.base_url + get_path(page, "_links.webui", f"/pages/{page['id']}"),
                            title=page.get("title"), content=html, mime_type="text/html", access_tags=tags,
                            owner=owner, updated_at=parse_time(get_path(page, "version.when")) or 0.0,
                            checksum=checksum_of(html + "|" + ",".join(tags)),
                            metadata={"space": key, "version": get_path(page, "version.number"),
                                      "ancestors": ancestors, "data_class": self.data_class.value},
                        )
                    if len(results) < self.page_size or not get_path(body, "_links.next"):
                        break
                    start += len(results)


@register("connector", "jira")
class JiraConnector(_HttpBase):
    """Jira issues found by a JQL query; project (and security level) mapped to access tags."""

    full_listing = True

    def __init__(self, source: str, base_url: str, jql: str, *, page_size: int = 50,
                 fields: list[str] | None = None, project_access: dict[str, list[str]] | None = None,
                 security_level_access: dict[str, list[str]] | None = None,
                 default_tags: list[str] | None = None, include_comments: bool = True,
                 data_class: str = "internal", api_version: str = "2", **http: Any):
        super().__init__(base_url, **http)
        self.source, self.jql, self.page_size = source, jql, page_size
        self.fields = fields or ["summary", "description", "status", "updated", "project", "reporter", "assignee",
                                 "comment", "security", "issuetype", "priority"]
        self.project_access = project_access or {}
        self.security_level_access = security_level_access or {}
        self.default_tags = default_tags or []
        self.include_comments = include_comments
        self.data_class = DataClass(data_class)
        self.api_version = api_version

    def to_document(self, issue: dict[str, Any]) -> Document | None:
        f = issue.get("fields", {})
        project = get_path(f, "project.key", "")
        level = get_path(f, "security.name")
        if level:
            # An issue security level narrows visibility: only its mapped groups, or nobody (fail closed).
            tags = list(self.security_level_access.get(level, []))
        else:
            tags = list(self.project_access.get(project, self.default_tags))
        if not tags:
            return None
        lines = [f"{issue['key']}: {f.get('summary', '')}", f"Status: {get_path(f, 'status.name', '')}",
                 f"Type: {get_path(f, 'issuetype.name', '')}", f"Priority: {get_path(f, 'priority.name', '')}", "",
                 str(f.get("description") or "")]
        if self.include_comments:
            comments = get_path(f, "comment.comments", []) or []
            if comments:
                lines += ["", "Comments:"]
                lines += [f"- {get_path(c, 'author.displayName', '?')}: {c.get('body', '')}" for c in comments]
        content = "\n".join(lines).strip()
        tags = sorted(set(tags))
        return Document(
            id=f"{self.source}:{issue['key']}", source=self.source, uri=f"{self.base_url}/browse/{issue['key']}",
            title=f"{issue['key']} {f.get('summary', '')}".strip(), content=content, mime_type="text/plain",
            access_tags=tags, owner=get_path(f, "reporter.name") or get_path(f, "reporter.accountId"),
            updated_at=parse_time(f.get("updated")) or 0.0, checksum=checksum_of(content + "|" + ",".join(tags)),
            metadata={"project": project, "data_class": self.data_class.value},
        )

    async def documents(self) -> AsyncIterator[Document]:
        async with self.client() as client:
            start = 0
            while True:
                body = await self.get_json(client, f"/rest/api/{self.api_version}/search", params={
                    "jql": self.jql, "startAt": start, "maxResults": self.page_size, "fields": ",".join(self.fields)})
                issues = body.get("issues", [])
                for issue in issues:
                    doc = self.to_document(issue)
                    if doc is None:
                        log.warning("jira issue %s skipped: no access mapping", issue.get("key"))
                        continue
                    yield doc
                start += len(issues)
                if not issues or start >= int(body.get("total", 0)):
                    break
