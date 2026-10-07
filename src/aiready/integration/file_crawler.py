"""File crawler: reads documents from file shares *with their access list*.

Slide "Legacy integration architecture": file shares (documents and exports) are read by a crawler that
"reads each file with its access list". Indexing a file without its access list would let the assistant
show a document to someone the file share would refuse, so the crawler **fails closed**: a file whose
access cannot be resolved is skipped (and counted) unless the configuration gives explicit default tags.

Backends:
  * ``local`` - a local directory, or an SMB/NFS share mounted on the worker (the usual way to read
    Windows shares without touching the file server);
  * ``s3`` - an S3-compatible bucket (``boto3`` imported lazily).

Access resolvers (plugin kind ``acl_resolver``) are applied in order and their tags are merged:
  * ``sidecar`` - ``<file>.acl.json`` next to the file, or ``.acl.json`` in the directory or any parent
    (nearest wins). Keys: ``groups``, ``users``, ``roles``, ``tags``, ``owner``, ``data_class``, ``inherit``;
  * ``path_map`` - glob patterns relative to the root mapped to groups (``"fraud/**": [fraud]``);
  * ``posix`` - owner, group and world-readable bits of the file mapped to tags.
Exotic ACL systems (NTFS ACLs exported by a share, SharePoint permissions...) plug in with
``class: their_pkg.module:Resolver``.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import mimetypes
import os
import stat
from collections.abc import AsyncIterator, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.audit import get_audit
from aiready.core.plugins import create, register
from aiready.core.telemetry import metrics, span
from aiready.core.types import DataClass, Document
from aiready.ingestion.documents import DocumentSink, checksum_of

log = logging.getLogger(__name__)

SIDECAR_SUFFIX = ".acl.json"
DIR_SIDECAR = ".acl.json"

mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("text/markdown", ".markdown")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")


def guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


# --------------------------------------------------------------------------- access resolution


class FileRef(BaseModel):
    """What a resolver may look at for one file."""

    relpath: str  # posix-style path relative to the crawl root
    abspath: str | None = None  # local backends
    size: int = 0
    mtime: float = 0.0
    uid: int | None = None
    gid: int | None = None
    mode: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # e.g. S3 object tags


class Access(BaseModel):
    tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    data_class: DataClass | None = None
    resolved: bool = False  # True when at least one resolver had an opinion


class AclResolver(Protocol):
    def resolve(self, ref: FileRef, reader: FileBackend) -> Access: ...


def _sidecar_access(data: dict[str, Any]) -> Access:
    tags = list(data.get("tags") or [])
    tags += list(data.get("groups") or [])
    tags += [f"user:{u}" for u in data.get("users") or []]
    tags += [f"role:{r}" for r in data.get("roles") or []]
    dc = data.get("data_class")
    return Access(tags=tags, owner=data.get("owner"), data_class=DataClass(dc) if dc else None, resolved=True)


@register("acl_resolver", "sidecar")
class SidecarResolver:
    """``<file>.acl.json`` beats ``<dir>/.acl.json``, which beats the parent directory's, and so on.

    A sidecar with ``"inherit": true`` merges with its parent's tags (useful for "parent groups plus one").
    """

    def resolve(self, ref: FileRef, reader: FileBackend) -> Access:
        chain: list[dict[str, Any]] = []
        own = reader.read_json(ref.relpath + SIDECAR_SUFFIX)
        if own is not None:
            chain.append(own)
        parts = PurePosixPath(ref.relpath).parts[:-1]
        if not chain or chain[-1].get("inherit"):
            for depth in range(len(parts), -1, -1):
                d = "/".join(parts[:depth])
                data = reader.read_json(f"{d}/{DIR_SIDECAR}" if d else DIR_SIDECAR)
                if data is None:
                    continue
                chain.append(data)
                if not data.get("inherit"):
                    break
        if not chain:
            return Access()
        merged = Access(resolved=True)
        for data in chain:  # nearest first: the first owner / data_class found wins
            acc = _sidecar_access(data)
            merged.tags += acc.tags
            merged.owner = merged.owner or acc.owner
            merged.data_class = merged.data_class or acc.data_class
        return merged


@register("acl_resolver", "path_map")
class PathMapResolver:
    """Glob patterns (relative to the root) -> groups. Every matching pattern contributes its groups."""

    def __init__(self, map: dict[str, list[str]] | None = None, owners: dict[str, str] | None = None,
                 data_classes: dict[str, str] | None = None):
        self.map = map or {}
        self.owners = owners or {}
        self.data_classes = data_classes or {}

    @staticmethod
    def _match(path: str, pattern: str) -> bool:
        return fnmatch.fnmatch(path, pattern) or (pattern.endswith("/**") and path.startswith(pattern[:-3] + "/"))

    def resolve(self, ref: FileRef, reader: FileBackend) -> Access:
        acc = Access()
        for pattern, groups in self.map.items():
            if self._match(ref.relpath, pattern):
                acc.tags += list(groups)
                acc.resolved = True
        for pattern, owner in self.owners.items():
            if self._match(ref.relpath, pattern):
                acc.owner = owner
                break
        for pattern, dc in self.data_classes.items():
            if self._match(ref.relpath, pattern):
                acc.data_class = DataClass(dc)
                break
        return acc


@register("acl_resolver", "posix")
class PosixResolver:
    """Map POSIX ownership to tags: owner -> ``user:<name>``, group -> ``group:<name>`` when group-readable,
    world-readable -> ``world_tag``. ``uid_map``/``gid_map`` translate ids when the share's accounts are not
    known to the worker (typical for NFS)."""

    def __init__(self, user_tag: str = "user:{name}", group_tag: str = "{name}", world_tag: str | None = None,
                 uid_map: dict[int, str] | None = None, gid_map: dict[int, str] | None = None,
                 include_owner: bool = True):
        self.user_tag, self.group_tag, self.world_tag = user_tag, group_tag, world_tag
        self.uid_map = {int(k): v for k, v in (uid_map or {}).items()}
        self.gid_map = {int(k): v for k, v in (gid_map or {}).items()}
        self.include_owner = include_owner

    def _user(self, uid: int) -> str:
        if uid in self.uid_map:
            return self.uid_map[uid]
        try:
            import pwd

            return pwd.getpwuid(uid).pw_name
        except (ImportError, KeyError):
            return str(uid)

    def _group(self, gid: int) -> str:
        if gid in self.gid_map:
            return self.gid_map[gid]
        try:
            import grp

            return grp.getgrgid(gid).gr_name
        except (ImportError, KeyError):
            return str(gid)

    def resolve(self, ref: FileRef, reader: FileBackend) -> Access:
        if ref.mode is None or ref.uid is None:
            return Access()
        acc = Access(resolved=True)
        owner = self._user(ref.uid)
        acc.owner = owner
        if self.include_owner and ref.mode & stat.S_IRUSR:
            acc.tags.append(self.user_tag.format(name=owner))
        if ref.gid is not None and ref.mode & stat.S_IRGRP:
            acc.tags.append(self.group_tag.format(name=self._group(ref.gid)))
        if self.world_tag and ref.mode & stat.S_IROTH:
            acc.tags.append(self.world_tag)
        return acc


# --------------------------------------------------------------------------- backends


class FileBackend(Protocol):
    def list(self) -> Iterator[FileRef]: ...
    def read_bytes(self, relpath: str) -> bytes: ...
    def read_json(self, relpath: str) -> dict[str, Any] | None: ...
    def uri(self, relpath: str) -> str: ...


class LocalBackend:
    """A local directory or a mounted SMB/NFS share. Symlinks are not followed unless allowed, so a link
    cannot pull files from outside the share into the index."""

    def __init__(self, root: str, follow_symlinks: bool = False):
        self.root = Path(root).resolve()
        self.follow_symlinks = follow_symlinks

    def list(self) -> Iterator[FileRef]:
        if not self.root.exists():
            raise FileNotFoundError(f"crawl root {self.root} does not exist")
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=self.follow_symlinks):
            dirnames.sort()
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.is_symlink() and not self.follow_symlinks:
                    continue
                if self.follow_symlinks and not p.resolve().is_relative_to(self.root):
                    continue
                st = p.stat()
                yield FileRef(relpath=p.relative_to(self.root).as_posix(), abspath=str(p), size=st.st_size,
                              mtime=st.st_mtime, uid=st.st_uid, gid=st.st_gid, mode=st.st_mode)

    def _path(self, relpath: str) -> Path:
        p = (self.root / relpath).resolve()
        if not p.is_relative_to(self.root):
            raise PermissionError(f"path escapes crawl root: {relpath}")
        return p

    def read_bytes(self, relpath: str) -> bytes:
        return self._path(relpath).read_bytes()

    def read_json(self, relpath: str) -> dict[str, Any] | None:
        p = self._path(relpath)
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            log.warning("unreadable ACL sidecar %s: %s", p, exc)
            return {"groups": [], "invalid": True}  # fail closed: a broken sidecar grants nothing
        return data if isinstance(data, dict) else None

    def uri(self, relpath: str) -> str:
        return (self.root / relpath).as_uri()


class S3Backend:
    """S3 or S3-compatible object storage. Object tags are exposed to resolvers as ``metadata.tags``."""

    def __init__(self, bucket: str, prefix: str = "", client: Any = None, endpoint_url: str | None = None,
                 region: str | None = None, read_object_tags: bool = False):
        if client is None:
            import boto3  # optional dependency

            client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        self.s3, self.bucket = client, bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self.read_object_tags = read_object_tags

    def list(self) -> Iterator[FileRef]:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                meta: dict[str, Any] = {"etag": obj.get("ETag")}
                if self.read_object_tags:
                    tags = self.s3.get_object_tagging(Bucket=self.bucket, Key=key).get("TagSet", [])
                    meta["tags"] = {t["Key"]: t["Value"] for t in tags}
                lm = obj.get("LastModified")
                yield FileRef(relpath=key[len(self.prefix):], size=int(obj.get("Size", 0)),
                              mtime=lm.timestamp() if hasattr(lm, "timestamp") else 0.0, metadata=meta)

    def read_bytes(self, relpath: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + relpath)["Body"].read()

    def read_json(self, relpath: str) -> dict[str, Any] | None:
        try:
            raw = self.read_bytes(relpath.lstrip("/"))
        except Exception:  # missing object: no sidecar
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return {"groups": [], "invalid": True}
        return data if isinstance(data, dict) else None

    def uri(self, relpath: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}{relpath}"


# --------------------------------------------------------------------------- crawler


class CrawlStats(BaseModel):
    files_seen: int = 0
    emitted: int = 0
    skipped_no_acl: int = 0
    skipped_filtered: int = 0
    skipped_too_large: int = 0
    errors: int = 0


class FileCrawler:
    """Lists files, resolves each file's access list and yields ``Document`` objects ready for ingestion."""

    def __init__(self, root: str | None = None, *, source: str = "files", backend: str = "local",
                 include: list[str] | None = None, exclude: list[str] | None = None,
                 acl: list[dict[str, Any] | str] | None = None, default_tags: list[str] | None = None,
                 default_data_class: str | DataClass = DataClass.INTERNAL, max_file_mb: float = 50.0,
                 follow_symlinks: bool = False, bucket: str | None = None, prefix: str = "",
                 s3_client: Any = None, endpoint_url: str | None = None, region: str | None = None,
                 read_object_tags: bool = False):
        self.source = source
        if backend == "local":
            if not root:
                raise ValueError("file crawler: 'root' is required for the local backend")
            self.backend: FileBackend = LocalBackend(root, follow_symlinks=follow_symlinks)
        elif backend == "s3":
            if not bucket:
                raise ValueError("file crawler: 'bucket' is required for the s3 backend")
            self.backend = S3Backend(bucket, prefix, client=s3_client, endpoint_url=endpoint_url, region=region,
                                     read_object_tags=read_object_tags)
        else:
            raise ValueError(f"unknown file crawler backend '{backend}' (local, s3)")
        self.include = include or ["**"]
        self.exclude = list(exclude or []) + ["*" + SIDECAR_SUFFIX, "**/.*", ".*"]
        self.resolvers: list[AclResolver] = [create("acl_resolver", spec) for spec in (acl or ["sidecar"])]
        self.default_tags = list(default_tags or [])
        self.default_data_class = DataClass(default_data_class)
        self.max_bytes = int(max_file_mb * 1024 * 1024)
        self.stats = CrawlStats()

    def _selected(self, relpath: str) -> bool:
        name = PurePosixPath(relpath).name
        inc = any(fnmatch.fnmatch(relpath, p) or fnmatch.fnmatch(name, p) or p == "**" for p in self.include)
        exc = any(fnmatch.fnmatch(relpath, p) or fnmatch.fnmatch(name, p) for p in self.exclude)
        return inc and not exc

    def access_for(self, ref: FileRef) -> Access:
        merged = Access()
        for r in self.resolvers:
            acc = r.resolve(ref, self.backend)
            if acc.resolved:
                merged.resolved = True
                merged.tags += acc.tags
                merged.owner = merged.owner or acc.owner
                merged.data_class = DataClass.max(merged.data_class or DataClass.PUBLIC, acc.data_class) \
                    if acc.data_class else merged.data_class
        if not merged.resolved and self.default_tags:
            merged.tags, merged.resolved = list(self.default_tags), True
        merged.tags = sorted(set(merged.tags))
        return merged

    def crawl(self) -> Iterator[Document]:
        """Yield one Document per readable file that has an access list."""
        self.stats = CrawlStats()
        with span("integration.file_crawl", source=self.source) as sp:
            for ref in self.backend.list():
                self.stats.files_seen += 1
                if not self._selected(ref.relpath):
                    self.stats.skipped_filtered += 1
                    continue
                if ref.size > self.max_bytes:
                    self.stats.skipped_too_large += 1
                    metrics.pipeline.labels(self.source, "crawl", "too_large").inc()
                    continue
                access = self.access_for(ref)
                if not access.resolved or not access.tags:
                    # Fail closed: never index a file whose readers we do not know.
                    self.stats.skipped_no_acl += 1
                    metrics.pipeline.labels(self.source, "crawl", "no_acl").inc()
                    log.warning("skipping %s: no access list resolved", ref.relpath)
                    continue
                try:
                    content = self.backend.read_bytes(ref.relpath)
                except OSError as exc:
                    self.stats.errors += 1
                    metrics.pipeline.labels(self.source, "crawl", "error").inc()
                    log.warning("cannot read %s: %s", ref.relpath, exc)
                    continue
                self.stats.emitted += 1
                metrics.pipeline.labels(self.source, "crawl", "ok").inc()
                yield Document(
                    id=f"{self.source}:{ref.relpath}", source=self.source, uri=self.backend.uri(ref.relpath),
                    title=PurePosixPath(ref.relpath).stem.replace("-", " ").replace("_", " "),
                    content=content, mime_type=guess_mime(ref.relpath), access_tags=access.tags,
                    owner=access.owner, updated_at=ref.mtime or 0.0,
                    checksum=checksum_of(content + b"|" + ",".join(access.tags).encode()),
                    metadata={"path": ref.relpath, "size": ref.size,
                              "data_class": (access.data_class or self.default_data_class).value},
                )
            sp.set(**self.stats.model_dump())

    async def documents(self) -> AsyncIterator[Document]:
        for doc in self.crawl():
            yield doc

    async def run(self, sink: DocumentSink, *, run_id: str | None = None) -> CrawlStats:
        """Crawl and send every document to ``sink`` (normally the ingestion work queue)."""
        for doc in self.crawl():
            await sink.emit(doc, source=self.source, run_id=run_id)
        get_audit().record("integration.file_crawl", actor="svc:file-crawler", action="crawl", resource=self.source,
                           decision="ok", run_id=run_id, **self.stats.model_dump())
        return self.stats
