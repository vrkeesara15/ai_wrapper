"""Landing store: raw copies of every fetched item, kept for replay.

Keeping the raw bytes means a parser fix, a new chunking strategy or a new embedding model can be applied by
re-running the pipeline from the landing store, without hitting the source system again (which may be a
fragile legacy system or a rate-limited SaaS API). Plugin kind ``landing_store``: ``local`` (a directory,
default) and ``s3`` (object storage, ``boto3`` imported lazily).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from aiready.core.plugins import create, register
from aiready.core.types import Document
from aiready.ingestion.documents import decode_document, encode_document


class LandingStore(Protocol):
    def put(self, doc: Document) -> str: ...
    def get(self, uri: str) -> Document: ...
    def versions(self, source: str, doc_id: str) -> list[str]: ...


def _doc_dir(source: str, doc_id: str) -> str:
    safe_source = "".join(c if c.isalnum() or c in "-_" else "_" for c in source)
    return f"{safe_source}/{hashlib.sha256(doc_id.encode()).hexdigest()[:32]}"


@register("landing_store", "local")
class LocalLanding:
    """``<path>/<source>/<sha(doc id)>/<checksum>.json`` - one file per version; old versions pruned."""

    def __init__(self, path: str = "data/landing", keep_versions: int = 3):
        self.root = Path(path)
        self.keep = max(1, keep_versions)

    def put(self, doc: Document) -> str:
        d = self.root / _doc_dir(doc.source, doc.id)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{(doc.checksum or 'nochecksum')[:40]}.json"
        target = d / name
        target.write_text(json.dumps(encode_document(doc)), encoding="utf-8")
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in files[:-self.keep]:
            if old != target:
                old.unlink(missing_ok=True)
        return target.resolve().as_uri()

    def get(self, uri: str) -> Document:
        path = Path(uri[len("file://"):]) if uri.startswith("file://") else Path(uri)
        return decode_document(json.loads(path.read_text(encoding="utf-8")))

    def versions(self, source: str, doc_id: str) -> list[str]:
        d = self.root / _doc_dir(source, doc_id)
        return [p.resolve().as_uri() for p in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)] if d.exists() else []


@register("landing_store", "s3")
class S3Landing:
    def __init__(self, bucket: str, prefix: str = "landing/", client: Any = None, endpoint_url: str | None = None,
                 region: str | None = None):
        if client is None:
            import boto3  # optional dependency

            client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        self.s3, self.bucket = client, bucket
        self.prefix = prefix if prefix.endswith("/") or not prefix else prefix + "/"

    def put(self, doc: Document) -> str:
        key = f"{self.prefix}{_doc_dir(doc.source, doc.id)}/{(doc.checksum or 'nochecksum')[:40]}.json"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(encode_document(doc)).encode(),
                           ContentType="application/json")
        return f"s3://{self.bucket}/{key}"

    def get(self, uri: str) -> Document:
        key = uri.split(f"s3://{self.bucket}/", 1)[-1]
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return decode_document(json.loads(body))

    def versions(self, source: str, doc_id: str) -> list[str]:
        prefix = f"{self.prefix}{_doc_dir(source, doc_id)}/"
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        objs = sorted(resp.get("Contents", []), key=lambda o: o.get("LastModified"))
        return [f"s3://{self.bucket}/{o['Key']}" for o in objs]


def make_landing(cfg: dict[str, Any]) -> LandingStore:
    spec = {k: v for k, v in cfg.items() if v is not None}
    return create("landing_store", spec)
