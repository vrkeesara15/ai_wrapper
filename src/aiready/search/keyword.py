"""Keyword indexes (plugin kind ``keyword_index``): exact terms, codes and names.

Dense vectors are weak on identifiers (policy numbers, error codes, product names); BM25 over the same
chunks catches them. Like the vector stores, keyword indexes filter by access and metadata before scoring.

Built-ins:
  * ``bm25``       - in-memory Okapi BM25 (k1/b configurable), tokenizer from ``core.textutil``;
  * ``opensearch`` - OpenSearch/Elasticsearch over REST (httpx; no client library required).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import httpx

from aiready.core.errors import NotFound, UpstreamError
from aiready.core.plugins import register
from aiready.core.textutil import tokenize
from aiready.core.types import Chunk
from aiready.search.filters import SearchFilter, field_path, matches
from aiready.search.stores import payload_of

log = logging.getLogger(__name__)


@runtime_checkable
class KeywordIndex(Protocol):
    blocking: bool

    def upsert(self, collection: str, chunks: list[Chunk]) -> int: ...
    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int: ...
    def search(
        self, collection: str, text: str, flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]: ...
    def count(self, collection: str) -> int: ...
    def snapshot(self, collection: str, directory: str) -> str: ...
    def restore(self, collection: str, location: str) -> int: ...


def index_text(chunk: Chunk) -> str:
    """Text indexed for a chunk: title and section boost exact-name matches."""
    return " ".join(x for x in (chunk.title, chunk.section, chunk.text) if x)


# =========================================================================== bm25


class _Bm25Collection:
    def __init__(self) -> None:
        self.payloads: dict[str, dict[str, Any]] = {}
        self.tf: dict[str, Counter[str]] = {}
        self.lengths: dict[str, int] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.total_len = 0

    def remove(self, cid: str) -> None:
        tf = self.tf.pop(cid, None)
        if tf is None:
            return
        for term in tf:
            ids = self.postings.get(term)
            if ids is not None:
                ids.discard(cid)
                if not ids:
                    del self.postings[term]
        self.total_len -= self.lengths.pop(cid, 0)
        self.payloads.pop(cid, None)

    def add(self, chunk: Chunk, tokens: list[str]) -> None:
        self.remove(chunk.id)
        tf = Counter(tokens)
        self.tf[chunk.id] = tf
        self.lengths[chunk.id] = len(tokens)
        self.total_len += len(tokens)
        self.payloads[chunk.id] = payload_of(chunk)
        for term in tf:
            self.postings[term].add(chunk.id)


@register("keyword_index", "bm25")
class Bm25Index:
    """In-memory Okapi BM25.

    score(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
    """

    blocking = False

    def __init__(
        self, k1: float = 1.2, b: float = 0.75, drop_stopwords: bool = True, persist_dir: str | None = None
    ):
        self.k1, self.b, self.drop_stopwords = k1, b, drop_stopwords
        self._cols: dict[str, _Bm25Collection] = {}
        self._lock = threading.RLock()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir and self.persist_dir.exists():
            for f in sorted(self.persist_dir.glob("*.json")):
                self.restore(f.stem, str(f))

    def tokens(self, text: str) -> list[str]:
        return tokenize(text, drop_stopwords=self.drop_stopwords)

    def _persist(self, collection: str) -> None:
        if self.persist_dir:
            self._write(collection, self.persist_dir / f"{collection}.json")

    def _write(self, collection: str, path: Path) -> None:
        col = self._cols.get(collection) or _Bm25Collection()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"collection": collection, "payloads": list(col.payloads.values())}))
        tmp.replace(path)

    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        with self._lock:
            col = self._cols.setdefault(collection, _Bm25Collection())
            for c in chunks:
                col.add(c, self.tokens(index_text(c)))
            if chunks:
                self._persist(collection)
        return len(chunks)

    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int:
        docs, ids = set(doc_ids or ()), set(chunk_ids or ())
        with self._lock:
            col = self._cols.get(collection)
            if col is None or not (docs or ids):
                return 0
            victims = [cid for cid, p in col.payloads.items() if cid in ids or p["doc_id"] in docs]
            for cid in victims:
                col.remove(cid)
            if victims:
                self._persist(collection)
            return len(victims)

    def search(
        self, collection: str, text: str, flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]:
        with self._lock:
            col = self._cols.get(collection)
            terms = list(dict.fromkeys(self.tokens(text)))
            if col is None or not col.tf or not terms or k <= 0:
                return []
            n_docs = len(col.tf)
            avgdl = (col.total_len / n_docs) or 1.0
            candidates: set[str] = set()
            for t in terms:
                candidates |= col.postings.get(t, set())
            # filter first: only chunks the caller may see are scored
            visible = [cid for cid in candidates if matches(col.payloads[cid], flt)]
            idf = {
                t: math.log(
                    1 + (n_docs - len(col.postings.get(t, ())) + 0.5) / (len(col.postings.get(t, ())) + 0.5)
                )
                for t in terms
            }
            scored: list[tuple[str, float]] = []
            for cid in visible:
                tf, dl = col.tf[cid], col.lengths[cid]
                s = 0.0
                for t in terms:
                    f = tf.get(t, 0)
                    if f:
                        s += idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
                if s > 0:
                    scored.append((cid, s))
            scored.sort(key=lambda x: (-x[1], x[0]))
            return [(Chunk.model_validate(col.payloads[cid]), s) for cid, s in scored[:k]]

    def count(self, collection: str) -> int:
        with self._lock:
            col = self._cols.get(collection)
            return len(col.tf) if col else 0

    def snapshot(self, collection: str, directory: str) -> str:
        with self._lock:
            if collection not in self._cols:
                raise NotFound(f"collection '{collection}' does not exist")
            path = (
                Path(directory)
                / f"{collection}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}.bm25.json"
            )
            self._write(collection, path)
        return str(path)

    def restore(self, collection: str, location: str) -> int:
        path = Path(location)
        if not path.is_file():
            raise NotFound(f"snapshot '{location}' not found")
        payloads = json.loads(path.read_text())["payloads"]
        col = _Bm25Collection()
        for p in payloads:
            c = Chunk.model_validate(p)
            col.add(c, self.tokens(index_text(c)))
        with self._lock:
            self._cols[collection] = col
            if self.persist_dir and path.parent.resolve() != self.persist_dir.resolve():
                self._persist(collection)
        return len(payloads)


# =========================================================================== opensearch

_SAFE = re.compile(r"[^a-z0-9_\-]")


@register("keyword_index", "opensearch")
class OpenSearchIndex:
    """OpenSearch / Elasticsearch BM25 over REST. One index per collection (``<prefix><collection>``).

    Snapshots use a snapshot repository registered on the cluster (``snapshot_repository``).
    """

    blocking = True

    def __init__(
        self,
        url: str = "http://localhost:9200",
        *,
        index_prefix: str = "aiready-kw-",
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        verify_tls: bool = True,
        timeout: float = 30.0,
        shards: int = 1,
        replicas: int = 1,
        k1: float = 1.2,
        b: float = 0.75,
        snapshot_repository: str = "aiready",
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"ApiKey {api_key}"
        auth = (username, password) if username and password else None
        self.http = httpx.Client(
            base_url=url.rstrip("/"),
            headers=headers,
            auth=auth,
            verify=verify_tls,
            timeout=timeout,
            transport=transport,
        )
        self.prefix, self.shards, self.replicas, self.k1, self.b = index_prefix, shards, replicas, k1, b
        self.repo = snapshot_repository
        self._known: set[str] = set()

    def index(self, collection: str) -> str:
        return (self.prefix + _SAFE.sub("-", collection.lower()))[:255]

    def _req(self, method: str, path: str, *, ok404: bool = False, **kw: Any) -> dict[str, Any]:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"opensearch unreachable: {exc}") from exc
        if r.status_code == 404 and ok404:
            return {}
        if r.status_code >= 400:
            raise UpstreamError(
                f"opensearch {method} {path} returned {r.status_code}", details={"body": r.text[:500]}
            )
        return r.json() if r.content else {}

    def _ensure(self, collection: str) -> None:
        if collection in self._known:
            return
        idx = self.index(collection)
        if self.http.head(f"/{idx}").status_code == 404:
            body = {
                "settings": {
                    "number_of_shards": self.shards,
                    "number_of_replicas": self.replicas,
                    "index": {"similarity": {"default": {"type": "BM25", "k1": self.k1, "b": self.b}}},
                },
                "mappings": {
                    "dynamic": False,
                    "properties": {
                        "id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "access_tags": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "data_class": {"type": "keyword"},
                        "owner": {"type": "keyword"},
                        "section": {"type": "keyword"},
                        "updated_at": {"type": "double"},
                        "title": {"type": "text"},
                        "text": {"type": "text"},
                        "metadata": {"type": "object", "dynamic": True},
                        "payload": {"type": "object", "enabled": False},
                    },
                },
            }
            self._req("PUT", f"/{idx}", json=body)
        self._known.add(collection)

    @staticmethod
    def _doc(chunk: Chunk) -> dict[str, Any]:
        p = payload_of(chunk)
        return {
            **{
                k: p.get(k)
                for k in (
                    "id",
                    "doc_id",
                    "access_tags",
                    "source",
                    "data_class",
                    "owner",
                    "section",
                    "updated_at",
                    "title",
                    "metadata",
                )
            },
            "text": index_text(chunk),
            "payload": p,
        }

    def _query_filter(self, flt: SearchFilter | None) -> tuple[list[dict], list[dict]]:
        must: list[dict] = []
        must_not: list[dict] = []
        if flt is None:
            return must, must_not
        if flt.access_groups is not None:
            must.append({"terms": {"access_tags": list(flt.access_groups) or ["\x00"]}})
        for key, cond in flt.metadata.items():
            f = ".".join(field_path(key))
            if isinstance(cond, dict):
                rng = {op: cond[op] for op in ("gt", "gte", "lt", "lte") if op in cond}
                if rng:
                    must.append({"range": {f: rng}})
                if "eq" in cond:
                    must.append({"term": {f: cond["eq"]}})
                if "in" in cond:
                    must.append({"terms": {f: list(cond["in"])}})
                if "ne" in cond:
                    must_not.append({"term": {f: cond["ne"]}})
                if "nin" in cond:
                    must_not.append({"terms": {f: list(cond["nin"])}})
            elif isinstance(cond, (list, tuple, set)):
                must.append({"terms": {f: list(cond)}})
            else:
                must.append({"term": {f: cond}})
        return must, must_not

    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._ensure(collection)
        idx = self.index(collection)
        lines = []
        for c in chunks:
            lines.append(json.dumps({"index": {"_index": idx, "_id": c.id}}))
            lines.append(json.dumps(self._doc(c), default=str))
        res = self._req(
            "POST",
            "/_bulk?refresh=wait_for",
            content="\n".join(lines) + "\n",
            headers={"Content-Type": "application/x-ndjson"},
        )
        if res.get("errors"):
            raise UpstreamError(
                "opensearch bulk upsert had errors",
                details={"first": next((i for i in res.get("items", []) if "error" in str(i)), None)},
            )
        return len(chunks)

    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int:
        if not (doc_ids or chunk_ids):
            return 0
        should = []
        if doc_ids:
            should.append({"terms": {"doc_id": list(doc_ids)}})
        if chunk_ids:
            should.append({"terms": {"id": list(chunk_ids)}})
        res = self._req(
            "POST",
            f"/{self.index(collection)}/_delete_by_query?refresh=true",
            ok404=True,
            json={"query": {"bool": {"should": should, "minimum_should_match": 1}}},
        )
        return int(res.get("deleted", 0))

    def search(
        self, collection: str, text: str, flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]:
        must, must_not = self._query_filter(flt)
        body = {
            "size": k,
            "_source": ["payload"],
            "query": {
                "bool": {
                    "must": [{"multi_match": {"query": text, "fields": ["title^2", "text"]}}],
                    "filter": must,
                    "must_not": must_not,
                }
            },
        }
        res = self._req("POST", f"/{self.index(collection)}/_search", ok404=True, json=body)
        return [
            (Chunk.model_validate(h["_source"]["payload"]), float(h["_score"]))
            for h in res.get("hits", {}).get("hits", [])
        ]

    def count(self, collection: str) -> int:
        return int(self._req("GET", f"/{self.index(collection)}/_count", ok404=True).get("count", 0))

    def snapshot(self, collection: str, directory: str) -> str:
        name = f"{self.index(collection)}-{time.strftime('%Y%m%dt%H%M%S')}"
        self._req(
            "PUT",
            f"/_snapshot/{self.repo}/{name}?wait_for_completion=true",
            json={"indices": self.index(collection), "include_global_state": False},
        )
        return f"opensearch://{self.repo}/{name}"

    def restore(self, collection: str, location: str) -> int:
        repo, name = location[len("opensearch://") :].split("/", 1)
        idx = self.index(collection)
        self._req("DELETE", f"/{idx}", ok404=True)
        self._req(
            "POST",
            f"/_snapshot/{repo}/{name}/_restore?wait_for_completion=true",
            json={
                "indices": "*",
                "rename_pattern": "(.+)",
                "rename_replacement": idx,
                "include_global_state": False,
            },
        )
        self._known.discard(collection)
        return self.count(collection)
