"""Dense vector stores (plugin kind ``vector_store``).

Every store keeps the chunk payload - including ``access_tags`` - next to its vector, and applies the access
and metadata filter *before* scoring (filter-first), so a user's top-k is computed only over chunks they may
see. Collections are created on first upsert with the dimension of the first vector.

Built-ins:
  * ``memory``   - numpy matrix, exact cosine; optional persistence to ``.npz`` snapshots (pilot, tests);
  * ``qdrant``   - Qdrant cluster: shards + replicas, HNSW, payload indexes on access_tags/doc_id, snapshots;
  * ``pgvector`` - PostgreSQL + pgvector: one table per collection, HNSW index, ``access_tags && groups``.

Methods are synchronous; the search service runs stores with ``blocking = True`` in a worker thread.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aiready.core.errors import ConfigError, NotFound, ValidationFailed
from aiready.core.plugins import register
from aiready.core.types import Chunk
from aiready.search.filters import OPERATORS, SearchFilter, field_path, matches

log = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    blocking: bool

    def upsert(self, collection: str, chunks: list[Chunk]) -> int: ...
    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int: ...
    def search(
        self, collection: str, vector: list[float], flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]: ...
    def count(self, collection: str) -> int: ...
    def collections(self) -> list[str]: ...
    def snapshot(self, collection: str, directory: str) -> str: ...
    def restore(self, collection: str, location: str) -> int: ...


def payload_of(chunk: Chunk) -> dict[str, Any]:
    return chunk.model_dump(mode="json", exclude={"vector"})


def _require_vectors(chunks: list[Chunk]) -> int:
    dims = {len(c.vector) for c in chunks if c.vector is not None}
    if any(c.vector is None for c in chunks):
        raise ValidationFailed("every chunk needs a vector for the dense index")
    if len(dims) > 1:
        raise ValidationFailed(f"mixed vector dimensions in one upsert: {sorted(dims)}")
    return dims.pop() if dims else 0


# =========================================================================== memory


class _MemCollection:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.payloads: list[dict[str, Any]] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, chunks: list[Chunk]) -> None:
        new_rows: list[np.ndarray] = []
        for c in chunks:
            v = np.asarray(c.vector, dtype=np.float32)
            n = float(np.linalg.norm(v))
            v = v / n if n else v
            if c.id in self.index:
                row = self.index[c.id]
                self.vectors[row] = v
                self.payloads[row] = payload_of(c)
            else:
                self.index[c.id] = len(self.ids)
                self.ids.append(c.id)
                self.payloads.append(payload_of(c))
                new_rows.append(v)
        if new_rows:
            self.vectors = np.vstack([self.vectors, np.stack(new_rows)])

    def delete(self, keep: list[bool]) -> int:
        removed = len(keep) - sum(keep)
        if removed:
            mask = np.asarray(keep, dtype=bool)
            self.vectors = self.vectors[mask]
            self.ids = [i for i, k in zip(self.ids, keep) if k]
            self.payloads = [p for p, k in zip(self.payloads, keep) if k]
            self.index = {cid: n for n, cid in enumerate(self.ids)}
        return removed


@register("vector_store", "memory")
class MemoryVectorStore:
    """Exact cosine search over a numpy matrix per collection.

    Good to a few hundred thousand chunks per process. ``persist_dir`` makes every write save the collection
    to ``<persist_dir>/<collection>.npz`` and reload it at start, so a single-node pilot survives restarts.
    """

    blocking = False

    def __init__(self, persist_dir: str | None = None):
        self._cols: dict[str, _MemCollection] = {}
        self._lock = threading.RLock()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir and self.persist_dir.exists():
            for f in sorted(self.persist_dir.glob("*.npz")):
                self._load(f.stem, f)

    # ---------------------------------------------------------------- persistence helpers
    def _save(self, collection: str, path: Path) -> None:
        col = self._cols[collection]
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = json.dumps(
            {"collection": collection, "dim": col.dim, "ids": col.ids, "payloads": col.payloads}
        )
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(tmp, vectors=col.vectors, meta=np.frombuffer(meta.encode(), dtype=np.uint8))
        tmp.replace(path)  # atomic: a crash mid-write never leaves a half snapshot

    def _load(self, collection: str, path: Path) -> int:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode())
            col = _MemCollection(int(meta["dim"]))
            col.vectors = np.asarray(data["vectors"], dtype=np.float32).reshape(-1, col.dim)
        col.ids, col.payloads = list(meta["ids"]), list(meta["payloads"])
        col.index = {cid: n for n, cid in enumerate(col.ids)}
        with self._lock:
            self._cols[collection] = col
        return len(col.ids)

    def _persist(self, collection: str) -> None:
        if self.persist_dir and collection in self._cols:
            self._save(collection, self.persist_dir / f"{collection}.npz")

    # ---------------------------------------------------------------- API
    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        dim = _require_vectors(chunks)
        with self._lock:
            col = self._cols.get(collection)
            if col is None:
                col = self._cols[collection] = _MemCollection(dim)
            elif col.dim != dim:
                raise ValidationFailed(f"collection '{collection}' has dim {col.dim}, got {dim}")
            col.upsert(chunks)
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
            keep = [not (p["doc_id"] in docs or p["id"] in ids) for p in col.payloads]
            removed = col.delete(keep)
            if removed:
                self._persist(collection)
            return removed

    def search(
        self, collection: str, vector: list[float], flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]:
        with self._lock:
            col = self._cols.get(collection)
            if col is None or not col.ids or k <= 0:
                return []
            # filter first: only rows the caller may see are scored
            rows = [i for i, p in enumerate(col.payloads) if matches(p, flt)]
            if not rows:
                return []
            q = np.asarray(vector, dtype=np.float32)
            if q.shape[0] != col.dim:
                raise ValidationFailed(f"query vector dim {q.shape[0]} != collection dim {col.dim}")
            n = float(np.linalg.norm(q))
            q = q / n if n else q
            idx = np.asarray(rows)
            scores = col.vectors[idx] @ q
            k = min(k, len(rows))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top], kind="stable")]
            return [(Chunk.model_validate(col.payloads[int(idx[t])]), float(scores[t])) for t in top]

    def count(self, collection: str) -> int:
        with self._lock:
            col = self._cols.get(collection)
            return len(col.ids) if col else 0

    def collections(self) -> list[str]:
        with self._lock:
            return sorted(self._cols)

    def dim(self, collection: str) -> int | None:
        col = self._cols.get(collection)
        return col.dim if col else None

    def snapshot(self, collection: str, directory: str) -> str:
        with self._lock:
            if collection not in self._cols:
                raise NotFound(f"collection '{collection}' does not exist")
            path = (
                Path(directory) / f"{collection}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}.npz"
            )
            self._save(collection, path)
        return str(path)

    def restore(self, collection: str, location: str) -> int:
        path = Path(location)
        if not path.is_file():
            raise NotFound(f"snapshot '{location}' not found")
        n = self._load(collection, path)
        with self._lock:
            self._persist(collection)
        return n


# =========================================================================== qdrant


@register("vector_store", "qdrant")
class QdrantVectorStore:
    """Qdrant cluster. One collection per application and tenant; access tags stored in every payload with a
    keyword payload index, so the filter is applied inside the HNSW traversal (filter-first)."""

    blocking = True

    def __init__(
        self,
        url: str = "http://localhost:6333",
        *,
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout: float = 30.0,
        shard_number: int = 1,
        replication_factor: int = 1,
        write_consistency_factor: int = 1,
        on_disk: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        search_ef: int | None = None,
        distance: str = "Cosine",
        client: Any = None,
    ):
        if client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise ConfigError("vector_store 'qdrant' needs `pip install aiready[qdrant]`") from exc
            client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=int(timeout))
        self.client, self.url = client, url
        self.shard_number, self.replication_factor = shard_number, replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk, self.hnsw_m, self.hnsw_ef_construct = on_disk, hnsw_m, hnsw_ef_construct
        self.search_ef, self.distance = search_ef, distance
        self._known: set[str] = set()

    @staticmethod
    def _models() -> Any:
        from qdrant_client import models

        return models

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))  # Qdrant ids must be uint or UUID

    def _ensure(self, collection: str, dim: int) -> None:
        if collection in self._known:
            return
        m = self._models()
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=m.VectorParams(
                    size=dim, distance=getattr(m.Distance, self.distance.upper()), on_disk=self.on_disk
                ),
                hnsw_config=m.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct),
                shard_number=self.shard_number,
                replication_factor=self.replication_factor,
                write_consistency_factor=self.write_consistency_factor,
            )
            for field, schema in (
                ("access_tags", m.PayloadSchemaType.KEYWORD),
                ("doc_id", m.PayloadSchemaType.KEYWORD),
                ("id", m.PayloadSchemaType.KEYWORD),
                ("source", m.PayloadSchemaType.KEYWORD),
                ("data_class", m.PayloadSchemaType.KEYWORD),
                ("updated_at", m.PayloadSchemaType.FLOAT),
            ):
                self.client.create_payload_index(collection, field_name=field, field_schema=schema)
        self._known.add(collection)

    def _filter(self, flt: SearchFilter | None) -> Any:
        if flt is None:
            return None
        m = self._models()
        must: list[Any] = []
        must_not: list[Any] = []
        if flt.access_groups is not None:
            must.append(
                m.FieldCondition(key="access_tags", match=m.MatchAny(any=list(flt.access_groups) or ["\x00"]))
            )
        for key, cond in flt.metadata.items():
            path = ".".join(field_path(key))
            if isinstance(cond, dict):
                rng = {op: cond[op] for op in ("gt", "gte", "lt", "lte") if op in cond}
                if rng:
                    must.append(m.FieldCondition(key=path, range=m.Range(**rng)))
                if "eq" in cond:
                    must.append(m.FieldCondition(key=path, match=m.MatchValue(value=cond["eq"])))
                if "in" in cond:
                    must.append(m.FieldCondition(key=path, match=m.MatchAny(any=list(cond["in"]))))
                if "ne" in cond:
                    must_not.append(m.FieldCondition(key=path, match=m.MatchValue(value=cond["ne"])))
                if "nin" in cond:
                    must_not.append(m.FieldCondition(key=path, match=m.MatchAny(any=list(cond["nin"]))))
            elif isinstance(cond, (list, tuple, set)):
                must.append(m.FieldCondition(key=path, match=m.MatchAny(any=list(cond))))
            else:
                must.append(m.FieldCondition(key=path, match=m.MatchValue(value=cond)))
        return m.Filter(must=must or None, must_not=must_not or None)

    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        dim = _require_vectors(chunks)
        self._ensure(collection, dim)
        m = self._models()
        points = [
            m.PointStruct(id=self.point_id(c.id), vector=list(c.vector or []), payload=payload_of(c))
            for c in chunks
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(collection_name=collection, points=points[i : i + 256], wait=True)
        return len(points)

    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int:
        if not self.client.collection_exists(collection) or not (doc_ids or chunk_ids):
            return 0
        m = self._models()
        should = []
        if doc_ids:
            should.append(m.FieldCondition(key="doc_id", match=m.MatchAny(any=list(doc_ids))))
        if chunk_ids:
            should.append(m.FieldCondition(key="id", match=m.MatchAny(any=list(chunk_ids))))
        flt = m.Filter(should=should)
        n = self.client.count(collection, count_filter=flt, exact=True).count
        self.client.delete(
            collection_name=collection, points_selector=m.FilterSelector(filter=flt), wait=True
        )
        return int(n)

    def search(
        self, collection: str, vector: list[float], flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]:
        if not self.client.collection_exists(collection):
            return []
        m = self._models()
        params = m.SearchParams(hnsw_ef=self.search_ef) if self.search_ef else None
        res = self.client.query_points(
            collection_name=collection,
            query=list(vector),
            query_filter=self._filter(flt),
            limit=k,
            with_payload=True,
            search_params=params,
        )
        return [(Chunk.model_validate(p.payload), float(p.score)) for p in res.points]

    def count(self, collection: str) -> int:
        if not self.client.collection_exists(collection):
            return 0
        return int(self.client.count(collection, exact=True).count)

    def collections(self) -> list[str]:
        return sorted(c.name for c in self.client.get_collections().collections)

    def snapshot(self, collection: str, directory: str) -> str:
        snap = self.client.create_snapshot(collection_name=collection, wait=True)
        return f"qdrant://{collection}/{snap.name}"

    def restore(self, collection: str, location: str) -> int:
        """``location``: a ``qdrant://<collection>/<snapshot>`` id from ``snapshot``, or a URL / file
        Qdrant can read."""
        if location.startswith("qdrant://"):
            src, name = location[len("qdrant://") :].split("/", 1)
            location = f"{self.url.rstrip('/')}/collections/{src}/snapshots/{name}"
        self.client.recover_snapshot(collection_name=collection, location=location, wait=True)
        self._known.discard(collection)
        return self.count(collection)


# =========================================================================== pgvector

_SAFE = re.compile(r"[^a-z0-9_]")


def pg_table(collection: str, prefix: str = "aiready_vec_") -> str:
    name = prefix + _SAFE.sub("_", collection.lower())
    return name[:63]


def pg_filter_sql(flt: SearchFilter | None, params: dict[str, Any]) -> str:
    """Translate a ``SearchFilter`` to a SQL ``WHERE`` fragment.

    Values are always bound parameters, never interpolated into the SQL string."""
    if flt is None:
        return "TRUE"
    clauses: list[str] = []
    if flt.access_groups is not None:
        params["groups"] = list(flt.access_groups)
        clauses.append("access_tags && CAST(:groups AS text[])")
    for n, (key, cond) in enumerate(flt.metadata.items()):
        path = field_path(key)
        if path == ("doc_id",):
            col = "doc_id"
        elif path == ("access_tags",):
            col = None
        else:
            params[f"p{n}"] = "{" + ",".join(path) + "}"
            col = f"(payload #>> CAST(:p{n} AS text[]))"
        if col is None:  # extra tag restriction
            vals = cond if isinstance(cond, (list, tuple, set)) else [cond]
            params[f"v{n}"] = [str(v) for v in vals]
            clauses.append(f"access_tags && CAST(:v{n} AS text[])")
            continue
        ops = (
            cond
            if isinstance(cond, dict)
            else ({"in": list(cond)} if isinstance(cond, (list, tuple, set)) else {"eq": cond})
        )
        for j, (op, val) in enumerate(ops.items()):
            if op not in OPERATORS:
                raise ValidationFailed(f"unknown operator {op}")
            pn = f"v{n}_{j}"
            if op in ("in", "nin"):
                params[pn] = [str(v) for v in val]
                clauses.append(f"{'NOT ' if op == 'nin' else ''}({col} = ANY(CAST(:{pn} AS text[])))")
            elif op in ("eq", "ne"):
                params[pn] = str(val.value if hasattr(val, "value") else val)
                clauses.append(f"{col} {'=' if op == 'eq' else '<>'} :{pn}")
            else:
                sym = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
                if isinstance(val, (int, float)):
                    params[pn] = float(val)
                    clauses.append(f"CAST({col} AS double precision) {sym} :{pn}")
                else:
                    params[pn] = str(val)
                    clauses.append(f"{col} {sym} :{pn}")
    return " AND ".join(clauses) or "TRUE"


@register("vector_store", "pgvector")
class PgVectorStore:
    """PostgreSQL + pgvector.

    One table per collection (``aiready_vec_<collection>``) with an HNSW index (``vector_cosine_ops``) and a
    GIN index on ``access_tags``; the access filter is ``access_tags && groups``.
    """

    blocking = True

    def __init__(
        self,
        url: str = "postgresql+psycopg://localhost/aiready",
        *,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        ef_search: int = 100,
        table_prefix: str = "aiready_vec_",
        pool_size: int = 5,
        engine: Any = None,
    ):
        from sqlalchemy import create_engine

        self.engine = engine or create_engine(url, pool_size=pool_size, pool_pre_ping=True, future=True)
        self.hnsw_m, self.hnsw_ef_construction, self.ef_search = hnsw_m, hnsw_ef_construction, ef_search
        self.prefix = table_prefix
        self._known: set[str] = set()

    def ddl(self, collection: str, dim: int) -> list[str]:
        t = pg_table(collection, self.prefix)
        return [
            "CREATE EXTENSION IF NOT EXISTS vector",
            (
                f"CREATE TABLE IF NOT EXISTS {t} (id text PRIMARY KEY, doc_id text NOT NULL,"
                " access_tags text[] NOT NULL, data_class text, updated_at double precision,"
                f" payload jsonb NOT NULL, embedding vector({int(dim)}) NOT NULL)"
            ),
            (
                f"CREATE INDEX IF NOT EXISTS {t}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops)"
                f" WITH (m = {int(self.hnsw_m)}, ef_construction = {int(self.hnsw_ef_construction)})"
            ),
            f"CREATE INDEX IF NOT EXISTS {t}_tags ON {t} USING gin (access_tags)",
            f"CREATE INDEX IF NOT EXISTS {t}_doc ON {t} (doc_id)",
        ]

    def _exists(self, collection: str) -> bool:
        from sqlalchemy import inspect

        return inspect(self.engine).has_table(pg_table(collection, self.prefix))

    def _ensure(self, collection: str, dim: int) -> None:
        if collection in self._known:
            return
        from sqlalchemy import text

        with self.engine.begin() as c:
            for stmt in self.ddl(collection, dim):
                c.execute(text(stmt))
        self._known.add(collection)

    @staticmethod
    def _vec(v: list[float]) -> str:
        return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"

    def upsert(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        from sqlalchemy import text

        dim = _require_vectors(chunks)
        self._ensure(collection, dim)
        t = pg_table(collection, self.prefix)
        stmt = text(
            f"INSERT INTO {t} (id, doc_id, access_tags, data_class, updated_at, payload, embedding) VALUES "
            "(:id, :doc_id, CAST(:tags AS text[]), :dc, :ts, CAST(:payload AS jsonb), CAST(:emb AS vector)) "
            "ON CONFLICT (id) DO UPDATE SET doc_id = EXCLUDED.doc_id, access_tags = EXCLUDED.access_tags, "
            "data_class = EXCLUDED.data_class, updated_at = EXCLUDED.updated_at, payload = EXCLUDED.payload, "
            "embedding = EXCLUDED.embedding"
        )
        rows = [
            {
                "id": c.id,
                "doc_id": c.doc_id,
                "tags": list(c.access_tags),
                "dc": c.data_class.value,
                "ts": c.updated_at,
                "payload": json.dumps(payload_of(c)),
                "emb": self._vec(c.vector or []),
            }
            for c in chunks
        ]
        with self.engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(rows)

    def delete(
        self, collection: str, *, doc_ids: list[str] | None = None, chunk_ids: list[str] | None = None
    ) -> int:
        if not (doc_ids or chunk_ids) or not self._exists(collection):
            return 0
        from sqlalchemy import text

        t = pg_table(collection, self.prefix)
        with self.engine.begin() as c:
            res = c.execute(
                text(
                    f"DELETE FROM {t} WHERE doc_id = ANY(CAST(:d AS text[])) OR id = ANY(CAST(:i AS text[]))"
                ),
                {"d": list(doc_ids or []), "i": list(chunk_ids or [])},
            )
        return int(res.rowcount or 0)

    def search(
        self, collection: str, vector: list[float], flt: SearchFilter | None, k: int
    ) -> list[tuple[Chunk, float]]:
        if not self._exists(collection):
            return []
        from sqlalchemy import text

        t = pg_table(collection, self.prefix)
        params: dict[str, Any] = {"q": self._vec(vector), "k": int(k)}
        where = pg_filter_sql(flt, params)
        sql = text(
            f"SELECT payload, 1 - (embedding <=> CAST(:q AS vector)) AS score FROM {t} WHERE {where} "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"
        )
        with self.engine.begin() as c:
            c.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))
            # iterative scans (pgvector >= 0.8) keep recall when the access filter is selective
            try:
                with c.begin_nested():
                    c.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            except Exception:
                log.debug("pgvector without iterative scan support")
            rows = c.execute(sql, params).all()
        return [
            (Chunk.model_validate(p if isinstance(p, dict) else json.loads(p)), float(s)) for p, s in rows
        ]

    def count(self, collection: str) -> int:
        if not self._exists(collection):
            return 0
        from sqlalchemy import text

        with self.engine.connect() as c:
            return int(
                c.execute(text(f"SELECT count(*) FROM {pg_table(collection, self.prefix)}")).scalar() or 0
            )

    def collections(self) -> list[str]:
        from sqlalchemy import inspect

        return sorted(
            n[len(self.prefix) :] for n in inspect(self.engine).get_table_names() if n.startswith(self.prefix)
        )

    def snapshot(self, collection: str, directory: str) -> str:
        """Logical export to JSONL (payload + vector).

        Use the database's own backups for point-in-time recovery."""
        from sqlalchemy import text

        path = Path(directory) / f"{collection}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.engine.connect() as c, path.open("w") as f:
            for payload, emb in c.execute(
                text(f"SELECT payload, embedding::text FROM {pg_table(collection, self.prefix)}")
            ):
                p = payload if isinstance(payload, dict) else json.loads(payload)
                f.write(json.dumps({"payload": p, "vector": json.loads(emb)}) + "\n")
        return str(path)

    def restore(self, collection: str, location: str) -> int:
        path = Path(location)
        if not path.is_file():
            raise NotFound(f"snapshot '{location}' not found")
        from sqlalchemy import text

        chunks = []
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                chunks.append(Chunk.model_validate({**rec["payload"], "vector": rec["vector"]}))
        if self._exists(collection):
            with self.engine.begin() as c:
                c.execute(text(f"TRUNCATE {pg_table(collection, self.prefix)}"))
        for i in range(0, len(chunks), 500):
            self.upsert(collection, chunks[i : i + 500])
        return len(chunks)
