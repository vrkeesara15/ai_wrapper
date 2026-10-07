"""Retriever: hybrid search with the user's access tags, one search per sub-query, merged and de-duplicated.

Access filtering happens *in* the search service (``SearchRequest.access_groups``) so a passage the user may
not see never leaves the index - it is not filtered afterwards. The retriever passes
``principal.access_groups`` on every call and never widens them.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from aiready.core.identity import Principal
from aiready.core.plugins import register
from aiready.core.telemetry import span
from aiready.core.types import SearchHit, SearchRequest

log = logging.getLogger(__name__)


def _text_key(hit: SearchHit) -> str:
    norm = " ".join(hit.chunk.text.lower().split())
    return hashlib.sha1(f"{hit.chunk.doc_id}|{norm}".encode()).hexdigest()


def merge_hits(result_sets: list[list[SearchHit]], top_k: int) -> list[SearchHit]:
    """Union of several result lists, best score per chunk, duplicates (same chunk id or same text in the
    same document - e.g. a chunk indexed twice across versions) removed, sorted by score."""
    by_id: dict[str, SearchHit] = {}
    for hits in result_sets:
        for h in hits:
            cur = by_id.get(h.chunk.id)
            if cur is None or h.score > cur.score:
                by_id[h.chunk.id] = h
    by_text: dict[str, SearchHit] = {}
    for h in by_id.values():
        k = _text_key(h)
        cur = by_text.get(k)
        if cur is None or h.score > cur.score:
            by_text[k] = h
    return sorted(by_text.values(), key=lambda h: -h.score)[:top_k]


@register("rag_retriever", "default")
class Retriever:
    def __init__(self, search: Any = None):
        self._search = search

    @property
    def search(self) -> Any:
        if self._search is not None:
            return self._search
        from aiready.core.wiring import connect

        return connect("search")

    async def retrieve(self, queries: list[str], vectors: list[list[float] | None], principal: Principal, *,
                       collection: str, top_k: int, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        groups = principal.access_groups
        with span("rag.search", collection=collection, sub_queries=len(queries), top_k=top_k) as sp:
            reqs = [SearchRequest(collection=collection, query_text=q, query_vector=v, access_groups=groups,
                                  filters=dict(filters or {}), top_k=top_k) for q, v in zip(queries, vectors)]
            results = await asyncio.gather(*(self.search.search(r, principal) for r in reqs))
            merged = merge_hits(list(results), top_k)
            sp.set(candidates=len(merged), raw=sum(len(r) for r in results))
            return merged
