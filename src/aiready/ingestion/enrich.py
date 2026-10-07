"""Enrich: source, date, owner and access list on every chunk, title / section path, deterministic ids.

Access tags travel on *every chunk* because the search service filters by them before scoring - a chunk
without tags would be invisible (or, worse, visible to everyone if a store defaulted to open), so the
pipeline refuses to index a document with no tags unless the source explicitly allows it.

Chunk ids are deterministic: hash(doc id) + position + hash(content). Re-running the pipeline on unchanged
content produces the same ids (idempotent upserts), a changed paragraph produces a new id, and stale ids of
the previous version can be computed and deleted precisely.
"""

from __future__ import annotations

from typing import Any

from aiready.core.textutil import sha256
from aiready.core.types import Chunk, DataClass, Document
from aiready.ingestion.chunking import ChunkDraft


def doc_prefix(doc_id: str) -> str:
    return sha256(doc_id)[:16] + "-"


def chunk_id(doc_id: str, position: int, text: str) -> str:
    return f"{doc_prefix(doc_id)}{position:05d}-{sha256(text)[:12]}"


def enrich(doc: Document, drafts: list[ChunkDraft], *, title: str | None, data_class: DataClass,
           access_tags: list[str], run_id: str | None, doc_version: int, parser: str, chunker: str,
           flags: dict[str, Any] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    doc_title = title or doc.title
    for pos, d in enumerate(drafts):
        content_hash = sha256(d.text)
        meta: dict[str, Any] = {
            "doc_version": doc_version, "doc_checksum": doc.checksum, "run_id": run_id, "parser": parser,
            "chunker": chunker, "mime_type": doc.mime_type, "content_hash": content_hash,
            "page": d.page, "pages": d.pages, "section_path": d.section.split(" > ") if d.section else [],
        }
        meta.update({k: v for k, v in d.metadata.items() if v is not None})
        meta.update(flags or {})
        for k in ("table", "space", "project", "path"):
            if k in doc.metadata:
                meta[k] = doc.metadata[k]
        chunks.append(Chunk(
            id=chunk_id(doc.id, pos, d.text), doc_id=doc.id, text=d.text, title=doc_title, source=doc.source,
            source_uri=doc.uri, section=d.section, position=pos, access_tags=list(access_tags),
            data_class=data_class, owner=doc.owner, updated_at=doc.updated_at, metadata=meta,
        ))
    return chunks


def embedding_text(chunk: Chunk, include_title: bool = True) -> str:
    """What gets embedded: title and section path give a short chunk the context it lacks on its own."""
    if not include_title:
        return chunk.text
    head = " > ".join(p for p in [chunk.title or "", chunk.section or ""] if p)
    return f"{head}\n{chunk.text}" if head else chunk.text
