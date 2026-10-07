"""Chunkers (plugin kind ``chunker``): split along headings and sections, with overlap.

* ``heading`` (default) - a chunk never crosses a section boundary, so each chunk is about one thing and
  carries its heading path ("Claims handbook > Settlement authority") for citations and re-ranking. Long
  sections are packed into token-sized windows (``core.tokens``) on sentence boundaries with an overlap so a
  fact split across two chunks is still found.
* ``fixed`` - a sliding token window over the whole text (logs, transcripts, text without structure).
* ``table_rows`` - groups of rows with the column header repeated, so every chunk of a table is
  self-describing.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from aiready.core.plugins import register
from aiready.core.textutil import split_sentences
from aiready.core.tokens import count_tokens
from aiready.ingestion.parsers import Block


class ChunkDraft(BaseModel):
    text: str
    section: str | None = None
    page: int | None = None
    pages: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunker(Protocol):
    name: str

    def chunk(self, blocks: list[Block], *, title: str | None = None) -> list[ChunkDraft]: ...


def _units(text: str, max_tokens: int) -> list[str]:
    """Sentence units; a sentence longer than the window is cut into word groups."""
    units: list[str] = []
    for para in text.split("\n"):
        for sent in split_sentences(para) or [para]:
            if count_tokens(sent) <= max_tokens:
                units.append(sent)
                continue
            words, buf = sent.split(), []
            for w in words:
                buf.append(w)
                if count_tokens(" ".join(buf)) >= max_tokens:
                    units.append(" ".join(buf))
                    buf = []
            if buf:
                units.append(" ".join(buf))
    return [u for u in units if u.strip()]


def pack(units: list[tuple[str, int | None]], max_tokens: int, overlap_tokens: int) -> list[tuple[str, list[int]]]:
    """Greedy packing of (text, page) units into windows of ``max_tokens`` with ``overlap_tokens`` carried over."""
    out: list[tuple[str, list[int]]] = []
    cur: list[tuple[str, int | None]] = []
    cur_tokens = 0
    for text, page in units:
        t = count_tokens(text)
        if cur and cur_tokens + t > max_tokens:
            out.append((" ".join(u for u, _ in cur), sorted({p for _, p in cur if p is not None})))
            carry: list[tuple[str, int | None]] = []
            carried = 0
            for u in reversed(cur):  # overlap: the tail of the previous window starts the next one
                ut = count_tokens(u[0])
                if carried + ut > overlap_tokens:
                    break
                carry.insert(0, u)
                carried += ut
            cur, cur_tokens = carry, carried
        cur.append((text, page))
        cur_tokens += t
    if cur:
        out.append((" ".join(u for u, _ in cur), sorted({p for _, p in cur if p is not None})))
    return out


@register("chunker", "heading")
class HeadingChunker:
    name = "heading"

    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 40, min_tokens: int = 20,
                 include_heading_in_text: bool = True):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens, self.overlap, self.min_tokens = max_tokens, overlap_tokens, min_tokens
        self.include_heading = include_heading_in_text

    def chunk(self, blocks: list[Block], *, title: str | None = None) -> list[ChunkDraft]:
        sections: list[tuple[list[str], list[Block]]] = []
        path: list[tuple[int, str]] = []
        current: list[Block] = []

        def close() -> None:
            if current:
                sections.append(([t for _, t in path], list(current)))
                current.clear()

        for b in blocks:
            if b.type == "heading":
                close()
                while path and path[-1][0] >= max(b.level, 1):
                    path.pop()
                path.append((max(b.level, 1), b.text))
            else:
                current.append(b)
        close()

        drafts: list[ChunkDraft] = []
        for heads, sec_blocks in sections:
            section = " > ".join(heads) if heads else None
            units: list[tuple[str, int | None]] = []
            for b in sec_blocks:
                units += [(u, b.page) for u in _units(b.text, self.max_tokens)]
            for text, pages in pack(units, self.max_tokens, self.overlap):
                body = f"{heads[-1]}\n{text}" if self.include_heading and heads else text
                drafts.append(ChunkDraft(text=body, section=section, page=pages[0] if pages else None, pages=pages))
        return _merge_small(drafts, self.min_tokens, self.max_tokens)


def _merge_small(drafts: list[ChunkDraft], min_tokens: int, max_tokens: int) -> list[ChunkDraft]:
    """Fold tiny chunks into the previous chunk of the same section (a lone line is a poor retrieval unit)."""
    out: list[ChunkDraft] = []
    for d in drafts:
        if (out and out[-1].section == d.section and count_tokens(d.text) < min_tokens
                and count_tokens(out[-1].text) + count_tokens(d.text) <= max_tokens):
            prev = out[-1]
            out[-1] = prev.model_copy(update={"text": prev.text + "\n" + d.text,
                                              "pages": sorted(set(prev.pages) | set(d.pages))})
        else:
            out.append(d)
    return out


@register("chunker", "fixed")
class FixedChunker:
    name = "fixed"

    def __init__(self, max_tokens: int = 300, overlap_tokens: int = 50):
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.max_tokens, self.overlap = max_tokens, overlap_tokens

    def chunk(self, blocks: list[Block], *, title: str | None = None) -> list[ChunkDraft]:
        units: list[tuple[str, int | None]] = []
        for b in blocks:
            units += [(u, b.page) for u in _units(b.text, self.max_tokens)]
        return [ChunkDraft(text=t, page=p[0] if p else None, pages=p) for t, p in pack(units, self.max_tokens, self.overlap)]


@register("chunker", "table_rows")
class TableRowsChunker:
    """Rows grouped ``rows_per_chunk`` at a time (and under ``max_tokens``), header repeated in each chunk.
    Non-row blocks (a heading or a paragraph around a table) are chunked with the heading chunker."""

    name = "table_rows"

    def __init__(self, rows_per_chunk: int = 20, max_tokens: int = 400, repeat_header: bool = True):
        self.rows_per_chunk, self.max_tokens, self.repeat_header = rows_per_chunk, max_tokens, repeat_header

    def chunk(self, blocks: list[Block], *, title: str | None = None) -> list[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        others: list[Block] = []
        group: list[Block] = []
        section: str | None = None

        def flush() -> None:
            if not group:
                return
            header = group[0].header
            prefix = f"Columns: {header}\n" if header and self.repeat_header else ""
            first = group[0].metadata.get("row", None)
            drafts.append(ChunkDraft(text=prefix + "\n".join(b.text for b in group), section=section,
                                     page=group[0].page, metadata={"rows": len(group), "first_row": first}))
            group.clear()

        for b in blocks:
            if b.type == "row":
                tokens = count_tokens("\n".join(x.text for x in group + [b]))
                if len(group) >= self.rows_per_chunk or (group and tokens > self.max_tokens):
                    flush()
                group.append(b)
            else:
                flush()
                if b.type == "heading":
                    section = b.text
                others.append(b)
        flush()
        if any(b.type != "heading" for b in others):
            drafts = HeadingChunker(max_tokens=self.max_tokens, overlap_tokens=min(40, self.max_tokens // 4)).chunk(
                others, title=title) + drafts
        return drafts
