"""Citation mapper: links each claim in the answer to the passages it cites.

The model marks sentences with ``[n]`` (also ``[1, 2]`` or ``[1][2]``). The mapper splits the answer into
claims, checks every marker points to a passage that was actually in the prompt (a model can invent ``[7]``
when only five sources exist), removes invented markers from the text, and builds one
``core.types.Citation`` per cited passage with the passage sentence that best supports the claims citing it.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from aiready.core.plugins import register
from aiready.core.textutil import overlap_score, split_sentences
from aiready.core.types import Citation, SearchHit

_GROUP = re.compile(r"\[(\s*\d+\s*(?:[,;]\s*\d+\s*)*)\]")


class Claim(BaseModel):
    index: int
    text: str  # without markers
    markers: list[int] = Field(default_factory=list)  # valid source numbers (1-based)
    invalid_markers: list[int] = Field(default_factory=list)


class CitationResult(BaseModel):
    answer: str  # answer with invalid markers removed
    claims: list[Claim]
    citations: list[Citation]
    invalid_markers: list[int] = Field(default_factory=list)

    @property
    def cited_ratio(self) -> float:
        return sum(1 for c in self.claims if c.markers) / len(self.claims) if self.claims else 0.0


def _numbers(group: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", group)]


def best_quote(claim: str, passage_text: str, max_chars: int = 300) -> str:
    sents = split_sentences(passage_text) or [passage_text]
    best = max(sents, key=lambda s: overlap_score(claim, s))
    return best if len(best) <= max_chars else best[: max_chars - 1].rstrip() + "…"


@register("rag_citation_mapper", "default")
class CitationMapper:
    def map(self, answer: str, passages: list[SearchHit]) -> CitationResult:
        n = len(passages)
        invalid_all: list[int] = []

        def fix(m: re.Match[str]) -> str:
            nums = _numbers(m.group(1))
            good = [x for x in nums if 1 <= x <= n]
            invalid_all.extend(x for x in nums if not 1 <= x <= n)
            return "".join(f"[{x}]" for x in dict.fromkeys(good))

        cleaned = _GROUP.sub(fix, answer or "")
        # "Claim. [1]" -> "Claim [1]." so the marker stays with the sentence it supports when splitting
        cleaned = re.sub(r"([.!?])((?:\s*\[\d+\])+)", lambda m: m.group(2) + m.group(1), cleaned)
        cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()

        claims: list[Claim] = []
        for i, sent in enumerate(split_sentences(cleaned)):
            markers = [int(x) for x in re.findall(r"\[(\d+)\]", sent)]
            text = re.sub(r"\s*\[\d+\]", "", sent).strip()
            if not text:
                continue
            claims.append(Claim(index=i, text=text, markers=list(dict.fromkeys(markers))))

        citations: list[Citation] = []
        for num in sorted({m for c in claims for m in c.markers}):
            hit = passages[num - 1]
            claim_text = " ".join(c.text for c in claims if num in c.markers)
            citations.append(Citation(marker=f"[{num}]", chunk_id=hit.chunk.id, doc_id=hit.chunk.doc_id,
                                      title=hit.chunk.title, source_uri=hit.chunk.source_uri,
                                      quote=best_quote(claim_text, hit.chunk.text)))
        return CitationResult(answer=cleaned, claims=claims, citations=citations,
                              invalid_markers=sorted(set(invalid_all)))

    @staticmethod
    def sources_only(passages: list[SearchHit]) -> list[Citation]:
        """Citations for a sources-only reply: every passage, quoting its opening sentence."""
        out = []
        for i, hit in enumerate(passages, start=1):
            first = (split_sentences(hit.chunk.text) or [hit.chunk.text])[0]
            out.append(Citation(marker=f"[{i}]", chunk_id=hit.chunk.id, doc_id=hit.chunk.doc_id,
                                title=hit.chunk.title, source_uri=hit.chunk.source_uri, quote=first[:300]))
        return out
