"""Context builder: fits the prompt to the token budget.

The prompt holds the instructions, a short summary of the conversation, the reranked passages as numbered
``[n]`` source blocks (title and source on the first line) and the question. Passages are added in rerank
order until the budget is spent, each trimmed to a per-passage limit, so the most relevant evidence always
fits and the prompt stays as small as the task needs (slide "Cost controls" - context trimming).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from aiready.core.plugins import register
from aiready.core.textutil import split_sentences
from aiready.core.tokens import count_tokens, truncate_to_tokens
from aiready.core.types import ChatMessage, SearchHit
from aiready.rag.config import ContextConfig
from aiready.rag.prompts import PromptTemplate

_MARKER = re.compile(r"\[(\d+)\]")


class ContextBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[ChatMessage]
    passages: list[SearchHit]  # passages[i] is source [i + 1]
    sources_text: str
    history_text: str
    prompt_ref: str
    tokens: int


def _clean(text: str) -> str:
    """Neutralise ``[n]`` inside passage text so only our numbering looks like a source marker."""
    return _MARKER.sub(r"(\1)", " ".join((text or "").split()))


def source_header(i: int, hit: SearchHit) -> str:
    c = hit.chunk
    title = c.title or c.doc_id
    where = c.source_uri or c.source
    section = f" - {c.section}" if c.section else ""
    return f"[{i}] {title}{section}" + (f" ({where})" if where else "")


def summarise_history(history: list[dict[str, Any]], *, recent_turns: int, max_tokens: int) -> str:
    """Older turns compressed to their first sentence, newest turns verbatim, all within ``max_tokens``.

    Deterministic on purpose: the memory service does model-based summarisation for long sessions; this only
    has to keep the prompt bounded for whatever history the caller sent.
    """
    turns = [h for h in history or [] if h.get("content")]
    if not turns or max_tokens <= 0:
        return ""
    lines: list[str] = []
    old, recent = turns[:-recent_turns] if recent_turns else turns, turns[-recent_turns:] if recent_turns else []
    for h in old:
        if h.get("role") == "system":  # a summary produced by the memory service
            lines.append(f"Earlier: {_MARKER.sub('', h['content']).strip()}")
            continue
        first = (split_sentences(_MARKER.sub("", h["content"])) or [""])[0]
        lines.append(f"{h.get('role', 'user')} (earlier): {first}")
    for h in recent:
        lines.append(f"{h.get('role', 'user')}: {_MARKER.sub('', h['content']).strip()}")
    # drop from the oldest until it fits
    while lines and count_tokens("\n".join(lines)) > max_tokens:
        if len(lines) == 1:
            lines[0] = truncate_to_tokens(lines[0], max_tokens)
            break
        lines.pop(0)
    return "\n".join(lines)


@register("rag_context_builder", "default")
class ContextBuilder:
    def __init__(self, config: ContextConfig | None = None):
        self.cfg = config or ContextConfig()

    def build(self, prompt: PromptTemplate, *, question: str, passages: list[SearchHit],
              history: list[dict[str, Any]] | None = None, **values: Any) -> ContextBundle:
        used: list[SearchHit] = []
        blocks: list[str] = []
        spent = 0
        for hit in passages:
            body = truncate_to_tokens(_clean(hit.chunk.text), self.cfg.max_passage_tokens)
            block = f"{source_header(len(used) + 1, hit)}\n{body}"
            cost = count_tokens(block) + 1
            if used and spent + cost > self.cfg.max_context_tokens:
                break
            if not used and cost > self.cfg.max_context_tokens:  # always keep the best passage, trimmed
                block = truncate_to_tokens(block, self.cfg.max_context_tokens)
                cost = count_tokens(block)
            used.append(hit)
            blocks.append(block)
            spent += cost
        sources_text = "\n\n".join(blocks)
        hist = summarise_history(history or [], recent_turns=self.cfg.recent_turns,
                                 max_tokens=self.cfg.history_tokens)
        history_text = f"Conversation so far:\n{hist}\n" if hist else ""
        system = prompt.render_system(sources=sources_text, history=history_text, question=question, **values)
        user = prompt.render_user(question=question, sources=sources_text, history=history_text, **values)
        messages = [ChatMessage(role="system", content=system.rstrip()), ChatMessage(role="user", content=user)]
        return ContextBundle(messages=messages, passages=used, sources_text=sources_text, history_text=history_text,
                             prompt_ref=prompt.ref, tokens=count_tokens(system) + count_tokens(user))
