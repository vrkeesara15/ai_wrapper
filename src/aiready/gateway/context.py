"""Prompt shaping before the call: prefix-cache friendly ordering and context trimming.

* **Prefix caching** - model servers (vLLM, TGI, SGLang) reuse the KV cache of a prompt prefix they have seen.
  Keeping the stable parts first and byte-identical (system prompts merged at the top, tool definitions in a
  fixed order) makes repeated system prompts and tool lists hit that cache instead of being recomputed.
* **Context trimming** - a request must fit the chosen model's window minus room for the answer. The oldest
  conversation turns go first; system prompts and the latest user turn are always kept; an assistant tool call
  is never separated from its tool results. As a last resort the longest kept message is truncated.
"""

from __future__ import annotations

from aiready.core.tokens import count_messages, count_tokens, truncate_to_tokens
from aiready.core.types import ChatMessage, ChatRequest


def prefix_friendly(req: ChatRequest) -> ChatRequest:
    systems = [m for m in req.messages if m.role == "system"]
    others = [m for m in req.messages if m.role != "system"]
    messages = systems + others
    tools = sorted(req.tools, key=lambda t: str(t.function.get("name", ""))) if req.tools else req.tools
    if messages == req.messages and tools == req.tools:
        return req
    return req.model_copy(update={"messages": messages, "tools": tools})


def _turn_groups(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Group an assistant message with the tool results answering it, so trimming keeps them together."""
    groups: list[list[ChatMessage]] = []
    for m in messages:
        if m.role == "tool" and groups and any(x.role == "assistant" and x.tool_calls for x in groups[-1]):
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups


def trim_to_window(req: ChatRequest, context_length: int, reserve_output: int) -> tuple[ChatRequest, int]:
    """Return (request that fits, number of messages dropped or truncated)."""
    tool_tokens = sum(count_tokens(str(t.function)) for t in req.tools or [])
    budget = max(256, context_length - min(reserve_output, context_length // 2) - tool_tokens)
    if count_messages(req.messages) <= budget:
        return req, 0
    systems = [m for m in req.messages if m.role == "system"]
    rest = [m for m in req.messages if m.role != "system"]
    last_user = max((i for i, m in enumerate(rest) if m.role == "user"), default=len(rest) - 1)
    head, tail = rest[:last_user], rest[last_user:]  # tail = latest user turn and anything after it
    groups = _turn_groups(head)
    changed = 0
    while groups and count_messages(systems + [m for g in groups for m in g] + tail) > budget:
        changed += len(groups.pop(0))
    kept = systems + [m for g in groups for m in g] + tail
    for _ in range(len(kept) * 4):  # bounded: each pass shrinks the longest message
        if count_messages(kept) <= budget:
            break
        longest = max(range(len(kept)), key=lambda i: count_tokens(kept[i].content or ""))
        excess = count_messages(kept) - budget
        cur = count_tokens(kept[longest].content or "")
        if cur <= 16:
            break
        kept[longest] = kept[longest].model_copy(
            update={"content": truncate_to_tokens(kept[longest].content or "", max(16, cur - excess - 8))})
        changed += 1
    return req.model_copy(update={"messages": kept}), changed
