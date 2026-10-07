"""Token counting. Uses the model's tokenizer when ``tiktoken``/``transformers`` is available, otherwise a
conservative estimate (about 4 characters per token) that is good enough for budgets and context trimming."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable


@lru_cache(maxsize=1)
def _encoder() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, (len(text) + 3) // 4)


def count_messages(messages: Iterable[Any]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        total += 4 + count_tokens(content or "")
    return total + 2


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    enc = _encoder()
    if enc is not None:
        return enc.decode(enc.encode(text, disallowed_special=())[:max_tokens])
    return text[: max_tokens * 4]
