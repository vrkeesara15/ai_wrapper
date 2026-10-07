"""Reranker scorers (plugin kind ``reranker``).

Search encodes question and passage separately: fast, approximate. A reranker reads each (question, passage)
pair together: slower and more precise, so it only sees the shortlist. A scorer receives the question and the
candidate passages and returns one relevance score per passage, in input order.

Built-ins:
  * ``lexical`` - word-overlap score, offline (tests, demos);
  * ``remote`` - a ``/rerank`` HTTP endpoint: Hugging Face TEI, Cohere, Jina, vLLM (``/v1/rerank``);
  * ``cross_encoder`` - sentence-transformers ``CrossEncoder`` on GPU/CPU (e.g. BAAI/bge-reranker-v2-m3).

Pairs are trimmed to the model's length limit before scoring and scored in batches.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import threading
from typing import Any, Protocol, runtime_checkable

import httpx

from aiready.core.errors import ConfigError, UpstreamError
from aiready.core.plugins import register
from aiready.core.textutil import overlap_score

log = logging.getLogger(__name__)

# Rough tokens-per-word for sub-word tokenizers. Only used to trim before sending;
# the model truncates exactly.
TOKENS_PER_WORD = 1.3


@runtime_checkable
class Scorer(Protocol):
    model_id: str
    max_length: int  # tokens per (query, passage) pair

    async def score(self, query: str, passages: list[str]) -> list[float]: ...


def trim_words(text: str, max_tokens: int) -> str:
    words = text.split()
    limit = max(1, int(max_tokens / TOKENS_PER_WORD))
    return text if len(words) <= limit else " ".join(words[:limit])


def build_pairs(query: str, passages: list[str], max_length: int) -> list[tuple[str, str]]:
    """One (question, passage) pair per candidate, trimmed so the pair fits the model's length limit.

    The question is kept whole up to half the budget; the passage gets the rest (minus special tokens).
    """
    q = trim_words(query, max_length // 2)
    q_tokens = math.ceil(len(q.split()) * TOKENS_PER_WORD)
    room = max(16, max_length - q_tokens - 4)
    return [(q, trim_words(p, room)) for p in passages]


class BatchedScorer(abc.ABC):
    """Base class: builds trimmed pairs and scores them ``batch_size`` at a time."""

    model_id: str = "reranker"
    max_length: int = 512
    batch_size: int = 32

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = build_pairs(query, passages, self.max_length)
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            out.extend(await self.score_batch(pairs[i : i + self.batch_size]))
        if len(out) != len(passages):
            raise UpstreamError(f"reranker returned {len(out)} scores for {len(passages)} passages")
        return out

    @abc.abstractmethod
    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score one batch of already-trimmed pairs, in order."""


@register("reranker", "lexical")
class LexicalScorer(BatchedScorer):
    """Word-overlap relevance in [0, 1]. Offline stand-in for a cross-encoder."""

    def __init__(self, model_id: str = "lexical-overlap", max_length: int = 4096, batch_size: int = 256):
        self.model_id, self.max_length, self.batch_size = model_id, max_length, batch_size

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [overlap_score(q, p) for q, p in pairs]


@register("reranker", "remote")
class RemoteScorer(BatchedScorer):
    """Calls a rerank HTTP endpoint.

    ``style``:
      * ``tei``    - ``POST {url}/rerank`` ``{"query", "texts"}`` -> ``[{"index", "score"}]``
        (Hugging Face TEI);
      * ``cohere`` - ``POST {url}/v1/rerank`` ``{"model", "query", "documents"}`` -> ``{"results": [{"index",
        "relevance_score"}]}`` (Cohere, Jina, vLLM, this platform's own ``/v1/rerank``).
    Response shapes of both styles are accepted whatever the style, so a misconfigured style still works.
    """

    def __init__(
        self,
        url: str,
        *,
        model: str = "reranker",
        style: str = "tei",
        api_key: str | None = None,
        timeout: float = 10.0,
        max_length: int = 512,
        batch_size: int = 32,
        path: str | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not url:
            raise ConfigError("remote reranker needs 'url'")
        if style not in ("tei", "cohere"):
            raise ConfigError("remote reranker style must be 'tei' or 'cohere'")
        self.style = style
        base = url.rstrip("/")
        self.endpoint = base + (
            path or ("/rerank" if style == "tei" else ("/rerank" if base.endswith("/v1") else "/v1/rerank"))
        )
        self.model_id, self.max_length, self.batch_size = model, max_length, batch_size
        h = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        self._headers, self._timeout, self._transport = h, timeout, transport
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers, transport=self._transport
            )
            self._loop = loop
        return self._client

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        query = pairs[0][0]
        texts = [p for _, p in pairs]
        if self.style == "tei":
            body: dict[str, Any] = {"query": query, "texts": texts, "raw_scores": False, "truncate": True}
        else:
            body = {"model": self.model_id, "query": query, "documents": texts, "top_n": len(texts)}
        try:
            r = await self._http().post(self.endpoint, json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"reranker unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(f"reranker returned {r.status_code}", details={"body": r.text[:500]})
        data = r.json()
        items = data.get("results", data) if isinstance(data, dict) else data
        scores = [float("-inf")] * len(texts)
        for it in items:
            scores[int(it["index"])] = float(it.get("relevance_score", it.get("score", 0.0)))
        return scores

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


@register("reranker", "cross_encoder")
class CrossEncoderScorer(BatchedScorer):
    """sentence-transformers ``CrossEncoder``; loaded on first use and run in a worker thread.

    Small enough to share a GPU with the embedding model. ``activation: sigmoid`` maps raw logits to [0, 1]
    for models whose CrossEncoder head has no activation (recent sentence-transformers already apply a
    sigmoid to single-label models, hence the default ``none``).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 32,
        activation: str = "none",
        trust_remote_code: bool = False,
    ):
        self.model_name = self.model_id = model
        self.device, self.max_length, self.batch_size = device, max_length, batch_size
        self.activation = activation
        self.trust_remote_code = trust_remote_code
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise ConfigError(
                        "reranker 'cross_encoder' needs `pip install sentence-transformers`"
                    ) from exc
                log.info("loading reranker %s on %s", self.model_name, self.device or "auto")
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                    trust_remote_code=self.trust_remote_code,
                )
            return self._model

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raw = self._load().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        vals = [float(x) for x in raw]
        if self.activation == "sigmoid":
            vals = [1.0 / (1.0 + math.exp(-v)) if abs(v) < 700 else (1.0 if v > 0 else 0.0) for v in vals]
        return vals

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await asyncio.to_thread(self._predict, pairs)
