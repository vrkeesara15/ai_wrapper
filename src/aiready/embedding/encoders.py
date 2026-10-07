"""Embedding encoders (plugin kind ``embedder``).

An encoder turns a batch of texts into vectors. It knows nothing about queues, caches or callers - the
service and the batcher handle those - so a client can plug in their own model with
``class: acme_ai.embed:MyEncoder`` and keep every other behaviour.

Each encoder exposes ``model_id``, ``version`` and ``dim``. Together they form the *fingerprint* stored
with every index: vectors from different models (or versions) cannot be compared, so changing any of them
means a full re-embed.

Built-ins:
  * ``hash`` - deterministic feature hashing, offline (tests, demos, air-gapped smoke tests);
  * ``openai`` - any OpenAI-compatible ``/v1/embeddings`` server (vLLM, TEI, Ollama, llama.cpp, LiteLLM...);
  * ``sentence_transformers`` - local model on GPU/CPU (lazy import of sentence-transformers).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from aiready.core.errors import ConfigError, UpstreamError
from aiready.core.plugins import register
from aiready.core.textutil import hash_embed

log = logging.getLogger(__name__)


@runtime_checkable
class Encoder(Protocol):
    model_id: str
    version: str
    max_batch_size: int

    @property
    def dim(self) -> int: ...

    async def encode(self, texts: list[str]) -> list[list[float]]: ...


def fingerprint(encoder: Encoder) -> str:
    """Identity of the vector space an encoder produces: ``model_id@version/dim``."""
    return f"{encoder.model_id}@{encoder.version}/{encoder.dim}"


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@register("embedder", "hash")
class HashEncoder:
    """Offline bag-of-words hashing encoder. Not semantic, but texts sharing words are close."""

    def __init__(
        self, dim: int = 384, model_id: str = "hash-embed", version: str = "1", max_batch_size: int = 256
    ):
        self.model_id, self.version, self._dim = model_id, version, int(dim)
        self.max_batch_size = max_batch_size
        self.calls = 0  # number of encode() invocations - lets tests prove the cache and batcher work

    @property
    def dim(self) -> int:
        return self._dim

    async def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [hash_embed(t, self._dim) for t in texts]


@register("embedder", "openai")
class OpenAICompatibleEncoder:
    """Calls an OpenAI-compatible embeddings endpoint (vLLM, TEI, Ollama, Azure/OpenAI, LiteLLM).

    ``url`` is the server root (``http://tei:80``) or its ``/v1`` base. ``model`` is what the server calls
    the model; ``version`` should change whenever the weights change (e.g. the HF revision hash).
    ``dim`` may be omitted: it is discovered from the first response.
    """

    def __init__(
        self,
        url: str,
        model: str,
        *,
        version: str = "1",
        dim: int | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        max_batch_size: int = 64,
        normalize: bool = True,
        dimensions: int | None = None,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not url or not model:
            raise ConfigError("openai embedder needs 'url' and 'model'")
        base = url.rstrip("/")
        self.endpoint = base + ("/embeddings" if base.endswith("/v1") else "/v1/embeddings")
        self.model_id, self.version = model, str(version)
        self.max_batch_size = max_batch_size
        self.normalize = normalize
        self.dimensions = dimensions  # Matryoshka models (OpenAI v3, nomic) can return shorter vectors
        self._dim = dimensions or dim
        h = {"Content-Type": "application/json", **(headers or {})}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        self._headers = h
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise ConfigError(
                f"dimension of '{self.model_id}' unknown until the first call; set 'dim' in config"
            )
        return self._dim

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, headers=self._headers, transport=self._transport
            )
            self._client_loop = loop
        return self._client

    async def encode(self, texts: list[str]) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model_id, "input": texts, "encoding_format": "float"}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        try:
            r = await self._http().post(self.endpoint, json=body)
        except httpx.HTTPError as exc:
            raise UpstreamError(f"embedding server unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise UpstreamError(f"embedding server returned {r.status_code}", details={"body": r.text[:500]})
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        arr = np.asarray([d["embedding"] for d in data], dtype=np.float32)
        if arr.shape[0] != len(texts):
            raise UpstreamError(f"embedding server returned {arr.shape[0]} vectors for {len(texts)} texts")
        if self._dim is None:
            self._dim = int(arr.shape[1])
        elif arr.shape[1] != self._dim:
            raise UpstreamError(
                f"embedding server returned dim {arr.shape[1]}, configured {self._dim}: "
                "a different model is being served"
            )
        if self.normalize:
            arr = _normalise(arr)
        return arr.tolist()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


@register("embedder", "sentence_transformers")
class SentenceTransformersEncoder:
    """Local sentence-transformers model (bge-m3, e5, gte, nomic...). Loaded on first use, run in a thread.

    ``prefix`` is prepended to every text (e5 needs ``"query: "``/``"passage: "``; use the same setting on
    both paths - documents and questions must be embedded by the same model with the same settings).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        version: str | None = None,
        device: str | None = None,
        max_batch_size: int = 64,
        normalize: bool = True,
        prefix: str = "",
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_folder: str | None = None,
    ):
        self.model_name = model
        self.model_id = model
        self.version = str(version or revision or "1")
        self.device, self.max_batch_size, self.normalize = device, max_batch_size, normalize
        self.prefix, self.trust_remote_code, self.revision = prefix, trust_remote_code, revision
        self.cache_folder = cache_folder
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ConfigError(
                        "embedder 'sentence_transformers' needs `pip install sentence-transformers`"
                    ) from exc
                log.info("loading embedding model %s on %s", self.model_name, self.device or "auto")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    revision=self.revision,
                    trust_remote_code=self.trust_remote_code,
                    cache_folder=self.cache_folder,
                )
            return self._model

    @property
    def dim(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        arr = model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.max_batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32).tolist()

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sync, texts)
