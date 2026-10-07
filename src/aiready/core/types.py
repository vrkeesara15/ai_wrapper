"""Shared data contracts.

Chat and embedding types are OpenAI-compatible on purpose: vLLM, TGI, Ollama, llama.cpp and most private
endpoints speak this API, so swapping a model is a routing change, never a code change.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex}"


# --------------------------------------------------------------------------- data classification


class DataClass(str, Enum):
    """Classification attached to inputs, chunks and egress rules. Ordered from least to most sensitive."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return list(DataClass).index(self)

    @classmethod
    def max(cls, *values: "DataClass") -> "DataClass":
        vals = [v for v in values if v is not None]
        return max(vals, key=lambda v: v.rank) if vals else cls.PUBLIC


# --------------------------------------------------------------------------- chat


class ToolCallFunction(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("call_"))
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = ""
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    type: Literal["function"] = "function"
    function: dict[str, Any]


class ChatRequest(BaseModel):
    """Request accepted by the gateway. `model` may be a concrete model or a logical route ("auto", "chat")."""

    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[ToolSpec] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    stop: list[str] | str | None = None
    user: str | None = None
    # AI-Ready extensions (ignored by upstream model servers; stripped before forwarding)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user" and m.content:
                return m.content
        return ""

    def all_text(self) -> str:
        return "\n".join(m.content or "" for m in self.messages)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("chatcmpl-"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)
    # AI-Ready extensions
    aiready: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.choices[0].message.content or "" if self.choices else ""


class EmbeddingRequest(BaseModel):
    model: str = "embedding"
    input: list[str] | str
    priority: Literal["live", "bulk"] = "live"

    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None
    model: str = "reranker"


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


# --------------------------------------------------------------------------- knowledge


class Chunk(BaseModel):
    """A retrievable unit. `access_tags` are the groups allowed to see it; enforced at search time."""

    id: str
    doc_id: str
    text: str
    title: str | None = None
    source: str = ""
    source_uri: str | None = None
    section: str | None = None
    position: int = 0
    access_tags: list[str] = Field(default_factory=list)
    data_class: DataClass = DataClass.INTERNAL
    owner: str | None = None
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


class SearchRequest(BaseModel):
    collection: str
    query_text: str
    query_vector: list[float] | None = None
    access_groups: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 50


class Citation(BaseModel):
    marker: str
    chunk_id: str
    doc_id: str
    title: str | None = None
    source_uri: str | None = None
    quote: str | None = None


class Document(BaseModel):
    """A source item as produced by a connector, before chunking."""

    id: str
    source: str
    uri: str | None = None
    title: str | None = None
    content: bytes | str = b""
    mime_type: str = "text/plain"
    access_tags: list[str] = Field(default_factory=list)
    owner: str | None = None
    updated_at: float = Field(default_factory=time.time)
    checksum: str | None = None
    deleted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
