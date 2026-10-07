"""Contracts between services.

Each service package implements its Protocol twice: an in-process class (``service.py``) and an HTTP client
(``client.py``) with identical method signatures. ``aiready.core.wiring.connect(name)`` returns one or the
other depending on configuration, so the same code runs as one process (pilot) or as separate pods
(production) - the architecture is the same at every stage, only the footprint changes.

Rules every implementation follows:
  * the ``principal`` argument is the signed-in user; access filtering and policy use it, never a service identity;
  * methods are async;
  * results are pydantic models from ``aiready.core.types`` or the models below.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from aiready.core.identity import Principal
from aiready.core.types import (
    ChatRequest,
    ChatResponse,
    Chunk,
    Citation,
    DataClass,
    RerankResponse,
    SearchHit,
    SearchRequest,
)

# =========================================================================== governance / policy


class PolicyInput(BaseModel):
    """Everything a policy may look at."""

    action: str  # e.g. model.invoke, tool.call, data.read, egress.send, memory.write, sql.execute
    principal: dict[str, Any]
    resource: dict[str, Any] = Field(default_factory=dict)  # model / tool / collection / table...
    context: dict[str, Any] = Field(default_factory=dict)  # data_class, feature, app, environment...


class PolicyDecision(BaseModel):
    allow: bool
    effect: Literal["allow", "deny", "require_approval"] = "allow"
    reasons: list[str] = Field(default_factory=list)
    matched_rules: list[str] = Field(default_factory=list)
    obligations: dict[str, Any] = Field(default_factory=dict)  # e.g. {"mask_pii": true, "max_tokens": 2000}
    policy_version: str = ""


@runtime_checkable
class PolicyEngine(Protocol):
    async def decide(self, inp: PolicyInput) -> PolicyDecision: ...


# =========================================================================== guardrails


class Finding(BaseModel):
    check: str  # injection, pii, toxicity, secret, schema, grounding...
    severity: Literal["low", "medium", "high"] = "medium"
    message: str = ""
    span: tuple[int, int] | None = None
    entity: str | None = None  # e.g. EMAIL, PHONE, IBAN
    score: float | None = None


class ScreenResult(BaseModel):
    allowed: bool
    text: str  # possibly masked
    findings: list[Finding] = Field(default_factory=list)
    data_class: DataClass = DataClass.INTERNAL
    masked_entities: dict[str, str] = Field(default_factory=dict)  # placeholder -> original (kept in memory only)


class ValidationResult(BaseModel):
    valid: bool
    text: str
    parsed: Any | None = None
    findings: list[Finding] = Field(default_factory=list)


@runtime_checkable
class Guardrails(Protocol):
    async def screen_input(self, text: str, *, principal: Principal | None = None,
                           context: dict[str, Any] | None = None) -> ScreenResult: ...

    async def validate_output(self, text: str, *, schema: dict[str, Any] | None = None,
                              context: dict[str, Any] | None = None) -> ValidationResult: ...

    async def mask(self, text: str) -> ScreenResult: ...

    async def screen_document(self, text: str, *, context: dict[str, Any] | None = None) -> ScreenResult:
        """Screen retrieved or ingested content for indirect prompt injection and personal data."""
        ...


# =========================================================================== gateway


@runtime_checkable
class Gateway(Protocol):
    """The only path to a model. Applications, agents and pipelines call this - never a model server."""

    async def chat(self, req: ChatRequest, principal: Principal) -> ChatResponse: ...

    def chat_stream(self, req: ChatRequest, principal: Principal) -> AsyncIterator[dict[str, Any]]: ...

    async def embed(self, texts: list[str], principal: Principal, *, model: str = "embedding",
                    priority: str = "live") -> list[list[float]]: ...

    async def rerank(self, query: str, documents: list[str], principal: Principal, *, top_n: int | None = None,
                     model: str = "reranker") -> RerankResponse: ...


# =========================================================================== search


@runtime_checkable
class SearchService(Protocol):
    async def search(self, req: SearchRequest, principal: Principal) -> list[SearchHit]: ...
    async def upsert(self, collection: str, chunks: list[Chunk]) -> int: ...
    async def delete(self, collection: str, *, doc_ids: list[str] | None = None,
                     chunk_ids: list[str] | None = None) -> int: ...
    async def stats(self, collection: str) -> dict[str, Any]: ...


# =========================================================================== RAG


class RagQuery(BaseModel):
    question: str
    collection: str | None = None  # default: the app's collection
    history: list[dict[str, Any]] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None  # candidates from search (default 50)
    top_n: int | None = None  # passages after rerank (default 5)
    mode: Literal["check_then_send", "stream_then_verify"] = "check_then_send"
    feature: str = "rag"
    session_id: str | None = None
    model: str = "auto"


class RagAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    found: bool = True
    grounded: bool = True
    grounding_score: float | None = None
    passages: list[SearchHit] = Field(default_factory=list)
    trace_id: str | None = None
    cached: bool = False
    model: str | None = None
    fallbacks: list[str] = Field(default_factory=list)


@runtime_checkable
class RagService(Protocol):
    async def answer(self, q: RagQuery, principal: Principal) -> RagAnswer: ...
    async def retrieve(self, q: RagQuery, principal: Principal) -> list[SearchHit]: ...


# =========================================================================== memory


class MemoryItem(BaseModel):
    id: str
    user: str
    kind: Literal["fact", "preference", "summary"] = "fact"
    text: str
    consent: bool = True
    created_at: float
    expires_at: float | None = None
    source: str | None = None
    app: str = "default"


@runtime_checkable
class MemoryService(Protocol):
    async def append_turn(self, session_id: str, principal: Principal, role: str, content: str) -> None: ...
    async def session_context(self, session_id: str, principal: Principal, max_tokens: int = 1500) -> list[dict[str, Any]]: ...
    async def remember(self, principal: Principal, text: str, *, kind: str = "fact", ttl_days: int | None = None,
                       consent: bool = False) -> MemoryItem: ...
    async def recall(self, principal: Principal, query: str | None = None, limit: int = 10) -> list[MemoryItem]: ...
    async def forget(self, principal: Principal, item_id: str | None = None) -> int: ...


# =========================================================================== structured data


class QueryRequest(BaseModel):
    question: str | None = None
    metric: str | None = None  # an approved metric name: always preferred to generated SQL
    params: dict[str, Any] = Field(default_factory=dict)
    max_rows: int = 200


class QueryResult(BaseModel):
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool = False
    summary: str | None = None
    source: Literal["metric", "generated"] = "generated"
    warnings: list[str] = Field(default_factory=list)


@runtime_checkable
class QueryService(Protocol):
    async def query(self, req: QueryRequest, principal: Principal) -> QueryResult: ...


# =========================================================================== tools (MCP)


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    effect: Literal["read", "write", "payment", "delete"] = "read"  # anything but read waits for a person
    data_class: DataClass = DataClass.INTERNAL
    required_roles: list[str] = Field(default_factory=list)
    timeout_seconds: float = 30.0
    source: str = "facade"  # facade / builtin / custom


class ToolResult(BaseModel):
    ok: bool
    content: Any = None
    error: str | None = None
    approval_id: str | None = None  # set when the call is waiting for a person


@runtime_checkable
class ToolExecutor(Protocol):
    async def list_tools(self, principal: Principal) -> list[ToolDefinition]: ...
    async def call(self, name: str, arguments: dict[str, Any], principal: Principal, *,
                   approval_id: str | None = None, context: dict[str, Any] | None = None) -> ToolResult: ...
