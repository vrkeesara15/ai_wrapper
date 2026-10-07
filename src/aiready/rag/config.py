"""Configuration of the RAG service (section ``rag`` in ``config/rag.yaml``).

Defaults run locally with no infrastructure: in-memory cache, lexical grounding check, prompts from
``config/prompts``. Every threshold is a starting value to tune per client against the evaluation set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from aiready.core.types import DataClass


class PromptsConfig(BaseModel):
    source: Literal["auto", "registry", "files"] = "auto"
    prompts_dir: str | None = None
    answer: str = "rag.answer"
    answer_strict: str = "rag.answer_strict"
    plan: str = "rag.plan"
    judge: str = "rag.judge"
    # pin a version per prompt name (e.g. {"rag.answer": "v1"}); unpinned prompts use their active version
    versions: dict[str, str] = Field(default_factory=dict)


class PlannerConfig(BaseModel):
    # auto: call the model only when there is history to resolve or the question looks multi-part
    use_llm: Literal["auto", "always", "never"] = "auto"
    route: str = "small"
    task: str = "rewrite"
    timeout_seconds: float = 3.0
    max_subqueries: int = 3
    max_history_turns: int = 6
    min_subquery_words: int = 3


class RerankConfig(BaseModel):
    enabled: bool = True
    via: Literal["rerank", "gateway"] = "rerank"  # reranking service, or the gateway's rerank route
    model: str | None = None  # None: the reranking service's default model / the gateway's "reranker" route
    timeout_seconds: float = 2.0
    max_passage_tokens: int = 512  # passages are trimmed to the reranker's length limit
    min_score: float | None = None  # drop passages scored under this (None keeps every reranked passage)


class RetrievalConfig(BaseModel):
    top_k: int = 50  # candidates from hybrid search
    top_n: int = 5  # passages kept after rerank
    # weak retrieval: when the best candidate scores under min_score the answer is "not found" - never a guess.
    # Fused hybrid scores (RRF) are rank-based and say nothing about relevance, so the default gate is the
    # dense (cosine) score; a hit without that field is judged on its fused score.
    min_score: float = 0.2
    score_field: Literal["score", "dense_score", "keyword_score"] = "dense_score"
    min_rerank_score: float | None = None  # optional second gate on the best reranked passage
    embed_query: bool = True  # embed through the gateway; when False search embeds (or runs keyword only)
    embedding_model: str = "embedding"


class ContextConfig(BaseModel):
    max_context_tokens: int = 3000  # budget for the numbered sources
    max_passage_tokens: int = 700
    history_tokens: int = 600  # summarised history placed before the sources
    recent_turns: int = 4  # newest turns kept verbatim; older ones are summarised


class GenerationConfig(BaseModel):
    model: str = "auto"  # route or model when the request says "auto"
    fallback_model: str = "small"  # used when the main model is slow or failing
    timeout_seconds: float = 20.0
    first_token_timeout_seconds: float = 5.0  # stream mode: fall back if the first token is this late
    max_tokens: int = 800
    temperature: float = 0.1


class GroundingConfig(BaseModel):
    method: Literal["lexical", "embedding", "hybrid"] = "lexical"
    min_support: float = 0.6  # a claim is supported when its best passage scores at least this
    min_grounded_ratio: float = 1.0  # share of claims that must be supported for the answer to pass
    min_claim_words: int = 3  # shorter sentences (greetings, "Yes.") are not checked
    require_citations: bool = False  # treat claims without a valid [n] marker as unsupported
    use_llm_judge: bool = False
    judge_route: str = "judge"
    judge_timeout_seconds: float = 8.0
    embedding_dim: int = 384


class StreamConfig(BaseModel):
    # verdict after streaming: grounded -> ok; ratio >= flag_above -> flag; otherwise withdraw
    flag_above: float = 0.5


class CacheConfig(BaseModel):
    enabled: bool = True
    kv: dict[str, Any] = Field(default_factory=lambda: {"type": "memory"})
    threshold: float = 0.92
    default_ttl_seconds: float = 3600
    ttl_by_content_type: dict[str, float] = Field(default_factory=dict)
    max_entries_per_scope: int = 5000
    content_type_field: str = "content_type"  # chunk metadata key naming the content type
    never_cache_data_classes: list[DataClass] = Field(
        default_factory=lambda: [DataClass.PERSONAL, DataClass.RESTRICTED])


class MemoryIntegrationConfig(BaseModel):
    enabled: bool = True  # use connect("memory") when a request carries session_id
    record_turns: bool = True


class RagConfig(BaseModel):
    # physical collection name (the search service's <base>__<app>__<tenant> layout). {collection} is the
    # request's collection, else default_collection, else apps.<app>.collection, else search.base_collection
    collection_template: str = "{collection}__{app}__{tenant}"
    default_collection: str | None = None
    not_found_message: str = "I could not find this in the available sources."
    sources_only_message: str = ("I could not produce an answer that is fully supported by the sources. "
                                 "These are the most relevant passages:")
    not_found_patterns: list[str] = Field(default_factory=lambda: [
        r"could not find (this|that|it|the answer)", r"not (found|available) in the (available )?sources",
        r"sources do not (contain|mention|say)"])
    feature: str = "rag"
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    memory: MemoryIntegrationConfig = Field(default_factory=MemoryIntegrationConfig)
    # design targets per stage in milliseconds (slide "Latency budget"); overruns are counted and traced
    latency_budget_ms: dict[str, float] = Field(default_factory=lambda: {
        "screen": 50, "embed": 30, "search": 70, "rerank": 150, "model_start": 500, "first_token": 800,
        "generate": 4000,
        "check": 500})
    # swappable building blocks, built with aiready.core.plugins.create(kind, spec); replace any of them with
    # {class: your_pkg.module:Class} (constructor receives config=<its section>, prompts=<PromptStore>)
    components: dict[str, dict[str, Any]] = Field(default_factory=lambda: {
        "planner": {"type": "default"},            # kind rag_planner
        "retriever": {"type": "default"},          # kind rag_retriever
        "reranker": {"type": "default"},           # kind rag_reranker
        "context_builder": {"type": "default"},    # kind rag_context_builder
        "citation_mapper": {"type": "default"},    # kind rag_citation_mapper
        "grounding": {"type": "default"},          # kind rag_grounding
    })
