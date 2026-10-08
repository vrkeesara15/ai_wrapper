# Customising the platform

Nothing client-specific requires a fork. There are four levels, from most to least common.

## 1. Configuration

All configuration lives in `config/` (`AIREADY_CONFIG_DIR`). Every `*.yaml` file below it is deep-merged into one
tree: mappings merge, **lists concatenate** (so `config/apps/<app>/*.yaml` can add validators, tool sources and
ingestion sources without editing the shared files). Environment overlays (`AIREADY_CONFIG_OVERLAY`, e.g.
`deploy/config-overlays/k8s`) are merged last; in an overlay `=key:` replaces instead of merging.

Values can reference the environment and secrets:

```yaml
url: ${QDRANT_URL:-http://qdrant:6333}       # env with default
api_key: ${secret:qdrant_api_key}            # AIREADY_SECRET_QDRANT_API_KEY or /run/secrets/qdrant_api_key
```

Services watch the configuration and apply changes **without a restart**: routes, models, budgets, rate limits,
policies, prompts, workflows, tools. `aiready config validate` checks every section against its schema and
compiles the policies; run it (and `aiready policy test`) in CI before a change is deployed.

| What to change | Where |
|---|---|
| Model pools, replicas, served model names, costs, tiers, data classes | `config/models.yaml` (`models.pools`, `models.catalog`) |
| Which model serves which task, eval-pass lists, fallbacks, response mode | `config/gateway.yaml` `gateway.routes` |
| Rate limits, token budgets, caching, guardrail use, egress switch | `config/gateway.yaml` |
| Policies (who may do what, obligations) | `config/policies/*.yaml` + `config/policies/tests/` |
| Egress destinations and allowed data classes | `config/security.yaml` `security.egress` |
| Token issuer, TTL, trusted IdPs (OIDC) | `config/security.yaml` `security.tokens` |
| Guardrail detectors, thresholds, per-app profiles | `config/guardrails.yaml` |
| Embedding / rerank / search backends and parameters | `config/knowledge.yaml` |
| RAG: top-k/top-n, thresholds, grounding method, latency budget, prompts | `config/rag.yaml`, `config/prompts/` |
| Memory: summarisation, consent, TTLs | `config/memory.yaml` |
| SQL: replica, row limits, access rules; approved metrics | `config/query.yaml`, `config/semantic_layer.yaml` |
| Agent limits, approval routing | `config/agents.yaml` |
| Workflows (patterns or explicit step graphs) | `config/workflows/*.yaml` |
| Tools: OpenAPI sources, HTTP tools, overrides (effect, roles, data class) | `config/tools.yaml` |
| Assistant widget per app (title, workflow, origins) | `config/assistant.yaml` |
| Identity bridge validators, role mapping; API facade endpoints; change capture | `config/integration.yaml` |
| Ingestion sources, schedules, chunking, cleaning, verification questions | `config/ingestion.yaml` |
| Observability: retention, masking, pricing, teams, budgets, alerts, access | `config/observability.yaml` |
| Golden sets, safety suites, release gate thresholds | `config/evals/` |
| Fine-tuning readiness, datasets, training, release | `config/finetune.yaml` |
| A whole new application | `aiready app init` -> `config/apps/<app>/` ([onboarding.md](onboarding.md)) |

## 2. Plugins

Every swappable component is built from a *spec*:

```yaml
vector_store: {type: qdrant, url: ...}                  # a registered implementation
vector_store: {class: acme_ai.stores:VespaStore, ...}   # or any importable class
```

| Kind | Built-ins | Contract |
|---|---|---|
| `model_backend` | `openai` (vLLM, TGI, Ollama, llama.cpp, LiteLLM, private endpoints), `anthropic`, `mock` | `aiready.models.backends.ModelBackend` |
| `vector_store` | `memory`, `qdrant`, `pgvector` | `aiready.search.stores` |
| `keyword_index` | `bm25`, `postgres`, `opensearch` | `aiready.search.keyword.KeywordIndex` |
| `embedder` | `hash`, `openai`, `sentence_transformers` | `aiready.embedding.encoders` |
| `reranker` | `lexical`, `remote`, `cross_encoder` | `aiready.rerank.scorers` |
| `detector` | `injection`, `pii`, `toxicity`, `guard_model`, `presidio` | `aiready.guardrails.detectors` |
| `connector` | `filesystem`, `sql`, `http_json`, `confluence`, `jira`, `s3` | `aiready.ingestion.connectors` |
| `parser` | text, markdown, html, pdf (+`ocr` hook), docx, csv, json | `aiready.ingestion.parsers` |
| `chunker` | `heading`, `fixed`, `table_rows` | `aiready.ingestion.chunking` |
| `landing_store`, `change_capture`, `acl_resolver` | local/s3; polling/debezium/outbox; sidecar/path_map/posix | `aiready.ingestion`, `aiready.integration` |
| `session_validator` | `introspect_http`, `oidc`, `shared_session_store`, `header_trust` | `aiready.integration.identity_bridge` |
| `tool`, `agent_step` | built-in tools; step types | `aiready.mcp.registry`, `aiready.agents.steps` |
| `rag_planner`, `rag_retriever`, `rag_reranker`, `rag_context_builder`, `rag_citation_mapper`, `rag_grounding` | defaults | `aiready.rag.*` |
| `scorer`, `eval_target`, `notifier`, `trace_store`, `experiment_tracker` | see `aiready.evals`, `aiready.observability`, `aiready.finetune` | |
| `policy_engine` / `policy_source` | `local` (YAML), `opa` | `aiready.governance.policy` |
| `audit_sink` | `jsonl`, `sqlite`, `http` (SIEM), `stdout`, `memory` | `aiready.core.audit.AuditSink` |
| `kv_store`, `queue` | `memory`, `redis` | `aiready.core.stores`, `aiready.core.queues` |

Ship client code as its own package installed next to the platform:

```python
# acme_ai/stores.py
from aiready.core.plugins import register

@register("vector_store", "vespa")
class VespaStore:
    def __init__(self, url: str, **kw): ...
    def upsert(self, collection, chunks): ...
    def search(self, collection, vector, flt, k): ...
    ...
```

```toml
# acme_ai/pyproject.toml - loaded automatically, then usable as `type: vespa`
[project.entry-points."aiready.plugins"]
acme = "acme_ai.stores"
```

Constructor parameters come from the spec; unknown keys are ignored unless the constructor takes `**kwargs`.

## 3. Service split and scaling

Every service implements a contract in `aiready/core/interfaces.py` twice: in-process and as an HTTP client.
`services.<name>.url` decides which one `connect(name)` returns:

```yaml
services:
  search: {url: http://search:8102}   # call the search pod
  rag: {}                             # run in this process
```

The pilot runs everything in one process (`aiready serve all`); production runs one deployment per service
(Helm chart). The code is the same; the architecture is the same at every stage, only the footprint changes.

## 4. Extending a service

When configuration and plugins are not enough, add a package beside the others following
[CONVENTIONS.md](CONVENTIONS.md): a `service.py` implementing a contract, `api.py` built with
`create_service_app`, a `client.py`, wiring providers in `__init__.py`, and the governance hooks (policy decision,
audit record, trace span, metrics) at every decision point.
