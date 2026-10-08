# AI-Ready: an AI foundation beside the applications a business already runs

A plug-and-play implementation of the *AI-ready applications* reference architecture (`AI-Ready.pdf`,
text in `docs/reference-architecture.txt`). It is deployed **beside** an existing application, inside the
client's own environment, and reaches the application only through its login, its existing API and its
documents / read replica. **The application's code is not modified.**

```
 Channels          app screens · assistant widget (one <script> tag) · outside agents (MCP) · browser agents (WebMCP)
 Agent interface   MCP server over the app's API · WebMCP add-on · every call under the user's own login
 Orchestration     agent runtime · workflows · human approval before any write
 AI gateway        one endpoint for every model: auth · limits/budgets · guardrails · cache · routing · tracing
 Knowledge/memory  ingestion pipeline · hybrid search + rerank · session and long-term memory
 Model serving     open-weight models on the client's GPUs (vLLM / TEI) · optional private endpoint
 Existing app      database, APIs, documents and login stay as they are
 ── across all ──  security & identity · policy as code · audit · observability · evaluation & release gate
```

## Quick start (pilot, no GPU, offline)

```bash
make install                                 # .venv with every extra
python -m examples.legacy_app &              # the sample "existing application" (claims portal) on :8000
aiready serve all --mock-models &            # every platform service in one process (stage-1 footprint)
python scripts/smoke_test.py                 # 24 end-to-end checks of the request path and every control
```

Or with containers: `make run-demo` (Docker Compose, zones as networks, Postgres/Redis/Qdrant/OTel/Grafana),
`make run-pilot-gpu` for vLLM + TEI on one GPU node, and the Helm chart in `deploy/helm/aiready` for
Kubernetes (stages 2 and 3). See [docs/deployment.md](docs/deployment.md).

## Onboard an application (configuration only)

```bash
aiready app init crm --base-url https://crm.internal --whoami /api/me --session-cookie CRMSESSION \
    --openapi /openapi.json --docs /mnt/shares/crm-docs --user-role rep --owner-team sales
aiready config validate && aiready policy test && aiready app check crm
```

This writes `config/apps/crm/`: login trust (identity bridge validator + role mapping), MCP tools generated from
the app's OpenAPI spec, ingestion sources with access tags, a router + agent workflow, app-specific policies,
budgets and an evaluation suite. Nothing else is edited; every file is commented. See
[docs/onboarding.md](docs/onboarding.md).

## How the architecture maps to the code

| Slide / component | Module | Port |
|---|---|---|
| AI gateway (8 steps: authenticate, limit, screen, reuse, route, call, validate, record) | `aiready.gateway` | 8080 |
| Gateway control plane (routes, models, budgets, keys - live, audited) | `aiready.gateway.admin` | 8081 |
| Model serving, inference router, pools, OpenAI-compatible / Anthropic / mock backends | `aiready.models` | (8200 mock) |
| Guardrail service (injection, PII masking, secrets, output schema) | `aiready.guardrails` | 8090 |
| Policy engine (policy as code), model / prompt / adapter registry, egress, access reviews, incidents | `aiready.governance` | 8091 |
| Embedding service (vector cache, dynamic batcher, live/bulk queues) | `aiready.embedding` | 8100 |
| Reranking service (cross-encoder, threshold, top-n, fallback) | `aiready.rerank` | 8101 |
| Vector search service (dense + BM25, RRF, access filter first, collection per app/tenant) | `aiready.search` | 8102 |
| RAG service (planner, retriever, context builder, citation mapper, hallucination check, modes A/B) | `aiready.rag` | 8110 |
| Memory service (session summaries, long-term with consent + expiry, right to erasure) | `aiready.memory` | 8111 |
| Query service (semantic layer, schema catalog, read-only validated SQL on the replica) | `aiready.query` | 8112 |
| Agent runtime (workflows, plan-act-check-decide, checkpoints, limits, approval inbox, sub-agents) | `aiready.agents` | 8120 |
| Assistant API + embeddable widget | `aiready.assistant`, `web/widget` | 8130 |
| MCP server (OpenAPI -> tools, user identity, approvals) + WebMCP add-on | `aiready.mcp`, `web/webmcp` | 8140 |
| Identity bridge (app session -> short-lived signed token) | `aiready.integration.bridge_api` | 8150 |
| API facade (clean endpoints over existing functions, generated OpenAPI) | `aiready.integration.facade_api` | 8151 |
| Change capture (replica polling / Debezium / outbox), file crawler with ACLs | `aiready.integration` | - |
| Ingestion platform (connectors, registry, landing, parse, clean, chunk, enrich, embed, index, verify, DLQ, lineage) | `aiready.ingestion` | 8160 |
| Observability & evaluation (trace store, replay, cost/chargeback, latency vs budget, live scorers, review queue, alerts, dashboard) | `aiready.observability` | 8170 |
| Evaluation suites, release gate, safety tests, routing evaluation | `aiready.evals` | CLI |
| Fine-tuning pipeline (readiness, dataset builder, adapter jobs, release gate, rollback) | `aiready.finetune` | CLI |
| Capacity & sizing method, load test, cost model | `aiready.sizing` | CLI |
| Semantic & exact cache (scope, expiry, never-cache rules, invalidation) | `aiready.cache` | - |
| Shared foundation: config, identity, plugins, audit, telemetry, stores, queues, wiring, contracts | `aiready.core` | - |
| Deployment topologies (compose pilot, Helm production/platform, AWS Terraform, air-gapped on-prem bundle) | `deploy/` | - |
| Operator CLI | `aiready` (`aiready.cli`) | - |

## Control and governance

Every control from the *Controls for security review* slide is built in and enforced by default. Details,
the role catalogue and the policy actions are in [docs/governance.md](docs/governance.md).

| Control | How it is enforced |
|---|---|
| Identity | Agents act as the signed-in user: the identity bridge turns the app session into a short-lived signed token that travels on every hop; tools call the app with the user's own session. No shared service accounts carry data rights. Revoked users/tokens are refused at authentication in every service. Signing keys rotate in stages (`aiready token rotate`). |
| Guardrails | Inputs and tool results screened for injection; personal data masked (reversible placeholders, Luhn/IBAN-checked); secrets never reach a model; outputs validated against a schema; ingested documents screened for indirect injection. |
| Approval gates | Reads run freely; writes, payments and deletions wait for a person (four eyes, expiry, bound to the exact arguments). |
| Policy as code | Versioned YAML rules (deny overrides, obligations), hot-reloaded, explained, tested in CI (`aiready policy test`); OPA adapter available. Default deny for sensitive actions; egress closed by default. |
| Evaluation | Golden sets + safety suites run before every release; the gate blocks a release that scores lower and records the verdict on the model registry entry. |
| Audit trail | Every prompt hash, retrieved record, tool call, approval, policy denial, config change and answer is written to a hash-chained, HMAC-signed, append-only log (`aiready audit verify`). |
| Data stays inside | Models, index and logs run in the client's environment; the model zone has no outbound network; public model routes need a policy entry naming the data classes allowed to leave. |
| Measured from day one | Traces for every request, cost and latency per feature/team/model, latency budget per stage, quality scores over time, alerts. |

## Customising without forking

* **Configuration** (`config/*.yaml`, deep-merged, hot-reloaded, with environment overlays) covers routes,
  models, budgets, policies, prompts, workflows, tools, sources, chunking, thresholds, latency budgets.
* **Plugins**: every swappable component (model backend, vector store, keyword index, connector, parser,
  chunker, detector, session validator, scorer, notifier, queue, audit sink, workflow step, RAG stage...) is
  created from a spec - `type: <registered name>` or `class: your_pkg.module:YourClass`. Ship client code as a
  separate package (optionally registered through the `aiready.plugins` entry point) and switch it on in config.
* **Service split**: every service runs in-process (pilot) or as its own pod (production) - set
  `services.<name>.url`. Same code, same contracts (`aiready/core/interfaces.py`).

See [docs/customization.md](docs/customization.md).

## Repository layout

```
src/aiready/     platform packages (one per component, each: service.py, api.py, client.py, config.py)
config/          all configuration: platform, models, gateway, policies/, workflows/, prompts/, evals/, apps/
web/             assistant widget and WebMCP add-on (zero-dependency JS)
deploy/          compose (pilot), helm (production/platform), terraform (AWS), onprem bundle, observability, security
examples/        legacy_app: the untouched sample application used by demos and tests
scripts/         smoke_test.py: end-to-end checks against a running deployment
tests/           unit, contract and integration tests (no network, no GPU)
docs/            deployment, onboarding, governance, customization, conventions, runbooks
```

## Development

```bash
make test           # pytest (489 tests, offline)
make lint           # ruff
make policy-test    # policy CI cases
make eval-gate      # evaluation suite + release gate
```
