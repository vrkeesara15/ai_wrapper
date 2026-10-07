# Engineering conventions

These rules apply to every package in `src/aiready`. They keep the platform plug-and-play: the client's
application is never modified, and anything client-specific is configuration or a plugin.

## 1. Package shape

Every service package `aiready/<pkg>/` has:

| File | Purpose |
|---|---|
| `__init__.py` | Registers wiring providers: `@provider("<name>", "local")` builds the in-process service, `@provider("<name>", "http")` builds the HTTP client. Keep imports light. |
| `service.py` | The in-process implementation of the Protocol in `aiready/core/interfaces.py` (async methods). |
| `api.py` | `create_app() -> FastAPI` built with `aiready.core.app.create_service_app(name)`. Run with `uvicorn aiready.<pkg>.api:create_app --factory`. |
| `client.py` | HTTP client with the same method signatures as the service, built on `aiready.core.http.ServiceClient`. |
| `config.py` | Pydantic models for the package's configuration section, with defaults that work out of the box. |
| other modules | Building blocks (stores, detectors, steps). Swappable ones are registered with `aiready.core.plugins.register(kind, name)`. |

The service calls other services only through `aiready.core.wiring.connect("<name>")`. Never import
another package's `service.py` directly from your own `service.py`, so a deployment can split them into pods.

## 2. Identity

* Every API endpoint (except `/healthz`, `/readyz`, `/metrics`) depends on
  `aiready.core.identity.current_principal` (or `current_principal_or_service` for pipeline/batch endpoints).
* Pass the `Principal` down every call. Access filters use `principal.access_groups`.
* There are no shared service accounts with data rights. Service tokens only prove *which service* relays
  the call.

## 3. Governance hooks (mandatory)

* **Policy**: before invoking a model, tool, data source, egress route, SQL or memory write, call
  `connect("policy").decide(PolicyInput(...))` and respect `allow`, `effect` and `obligations`.
* **Audit**: write a record through `aiready.core.audit.get_audit().for_principal(principal, "<pkg>.<event>", ...)`
  for every decision that matters: model call, retrieval (record ids), tool call, approval, policy denial,
  config change, memory write, SQL executed.
* **Tracing**: wrap each stage in `with span("<pkg>.<stage>", ...)`. Record stage latency with
  `metrics.time_stage(service, stage)`.
* **No direct model calls**: anything that needs a model (chat, embedding, rerank, guard model) goes through
  `connect("gateway")`, except the model-serving layer itself and the gateway's own upstream calls.

## 4. Configuration

* All configuration lives in `config/*.yaml` (see `config/README.md`). Each package reads its own top-level
  section through `get_config().section("<section>", Model)`.
* Defaults must make the package run locally with no external infrastructure (memory/sqlite/mock backends).
* Subscribe to `get_config().subscribe(cb)` when a component must pick up changes without restart
  (routes, budgets, policies, models).

## 5. Errors

Raise `aiready.core.errors.*` (they map to HTTP status codes and a uniform JSON body). Never return a
stack trace to a caller.

## 6. Tests

`tests/<pkg>/test_*.py`, pytest + pytest-asyncio (`asyncio_mode = auto`). The shared fixtures in
`tests/conftest.py` give an isolated config, token service, audit log and doc store. Tests must run with no
network and no GPU: use the `mock` model backend and in-memory stores.

## 7. Service ports (defaults)

| Service | Port | Wiring name |
|---|---|---|
| gateway (data plane) | 8080 | `gateway` |
| gateway admin (control plane) | 8081 | - |
| guardrails | 8090 | `guardrails` |
| governance (policy engine, registries) | 8091 | `policy`, `registry` |
| embedding | 8100 | `embedding` |
| rerank | 8101 | `rerank` |
| search (vector + keyword) | 8102 | `search` |
| rag | 8110 | `rag` |
| memory | 8111 | `memory` |
| query (structured data) | 8112 | `query` |
| agent runtime | 8120 | `agents` |
| assistant API + widget | 8130 | - |
| MCP server | 8140 | `tools` |
| identity bridge | 8150 | - |
| API facade | 8151 | - |
| ingestion | 8160 | - |
| observability (traces, feedback, scores, cost) | 8170 | `observability` |
| mock model server (dev only) | 8200 | - |
