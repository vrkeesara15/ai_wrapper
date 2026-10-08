# Onboarding an existing application

The foundation is built once and reused by every application. Adding an application is configuration only;
its code, database and login are not changed.

## What the foundation needs from the application

| Need | Usually already there | If missing (built first, per the *Legacy integration* slide) |
|---|---|---|
| A way to know who the user is | A "who am I" endpoint, an IdP (OIDC/SAML via the client's SSO), or an authenticating reverse proxy | Identity bridge `header_trust` behind the proxy, or a thin endpoint |
| Actions the assistant may take | Its REST/JSON API with an OpenAPI spec | API facade: declare clean endpoints over existing functions in `integration.facade.endpoints`; it publishes an OpenAPI spec |
| Knowledge | Document shares, wiki, tickets, database | Read replica or change capture for rows (never the live database) |
| A place for the assistant | Its web pages | One `<script>` tag (or the app keeps its UI and calls the assistant API) |

## Steps

### 1. Generate the configuration

```bash
aiready app init crm \
  --base-url https://crm.internal \
  --whoami /api/me --session-cookie CRMSESSION \
  --openapi /openapi.json \
  --docs /mnt/shares/crm-docs \
  --replica-url postgresql+psycopg://reader@crm-replica/crm \
  --user-role rep --user-role manager \
  --owner-team sales-platform --tenant acme --origin https://crm.internal
```

`config/apps/crm/` now holds `app.yaml`, `identity.yaml`, `tools.yaml`, `ingestion.yaml`, `workflows.yaml`,
`policies.yaml`, `evals.yaml` and a README. Files are plain configuration: review and commit them.

### 2. Fill in the details

* **identity.yaml**: map the who-am-I response with JSONPath (`sub`, `email`, `tenant`, `roles`), then map
  application roles to platform roles and groups. Unmapped roles grant nothing.
* **tools.yaml**: include/exclude operations; set `tools.overrides.<tool>.effect` for anything the HTTP method
  does not express (e.g. a `POST /search` that only reads, or a `payment`).
* **ingestion.yaml**: give every document an access list (`.acl.json` sidecars, a path map, or POSIX ACLs) -
  documents without one are not indexed. Map replica tables (`key_columns`, `title_template`, row-level
  `access_tag_columns`), then set `enabled: true`. Write at least one verification question.
* **policies.yaml**: app-specific rules on top of the platform rules.
* **evals.yaml**: 20-50 real questions with expected sources; the release gate uses them.

### 3. Check and test

```bash
aiready config validate          # every section against its schema, policies compiled
aiready policy test              # policy CI cases (add cases for the app's rules)
aiready app check crm            # login endpoint, API spec and documents reachable; placeholders filled in
aiready app list                 # what each application uses
```

### 4. Load knowledge and evaluate

```bash
aiready ingest run --source crm-docs [--url http://ingestion:8160]
aiready eval run crm --release 1.0.0
aiready eval gate crm --release 1.0.0   # blocks if scores are below thresholds or the safety suites fail
```

### 5. Embed the assistant

```html
<script src="https://assistant.internal/widget/assistant-widget.js"
        data-app="crm" data-endpoint="https://assistant.internal"
        data-token-endpoint="https://bridge.internal/v1/token"></script>
```

The widget exchanges the user's existing session for a short-lived token at the identity bridge. Outside agents
use the MCP server (`https://mcp.internal/mcp`) with the same token; browser agents get the WebMCP add-on where
the browser supports it, and nothing changes where it does not.

### 6. Go live

* Budget action from `warn` to `block` once usage is understood (`app.yaml`).
* Register and approve the models the app's routes use (`aiready registry ...`); set
  `governance.registry.require_registration: true` in production.
* Watch the dashboard (`/dashboard` on observability): spend per feature, latency against the budget, quality,
  review queue. Every reviewed failure becomes a new evaluation case.
