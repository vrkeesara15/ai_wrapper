# Control and governance

The controls are built before the first agent ships, so the security team reviews a finished design. This
document lists what is enforced, where, and how to operate it.

## 1. Enforcement points

```
 user ─► app session ─► identity bridge ──signed token──► every service
                                             │
            ┌────────────────────────────────┼─────────────────────────────────────┐
            │ authentication   revocation deny-list (subject / token id), key ring    │ every request
            │ policy engine    PolicyInput(action, principal, resource, context)      │ every decision point
            │ guardrails       screen input · mask PII · block injection · validate   │ gateway, RAG, tools, ingestion
            │ approval gate    writes / payments / deletes wait for a person          │ MCP executor, agent runtime, facade
            │ access tags      chunk-level visibility, filtered before scoring        │ search
            │ egress           public routes closed unless a rule opens a data class  │ gateway, egress proxy
            │ audit            hash-chained, HMAC-signed, append-only                 │ all of the above
            └─────────────────────────────────────────────────────────────────────────┘
```

| Decision point | Policy action | Module |
|---|---|---|
| Token issued for an app session | `identity.token_issue` | `integration.identity_bridge` |
| Any model call (chat / embed / rerank) | `model.invoke`, `model.embed`, `model.rerank` | `gateway.service` |
| Prompt leaves the boundary | `egress.send` | `gateway.service`, `governance.egress` |
| Search / retrieval on a collection | `data.read`, `rag.*`, `search.*` | `search.service`, `rag.service` |
| Source read / document indexed | `data.ingest`, `data.index` | `ingestion.pipeline` |
| Tool call / facade call | `tool.call`, `facade.call` | `mcp.executor`, `integration.facade` |
| SQL | `sql.execute`, `sql.metric`, `query.metric` | `query.service` |
| Memory | `memory.write`, `memory.read`, `memory.forget`, `memory.session` | `memory.service` |
| Trace replay | `trace.replay` | `observability.service` |
| Administration | `admin.*` (`admin.registry.*`, `admin.policy.*`, `admin.config.*`, `admin.egress.*`, `admin.incident.*`, `admin.access_review.*`) | governance and gateway admin APIs |

Default effect per action prefix is set in `config/governance.yaml` (`governance.policy.defaults`): sensitive
prefixes (`model.`, `tool.`, `data.`, `sql.`, `memory.`, `egress.`, `admin.`, `identity.`, `facade.`) default
to **deny**; a rule must allow them.

## 2. Policy as code

Rules live in `config/policies/<domain>.yaml`:

```yaml
policies:
  tool:
    version: 1.0.0
    rules:
      - id: tool.side_effect.require_approval
        action: [tool.call, facade.call]
        effect: require_approval          # allow | deny | require_approval | obligate
        when:
          resource.effect: {in: [write, payment, delete]}
          resource.approval_id: {exists: false}
        obligations: {approval_roles: [approver]}
```

* Conditions use a small safe expression language (`eq ne in not_in contains any_in glob regex gt lt exists`,
  combinators `all any not`, `{ref: principal.access_groups}`) - never Python `eval`.
* Combining: **deny > require_approval > allow**; `obligate` only adds obligations (strictest value wins).
* Facts that callers cannot fake are filled in by the engine itself (model registry status, boundary,
  whether egress is open for the data class, revocation).
* Bundles are versioned (`semver+content hash`), hot-reloaded, and a bundle that fails to compile is refused
  while the previous one stays active. `aiready policy decide <action> --explain` shows the rule trace.
* CI: `aiready policy test` runs `config/policies/tests/*.yaml`. Add a case for every new rule.
* Prefer Rego? Set `governance.policy.engine: opa` to query an OPA server instead.

### Shipped rules

| Rule | Action | Effect | Purpose |
|---|---|---|---|
| base.authenticated | `*` | deny | Every governed action needs an authenticated principal |
| base.tenant_isolation | `*` | deny | Never act on another tenant's resources |
| model.registry.not_approved / .disabled / .data_class | `model.*` | deny | Only approved, open models, for allowed data classes |
| model.public.data_class / .egress_closed | `model.*` | deny | Public models get PUBLIC/INTERNAL data only, through an opened egress route |
| model.obligations.default / .personal | `model.*` | obligate | Output ceiling; no full-prompt logging for personal data |
| egress.send.closed / .opened | `egress.send` | deny / allow | Egress closed unless a destination is opened for the data class |
| data.read.access_tags / .restricted / .personal_mask | `data.read` | allow / deny / obligate | Access tags, restricted data, PII masking |
| ingestion.index.injection / .restricted / .personal_unmasked | `data.index` | deny | Poisoned, restricted or unmasked-PII documents never enter the index |
| tool.read.allow | `tool.call` | allow | Reads run freely |
| tool.side_effect.require_approval | `tool.call` | require_approval | Writes, payments, deletions wait for a person |
| tool.required_roles / .unknown_effect | `tool.call` | deny | Role-gated tools; unknown effects never run |
| tool.obligations.personal | `tool.call` | obligate | Tool results with PII are masked before a model sees them |
| sql.write.deny / sql.analyst.read_only / sql.approved_metric | `sql.execute` | deny / allow | Generated SQL is read-only and analyst-only; approved metrics for everyone |
| memory.write.consent / .no_consent / memory.other_user | `memory.*` | allow / deny | Consent and per-user isolation |
| admin.no_service_principals | `admin.*` | deny | Administration is done by people |
| admin.registry.approval | `admin.registry.approve` | deny | Only model approvers approve |

## 3. Roles

Roles come from the client's login through the identity bridge role mapping (`integration.identity_bridge.
role_mapping`, or per app in `config/apps/<app>/identity.yaml`). Unmapped application roles grant nothing.

| Role | Grants |
|---|---|
| `user` | Default for any mapped user: ask the assistant, read tools, own memory |
| `<app>-user` (e.g. `claims-user`) | The app's tools (generated per app) |
| `approver`, `claims-lead`, `finance-approver` | Decide pending writes / status changes / payments (routing in `config/agents.yaml` `approvals.rules`) |
| `analyst` | Generated read-only SQL (approved metrics are open to all) |
| `pii-reader`, `restricted-data-reader` | See unmasked personal data / restricted collections |
| `platform-admin` | Gateway control plane (routes, models, budgets, keys), usage reports, dashboards, replay |
| `security-admin` | Incidents (revoke, close routes, trace impact), egress routes, audit, access reviews |
| `model-approver` | Approve / deprecate models and adapters (four eyes: not the registrant) |
| `ai-engineer` | Register models, prompts, adapters; review queue |
| `ingestion-admin` | Trigger runs, inspect lineage, replay dead letters |
| `access-reviewer`, `auditor`, `ops`, `finance`, `reviewer` | Reviews, read-only dashboards, chargeback, answer review |
| `service`, `ingestion`, `verifier` | Service identities: prove which service relays a call; no data rights of their own |

## 4. Identity

* The identity bridge validates the existing app session (introspection of the app's "who am I" endpoint,
  OIDC token exchange, shared session store or trusted headers) and issues a token valid for 15 minutes,
  carrying `sub`, `tenant`, `app`, `roles`, `groups`, `jti`. One bridge serves every app: each app's
  validator names its `app` and role mapping.
* Every service verifies the token, then checks the revocation deny-list. Agents and tools forward the
  user's token (and, for the facade, the user's own app session), so the existing application applies its
  own permissions too.
* Service tokens (`X-Service-Token`) only prove which service relays a call.
* Key rotation without downtime: `aiready token rotate --stage add` → wait one reload → `--stage activate` →
  wait longer than the token TTL → `--stage retire`. See `docs/runbooks/rotate-signing-keys.md`.

## 5. Approvals

* `ToolResult(ok=false, approval_id=...)` / HTTP 202 for any write, payment or delete.
* The approval is bound to the requester, tool and a hash of the exact arguments; it expires; the requester
  cannot approve it (four eyes); approver roles are routed per tool / effect.
* The agent runtime pauses the run (`waiting_approval`) and resumes on the decision; waiting time does not
  count against the run's time limit.

## 6. Audit

* Records: `gateway.request` (prompt hash, model, decision, data class, tokens, cost), `search.query`
  (returned chunk ids), `rag.answer` (question, answer, chunk ids, grounding), `tool.call`,
  `approval.request/decision/use`, `policy.decision` (denials always), `identity.token_issue/denied`,
  `config.change` (before/after), `registry.*`, `incident.*`, `ingestion.*`, `memory.*`, `cli.break_glass`.
* Integrity: each record's hash covers the previous one; HMAC with a client-held key; SQL sink refuses
  UPDATE/DELETE; appends are serialised across processes. `aiready audit verify [--all-chains]`.
* Forwarding: add an `http` sink (SIEM) or `stdout` (log shipper) next to the system of record.
* Key rotation: `aiready audit rotate-key` (older records keep verifying with the previous keys). A confirmed
  break: `aiready audit reset-chain --incident INC-n` quarantines the segment as evidence.

## 7. Model, prompt and adapter registry

* Models: draft → approved (needs licence and weights checksum, four eyes) → deprecated; `disabled` by an
  incident overrides approval. Only approved models are routable (`governance.registry.require_registration`
  makes unknown models unroutable in production).
* Release-gate verdicts (`aiready eval gate`) are attached to the model version's `eval_scores` and history.
* Prompts: every version kept, active pointer per environment, diff. Adapters: linked to their base model
  version and dataset version; approved only if they beat the base on held-out data.

## 8. Incidents

| Need | Command |
|---|---|
| Open an incident | `aiready incident open --title ... --severity sev1` |
| Cut off a user / token | `aiready incident revoke --subject <user>` / `--jti <id>` (refused everywhere within seconds) |
| Revoke an API key | `aiready token revoke --key-id <id> --reason INC-n` |
| Close a model, route or egress destination | `aiready incident close-route --model <m> / --route <r> / --egress <dest>` |
| Who and what was affected | `aiready incident trace-impact --since <t> [--subject] [--model]` |
| Evidence | `aiready audit export --since --until --verify --out evidence.jsonl` |

## 9. Access reviews

`POST /v1/access-reviews` on the governance service generates who may use which model and data (from policies,
registry and group mappings), records reviewer sign-off and flags stale grants; overdue reviews are listed.
