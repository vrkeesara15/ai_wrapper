# Incident response: revoke keys, close routes, trace impact

**Owner:** platform engineer on call, with the client's security lead. **Severity** is set by data exposure
first, availability second.

## 0. Open the incident (5 minutes)

```bash
aiready incident open --title "<short description>" --severity sev1|sev2|sev3   # prints INC id; audit event incident.opened
```
Freeze changes: pause `deploy.yml` (disable the workflow) and stop running config changes through the admin API.

## 1. Contain - revoke keys

| What leaked / misbehaves | Action |
|---|---|
| A user's or app's API key | `aiready token revoke --key-id <id> --reason INC-<n>`; or `DELETE /admin/keys/{key_id}` on gateway-admin :8081 |
| A user's session/token | `aiready token revoke --subject <user> --reason INC-<n>` (adds the subject to the deny list until tokens expire) |
| A service token key | Rotate it: [rotate-signing-keys.md](rotate-signing-keys.md), section "service tokens" |
| The token signing key | Rotate it immediately: [rotate-signing-keys.md](rotate-signing-keys.md). Every session must sign in again |
| Vendor API key (`llm_api_key`) | Revoke it at the vendor, set an empty secret, and close the public route (step 2) |
| Database credentials (Vault dynamic) | `vault lease revoke -prefix database/creds/aiready-` (operator policy) |

## 2. Contain - close routes

* **Public model / egress route:**
  `aiready incident close-route --route <route> --reason INC-<n>`
  This is a policy change: it removes the egress route in governance (`POST /v1/egress/routes` with
  `enabled: false`). Also delete the domain from `egressProxy.allowlist` and run `helm upgrade`. Either one
  alone closes the route; do both.
* **A model or adapter:**
  `aiready registry deprecate model <name> <version>`
  Routes to it stop at the next config reload (seconds).
* **A tool (MCP):** disable it in `config/` (tool registry) or deny it in policy (`config/policies/tool.yaml`).
  Run `aiready policy test`, then apply. Policy changes are hot-reloaded.
* **Everything, as a last resort:** scale the gateway to zero
  (`kubectl -n aiready scale deploy/gateway --replicas=0`; disable its HPA first with
  `helm upgrade --reuse-values --set components.gateway.hpa.enabled=false`). The existing application keeps
  working, because the assistant is an add-on.

## 3. Trace impact

```bash
aiready incident trace-impact --incident INC-<n> --since <time> [--subject <user>] [--key-id <id>] [--model <m>] [--route <r>]
```
This joins the audit log (who, which model, which records were retrieved, tool calls, policy decisions) with
the traces to produce:
* the requests involved, and the records/documents each one retrieved (`rag.retrieval` record ids);
* the data classes that left the boundary (`gateway.egress` events);
* the writes made by tools or agents (`agents.approval`, `mcp.tool_call` events).

Before you trust that output, check that the audit chain is intact: `aiready audit verify`
([audit-chain-verification.md](audit-chain-verification.md)). If you need to replay a request, use the
trace viewer in Grafana (Tempo); prompt bodies are not stored there by design.

## 4. Recover and close

1. Fix the cause: a policy rule, a guardrail detector, prompt or tool hardening.
2. Add a regression case: a safety suite case in `config/evals/safety/`, or a policy test in
   `config/policies/tests/`.
3. Release through the normal pipeline (the release gate must pass), and re-open the routes you closed.
4. `aiready incident close --incident INC-<n> --summary "..."`. Write a blameless post-mortem within 5 working
   days.
