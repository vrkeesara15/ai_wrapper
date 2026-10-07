# WebMCP add-on

> **Status (October 2026): emerging, experimental.** WebMCP (`navigator.modelContext`) is available as a
> Chrome origin trial; the specification is a W3C community group draft, **not a standard**. The API shape
> may still change. This add-on is progressive enhancement: where WebMCP is absent the script returns
> immediately and the page behaves exactly the same.

The add-on registers the application's tools with a browser agent, so the agent calls the app's actions
directly instead of scraping the screen. The tools are the same ones the MCP server exposes - one tool per
API facade endpoint - under the permissions the application already enforces.

```html
<script src="https://ai.example.com/widget/webmcp-tools.js"
        data-endpoint="https://ai.example.com/mcp-api"
        data-token-endpoint="https://ai.example.com/v1/assistant/session"
        data-app="claims"
        defer></script>
```

| Attribute | Meaning |
|---|---|
| `data-endpoint` | MCP server base URL (port 8140 service): `GET /v1/tools?format=webmcp`, `POST /v1/tools/{name}:call` |
| `data-token-endpoint` | Session exchange (assistant API) giving the signed-in user's short-lived token |
| `data-app` | App id |
| `data-manifest` | Optional. A URL, or `inline` to read `<script type="application/json" id="aiready-webmcp-manifest">` |

## How it works

1. Feature-detects `navigator.modelContext` (`registerTool`, or `provideContext` on earlier drafts).
2. Loads the tool manifest the user may see - role-filtered by the MCP server - from
   `/v1/tools?format=webmcp` (or an inline manifest).
3. Registers each tool with an `execute` that calls `/v1/tools/{name}:call` with the **user's own token**.
4. Read tools run immediately. Write, payment and delete tools first ask the user in the page (via
   `client.requestUserInteraction` when the browser offers it), and then still **wait for an approver** on the
   server: the agent is told the action is pending, with its approval id.

`window.AIReadyAuth.getToken()` (if the page defines it) is used instead of the session endpoint.
`window.AIReadyWebMCP.reload()` re-reads the manifest after the user's roles change.

The page's origin must be allowed in `tools.cors_allowed_origins` (`config/tools.yaml`).
