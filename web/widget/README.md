# Assistant widget

A zero-dependency chat widget added to an existing application with **one script tag** - no build step,
no framework, no change to the host app's code or CSS.

```html
<script src="https://ai.example.com/widget/assistant-widget.js"
        data-endpoint="https://ai.example.com"
        data-app="claims"
        data-token-endpoint="https://ai.example.com/v1/assistant/session"
        data-theme="auto"
        defer></script>
```

| Attribute | Default | Meaning |
|---|---|---|
| `data-endpoint` | origin of the script | Assistant API base URL (port 8130 service) |
| `data-app` | `default` | App id: selects the workflow, knowledge collection and CORS allow-list (`config/assistant.yaml`) |
| `data-token-endpoint` | `<endpoint>/v1/assistant/session` | Exchanges the app session (cookie) for a short-lived AI-Ready token via the identity bridge |
| `data-theme` | `auto` | `light`, `dark` or `auto` (follows `prefers-color-scheme`) |
| `data-feature` | - | Screen / use case; mapped to a workflow by `assistant.apps.<app>.features` |
| `data-title` | server config | Panel title |
| `data-position` | `right` | `right` or `left` |

## What it does

* **Isolated UI** - everything renders inside a Shadow DOM: the host's CSS cannot break it, its CSS cannot leak.
* **Streaming** - `fetch` + `ReadableStream` SSE parsing of `/v1/assistant/chat`; words appear as they are
  generated. Events: `run`, `token`, `step`, `approval`, `waiting`, `done`, `error`.
* **Citations** - sources listed under each answer with links (only `http(s)` URLs are linked).
* **Approval notices** - when the assistant tries a write, payment or deletion, the run pauses and the widget
  says that an approver was asked; "Check status" shows the result once decided.
* **Feedback** - thumbs up/down and "Edit answer" (the corrected text and a comment) go to
  `/v1/assistant/feedback`, which forwards them to the observability service (or stores them) for the
  evaluation set.
* **Accessible** - real buttons, `role="dialog"`, `role="log"` with `aria-live`, visible focus, Escape closes,
  Enter sends (Shift+Enter for a new line), focus returns to the launcher, respects reduced motion.
* **Mobile** - full-screen panel under 480px.
* **Safe rendering** - model output is inserted with `textContent` only; never as HTML.

## Identity

The widget never sees the app's credentials. On first use it POSTs to `data-token-endpoint` with the
browser's cookies (`credentials: "include"`); the assistant API relays the configured session cookie to the
identity bridge, which returns a signed, short-lived token for the signed-in user. Every chat call carries
that token, so the assistant (and every tool it calls) acts with the user's own permissions.

Hosts that already hold a token can skip the exchange:

```js
AIReadyAssistant.setTokenProvider(async () => (await fetch("/my/token")).text());
```

## Page API

`AIReadyAssistant.open() / close() / toggle() / send(text) / setToken(token, expiresAt) /
setTokenProvider(fn) / setContext(obj) / setFeature(name)`

## Server-side configuration

The page's origin must be listed in `assistant.apps.<app>.allowed_origins` (`config/assistant.yaml`).
See `demo.html` for a runnable example.
