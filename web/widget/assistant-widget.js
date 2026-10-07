/*!
 * AI-Ready assistant widget - zero dependencies, one script tag, no change to the host app.
 *
 *   <script src="https://ai.example.com/widget/assistant-widget.js"
 *           data-endpoint="https://ai.example.com"            (assistant API base; default: this script's origin)
 *           data-app="claims"                                 (app id: routing, CORS allow-list, knowledge collection)
 *           data-token-endpoint="https://ai.example.com/v1/assistant/session"   (app session -> token)
 *           data-theme="auto"                                 (light | dark | auto)
 *           data-feature="claim-detail"                       (optional: screen / use case -> workflow)
 *           data-title="Claims helper"                        (optional: overrides the server title)
 *           data-position="right"                             (right | left)
 *           defer></script>
 *
 * The UI lives in a Shadow DOM, so the host page's CSS cannot break it and its CSS cannot leak into the page.
 * Model output is only ever inserted as text (never as HTML). Answers stream over SSE (fetch + ReadableStream).
 *
 * Optional page API: window.AIReadyAssistant.{open, close, toggle, send(text), setToken(token, expiresAt),
 * setTokenProvider(async () => token), setContext(object), setFeature(name)}.
 */
(function () {
  "use strict";
  if (window.AIReadyAssistant && window.AIReadyAssistant.__loaded) return;

  var script = document.currentScript || (function () {
    var all = document.querySelectorAll("script[src*='assistant-widget']");
    return all[all.length - 1];
  })();
  var data = (script && script.dataset) || {};
  var scriptOrigin = script && script.src ? new URL(script.src, location.href).origin : location.origin;
  var cfg = {
    endpoint: (data.endpoint || scriptOrigin).replace(/\/+$/, ""),
    app: data.app || "default",
    feature: data.feature || null,
    theme: data.theme || "auto",
    title: data.title || null,
    position: data.position === "left" ? "left" : "right",
    maxHistory: 12
  };
  cfg.tokenEndpoint = data.tokenEndpoint || cfg.endpoint + "/v1/assistant/session";

  // ------------------------------------------------------------------ small helpers
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "text") node.textContent = v;
      else if (k === "class") node.className = v;
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : String(v));
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  function store(kind) {
    try { var s = window[kind]; var k = "__aiready_probe"; s.setItem(k, "1"); s.removeItem(k); return s; }
    catch (e) { return null; }
  }
  var session = store("sessionStorage");
  function sget(key, fallback) {
    try { var v = session && session.getItem("aiready:" + cfg.app + ":" + key); return v ? JSON.parse(v) : fallback; }
    catch (e) { return fallback; }
  }
  function sset(key, value) {
    try { if (session) session.setItem("aiready:" + cfg.app + ":" + key, JSON.stringify(value)); } catch (e) { /* full / blocked */ }
  }
  function uid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "s" + Date.now().toString(36) + Math.random().toString(36).slice(2);
  }
  function safeUrl(u) {
    try { var p = new URL(u, location.href); return (p.protocol === "https:" || p.protocol === "http:") ? p.href : null; }
    catch (e) { return null; }
  }

  // ------------------------------------------------------------------ identity
  var auth = { token: null, expiresAt: 0, provider: null };
  function getToken(force) {
    var now = Date.now() / 1000;
    if (!force && auth.token && (!auth.expiresAt || auth.expiresAt - 30 > now)) return Promise.resolve(auth.token);
    if (auth.provider) {
      return Promise.resolve(auth.provider()).then(function (t) { auth.token = t; auth.expiresAt = 0; return t; });
    }
    // the browser's own app session (cookie) is exchanged for a short-lived token by the identity bridge
    return fetch(cfg.tokenEndpoint, { method: "POST", credentials: "include", headers: { "X-App": cfg.app } })
      .then(function (r) {
        if (!r.ok) throw new Error(r.status === 401 ? "Please sign in to the application first." : "Sign-in failed (" + r.status + ")");
        return r.json();
      })
      .then(function (body) { auth.token = body.token; auth.expiresAt = body.expires_at || 0; return auth.token; });
  }
  function authedFetch(path, opts, retried) {
    return getToken(false).then(function (token) {
      var headers = Object.assign({ "Authorization": "Bearer " + token, "X-App": cfg.app }, (opts && opts.headers) || {});
      if (state.feature) headers["X-Feature"] = state.feature;
      return fetch(cfg.endpoint + path, Object.assign({}, opts, { headers: headers, credentials: "omit" }));
    }).then(function (r) {
      if (r.status === 401 && !retried) { auth.token = null; return authedFetch(path, opts, true); }
      return r;
    });
  }

  // ------------------------------------------------------------------ SSE over fetch
  function readSSE(response, onEvent) {
    if (!response.body || !response.body.getReader) {
      return response.text().then(function (t) { parseChunk(t + "\n\n", onEvent); });
    }
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = "";
    function pump() {
      return reader.read().then(function (res) {
        if (res.done) { if (buffer.trim()) parseChunk(buffer + "\n\n", onEvent); return; }
        buffer += decoder.decode(res.value, { stream: true });
        var parts = buffer.split(/\r?\n\r?\n/);
        buffer = parts.pop();
        parts.forEach(function (block) { parseBlock(block, onEvent); });
        return pump();
      });
    }
    return pump();
  }
  function parseChunk(text, onEvent) { text.split(/\r?\n\r?\n/).forEach(function (b) { parseBlock(b, onEvent); }); }
  function parseBlock(block, onEvent) {
    var event = "message", lines = [];
    block.split(/\r?\n/).forEach(function (line) {
      if (line.indexOf("event:") === 0) event = line.slice(6).trim();
      else if (line.indexOf("data:") === 0) lines.push(line.slice(5).replace(/^ /, ""));
    });
    if (!lines.length) return;  // comments / pings
    var payload;
    try { payload = JSON.parse(lines.join("\n")); } catch (e) { payload = { text: lines.join("\n") }; }
    onEvent(event, payload);
  }

  // ------------------------------------------------------------------ state
  var state = {
    open: false, busy: false, feature: cfg.feature, context: {},
    sessionId: sget("session", null) || uid(),
    history: sget("history", []),
    serverTitle: null, greeting: null
  };
  sset("session", state.sessionId);

  // ------------------------------------------------------------------ UI
  var host = el("div", { id: "aiready-assistant", "data-position": cfg.position });
  var root = host.attachShadow ? host.attachShadow({ mode: "open" }) : host;
  var style = el("style", { text: [
    ":host{all:initial}",
    ".wrap{--bg:#ffffff;--fg:#1d2330;--muted:#5d6678;--line:#e3e6ec;--accent:#2f5bea;--accent-fg:#fff;--bubble:#f3f5f9;",
    "--warn-bg:#fff6e0;--warn-fg:#6b4a00;--shadow:0 12px 40px rgba(15,23,42,.18);font:14px/1.45 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:var(--fg)}",
    ".wrap.dark{--bg:#161a22;--fg:#e8ebf2;--muted:#9aa3b5;--line:#2a303c;--accent:#7b9bff;--accent-fg:#0b1020;--bubble:#202633;--warn-bg:#3a2f12;--warn-fg:#ffd98a;--shadow:0 12px 40px rgba(0,0,0,.5)}",
    ".launcher{position:fixed;bottom:20px;width:56px;height:56px;border-radius:50%;border:0;cursor:pointer;background:var(--accent);",
    "color:var(--accent-fg);box-shadow:var(--shadow);display:flex;align-items:center;justify-content:center;z-index:2147483000}",
    ".launcher:focus-visible,button:focus-visible,textarea:focus-visible,a:focus-visible{outline:3px solid var(--accent);outline-offset:2px}",
    ".right .launcher,.right .panel{right:20px}.left .launcher,.left .panel{left:20px}",
    ".panel{position:fixed;bottom:88px;width:380px;max-width:calc(100vw - 40px);height:560px;max-height:calc(100vh - 120px);",
    "background:var(--bg);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);display:flex;flex-direction:column;",
    "overflow:hidden;z-index:2147483000}",
    ".panel[hidden]{display:none}",
    "header{display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid var(--line)}",
    "header h2{font-size:15px;margin:0;flex:1;font-weight:600}",
    ".icon-btn{background:none;border:0;color:var(--muted);cursor:pointer;padding:6px;border-radius:6px;font-size:16px;line-height:1}",
    ".icon-btn:hover{background:var(--bubble)}",
    ".log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}",
    ".msg{max-width:88%;padding:9px 12px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}",
    ".msg.user{align-self:flex-end;background:var(--accent);color:var(--accent-fg);border-bottom-right-radius:4px}",
    ".msg.bot{align-self:flex-start;background:var(--bubble);border-bottom-left-radius:4px}",
    ".msg.bot.streaming .text::after{content:'\\2589';animation:blink 1s steps(2) infinite;margin-left:2px;opacity:.6}",
    "@keyframes blink{50%{opacity:0}}",
    ".status{font-size:12px;color:var(--muted);margin-top:4px}",
    ".cites{margin:8px 0 0;padding:0;list-style:none;font-size:12px;border-top:1px solid var(--line);padding-top:6px}",
    ".cites li{margin:2px 0}.cites a{color:var(--accent)}",
    ".notice{background:var(--warn-bg);color:var(--warn-fg);border-radius:10px;padding:9px 12px;font-size:13px}",
    ".notice button,.fb button,.edit button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--bg);color:var(--fg);",
    "border-radius:6px;padding:3px 8px;cursor:pointer;margin:6px 6px 0 0}",
    ".fb{display:flex;gap:2px;align-items:center;margin-top:4px}",
    ".fb button[aria-pressed=true]{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}",
    ".edit textarea{width:100%;box-sizing:border-box;min-height:70px;margin-top:6px}",
    ".composer{display:flex;gap:8px;padding:10px;border-top:1px solid var(--line)}",
    ".composer textarea{flex:1;resize:none;border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;",
    "color:var(--fg);background:var(--bg);max-height:120px;min-height:38px}",
    ".send{background:var(--accent);color:var(--accent-fg);border:0;border-radius:10px;padding:0 14px;cursor:pointer;font:inherit;font-weight:600}",
    ".send:disabled{opacity:.5;cursor:default}",
    ".sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:480px){.panel{left:0!important;right:0!important;bottom:0;width:100vw;max-width:100vw;height:100%;max-height:100%;border-radius:0}",
    ".launcher{bottom:16px}}",
    "@media (prefers-reduced-motion:reduce){.msg.bot.streaming .text::after{animation:none}}"
  ].join("") });

  var wrap = el("div", { class: "wrap " + cfg.position });
  var launcher = el("button", { class: "launcher", type: "button", "aria-label": "Open assistant", "aria-expanded": "false",
    "aria-controls": "aiready-panel", onclick: function () { toggle(); } });
  launcher.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8a2.5 2.5 0 0 1-2.5 2.5H10l-4.2 3.6c-.5.4-1.3.1-1.3-.6V16A2.5 2.5 0 0 1 4 13.5v-8Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>';

  var titleEl = el("h2", { id: "aiready-title", text: cfg.title || "Assistant" });
  var log = el("div", { class: "log", role: "log", "aria-live": "polite", "aria-relevant": "additions", "aria-labelledby": "aiready-title" });
  var input = el("textarea", { rows: "1", "aria-label": "Message", placeholder: "Ask a question…", maxlength: "8000" });
  var sendBtn = el("button", { class: "send", type: "submit", text: "Send" });
  var form = el("form", { class: "composer", onsubmit: function (e) { e.preventDefault(); send(input.value); } }, [input, sendBtn]);
  var closeBtn = el("button", { class: "icon-btn", type: "button", "aria-label": "Close assistant", text: "✕", onclick: function () { close(); } });
  var clearBtn = el("button", { class: "icon-btn", type: "button", "aria-label": "Start a new conversation", title: "New conversation", text: "↺",
    onclick: function () { resetConversation(); } });
  var panel = el("div", { class: "panel", id: "aiready-panel", role: "dialog", "aria-modal": "false", "aria-labelledby": "aiready-title", hidden: true },
    [el("header", null, [titleEl, clearBtn, closeBtn]), log, form]);
  wrap.appendChild(launcher);
  wrap.appendChild(panel);
  root.appendChild(style);
  root.appendChild(wrap);

  function applyTheme(theme) {
    var dark = theme === "dark" || (theme === "auto" && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
    wrap.classList.toggle("dark", !!dark);
  }
  applyTheme(cfg.theme);
  if (cfg.theme === "auto" && window.matchMedia) {
    var mq = matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", function () { applyTheme("auto"); });
  }

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(input.value); }
  });
  input.addEventListener("input", function () { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });
  panel.addEventListener("keydown", function (e) { if (e.key === "Escape") { e.stopPropagation(); close(); } });

  function scrollDown() { log.scrollTop = log.scrollHeight; }
  function addUser(text) { log.appendChild(el("div", { class: "msg user", text: text })); scrollDown(); }
  function addNotice(text, actions) {
    var n = el("div", { class: "notice", role: "status" }, [el("div", { text: text })]);
    (actions || []).forEach(function (a) { n.appendChild(el("button", { type: "button", text: a.label, onclick: a.onclick })); });
    log.appendChild(n); scrollDown(); return n;
  }
  function addBot() {
    var text = el("span", { class: "text" });
    var status = el("div", { class: "status", "aria-live": "off" });
    var bubble = el("div", { class: "msg bot streaming" }, [text, status]);
    log.appendChild(bubble); scrollDown();
    return { bubble: bubble, text: text, status: status, answer: "" };
  }

  // ------------------------------------------------------------------ feedback
  function attachFeedback(view, runId, question) {
    var up = el("button", { type: "button", "aria-pressed": "false", "aria-label": "Helpful", text: "👍" });
    var down = el("button", { type: "button", "aria-pressed": "false", "aria-label": "Not helpful", text: "👎" });
    var edit = el("button", { type: "button", text: "Edit answer" });
    var bar = el("div", { class: "fb" }, [up, down, edit]);
    function rate(value, btn, other) {
      btn.setAttribute("aria-pressed", "true"); other.setAttribute("aria-pressed", "false");
      sendFeedback({ run_id: runId, rating: value, original_answer: view.answer, message: question });
    }
    up.addEventListener("click", function () { rate("up", up, down); });
    down.addEventListener("click", function () { rate("down", down, up); });
    edit.addEventListener("click", function () {
      if (view.bubble.querySelector(".edit")) return;
      var area = el("textarea", { "aria-label": "Corrected answer" });
      area.value = view.answer;
      var comment = el("textarea", { "aria-label": "What was wrong? (optional)", placeholder: "What was wrong? (optional)" });
      var box = el("div", { class: "edit" }, [area, comment]);
      box.appendChild(el("button", { type: "button", text: "Send correction", onclick: function () {
        sendFeedback({ run_id: runId, rating: "down", original_answer: view.answer, edited_answer: area.value,
          comment: comment.value || null, message: question });
        box.replaceWith(el("div", { class: "status", text: "Thanks - your correction was sent." }));
      } }));
      box.appendChild(el("button", { type: "button", text: "Cancel", onclick: function () { box.remove(); } }));
      view.bubble.appendChild(box); area.focus();
    });
    view.bubble.appendChild(bar);
  }
  function sendFeedback(body) {
    body.app = cfg.app; body.feature = state.feature;
    authedFetch("/v1/assistant/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .catch(function () { /* feedback is best effort */ });
  }

  // ------------------------------------------------------------------ rendering a run
  var STEP_LABEL = { retrieve: "Searching the knowledge base…", llm: "Thinking…", tool: "Working…",
    query: "Looking up data…", subagent: "Delegating…", approval: "Waiting for approval…" };

  function renderCitations(view, citations) {
    if (!citations || !citations.length) return;
    var list = el("ul", { class: "cites", "aria-label": "Sources" });
    citations.forEach(function (c) {
      var label = (c.marker ? c.marker + " " : "") + (c.title || c.doc_id || "Source");
      var href = c.source_uri && safeUrl(c.source_uri);
      list.appendChild(el("li", null, [href ? el("a", { href: href, target: "_blank", rel: "noopener noreferrer", text: label })
        : el("span", { text: label })]));
    });
    view.bubble.appendChild(list);
  }

  function handlers(view, question, onRun) {
    return function (event, d) {
      if (event === "run") { if (d.run_id) { view.runId = d.run_id; onRun && onRun(d.run_id); } }
      else if (event === "token") { view.answer += d.text || ""; view.text.textContent = view.answer; scrollDown(); }
      else if (event === "step") { if (d.status === "ok" || d.status === "waiting") view.status.textContent = STEP_LABEL[d.type] || ""; }
      else if (event === "approval") {
        view.bubble.classList.remove("streaming"); view.status.textContent = "";
        var names = (d.tools || []).map(function (t) { return t.name; }).filter(Boolean).join(", ");
        addNotice((d.message || "This action needs approval before it runs.") + (names ? " (" + names + ")" : "") +
          " An approver has been asked; you can check back here.", [{ label: "Check status", onclick: function () { checkRun(view.runId || d.run_id); } }]);
      }
      else if (event === "waiting") { addNotice("Your session expired. Sign in again, then press Continue.",
        [{ label: "Continue", onclick: function () { auth.token = null; resumeRun(d.run_id); } }]); }
      else if (event === "done") { finish(view, d, question); }
      else if (event === "error") { view.bubble.classList.remove("streaming"); view.text.textContent = d.message || "Something went wrong."; }
    };
  }

  function finish(view, d, question) {
    view.bubble.classList.remove("streaming");
    view.status.textContent = "";
    if (d.status && d.status !== "completed") {
      view.text.textContent = view.answer || (d.status === "rejected" ? "The request was rejected by an approver." :
        "I could not finish this (" + (d.stop_reason || d.error || d.status) + ").");
      return;
    }
    if (d.answer) { view.answer = d.answer; view.text.textContent = d.answer; }
    renderCitations(view, d.citations);
    attachFeedback(view, d.run_id, question);
    state.history.push({ role: "user", content: question }, { role: "assistant", content: view.answer });
    state.history = state.history.slice(-cfg.maxHistory);
    sset("history", state.history);
    scrollDown();
  }

  function stream(path, body, view, question) {
    setBusy(true);
    return authedFetch(path, { method: "POST", headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
      body: JSON.stringify(body || {}) })
      .then(function (r) {
        if (!r.ok) return r.json().catch(function () { return {}; }).then(function (b) {
          throw new Error((b.error && b.error.message) || ("Request failed (" + r.status + ")"));
        });
        return readSSE(r, handlers(view, question));
      })
      .catch(function (err) { view.bubble.classList.remove("streaming"); view.text.textContent = err.message || "Network error."; })
      .then(function () { setBusy(false); });
  }

  function send(text) {
    text = (text || "").trim();
    if (!text || state.busy) return;
    input.value = ""; input.style.height = "auto";
    addUser(text);
    var view = addBot();
    stream("/v1/assistant/chat", { message: text, history: state.history, app: cfg.app, feature: state.feature,
      session_id: state.sessionId, context: state.context }, view, text);
  }

  function checkRun(runId) {
    if (!runId) return;
    authedFetch("/v1/assistant/runs/" + encodeURIComponent(runId)).then(function (r) { return r.json(); }).then(function (run) {
      if (run.status === "waiting_approval") { addNotice("Still waiting for an approver."); return; }
      if (run.status === "running" || run.status === "queued") {
        var view = addBot();
        setBusy(true);
        authedFetch("/v1/assistant/runs/" + encodeURIComponent(runId) + "/events?after=" + (run.event_seq || 0),
          { headers: { "Accept": "text/event-stream" } })
          .then(function (resp) { return readSSE(resp, handlers(view, "")); })
          .then(function () { setBusy(false); });
        return;
      }
      var v = addBot();
      finish(v, { run_id: runId, status: run.status, answer: run.answer, citations: run.citations, error: run.error }, "");
    }).catch(function () { addNotice("Could not check the status right now."); });
  }

  function resumeRun(runId) { stream("/v1/assistant/runs/" + encodeURIComponent(runId) + "/resume", {}, addBot(), ""); }

  function setBusy(b) { state.busy = b; sendBtn.disabled = b; log.setAttribute("aria-busy", b ? "true" : "false"); }

  function resetConversation() {
    state.history = []; sset("history", []);
    state.sessionId = uid(); sset("session", state.sessionId);
    log.textContent = "";
    greet();
  }
  function greet() { if (state.greeting) log.appendChild(el("div", { class: "msg bot", text: state.greeting })); }

  // ------------------------------------------------------------------ open / close
  var loadedConfig = false;
  function loadConfig() {
    if (loadedConfig) return; loadedConfig = true;
    fetch(cfg.endpoint + "/v1/assistant/config?app=" + encodeURIComponent(cfg.app), { headers: { "X-App": cfg.app } })
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (c) {
        if (!cfg.title && c.title) titleEl.textContent = c.title;
        state.greeting = c.greeting || null;
        if (c.theme && c.theme.accent) wrap.style.setProperty("--accent", c.theme.accent);
        if (!log.childNodes.length) greet();
      }).catch(function () { /* defaults are fine */ });
  }
  function open() {
    if (state.open) return;
    state.open = true; panel.hidden = false; launcher.setAttribute("aria-expanded", "true");
    launcher.setAttribute("aria-label", "Close assistant");
    loadConfig(); setTimeout(function () { input.focus(); }, 0);
  }
  function close() {
    if (!state.open) return;
    state.open = false; panel.hidden = true; launcher.setAttribute("aria-expanded", "false");
    launcher.setAttribute("aria-label", "Open assistant"); launcher.focus();
  }
  function toggle() { state.open ? close() : open(); }

  function mount() { (document.body || document.documentElement).appendChild(host); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount); else mount();

  window.AIReadyAssistant = {
    __loaded: true,
    open: open, close: close, toggle: toggle,
    send: function (text) { open(); send(text); },
    setToken: function (token, expiresAt) { auth.token = token; auth.expiresAt = expiresAt || 0; },
    setTokenProvider: function (fn) { auth.provider = fn; auth.token = null; },
    setContext: function (ctx) { state.context = ctx || {}; },
    setFeature: function (name) { state.feature = name || null; }
  };
})();
