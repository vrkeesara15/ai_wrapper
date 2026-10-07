/*!
 * AI-Ready WebMCP add-on - progressive enhancement, zero dependencies.
 *
 * If the browser exposes WebMCP (`navigator.modelContext`, Chrome origin trial / W3C Web Machine Learning
 * community group draft - NOT a standard yet), register the application's tools so a browser agent can call
 * them directly instead of scraping the screen. If it is absent, this script does nothing at all: the page
 * behaves exactly the same.
 *
 *   <script src="https://ai.example.com/widget/webmcp-tools.js"
 *           data-endpoint="https://ai.example.com/mcp-api"     (MCP server REST base: /v1/tools, /v1/tools/{name}:call)
 *           data-token-endpoint="https://ai.example.com/v1/assistant/session"
 *           data-app="claims"
 *           data-manifest="inline"                              (optional: read <script type="application/json"
 *                                                                id="aiready-webmcp-manifest"> instead of fetching)
 *           defer></script>
 *
 * Each tool's `execute` calls the tool REST endpoint with the signed-in user's own token, so the browser agent
 * gets exactly the user's permissions - and writes, payments and deletions still wait for an approver.
 */
(function () {
  "use strict";
  var mc = typeof navigator !== "undefined" ? navigator.modelContext : undefined;
  if (!mc || (typeof mc.registerTool !== "function" && typeof mc.provideContext !== "function")) return;  // no WebMCP

  var script = document.currentScript;
  var data = (script && script.dataset) || {};
  var origin = script && script.src ? new URL(script.src, location.href).origin : location.origin;
  var endpoint = (data.endpoint || origin).replace(/\/+$/, "");
  var tokenEndpoint = data.tokenEndpoint || origin + "/v1/assistant/session";
  var app = data.app || "default";
  var token = null, tokenExp = 0;

  function getToken(force) {
    var now = Date.now() / 1000;
    if (!force && token && (!tokenExp || tokenExp - 30 > now)) return Promise.resolve(token);
    if (window.AIReadyAuth && typeof window.AIReadyAuth.getToken === "function") {
      return Promise.resolve(window.AIReadyAuth.getToken()).then(function (t) { token = t; tokenExp = 0; return t; });
    }
    return fetch(tokenEndpoint, { method: "POST", credentials: "include", headers: { "X-App": app } })
      .then(function (r) { if (!r.ok) throw new Error("not signed in (" + r.status + ")"); return r.json(); })
      .then(function (b) { token = b.token; tokenExp = b.expires_at || 0; return token; });
  }

  function call(name, args, retried) {
    return getToken(false).then(function (t) {
      return fetch(endpoint + "/v1/tools/" + encodeURIComponent(name) + ":call", {
        method: "POST", credentials: "omit",
        headers: { "Authorization": "Bearer " + t, "Content-Type": "application/json", "X-App": app },
        body: JSON.stringify({ arguments: args || {}, context: { channel: "webmcp", page: location.pathname } })
      });
    }).then(function (r) {
      if (r.status === 401 && !retried) { token = null; return call(name, args, true); }
      return r.json().then(function (body) { return { status: r.status, body: body }; });
    });
  }

  function asContent(res) {
    var b = res.body || {};
    if (b.ok) {
      var text = typeof b.content === "string" ? b.content : JSON.stringify(b.content);
      return { content: [{ type: "text", text: text }] };
    }
    if (b.approval_id) {
      return { content: [{ type: "text", text: "Not executed yet: this action waits for approval (request " +
        b.approval_id + "). The user will be told when an approver decides." }] };
    }
    var msg = (b.error && (b.error.message || b.error)) || b.error || ("tool call failed (" + res.status + ")");
    return { content: [{ type: "text", text: String(msg) }], isError: true };
  }

  function toWebMCP(t) {
    var readOnly = !!(t.annotations && t.annotations.readOnlyHint);
    return {
      name: t.name,
      description: t.description + (readOnly ? "" : " (Changes data: waits for a person to approve it.)"),
      inputSchema: t.inputSchema || { type: "object", properties: {} },
      annotations: { readOnlyHint: readOnly },
      execute: function (input, client) {
        var run = function () { return call(t.name, input).then(asContent); };
        // let the user see a write being requested from their own page before it is even submitted
        if (!readOnly && client && typeof client.requestUserInteraction === "function") {
          return client.requestUserInteraction(function () {
            return Promise.resolve(window.confirm("Allow the assistant to request: " + t.name + "?"));
          }).then(function (ok) {
            return ok ? run() : { content: [{ type: "text", text: "The user declined." }], isError: true };
          });
        }
        return run().catch(function (e) { return { content: [{ type: "text", text: String(e.message || e) }], isError: true }; });
      }
    };
  }

  function loadManifest() {
    if (data.manifest === "inline") {
      var node = document.getElementById("aiready-webmcp-manifest");
      return Promise.resolve(node ? JSON.parse(node.textContent || "{}") : { tools: [] });
    }
    var url = data.manifest || endpoint + "/v1/tools?format=webmcp";
    return getToken(false).then(function (t) {
      return fetch(url, { headers: { "Authorization": "Bearer " + t, "X-App": app }, credentials: "omit" });
    }).then(function (r) { if (!r.ok) throw new Error("manifest " + r.status); return r.json(); });
  }

  var registered = [];
  function register(manifest) {
    var tools = (manifest.tools || []).map(toWebMCP);
    if (typeof mc.registerTool === "function") {
      tools.forEach(function (tool) {
        try { var reg = mc.registerTool(tool); registered.push(reg || tool.name); }
        catch (e) { if (window.console) console.warn("[aiready-webmcp] could not register", tool.name, e); }
      });
    } else {
      mc.provideContext({ tools: tools });  // earlier drafts: replace the whole tool set at once
      registered = tools.map(function (t) { return t.name; });
    }
  }

  window.AIReadyWebMCP = {
    reload: function () {
      registered.forEach(function (r) {
        try {
          if (r && typeof r.unregister === "function") r.unregister();
          else if (typeof mc.unregisterTool === "function") mc.unregisterTool(r);
        } catch (e) { /* already gone */ }
      });
      registered = [];
      return loadManifest().then(register);
    },
    tools: function () { return registered.slice(); }
  };

  loadManifest().then(register).catch(function (e) {
    // never break the page: without tools the page simply behaves as before
    if (window.console) console.info("[aiready-webmcp] tools not registered:", e.message || e);
  });
})();
