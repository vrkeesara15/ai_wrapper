# Brief for build agents (shared by every work package)

Repository: `/Users/vamshikeesara/Projects/ai_ready` - a Python 3.12 monorepo, package `aiready` in `src/aiready`.
We are implementing the reference architecture in `docs/reference-architecture.txt` (text of `AI-Ready.pdf`)
as a plug-and-play platform deployed *beside* an existing application without modifying it.

Before writing code, read in full:
1. `docs/CONVENTIONS.md` - package shape, identity, governance hooks, config, ports. Mandatory.
2. `src/aiready/core/` - especially `interfaces.py` (service contracts), `types.py`, `identity.py`,
   `plugins.py`, `config.py`, `wiring.py`, `audit.py`, `telemetry.py`, `stores.py`, `queues.py`, `app.py`, `http.py`.
3. `src/aiready/testing.py` - fakes (`FakeGateway`, `AllowAllPolicy`, `PassThroughGuardrails`, `make_principal`)
   you use with `aiready.core.wiring.override(...)` to test your package without the others.
4. `src/aiready/cache/semantic.py` - shared exact + semantic cache.
5. The slides in `docs/reference-architecture.txt` relevant to your package.

Rules:
* Only create/modify files inside the paths your work package owns. Other agents are writing the other packages
  at the same time. Do NOT edit `src/aiready/core/*`, `src/aiready/testing.py`, `src/aiready/cache/*`,
  `tests/conftest.py`, `pyproject.toml`. If you need a change there, describe it in your final report instead
  (and work around it locally).
* Use the venv: `.venv/bin/python -m pytest -q tests/<your pkg>`. Do not install packages. Installed:
  fastapi, uvicorn, httpx, pydantic v2, pyyaml, numpy, sqlalchemy 2, sqlglot, jsonschema, pyjwt, prometheus_client,
  sse-starlette, typer, rich, mcp (official MCP python SDK), pypdf, beautifulsoup4, python-docx.
  Optional heavy dependencies (redis, qdrant-client, psycopg, opentelemetry, sentence-transformers, torch, peft,
  kafka...) must be imported lazily inside the function/class that needs them, so the package imports without them.
* Production quality: typed, docstrings that explain *why*, clear module boundaries, no placeholder `pass`
  bodies or TODO stubs for required behaviour. Every swappable component is registered with
  `aiready.core.plugins.register(kind, name)` and constructed from config with `plugins.create(kind, spec)` so a
  client customises it with `class: their_pkg.module:Class` - no fork needed.
* Defaults must run locally with no external infrastructure (in-memory/sqlite/mock).
* Governance is non-negotiable: policy check, audit record, trace span, metrics at every decision point
  (see CONVENTIONS section 3).
* Write meaningful tests (happy path + governance/failure paths). All tests must pass before you finish.
* Config: put your package's default configuration in the config files you own under `config/`, fully commented
  so a client can customise it. Config is deep-merged from all YAML files; use a top-level key named after your
  section.
* Final report (your last message): list files created, public entry points (classes, `create_app` factories,
  CLI-relevant functions with signatures), config sections + keys, anything you needed from core but could not
  change, and test results. Keep it under 600 words.
