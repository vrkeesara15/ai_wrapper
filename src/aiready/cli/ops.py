"""Operations commands: serve, config, ingest, sizing, cost, bundle."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from aiready.cli.common import console, emit, fail, ok, run, table

config_app = typer.Typer(help="Validate and inspect the merged configuration.", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingestion runs, dead letters and verification.", no_args_is_help=True)
dlq_app = typer.Typer(help="Dead-letter queue.", no_args_is_help=True)
ingest_app.add_typer(dlq_app, name="dlq")
sizing_app = typer.Typer(help="Capacity sizing worksheet and load tests.", no_args_is_help=True)
cost_app = typer.Typer(help="Self-hosted vs API cost model.", no_args_is_help=True)
bundle_app = typer.Typer(help="Air-gapped release bundles (build, verify, import).", no_args_is_help=True)

# service -> (uvicorn factory, default port)
SERVICES: dict[str, tuple[str, int]] = {
    "gateway": ("aiready.gateway.api:create_app", 8080),
    "gateway-admin": ("aiready.gateway.admin:create_admin_app", 8081),
    "guardrails": ("aiready.guardrails.api:create_app", 8090),
    "governance": ("aiready.governance.api:create_app", 8091),
    "embedding": ("aiready.embedding.api:create_app", 8100),
    "rerank": ("aiready.rerank.api:create_app", 8101),
    "search": ("aiready.search.api:create_app", 8102),
    "rag": ("aiready.rag.api:create_app", 8110),
    "memory": ("aiready.memory.api:create_app", 8111),
    "query": ("aiready.query.api:create_app", 8112),
    "agents": ("aiready.agents.api:create_app", 8120),
    "assistant": ("aiready.assistant.api:create_app", 8130),
    "mcp": ("aiready.mcp.api:create_app", 8140),
    "identity-bridge": ("aiready.integration.bridge_api:create_app", 8150),
    "facade": ("aiready.integration.facade_api:create_app", 8151),
    "ingestion": ("aiready.ingestion.api:create_app", 8160),
    "observability": ("aiready.observability.api:create_app", 8170),
    "mock-models": ("aiready.models.mock_server:create_app", 8200),
}
PILOT_SET = [s for s in SERVICES if s not in {"mock-models"}]


# =========================================================================== serve



def serve(services: list[str] = typer.Argument(None, help=f"Service names, or 'all'. One of: {', '.join(SERVICES)}"),
          host: str = "0.0.0.0", port: Optional[int] = typer.Option(None, help="Port (single service only)."),
          mock_models: bool = typer.Option(False, help="Also start the mock model server (offline demo)."),
          log_level: str = "info"):
    """Start services. ``aiready serve all`` runs every service in one process on its default port: the
    stage-1 pilot footprint. In-process wiring is used between them unless services.<name>.url is set."""
    import uvicorn

    from aiready.core.plugins import import_object

    names = list(services or ["all"])
    if names == ["all"]:
        names = list(PILOT_SET)
    if mock_models and "mock-models" not in names:
        names.append("mock-models")
    unknown = [n for n in names if n not in SERVICES]
    if unknown:
        fail(f"unknown service(s) {unknown}; choose from {list(SERVICES)}")
    if port and len(names) > 1:
        fail("--port applies to a single service")
    os.environ.setdefault("AIREADY_SERVICE", names[0] if len(names) == 1 else "aiready-pilot")
    # long-running servers hot-reload configuration (routes, budgets, policies) unless explicitly disabled
    os.environ["AIREADY_CONFIG_WATCH"] = os.environ.get("AIREADY_SERVE_CONFIG_WATCH", "1")

    async def main() -> None:
        servers = []
        for n in names:
            factory, default_port = SERVICES[n]
            app = import_object(factory)()
            cfg = uvicorn.Config(app, host=host, port=port or default_port, log_level=log_level, lifespan="on")
            servers.append(uvicorn.Server(cfg))
            console.print(f"[cyan]{n}[/cyan] on http://{host}:{port or default_port}")
        await asyncio.gather(*(s.serve() for s in servers))

    asyncio.run(main())


# =========================================================================== config


def _known_sections() -> dict[str, str]:
    """Config section -> pydantic model, for validation."""
    return {
        "gateway": "aiready.gateway.config:GatewayConfig",
        "rag": "aiready.rag.config:RagConfig",
        "memory": "aiready.memory.config:MemoryConfig",
        "query": "aiready.query.config:QueryConfig",
        "embedding": "aiready.embedding.config:EmbeddingConfig",
        "rerank": "aiready.rerank.config:RerankConfig",
        "search": "aiready.search.config:SearchConfig",
        "ingestion": "aiready.ingestion.config:IngestionConfig",
        "integration": "aiready.integration.config:IntegrationConfig",
        "observability": "aiready.observability.config:ObservabilityConfig",
        "finetune": "aiready.finetune.config:FinetuneConfig",
        "guardrails": "aiready.guardrails.config:GuardrailsConfig",
        "governance": "aiready.governance.config:GovernanceConfig",
    }


@config_app.command("validate")
def config_validate(config_dir: Path = typer.Option(Path(os.environ.get("AIREADY_CONFIG_DIR", "config"))),
                    overlay: Optional[Path] = None):
    """Load and merge every YAML file, then validate each section against its schema and compile policies."""
    from aiready.core.config import ConfigStore, set_config
    from aiready.core.plugins import import_object

    store = ConfigStore(config_dir, overlay)
    set_config(store)
    problems = 0
    rows = []
    for section, path in _known_sections().items():
        try:
            model = import_object(path)
        except Exception as exc:
            rows.append({"section": section, "result": f"schema not importable: {exc}"})
            continue
        try:
            store.section(section, model)
            rows.append({"section": section, "result": "ok"})
        except Exception as exc:
            problems += 1
            rows.append({"section": section, "result": f"INVALID: {str(exc)[:200]}"})
    try:
        from aiready.governance.policy import engine_from_tree

        engine_from_tree(store.tree)
        rows.append({"section": "policies", "result": "ok (compiled)"})
    except Exception as exc:
        problems += 1
        rows.append({"section": "policies", "result": f"INVALID: {exc}"})
    table(rows, ["section", "result"], f"config {store.version[:12]} from {config_dir}")
    raise typer.Exit(1 if problems else 0)


_SECRETISH = re.compile(r"(secret|password|token|api_key|apikey|private_key|hmac)", re.I)


def _redact(node: Any, key: str = "") -> Any:
    if isinstance(node, dict):
        return {k: _redact(v, str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [_redact(v, key) for v in node]
    if _SECRETISH.search(key) and isinstance(node, str) and node:
        return "***"
    return node


@config_app.command("show")
def config_show(section: Optional[str] = typer.Argument(None), config_dir: Optional[Path] = None):
    """Print the merged configuration (secrets redacted)."""
    from aiready.core.config import ConfigStore, get_config

    store = ConfigStore(config_dir) if config_dir else get_config()
    data = store.get(section) if section else store.tree
    typer.echo(yaml.safe_dump(_redact(data), sort_keys=False))


# =========================================================================== ingest


def _pipeline() -> Any:
    from aiready.ingestion.pipeline import IngestionPipeline

    return IngestionPipeline()


async def _ingestion_http(method: str, path: str, url: str, **kw: Any) -> Any:
    from aiready.cli.common import operator
    from aiready.core.http import ServiceClient

    client = ServiceClient(url)
    try:
        return await client.json(method, path, principal=operator(None, ["ingestion-admin"]), **kw)
    finally:
        await client.aclose()


@ingest_app.command("run")
def ingest_run(source: Optional[str] = None, all_sources: bool = typer.Option(False, "--all-sources"),
               incremental: bool = typer.Option(True, "--incremental/--full",
                                                help="--full re-reads every item (ignores checksums)."),
               url: Optional[str] = typer.Option(None, envvar="AIREADY_INGESTION_URL",
                                                 help="Ingestion API; omit to run in this process.")):
    """Extract -> transform -> load one or every source, then verify."""
    if not source and not all_sources:
        fail("give --source NAME or --all-sources")
    if url:
        if all_sources:
            sources = run(_ingestion_http("GET", "/v1/sources", url))
            names = [s["name"] for s in (sources.get("sources", sources) if isinstance(sources, dict) else sources)]
        else:
            names = [source]
        emit([run(_ingestion_http("POST", f"/v1/sources/{n}/runs", url, json={"full": not incremental})) for n in names])
        return
    pipe = _pipeline()
    names = [s.name for s in pipe.cfg.sources if getattr(s, "enabled", True)] if all_sources else [source]
    out = []
    for n in names:
        if not incremental:
            pipe.registry.forget_source(n) if hasattr(pipe.registry, "forget_source") else None
        out.append(run(pipe.run_source(n, trigger="manual")).model_dump())
    emit(out)


@ingest_app.command("runs")
def ingest_runs(source: Optional[str] = None, last: int = typer.Option(10, "--last")):
    rows = [r.model_dump() for r in _pipeline().registry.runs(source=source, limit=last)]
    for r in rows:
        r["counts"] = json.dumps(r["counts"])
    table(rows, ["run_id", "source", "trigger", "status", "counts"], "ingestion runs")


@ingest_app.command("verify")
def ingest_verify(source: Optional[str] = None, collection: Optional[str] = None):
    """Run the verification questions of a source (or every source loading into a collection)."""
    pipe = _pipeline()
    srcs = [s for s in pipe.cfg.sources if (source and s.name == source) or (collection and s.collection == collection)]
    if not srcs:
        fail("no matching source")
    results = []
    for s in srcs:
        results += [r.model_dump() for r in run(pipe.verifier().verify(s.name, s.collection, s.verify))]
    emit(results)
    if any(not r.get("passed") for r in results):
        raise typer.Exit(1)


@dlq_app.command("list")
def dlq_list(topic: Optional[str] = None, url: Optional[str] = typer.Option(None, envvar="AIREADY_INGESTION_URL")):
    if url:
        emit(run(_ingestion_http("GET", "/v1/dlq", url)))
        return
    pipe = _pipeline()
    if topic:
        pipe.topic = topic
    emit(run(pipe.dead_letters()))


@dlq_app.command("replay")
def dlq_replay(topic: Optional[str] = None, url: Optional[str] = typer.Option(None, envvar="AIREADY_INGESTION_URL")):
    if url:
        emit(run(_ingestion_http("POST", "/v1/dlq/replay", url)))
        return
    pipe = _pipeline()
    if topic:
        pipe.topic = topic
    emit(run(pipe.replay_dead_letters()))


# =========================================================================== sizing / cost


@sizing_app.command("worksheet")
def sizing_worksheet(file: Optional[Path] = typer.Argument(None, help="YAML: pools: [...], ingestion: {...}"),
                     out: Optional[Path] = None):
    """Fill in the capacity worksheet per pool (demand / measured supply, +1 node, + headroom)."""
    from aiready.sizing import report
    from aiready.sizing.capacity import IngestionWorksheet, PoolWorksheet, default_pools, size_ingestion

    data = yaml.safe_load(file.read_text()) if file else {}
    pools = [PoolWorksheet.model_validate(p) for p in data.get("pools", [])] or default_pools()
    ing = None
    if data.get("ingestion"):
        iw = IngestionWorksheet.model_validate(data["ingestion"])
        ing = (iw, size_ingestion(iw))
    md = report.capacity_report(pools, ing)
    if out:
        out.write_text(md)
        ok(f"worksheet -> {out}")
    else:
        typer.echo(md)


@sizing_app.command("loadtest")
def sizing_loadtest(base_url: str, model: str, concurrency: list[int] = typer.Option([1, 4, 16]),
                    requests: int = 50, api_key: Optional[str] = typer.Option(None, envvar="AIREADY_LOADTEST_KEY"),
                    max_p95_ms: Optional[float] = None):
    """Measure tokens/sec per node on the client's hardware (the supply figure for the worksheet)."""
    from aiready.sizing import loadtest, report

    results = []
    for c in concurrency:
        fields = loadtest.LoadTestConfig.model_fields
        cfg_data: dict[str, Any] = {"base_url": base_url, "model": model}
        for k, v in (("concurrency", c), ("requests", requests), ("total_requests", requests), ("api_key", api_key)):
            if k in fields and v is not None:
                cfg_data[k] = v
        results.append(run(loadtest.run_load_test(loadtest.LoadTestConfig(**cfg_data))))
    best = loadtest.supply_at_target(results, max_latency_p95_ms=max_p95_ms)
    typer.echo(report.loadtest_report(results, best))


@cost_app.command("breakeven")
def cost_breakeven(file: Optional[Path] = typer.Argument(None, help="YAML CostWorksheet"),
                   gpu_platform_monthly: Optional[float] = None, run_support_monthly: Optional[float] = None,
                   api_price_per_request: Optional[float] = None, forecast: Optional[float] = None,
                   currency: str = "USD", svg: Optional[Path] = None):
    """Break-even requests per month = fixed self-hosted cost / API price per request."""
    from aiready.sizing import report
    from aiready.sizing.cost import CostWorksheet, break_even_svg

    data: dict[str, Any] = yaml.safe_load(file.read_text()) if file else {}
    for k, v in (("gpu_platform_monthly", gpu_platform_monthly), ("run_support_monthly", run_support_monthly),
                 ("api_price_per_request", api_price_per_request), ("forecast_requests_per_month", forecast)):
        if v is not None:
            data[k] = v
    data.setdefault("currency", currency)
    try:
        ws = CostWorksheet.model_validate(data)
    except Exception as exc:
        fail(str(exc))
        raise
    typer.echo(report.cost_report(ws, chart=False))
    if svg:
        svg.write_text(break_even_svg(ws))
        ok(f"chart -> {svg}")


# =========================================================================== bundle


def _script(name: str) -> Path:
    for base in (Path.cwd(), Path(__file__).resolve().parents[3]):
        p = base / "deploy" / "onprem" / name
        if p.exists():
            return p
    fail(f"deploy/onprem/{name} not found (run from the repository or release bundle root)")
    raise AssertionError


def _bundle(script: str, args: list[str]) -> None:
    raise typer.Exit(subprocess.call(["bash", str(_script(script)), *args]))


@bundle_app.command("build", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def bundle_build(ctx: typer.Context):
    """Build a signed release bundle (images, chart, model weights manifest, SBOM, checksums)."""
    _bundle("build-bundle.sh", ctx.args)


@bundle_app.command("verify", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def bundle_verify(ctx: typer.Context):
    """Verify signatures and checksums before anything is imported."""
    _bundle("verify-bundle.sh", ctx.args)


@bundle_app.command("import", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def bundle_import(ctx: typer.Context):
    """Import a verified bundle into the internal registry and model store."""
    _bundle("import-bundle.sh", ctx.args)
