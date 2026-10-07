"""Markdown renderers for the assessment's sizing output: capacity worksheet, ingestion, load test, cost."""

from __future__ import annotations

from aiready.sizing.capacity import IngestionSizing, IngestionWorksheet, PoolSizing, PoolWorksheet, size_pool
from aiready.sizing.cost import CostWorksheet, evaluate
from aiready.sizing.loadtest import LoadTestResult


def _v(x: float | None, fmt: str = "{:,.2f}") -> str:
    return "[__]" if x is None else fmt.format(x)


def pool_worksheet(ws: PoolWorksheet, res: PoolSizing | None = None) -> str:
    """One pool, laid out as on the slide (unfilled values stay as [__])."""
    res = res or size_pool(ws)
    lines = [f"### Pool: {ws.pool}" + (f" ({ws.model})" if ws.model else ""), "",
             "| Worksheet for one pool | Value |", "|---|---|",
             f"| Peak users at once | {_v(ws.peak_users, '{:,.0f}')} |",
             f"| Requests per user per minute | {_v(ws.requests_per_user_per_min)} |",
             f"| Tokens per request (in + out) | {_v(ws.tokens_per_request, '{:,.0f}')} |",
             f"| Demand, tokens per second (product / 60) | {_v(res.demand_tokens_per_sec)} |",
             f"| Measured supply per node, tokens/s | {_v(ws.supply_tokens_per_sec_per_node)} |",
             f"| Nodes for load (demand / supply) | {_v(res.nodes_for_load)} -> {_v(res.load_nodes, '{}')} |",
             f"| Nodes to order (load + {ws.redundancy_nodes} + headroom {ws.headroom_pct:g}%) | "
             f"{_v(res.nodes_to_order, '{}')} |"]
    if res.status == "pending_load_test":
        lines += ["", "_No node count is quoted before the load test on the client's hardware._"]
    elif res.status == "incomplete":
        lines += ["", f"_To fill in during the assessment: {', '.join(res.missing)}._"]
    elif ws.supply_source:
        lines += ["", f"Supply source: {ws.supply_source}" + (f" (target {ws.target_latency})" if ws.target_latency else "")]
    return "\n".join(lines) + "\n"


def capacity_report(pools: list[PoolWorksheet], ingestion: tuple[IngestionWorksheet, IngestionSizing] | None = None) -> str:
    sized = [(p, size_pool(p)) for p in pools]
    total = sum(r.nodes_to_order or 0 for _, r in sized)
    pending = [p.pool for p, r in sized if r.nodes_to_order is None]
    out = ["# Capacity and sizing", "",
           "Method: demand = peak users x requests/user/min x tokens/request / 60; nodes = demand / measured supply, "
           "rounded up, + 1 for failure + headroom. Supply is taken from a load test on the client's hardware.", "",
           "| Pool | Demand tok/s | Supply tok/s/node | Load nodes | +1 | Headroom | Nodes to order |",
           "|---|---|---|---|---|---|---|"]
    for p, r in sized:
        out.append(f"| {p.pool} | {_v(r.demand_tokens_per_sec)} | {_v(p.supply_tokens_per_sec_per_node)} | "
                   f"{_v(r.load_nodes, '{}')} | {r.redundancy_nodes} | {_v(r.headroom_nodes, '{}')} | {_v(r.nodes_to_order, '{}')} |")
    out += ["", f"**GPU nodes to order: {total}**" + (f" (pending load test: {', '.join(pending)})" if pending else ""), ""]
    out += [pool_worksheet(p, r) for p, r in sized]
    if ingestion:
        ws, ing = ingestion
        out += ["## Ingestion", "",
                "| Item | Value |", "|---|---|",
                f"| Backlog | {ws.backlog_documents:,.0f} documents, {ing.backlog_tokens:,.0f} tokens |",
                f"| Backfill over {ws.backfill_days:g} nights of {ws.nightly_window_hours:g} h | {ing.backfill_tokens_per_sec:,.1f} tokens/s |",
                f"| Nightly changes | {ws.daily_changed_documents:,.0f} documents, {ing.nightly_tokens_per_sec:,.1f} tokens/s |",
                f"| Embedding nodes (backfill / nightly) | {_v(ing.embed_nodes_backfill, '{}')} / {_v(ing.embed_nodes_nightly, '{}')} |",
                f"| Parser workers (backfill / nightly) | {_v(ing.parse_workers_backfill, '{}')} / {_v(ing.parse_workers_nightly, '{}')} |", ""]
    return "\n".join(out)


def loadtest_report(results: list[LoadTestResult], best: LoadTestResult | None = None) -> str:
    if not results:
        return "No load test results.\n"
    r0 = results[0]
    out = [f"# Load test: {r0.model} ({r0.mode})", "", f"Endpoint {r0.base_url} · hardware: {r0.hardware or '[not labelled]'}", "",
           "| Concurrency | Requests | Errors | Req/s | Out tok/s | Total tok/s | p50 ms | p95 ms | TTFT p50 | TTFT p95 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        out.append(f"| {r.concurrency} | {r.requests} | {r.errors} | {r.requests_per_sec:.2f} | {r.output_tokens_per_sec:,.1f} | "
                   f"{r.total_tokens_per_sec:,.1f} | {r.latency_p50_ms:,.0f} | {r.latency_p95_ms:,.0f} | "
                   f"{_v(r.ttft_p50_ms, '{:,.0f}')} | {_v(r.ttft_p95_ms, '{:,.0f}')} |")
    if best:
        out += ["", f"**Measured supply per node: {best.total_tokens_per_sec:,.1f} tokens/s** "
                    f"(concurrency {best.concurrency}, p95 {best.latency_p95_ms:,.0f} ms) - {best.source}"]
    return "\n".join(out) + "\n"


def cost_report(ws: CostWorksheet, *, chart: bool = True) -> str:
    res = evaluate(ws)
    c = ws.currency
    out = ["# Cost model", "", "| Worksheet | Value |", "|---|---|",
           f"| GPU and platform cost per month | {ws.gpu_platform_monthly:,.2f} {c} |",
           f"| Run and support cost per month | {ws.run_support_monthly:,.2f} {c} |",
           f"| API price for a typical request | {res.api_price_per_request:,.5f} {c} |",
           f"| Break-even requests per month (fixed cost / API price) | {res.break_even_requests_per_month:,.0f} |",
           f"| Client's forecast requests per month | {_v(ws.forecast_requests_per_month, '{:,.0f}')} |", ""]
    if res.at_forecast:
        f = res.at_forecast
        out += [f"At the forecast, **{f.cheaper}** is cheaper by {abs(f.difference):,.2f} {c} per month.", ""]
    out += ["| Requests / month | Vendor API | Self-hosted | Added nodes | Cheaper |", "|---|---|---|---|---|"]
    for r in res.table:
        out.append(f"| {r.requests_per_month:,.0f} | {r.api_cost:,.2f} | {r.self_hosted_cost:,.2f} | {r.added_nodes} | {r.cheaper} |")
    out += ["", "_Cost is one input: data control, latency and model quality can settle the choice first._"]
    if chart:
        from aiready.sizing.cost import break_even_svg

        out += ["", break_even_svg(ws)]
    return "\n".join(out) + "\n"
