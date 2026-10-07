import json
import math

import httpx
import pytest

from aiready.sizing.capacity import (
    IngestionWorksheet,
    PoolWorksheet,
    default_pools,
    demand_tokens_per_sec,
    nodes_to_order,
    size_ingestion,
    size_pool,
)
from aiready.sizing.cost import CostWorksheet, break_even_requests, break_even_svg, evaluate, self_hosted_cost
from aiready.sizing.loadtest import LoadTestConfig, run_load_test, supply_at_target, sweep
from aiready.sizing.report import capacity_report, cost_report, loadtest_report

# ------------------------------------------------------------------ capacity (slide formula)

def test_demand_and_nodes_follow_the_slide_formula():
    # 400 users x 2 req/min x 1500 tokens / 60 = 20,000 tokens/s
    assert demand_tokens_per_sec(400, 2, 1500) == 20000
    raw, load, head, total = nodes_to_order(20000, 2500, redundancy=1, headroom_pct=20)
    assert (raw, load, head, total) == (8.0, 8, 2, 11)  # 8 + 1 + ceil(1.6)
    raw, load, head, total = nodes_to_order(20000, 3000, headroom_pct=0)
    assert load == 7 and total == 8  # ceil(6.67) + 1


@pytest.mark.parametrize("users,rpm,tok,supply,head,expected", [
    (50, 1, 1000, 2000, 25, 1 + 1 + 1),
    (1000, 3, 2000, 4000, 30, 25 + 1 + 8),
    (10, 0.5, 800, 10000, 20, 1 + 1 + 1),
])
def test_size_pool_matches_hand_calculation(users, rpm, tok, supply, head, expected):
    res = size_pool(PoolWorksheet(pool="chat", peak_users=users, requests_per_user_per_min=rpm, tokens_per_request=tok,
                                  supply_tokens_per_sec_per_node=supply, supply_source="load test 2026-10-01", headroom_pct=head))
    demand = users * rpm * tok / 60
    load = math.ceil(demand / supply)
    assert res.status == "sized" and res.demand_tokens_per_sec == pytest.approx(demand, abs=1e-3)
    assert res.load_nodes == load and res.nodes_to_order == load + 1 + math.ceil(load * head / 100) == expected
    assert 0 < res.utilisation_at_peak < 1


def test_tokens_in_and_out_are_summed():
    ws = PoolWorksheet(pool="chat", peak_users=60, requests_per_user_per_min=1, tokens_in=1200, tokens_out=300)
    assert ws.tokens_per_request == 1500


def test_no_node_count_before_load_test_and_no_brochures():
    pending = size_pool(PoolWorksheet(pool="chat", peak_users=100, requests_per_user_per_min=1, tokens_per_request=1000))
    assert pending.status == "pending_load_test" and pending.nodes_to_order is None
    assert pending.demand_tokens_per_sec == pytest.approx(1666.667, abs=1e-3)
    incomplete = size_pool(default_pools()[0])
    assert incomplete.status == "incomplete" and "peak_users" in incomplete.missing
    with pytest.raises(ValueError, match="brochure"):
        PoolWorksheet(pool="chat", supply_tokens_per_sec_per_node=5000, supply_source="vendor datasheet")
    report = capacity_report(default_pools())
    assert "[__]" in report and "Nodes to order" in report


def test_ingestion_sizing_by_backlog_and_window():
    ws = IngestionWorksheet(backlog_documents=200_000, avg_chunks_per_document=10, tokens_per_chunk=400,
                            daily_changed_documents=2_000, nightly_window_hours=6, backfill_days=5,
                            embed_supply_tokens_per_sec_per_node=20_000, parse_docs_per_sec_per_worker=2,
                            supply_source="load test 2026-09-30", headroom_pct=0)
    res = size_ingestion(ws)
    assert res.backlog_tokens == 800_000_000
    assert res.backfill_tokens_per_sec == pytest.approx(800e6 / (6 * 3600 * 5), abs=1e-3)  # 7407 tok/s
    assert res.embed_nodes_backfill == 1 and res.embed_nodes_nightly == 1
    assert res.parse_workers_backfill == math.ceil(200_000 / (6 * 3600 * 5) / 2)
    assert res.status == "sized"
    assert "Ingestion" in capacity_report([], (ws, res))


# ------------------------------------------------------------------ cost model

def test_break_even_is_fixed_cost_over_api_price():
    ws = CostWorksheet(gpu_platform_monthly=12_000, run_support_monthly=8_000, api_price_per_request=0.01,
                       forecast_requests_per_month=3_000_000)
    assert break_even_requests(ws) == 2_000_000
    res = evaluate(ws)
    assert res.break_even_requests_per_month == 2_000_000
    assert res.at_forecast.cheaper == "self-hosted" and res.at_forecast.difference == 10_000
    be_row = next(r for r in res.table if r.requests_per_month == 2_000_000)
    assert be_row.cheaper == "equal"
    assert next(r for r in res.table if r.requests_per_month == 1_000_000).cheaper == "api"


def test_api_price_from_tokens_and_node_steps():
    ws = CostWorksheet(gpu_platform_monthly=10_000, run_support_monthly=0, tokens_in_per_request=2000,
                       tokens_out_per_request=500, api_price_in_per_1k=0.003, api_price_out_per_1k=0.015,
                       capacity_requests_per_month=1_000_000, node_monthly_cost=4_000, capacity_per_added_node=500_000)
    assert ws.api_price_per_request == pytest.approx(0.0135)
    assert self_hosted_cost(ws, 900_000) == 10_000
    assert self_hosted_cost(ws, 1_200_000) == 14_000  # a node is added
    assert self_hosted_cost(ws, 1_600_000) == 18_000
    with pytest.raises(ValueError):
        CostWorksheet(gpu_platform_monthly=1, run_support_monthly=1)


def test_cost_chart_and_report():
    ws = CostWorksheet(gpu_platform_monthly=12_000, run_support_monthly=8_000, api_price_per_request=0.01,
                       forecast_requests_per_month=3_000_000)
    svg = break_even_svg(ws)
    assert svg.startswith("<svg") and "break-even 2,000,000" in svg and "Self-hosted" in svg
    md = cost_report(ws)
    assert "fixed cost / API price" in md and "2,000,000" in md and "self-hosted" in md


# ------------------------------------------------------------------ load test

def sse_transport(tokens_per_reply=20, fail_every=0):
    count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        if fail_every and count["n"] % fail_every == 0:
            return httpx.Response(503, json={"error": "busy"})
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 50}})
        assert body["stream"] is True and body["model"] == "llama-8b"
        lines = [f'data: {json.dumps({"choices": [{"index": 0, "delta": {"content": f"tok{i} "}}]})}'
                 for i in range(tokens_per_reply)]
        lines.append(f'data: {json.dumps({"choices": [], "usage": {"prompt_tokens": 60, "completion_tokens": tokens_per_reply}})}')
        lines.append("data: [DONE]")
        return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode(),
                              headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler), count


async def test_load_test_measures_tokens_ttft_and_latency():
    transport, count = sse_transport()
    cfg = LoadTestConfig(base_url="http://vllm.local", model="llama-8b", concurrency=4, requests=40, hardware="test-node",
                         warmup_requests=1)
    r = await run_load_test(cfg, transport=transport)
    assert r.requests == 40 and r.errors == 0 and count["n"] == 41  # warm-up not counted
    assert r.output_tokens_per_sec > 0 and r.total_tokens_per_sec > r.output_tokens_per_sec
    assert r.output_tokens_per_sec == pytest.approx(40 * 20 / r.duration_s, rel=0.01)
    assert r.ttft_p50_ms is not None and r.ttft_p95_ms >= r.ttft_p50_ms and r.latency_p95_ms >= r.latency_p50_ms
    assert "test-node" in r.source


async def test_load_test_errors_and_supply_selection():
    transport, _ = sse_transport(fail_every=5)
    cfg = LoadTestConfig(base_url="http://vllm.local", model="llama-8b", requests=20, warmup_requests=0)
    results = await sweep(cfg, [1, 2], transport=transport)
    assert all(r.errors > 0 for r in results)
    assert supply_at_target(results, max_error_rate=0.01) is None
    best = supply_at_target(results, max_error_rate=0.5)
    assert best in results
    md = loadtest_report(results, best)
    assert "Measured supply per node" in md
    ws = PoolWorksheet(pool="chat", peak_users=10, requests_per_user_per_min=1, tokens_per_request=80,
                       supply_tokens_per_sec_per_node=best.total_tokens_per_sec, supply_source=best.source)
    assert size_pool(ws).status == "sized"


async def test_embedding_load_test():
    transport, _ = sse_transport()
    r = await run_load_test(LoadTestConfig(base_url="http://emb.local", model="bge", mode="embeddings", requests=10,
                                           warmup_requests=0), transport=transport)
    assert r.errors == 0 and r.total_tokens_per_sec == pytest.approx(10 * 50 / r.duration_s, rel=0.01)
    assert r.ttft_p50_ms is None
