"""Async load generator for any OpenAI-compatible endpoint (vLLM, TGI, Ollama, llama.cpp, the AI-Ready
gateway itself).

"Measured supply: tokens per second one GPU node sustains for the chosen model at the target latency. Taken
from a load test on the client's hardware, never from a brochure." This produces that number: at a given
concurrency it measures output tokens/s, time to first token (streaming) and p50/p95 latency. ``sweep`` runs
several concurrencies and ``supply_at_target`` picks the highest throughput that still meets the latency
target - the figure that goes into the capacity worksheet.

    python -m aiready.sizing.loadtest --url http://vllm:8000 --model llama-3.1-8b -c 1 -c 8 -c 32 -n 200
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from aiready.core.tokens import count_tokens
from aiready.observability.usage import percentile

DEFAULT_PROMPT = ("You are assisting a claims adjuster. Summarise the following note in five bullet points: "
                  "the customer reported water damage in the kitchen after a pipe burst on Saturday night; a plumber "
                  "repaired the pipe on Sunday; the floor and two cabinets need replacing; photos were uploaded.")


class LoadTestConfig(BaseModel):
    base_url: str
    model: str
    api_key: str | None = None
    mode: Literal["chat", "embeddings"] = "chat"
    concurrency: int = 8
    requests: int | None = 100  # total requests (or use duration_s)
    duration_s: float | None = None
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = 256
    temperature: float = 0.0
    stream: bool = True  # needed to measure time to first token
    timeout_s: float = 120.0
    hardware: str = ""  # e.g. "client DC rack 4, 1x H100 80GB" - recorded with the result
    warmup_requests: int = 2
    extra_headers: dict[str, str] = Field(default_factory=dict)


class RequestSample(BaseModel):
    ok: bool
    latency_ms: float
    ttft_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class LoadTestResult(BaseModel):
    model: str
    base_url: str
    mode: str
    hardware: str
    measured_at: str
    concurrency: int
    requests: int
    errors: int
    error_rate: float
    duration_s: float
    requests_per_sec: float
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None

    @property
    def source(self) -> str:
        """Provenance string for ``PoolWorksheet.supply_source``."""
        return f"load test {self.measured_at} on {self.hardware or 'unlabelled hardware'} at concurrency {self.concurrency}"


def _headers(cfg: LoadTestConfig) -> dict[str, str]:
    h = {"content-type": "application/json", **cfg.extra_headers}
    if cfg.api_key:
        h["authorization"] = f"Bearer {cfg.api_key}"
    return h


async def _one_chat(client: httpx.AsyncClient, cfg: LoadTestConfig) -> RequestSample:
    body: dict[str, Any] = {"model": cfg.model, "messages": [{"role": "user", "content": cfg.prompt}],
                            "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "stream": cfg.stream}
    if cfg.stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        if cfg.stream:
            async with client.stream("POST", "/v1/chat/completions", json=body, headers=_headers(cfg)) as r:
                if r.status_code >= 400:
                    await r.aread()
                    return RequestSample(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, error=f"http {r.status_code}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            text_parts.append(piece)
        else:
            r = await client.post("/v1/chat/completions", json=body, headers=_headers(cfg))
            if r.status_code >= 400:
                return RequestSample(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, error=f"http {r.status_code}")
            data = r.json()
            usage = data.get("usage") or {}
            text_parts.append(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    except httpx.HTTPError as exc:
        return RequestSample(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, error=type(exc).__name__)
    latency = (time.perf_counter() - t0) * 1000
    out_tokens = int(usage.get("completion_tokens") or count_tokens("".join(text_parts)))
    in_tokens = int(usage.get("prompt_tokens") or count_tokens(cfg.prompt))
    return RequestSample(ok=True, latency_ms=latency, ttft_ms=ttft, input_tokens=in_tokens, output_tokens=out_tokens)


async def _one_embed(client: httpx.AsyncClient, cfg: LoadTestConfig) -> RequestSample:
    t0 = time.perf_counter()
    try:
        r = await client.post("/v1/embeddings", json={"model": cfg.model, "input": [cfg.prompt]}, headers=_headers(cfg))
    except httpx.HTTPError as exc:
        return RequestSample(ok=False, latency_ms=(time.perf_counter() - t0) * 1000, error=type(exc).__name__)
    latency = (time.perf_counter() - t0) * 1000
    if r.status_code >= 400:
        return RequestSample(ok=False, latency_ms=latency, error=f"http {r.status_code}")
    usage = r.json().get("usage") or {}
    return RequestSample(ok=True, latency_ms=latency, input_tokens=int(usage.get("prompt_tokens") or count_tokens(cfg.prompt)))


async def run_load_test(cfg: LoadTestConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> LoadTestResult:
    """Closed-loop load: ``concurrency`` workers each send the next request as soon as the previous returns."""
    if not cfg.requests and not cfg.duration_s:
        raise ValueError("set requests or duration_s")
    one = _one_chat if cfg.mode == "chat" else _one_embed
    samples: list[RequestSample] = []
    async with httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s, transport=transport,
                                 limits=httpx.Limits(max_connections=cfg.concurrency * 2)) as client:
        for _ in range(cfg.warmup_requests):  # load weights / warm caches; not counted
            await one(client, cfg)
        issued = 0
        deadline = time.perf_counter() + cfg.duration_s if cfg.duration_s else None

        async def worker() -> None:
            nonlocal issued
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    return
                if cfg.requests and issued >= cfg.requests:
                    return
                issued += 1
                samples.append(await one(client, cfg))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(max(1, cfg.concurrency))))
        wall = time.perf_counter() - t0
    ok = [s for s in samples if s.ok]
    lat = [s.latency_ms for s in ok]
    ttft = [s.ttft_ms for s in ok if s.ttft_ms is not None]
    out_tok = sum(s.output_tokens for s in ok)
    all_tok = out_tok + sum(s.input_tokens for s in ok)
    wall = max(wall, 1e-9)
    return LoadTestResult(
        model=cfg.model, base_url=cfg.base_url, mode=cfg.mode, hardware=cfg.hardware,
        measured_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), concurrency=cfg.concurrency,
        requests=len(samples), errors=len(samples) - len(ok),
        error_rate=round((len(samples) - len(ok)) / len(samples), 4) if samples else 0.0, duration_s=round(wall, 6),
        requests_per_sec=round(len(ok) / wall, 3),
        output_tokens_per_sec=round(out_tok / wall, 2) if cfg.mode == "chat" else round(all_tok / wall, 2),
        total_tokens_per_sec=round(all_tok / wall, 2),
        latency_p50_ms=round(percentile(lat, 50), 2), latency_p95_ms=round(percentile(lat, 95), 2),
        ttft_p50_ms=round(percentile(ttft, 50), 2) if ttft else None,
        ttft_p95_ms=round(percentile(ttft, 95), 2) if ttft else None,
    )


async def sweep(cfg: LoadTestConfig, concurrencies: list[int], *,
                transport: httpx.AsyncBaseTransport | None = None) -> list[LoadTestResult]:
    out = []
    for c in concurrencies:
        out.append(await run_load_test(cfg.model_copy(update={"concurrency": c}), transport=transport))
    return out


def supply_at_target(results: list[LoadTestResult], *, max_latency_p95_ms: float | None = None,
                     max_ttft_p95_ms: float | None = None, max_error_rate: float = 0.01,
                     metric: Literal["output", "total"] = "total") -> LoadTestResult | None:
    """The highest-throughput run that meets the latency target: the measured supply per node.

    ``metric``: the capacity worksheet's tokens per request counts tokens *in and out*, so the matching supply
    is total tokens/s; use ``output`` when the worksheet counts output tokens only."""
    ok = [r for r in results if r.error_rate <= max_error_rate
          and (max_latency_p95_ms is None or r.latency_p95_ms <= max_latency_p95_ms)
          and (max_ttft_p95_ms is None or (r.ttft_p95_ms is not None and r.ttft_p95_ms <= max_ttft_p95_ms))]
    if not ok:
        return None
    key = (lambda r: r.total_tokens_per_sec) if metric == "total" else (lambda r: r.output_tokens_per_sec)
    return max(ok, key=key)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    import argparse

    ap = argparse.ArgumentParser(description="Load test an OpenAI-compatible endpoint.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key")
    ap.add_argument("--mode", default="chat", choices=["chat", "embeddings"])
    ap.add_argument("-c", "--concurrency", type=int, action="append", default=[])
    ap.add_argument("-n", "--requests", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--hardware", default="")
    ap.add_argument("--target-ttft-ms", type=float)
    ap.add_argument("--target-p95-ms", type=float)
    a = ap.parse_args(argv)
    cfg = LoadTestConfig(base_url=a.url, model=a.model, api_key=a.api_key, mode=a.mode, requests=a.requests,
                         max_tokens=a.max_tokens, hardware=a.hardware)
    results = asyncio.run(sweep(cfg, a.concurrency or [1, 4, 16]))
    from aiready.sizing.report import loadtest_report

    print(loadtest_report(results, best=supply_at_target(results, max_latency_p95_ms=a.target_p95_ms,
                                                         max_ttft_p95_ms=a.target_ttft_ms)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
