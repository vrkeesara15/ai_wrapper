"""Cost model: self-hosted fixed monthly cost vs a vendor API's price per request.

Cost model slide: "Self-hosted is a fixed cost: hardware, platform and the people who run it. An API is a
variable cost: a price for every request." Break-even requests per month = fixed cost / API price. Above it,
self-hosting is cheaper - until capacity runs out and "a node is added" (a step in the self-hosted line).
"Cost is one input: data control, latency and model quality can settle the choice first."
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, model_validator


class CostWorksheet(BaseModel):
    gpu_platform_monthly: float  # GPU and platform cost per month
    run_support_monthly: float  # run and support cost per month (people, on-call)
    api_price_per_request: float | None = None  # price for a typical request
    # ...or derived from tokens and the vendor's per-1k prices
    tokens_in_per_request: float | None = None
    tokens_out_per_request: float | None = None
    api_price_in_per_1k: float | None = None
    api_price_out_per_1k: float | None = None
    forecast_requests_per_month: float | None = None  # client's forecast
    # optional step function: capacity of the self-hosted setup and the cost of one more node
    capacity_requests_per_month: float | None = None
    node_monthly_cost: float | None = None
    capacity_per_added_node: float | None = None
    currency: str = "EUR"

    @model_validator(mode="after")
    def _price(self) -> CostWorksheet:
        if self.api_price_per_request is None:
            if None in (self.tokens_in_per_request, self.tokens_out_per_request, self.api_price_in_per_1k,
                        self.api_price_out_per_1k):
                raise ValueError("give api_price_per_request or tokens and per-1k prices")
            self.api_price_per_request = ((self.tokens_in_per_request or 0) / 1000 * (self.api_price_in_per_1k or 0)
                                          + (self.tokens_out_per_request or 0) / 1000 * (self.api_price_out_per_1k or 0))
        if self.api_price_per_request <= 0:
            raise ValueError("API price per request must be positive")
        return self

    @property
    def fixed_monthly(self) -> float:
        return self.gpu_platform_monthly + self.run_support_monthly


def break_even_requests(ws: CostWorksheet) -> float:
    """Fixed cost / API price (the slide's formula)."""
    return ws.fixed_monthly / (ws.api_price_per_request or 1)


def api_cost(ws: CostWorksheet, requests: float) -> float:
    return requests * (ws.api_price_per_request or 0)


def added_nodes(ws: CostWorksheet, requests: float) -> int:
    if not ws.capacity_requests_per_month or requests <= ws.capacity_requests_per_month or not ws.node_monthly_cost:
        return 0
    per_node = ws.capacity_per_added_node or ws.capacity_requests_per_month
    return math.ceil((requests - ws.capacity_requests_per_month) / per_node)


def self_hosted_cost(ws: CostWorksheet, requests: float) -> float:
    return ws.fixed_monthly + added_nodes(ws, requests) * (ws.node_monthly_cost or 0)


class CostRow(BaseModel):
    requests_per_month: float
    api_cost: float
    self_hosted_cost: float
    added_nodes: int
    cheaper: str
    difference: float  # api - self-hosted (positive = self-hosting saves this much)


class CostResult(BaseModel):
    currency: str
    fixed_monthly: float
    api_price_per_request: float
    break_even_requests_per_month: float
    forecast_requests_per_month: float | None = None
    at_forecast: CostRow | None = None
    table: list[CostRow]


def row(ws: CostWorksheet, requests: float) -> CostRow:
    a, s = api_cost(ws, requests), self_hosted_cost(ws, requests)
    return CostRow(requests_per_month=requests, api_cost=round(a, 2), self_hosted_cost=round(s, 2),
                   added_nodes=added_nodes(ws, requests),
                   cheaper="equal" if abs(a - s) < 0.005 else ("self-hosted" if s < a else "api"), difference=round(a - s, 2))


def comparison_table(ws: CostWorksheet, volumes: list[float] | None = None) -> list[CostRow]:
    be = break_even_requests(ws)
    vols = volumes or sorted({round(be * f) for f in (0.25, 0.5, 1.0, 2.0, 4.0)} |
                             ({ws.forecast_requests_per_month} if ws.forecast_requests_per_month else set()))
    return [row(ws, v) for v in vols]


def evaluate(ws: CostWorksheet, volumes: list[float] | None = None) -> CostResult:
    return CostResult(currency=ws.currency, fixed_monthly=ws.fixed_monthly, api_price_per_request=ws.api_price_per_request or 0,
                      break_even_requests_per_month=round(break_even_requests(ws), 2),
                      forecast_requests_per_month=ws.forecast_requests_per_month,
                      at_forecast=row(ws, ws.forecast_requests_per_month) if ws.forecast_requests_per_month else None,
                      table=comparison_table(ws, volumes))


def break_even_svg(ws: CostWorksheet, *, max_requests: float | None = None, width: int = 560, height: int = 300,
                   points: int = 200) -> str:
    """Cost per month vs requests per month: the API line, the self-hosted step line and the break-even mark.
    Self-contained SVG (no fonts or scripts loaded)."""
    be = break_even_requests(ws)
    xmax = max_requests or max(be * 2.5, (ws.forecast_requests_per_month or 0) * 1.2)
    xs = [xmax * i / points for i in range(points + 1)]
    ymax = max(max(api_cost(ws, x) for x in xs), max(self_hosted_cost(ws, x) for x in xs)) * 1.05
    pl, pb, pt, pr = 64, 40, 16, 16
    pw, ph = width - pl - pr, height - pb - pt

    def px(x: float) -> float:
        return pl + pw * x / xmax

    def py(y: float) -> float:
        return pt + ph * (1 - y / ymax)

    def path(fn: Any) -> str:
        return " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(fn(ws, x)):.1f}" for i, x in enumerate(xs))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'font-family="system-ui,sans-serif" font-size="11">',
             f'<line x1="{pl}" y1="{pt + ph}" x2="{pl + pw}" y2="{pt + ph}" stroke="#888"/>',
             f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt + ph}" stroke="#888"/>',
             f'<path d="{path(api_cost)}" fill="none" stroke="#e0793a" stroke-width="2"/>',
             f'<path d="{path(self_hosted_cost)}" fill="none" stroke="#2f6fdb" stroke-width="2"/>']
    if be <= xmax:
        bx, by = px(be), py(api_cost(ws, be))
        parts += [f'<line x1="{bx:.1f}" y1="{pt}" x2="{bx:.1f}" y2="{pt + ph}" stroke="#555" stroke-dasharray="4 3"/>',
                  f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4" fill="#555"/>',
                  f'<text x="{bx + 6:.1f}" y="{pt + 12}" fill="#555">break-even {be:,.0f} req/month</text>']
    if ws.forecast_requests_per_month and ws.forecast_requests_per_month <= xmax:
        fx = px(ws.forecast_requests_per_month)
        parts.append(f'<line x1="{fx:.1f}" y1="{pt}" x2="{fx:.1f}" y2="{pt + ph}" stroke="#3a9e6f" stroke-dasharray="2 2"/>'
                     f'<text x="{fx + 6:.1f}" y="{pt + 26}" fill="#3a9e6f">forecast</text>')
    for i in range(5):
        x = xmax * i / 4
        parts.append(f'<text x="{px(x):.1f}" y="{pt + ph + 14}" text-anchor="middle" fill="#666">{x:,.0f}</text>')
        y = ymax * i / 4
        parts.append(f'<text x="{pl - 6}" y="{py(y) + 4:.1f}" text-anchor="end" fill="#666">{y:,.0f}</text>')
    parts += [f'<text x="{pl + pw / 2}" y="{height - 6}" text-anchor="middle" fill="#444">Requests per month</text>',
              f'<text x="14" y="{pt + ph / 2}" transform="rotate(-90 14 {pt + ph / 2})" text-anchor="middle" fill="#444">'
              f'Cost per month ({ws.currency})</text>',
              f'<rect x="{pl + 10}" y="{pt + 34}" width="12" height="3" fill="#e0793a"/><text x="{pl + 26}" y="{pt + 38}" fill="#444">Vendor API</text>',
              f'<rect x="{pl + 10}" y="{pt + 48}" width="12" height="3" fill="#2f6fdb"/><text x="{pl + 26}" y="{pt + 52}" fill="#444">Self-hosted</text>',
              "</svg>"]
    return "".join(parts)
