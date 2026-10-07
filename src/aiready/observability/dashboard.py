"""Self-contained HTML dashboard (``GET /dashboard``).

No CDN, no external fonts, no JavaScript framework: the platform runs air-gapped, and a dashboard that pulls
from the internet would either break there or leak that it exists. Charts are inline SVG rendered on the
server from the same data as ``GET /v1/dashboard/data``; a few lines of vanilla JS only sort the tables.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from typing import Any

PALETTE = ["#2f6fdb", "#e0793a", "#3a9e6f", "#a259c9", "#c9a227", "#d64f6b", "#4bb0c9", "#7a7f87"]


def _e(v: Any) -> str:
    return html.escape(str(v), quote=True)


def _fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def bar_chart(rows: list[tuple[str, float]], *, unit: str = "", width: int = 520, bar_h: int = 22,
              marker: dict[str, float] | None = None) -> str:
    """Horizontal bars; ``marker`` draws a budget line per label."""
    if not rows:
        return '<p class="empty">No data in this period.</p>'
    label_w = 150
    vmax = max([v for _, v in rows] + list((marker or {}).values()) + [1e-9])
    plot_w = width - label_w - 70
    h = len(rows) * (bar_h + 8) + 8
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" class="chart">']
    for i, (label, v) in enumerate(rows):
        y = 8 + i * (bar_h + 8)
        w = max(1.0, v / vmax * plot_w)
        over = marker and label in marker and v > marker[label]
        color = "#d64f6b" if over else PALETTE[0]
        parts.append(f'<text x="{label_w - 8}" y="{y + bar_h * 0.7}" text-anchor="end" class="lbl">{_e(label[:22])}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" fill="{color}">'
                     f'<title>{_e(label)}: {_fmt(v)} {unit}</title></rect>')
        parts.append(f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h * 0.7}" class="val">{_fmt(v)}{unit}</text>')
        if marker and label in marker:
            mx = label_w + marker[label] / vmax * plot_w
            parts.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{y - 3}" y2="{y + bar_h + 3}" class="budget">'
                         f'<title>budget {marker[label]:g}{unit}</title></line>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(points: list[dict[str, Any]], series: list[str], *, width: int = 520, height: int = 200,
               ymin: float = 0.0, ymax: float = 1.0) -> str:
    if not points:
        return '<p class="empty">No scored answers yet.</p>'
    pad_l, pad_b, pad_t = 36, 26, 10
    pw, ph = width - pad_l - 10, height - pad_b - pad_t
    n = len(points)

    def xy(i: int, v: float) -> tuple[float, float]:
        x = pad_l + (pw * i / (n - 1) if n > 1 else pw / 2)
        y = pad_t + ph * (1 - (v - ymin) / (ymax - ymin or 1))
        return x, y

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" class="chart">']
    for t in (0.0, 0.5, 1.0):
        _, y = xy(0, ymin + t * (ymax - ymin))
        parts.append(f'<line x1="{pad_l}" x2="{width - 10}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end" class="lbl">{ymin + t * (ymax - ymin):.1f}</text>')
    for si, name in enumerate(series):
        pts = [(i, p[name]) for i, p in enumerate(points) if isinstance(p.get(name), (int, float))]
        if not pts:
            continue
        color = PALETTE[si % len(PALETTE)]
        path = " ".join(f"{'M' if j == 0 else 'L'}{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for j, (i, v) in enumerate(pts))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, v in pts:
            x, y = xy(i, v)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"><title>{_e(name)} {_e(points[i]["bucket"])}: {v:.3f}</title></circle>')
    for i in {0, n - 1}:
        x, _ = xy(i, ymin)
        parts.append(f'<text x="{x:.1f}" y="{height - 6}" text-anchor="middle" class="lbl">{_e(points[i]["bucket"])}</text>')
    legend_x = pad_l
    for si, name in enumerate(series):
        parts.append(f'<rect x="{legend_x}" y="0" width="10" height="3" fill="{PALETTE[si % len(PALETTE)]}"/>'
                     f'<text x="{legend_x + 14}" y="6" class="lbl">{_e(name)}</text>')
        legend_x += 14 + 8 * len(name) + 12
    parts.append("</svg>")
    return "".join(parts)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return '<p class="empty">No data.</p>'
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="sortable"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2129;--muted:#5d6470;--line:#e2e5ea;--accent:#2f6fdb}
@media (prefers-color-scheme: dark){:root{--bg:#121417;--card:#1b1e23;--fg:#e8eaed;--muted:#9aa1ab;--line:#2c3138}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 24px 8px}h1{font-size:20px;margin:0}header p{color:var(--muted);margin:4px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px;padding:16px 24px 32px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0}
.card h2{font-size:14px;margin:0 0 12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;padding:8px 24px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--muted);font-size:12px}
.chart text{fill:var(--fg);font-size:11px}.chart .lbl{fill:var(--muted)}.chart .grid{stroke:var(--line)}
.chart .budget{stroke:var(--fg);stroke-width:2;stroke-dasharray:3 2}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}
th{cursor:pointer;color:var(--muted)}.empty{color:var(--muted)}.bad{color:#d64f6b}.ok{color:#3a9e6f}
@media (max-width:600px){.grid,.kpis{padding-left:16px;padding-right:16px}}
"""

JS = """
document.querySelectorAll('table.sortable th').forEach(function(th,i){th.addEventListener('click',function(){
var tb=th.closest('table').tBodies[0];var rows=Array.from(tb.rows);var asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
rows.sort(function(a,b){var x=a.cells[i].textContent,y=b.cells[i].textContent;var nx=parseFloat(x.replace(/,/g,'')),ny=parseFloat(y.replace(/,/g,''));
var c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c;});rows.forEach(function(r){tb.appendChild(r);});});});
"""


def render_dashboard(data: dict[str, Any], title: str = "AI-Ready observability") -> str:
    cur = data.get("currency", "")
    gen = datetime.fromtimestamp(data.get("generated_at", 0), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    feat = [(r["key"].get("feature", "-"), r["cost"]) for r in data.get("spend_by_feature", [])][:12]
    team = [(r["key"].get("team", "-"), r["cost"]) for r in data.get("spend_by_team", [])][:12]
    lat = [r for r in data.get("latency", []) if r.get("count")]
    lat_rows = [(r["stage"], r["p95_ms"]) for r in lat]
    budgets = {r["stage"]: r["budget_ms"] for r in lat if r.get("budget_ms")}
    quality = data.get("quality", [])
    series = ["overall"] + sorted({k for p in quality for k in p if k not in ("bucket", "samples", "overall")})[:5]
    chr_ = data.get("cache_hit_rate")
    alerts = data.get("alerts", [])
    lat_table = _table(["stage", "samples", "p50 ms", "p95 ms", "budget ms", "within budget"],
                       [[r["stage"], r["count"], _fmt(r["p50_ms"], 1), _fmt(r["p95_ms"], 1), _fmt(r.get("budget_ms"), 0),
                         "yes" if r.get("within_budget") else ("no" if r.get("within_budget") is False else "-")]
                        for r in data.get("latency", [])])
    team_table = _table(["team", "requests", "tokens", f"cost {cur}", "p95 ms", "errors"],
                        [[r["key"].get("team"), r["requests"], r["total_tokens"], _fmt(r["cost"], 4), _fmt(r["latency_p95_ms"], 0),
                          r["errors"]] for r in data.get("spend_by_team", [])])
    alert_html = ("".join(f'<p class="bad">&#9679; {_e(a.get("last", {}).get("rule", ""))}: {_e(a.get("last", {}).get("message", ""))}</p>'
                          for a in alerts) or '<p class="ok">No active alerts.</p>')
    causes = data.get("review_causes", {})
    embedded = json.dumps(data, default=str).replace("</", "<\\/")  # safe inside <script>
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>{CSS}</style></head>
<body><header><h1>{_e(title)}</h1><p>Last {_e(data.get("days"))} days &middot; generated {_e(gen)}</p></header>
<section class="kpis">
<div class="kpi"><b>{_e(data.get("requests", 0))}</b><span>requests</span></div>
<div class="kpi"><b>{_fmt(data.get("total_cost"), 2)} {_e(cur)}</b><span>spend</span></div>
<div class="kpi"><b>{"-" if chr_ is None else f"{chr_ * 100:.1f}%"}</b><span>cache hit rate</span></div>
<div class="kpi"><b>{_e(data.get("review_queue", 0))}</b><span>open review items</span></div>
<div class="kpi"><b>{len(alerts)}</b><span>active alerts</span></div>
</section>
<main class="grid">
<div class="card"><h2>Spend per feature ({_e(cur)})</h2>{bar_chart(feat)}</div>
<div class="card"><h2>Spend per team ({_e(cur)})</h2>{bar_chart(team)}</div>
<div class="card"><h2>Latency p95 vs budget (ms)</h2>{bar_chart(lat_rows, unit=" ms", marker=budgets)}{lat_table}</div>
<div class="card"><h2>Quality trend</h2>{line_chart(quality, series)}</div>
<div class="card"><h2>Chargeback by team</h2>{team_table}</div>
<div class="card"><h2>Alerts</h2>{alert_html}<h2>Review causes</h2>{_table(["cause", "count"], [[k, v] for k, v in sorted(causes.items())])}</div>
</main>
<script type="application/json" id="data">{embedded}</script>
<script>{JS}</script></body></html>"""
