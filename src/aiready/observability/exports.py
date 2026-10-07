"""Generators for the client's own monitoring stack: a Grafana dashboard and Prometheus alert rules.

Both are built from the metric names in ``aiready.core.telemetry`` and from this package's configuration
(latency budget, quality threshold, budgets), so the external dashboards and alerts never drift from the
platform's own. The deployment tooling writes them to ``deploy/observability/``; they are plain dicts here.
"""

from __future__ import annotations

from typing import Any

from aiready.observability.config import ObservabilityConfig

DS = {"type": "prometheus", "uid": "${datasource}"}

#: budget stage -> (histogram, label filter) used for the p95-vs-budget panels and alerts
STAGE_METRICS: dict[str, tuple[str, str]] = {
    "screen": ("aiready_stage_latency_seconds_bucket", 'stage=~"screen.*|screen_input"'),
    "embed": ("aiready_stage_latency_seconds_bucket", 'stage=~"embed.*"'),
    "search": ("aiready_stage_latency_seconds_bucket", 'stage=~"search|hybrid_search"'),
    "rerank": ("aiready_stage_latency_seconds_bucket", 'stage=~"rerank"'),
    "check": ("aiready_stage_latency_seconds_bucket", 'stage=~"verify|check|grounding"'),
    "first_token": ("aiready_time_to_first_token_seconds_bucket", ""),
}


def _cfg(cfg: ObservabilityConfig | None) -> ObservabilityConfig:
    if cfg is not None:
        return cfg
    try:
        from aiready.observability.config import load_config

        return load_config()
    except Exception:
        return ObservabilityConfig()


def _p95(metric: str, flt: str, by: str = "") -> str:
    sel = f"{{{flt}}}" if flt else ""
    grp = f"le{',' + by if by else ''}"
    return f"histogram_quantile(0.95, sum by ({grp}) (rate({metric}{sel}[5m])))"


def _panel(pid: int, title: str, exprs: list[tuple[str, str]], *, x: int, y: int, w: int = 12, h: int = 8,
           unit: str = "short", kind: str = "timeseries", thresholds: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    p: dict[str, Any] = {
        "id": pid, "type": kind, "title": title, "datasource": DS, "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": lg, "datasource": DS} for i, (e, lg) in enumerate(exprs)],
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": thresholds}
    return p


def grafana_dashboard(cfg: ObservabilityConfig | None = None) -> dict[str, Any]:
    """Grafana dashboard JSON (schema 39): spend per feature/team/model, latency p95 vs budget per stage,
    quality scores, cache hit rate, error rate, review queue depth. Template vars: datasource, app, feature."""
    c = _cfg(cfg)
    flt = 'app=~"$app",feature=~"$feature"'
    panels: list[dict[str, Any]] = []
    pid = 1
    panels.append(_panel(pid, f"Spend per feature ({c.currency}/h)", [
        (f"sum by (feature) (increase(aiready_cost_total{{{flt}}}[1h]))", "{{feature}}")], x=0, y=0, unit="currencyEUR" if c.currency == "EUR" else "short"))
    pid += 1
    panels.append(_panel(pid, f"Spend per team ({c.currency}/h)", [
        (f"sum by (team) (increase(aiready_cost_total{{{flt}}}[1h]))", "{{team}}")], x=12, y=0))
    pid += 1
    panels.append(_panel(pid, "Tokens per model", [
        (f"sum by (model, direction) (rate(aiready_tokens_total{{{flt}}}[5m]))", "{{model}} {{direction}}")], x=0, y=8))
    pid += 1
    panels.append(_panel(pid, "Error rate", [
        (f'sum(rate(aiready_requests_total{{{flt},status=~"5.."}}[5m])) / sum(rate(aiready_requests_total{{{flt}}}[5m]))',
         "errors")], x=12, y=8, unit="percentunit"))
    pid += 1
    y = 16
    for i, (stage, budget) in enumerate(c.latency_budget_ms.items()):
        if stage not in STAGE_METRICS:
            continue
        metric, sel = STAGE_METRICS[stage]
        panels.append(_panel(pid, f"{stage} p95 vs budget {budget:g} ms", [(_p95(metric, sel), "p95")], x=(i % 3) * 8,
                             y=y + (i // 3) * 7, w=8, h=7, unit="s", kind="timeseries",
                             thresholds=[{"color": "green", "value": None}, {"color": "red", "value": budget / 1000}]))
        pid += 1
    y += 14
    panels.append(_panel(pid, "Quality score (live scorers)", [
        (f'avg by (feature, scorer) (aiready_quality_score{{{flt}}})', "{{feature}} {{scorer}}")], x=0, y=y, unit="percentunit",
        thresholds=[{"color": "red", "value": None}, {"color": "green", "value": c.scoring.low_score_threshold}]))
    pid += 1
    panels.append(_panel(pid, "Cache hit rate", [
        ('sum(rate(aiready_cache_total{result="hit"}[15m])) / sum(rate(aiready_cache_total{result=~"hit|miss"}[15m]))', "hit rate")],
        x=12, y=y, unit="percentunit"))
    pid += 1
    panels.append(_panel(pid, "Review queue", [('max(aiready_queue_depth{queue="review"})', "open items")],
                         x=0, y=y + 8, kind="stat"))
    return {
        "uid": "aiready-overview", "title": "AI-Ready overview", "schemaVersion": 39, "version": 1,
        "tags": ["aiready"], "timezone": "browser", "refresh": "1m", "time": {"from": "now-24h", "to": "now"},
        "templating": {"list": [
            {"name": "datasource", "type": "datasource", "query": "prometheus", "label": "Data source"},
            {"name": "app", "type": "query", "datasource": DS, "query": "label_values(aiready_requests_total, app)",
             "includeAll": True, "multi": True, "current": {"text": "All", "value": "$__all"}},
            {"name": "feature", "type": "query", "datasource": DS,
             "query": 'label_values(aiready_requests_total{app=~"$app"}, feature)', "includeAll": True, "multi": True,
             "current": {"text": "All", "value": "$__all"}},
        ]},
        "panels": panels,
    }


def prometheus_rules(cfg: ObservabilityConfig | None = None) -> dict[str, Any]:
    """Prometheus rule file (``groups:``) mirroring the built-in alert kinds: latency over budget per stage,
    quality below threshold, error rate, spend over budget, review queue backlog."""
    c = _cfg(cfg)
    rules: list[dict[str, Any]] = []
    for stage, budget in c.latency_budget_ms.items():
        if stage not in STAGE_METRICS:
            continue
        metric, sel = STAGE_METRICS[stage]
        rules.append({
            "alert": f"AIReadyLatencyOverBudget_{stage}", "expr": f"{_p95(metric, sel)} > {budget / 1000:g}",
            "for": "10m", "labels": {"severity": "warning", "stage": stage},
            "annotations": {"summary": f"{stage} p95 latency over the {budget:g} ms budget",
                            "description": "Design target from the latency budget; confirm against the load test."}})
    rules.append({
        "alert": "AIReadyQualityLow", "expr": f"min by (app, feature) (aiready_quality_score{{scorer=\"overall\"}}) < {c.scoring.low_score_threshold:g}",
        "for": "30m", "labels": {"severity": "warning"},
        "annotations": {"summary": "Live quality score below threshold for {{ $labels.feature }}"}})
    rules.append({
        "alert": "AIReadyErrorRateHigh",
        "expr": 'sum by (service) (rate(aiready_requests_total{status=~"5.."}[5m])) / sum by (service) (rate(aiready_requests_total[5m])) > 0.05',
        "for": "10m", "labels": {"severity": "critical"},
        "annotations": {"summary": "More than 5% errors on {{ $labels.service }}"}})
    for key, budget in c.budgets.items():
        dim, _, val = key.partition(":")
        if dim not in ("team", "feature", "app") or not val:
            continue
        rules.append({
            "alert": f"AIReadySpendOverBudget_{dim}_{val}".replace("-", "_"),
            "expr": f'sum(increase(aiready_cost_total{{{dim}="{val}"}}[30d])) > {budget:g}',
            "for": "15m", "labels": {"severity": "warning", dim: val},
            "annotations": {"summary": f"{dim} {val} spend over its monthly budget of {budget:g} {c.currency}"}})
    rules.append({
        "alert": "AIReadyReviewBacklog", "expr": 'max(aiready_queue_depth{queue="review"}) > 200', "for": "1d",
        "labels": {"severity": "info"}, "annotations": {"summary": "Review queue backlog above 200 items"}})
    for r in c.alerts.rules:
        if r.kind == "pipeline_lag":
            rules.append({
                "alert": f"AIReadyPipelineLag_{r.name}".replace("-", "_"),
                "expr": 'max(aiready_queue_depth{queue=~"ingest.*"}) > 1000', "for": f"{max(r.window_minutes, 5)}m",
                "labels": {"severity": r.severity}, "annotations": {"summary": "Ingestion backlog growing"}})
    return {"groups": [{"name": "aiready", "interval": "1m", "rules": rules}]}
