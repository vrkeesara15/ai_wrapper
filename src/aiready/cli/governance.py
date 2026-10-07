"""Governance commands: policy, token, audit, registry, incident."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from aiready.cli.common import console, emit, fail, ok, operator, run, table
from aiready.core.audit import AuditLog, JsonlSink, get_audit
from aiready.core.config import SECRET_DIRS, get_config, resolve_secret
from aiready.core.identity import get_token_service, load_keyring
from aiready.core.stores import get_docstore

policy_app = typer.Typer(help="Policy as code: test, decide, explain.", no_args_is_help=True)
token_app = typer.Typer(help="Tokens: issue (break-glass/dev), verify, revoke, staged key rotation.", no_args_is_help=True)
audit_app = typer.Typer(help="Tamper-evident audit log: verify, export, rotate key, reset chain.", no_args_is_help=True)
registry_app = typer.Typer(help="Model, prompt and adapter registries.", no_args_is_help=True)
incident_app = typer.Typer(help="Incident runbook actions: open, revoke, close routes, trace impact, close.",
                           no_args_is_help=True)
registry_model_app = typer.Typer(help="Model registry.", no_args_is_help=True)
registry_app.add_typer(registry_model_app, name="model")

TokenOpt = typer.Option(None, "--token", envvar="AIREADY_TOKEN", help="Operator token (else break-glass).")


def _parse_time(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _governance() -> Any:
    from aiready.core.wiring import connect

    return connect("policy")


def _secret_dir() -> Path:
    for d in SECRET_DIRS:
        if d.is_dir() and os.access(d, os.W_OK):
            return d
    d = Path(os.environ.get("AIREADY_LOCAL_SECRET_DIR", "data/secrets"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_secret(name: str, value: str, out_dir: Path | None) -> Path:
    path = (out_dir or _secret_dir()) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(0o600)
    return path


# =========================================================================== policy


@policy_app.command("test")
def policy_test(path: Path = typer.Argument(Path("config/policies/tests")),
                config_dir: Optional[Path] = typer.Option(None, help="Policies to test (default: active config).")):
    """Run YAML policy test cases. Non-zero exit when any case fails (CI gate)."""
    from aiready.governance.policy import run_policy_tests

    report = run_policy_tests(path, config_dir=config_dir)
    console.print(report.summary())
    if not report.ok:
        raise typer.Exit(1)


@policy_app.command("decide")
def policy_decide(action: str, principal: str = typer.Option("{}", help="JSON principal"),
                  resource: str = typer.Option("{}", help="JSON resource"),
                  context: str = typer.Option("{}", help="JSON context"),
                  explain: bool = typer.Option(False, help="Show the matched-rule trace.")):
    """Evaluate one decision against the active policy bundle."""
    from aiready.core.interfaces import PolicyInput

    inp = PolicyInput(action=action, principal=json.loads(principal), resource=json.loads(resource),
                      context=json.loads(context))
    gov = _governance()
    res = run(gov.explain(inp)) if explain and hasattr(gov, "explain") else run(gov.decide(inp))
    emit(res)


@policy_app.command("bundles")
def policy_bundles():
    """List versioned policy bundles."""
    gov = _governance()
    res = gov.bundles()
    emit(run(res) if hasattr(res, "__await__") else res)


# =========================================================================== token


@token_app.command("issue")
def token_issue(sub: str, roles: list[str] = typer.Option([], "--role"), groups: list[str] = typer.Option([], "--group"),
                tenant: str = "default", app: str = "default", ttl: int = 900):
    """Mint a token (dev environments and break-glass only; production users get tokens from the bridge)."""
    if get_config().get("platform.environment", "dev") == "production" and not os.environ.get("AIREADY_ALLOW_ISSUE"):
        fail("token issue is disabled in production (set AIREADY_ALLOW_ISSUE=1 for a documented break-glass)")
    tok = get_token_service().issue(sub=sub, roles=roles, groups=groups, tenant=tenant, app=app, ttl=ttl)
    get_audit().record("token.issue", actor=os.environ.get("USER", "cli"), resource=f"user:{sub}",
                       decision="issued", roles=roles, ttl=ttl)
    typer.echo(tok)


@token_app.command("verify")
def token_verify(token: Optional[str] = typer.Argument(None),
                 sample: bool = typer.Option(False, help="Mint and verify a sample token with the active key.")):
    """Verify a token (or a freshly minted sample) against the current keyring."""
    ts = get_token_service()
    if sample:
        token = ts.issue(sub="svc:keycheck", roles=["service"], ttl=60)
    if not token:
        fail("give a token or --sample")
    p = ts.verify(token)  # type: ignore[arg-type]
    import jwt

    emit({"valid": True, "sub": p.sub, "roles": p.roles, "groups": p.groups, "tenant": p.tenant,
          "kid": jwt.get_unverified_header(token).get("kid"), "exp": p.claims.get("exp")})


@token_app.command("revoke")
def token_revoke(subject: Optional[str] = typer.Option(None), jti: Optional[str] = typer.Option(None),
                 key_id: Optional[str] = typer.Option(None, help="Gateway service API key id."),
                 reason: str = typer.Option(..., help="Incident reference, e.g. INC-42"),
                 incident: Optional[str] = None, token: Optional[str] = TokenOpt):
    """Put a subject / token id on the deny-list (enforced at authentication everywhere), or revoke an API key."""
    op = operator(token, ["security-admin"])
    out: dict[str, Any] = {}
    if subject or jti:
        out["deny_list"] = _governance().incidents.revoke(op, subject=subject, jti=jti, reason=reason,
                                                           incident_id=incident or reason)
    if key_id:
        from aiready.gateway.keys import KeyService

        ks = KeyService()
        fn = getattr(ks, "revoke", None) or getattr(ks, "revoke_key")
        res = fn(key_id, principal=op) if "principal" in fn.__code__.co_varnames else fn(key_id)
        out["api_key"] = res.model_dump() if hasattr(res, "model_dump") else res
    if not out:
        fail("give --subject, --jti or --key-id")
    emit(out)


@token_app.command("rotate")
def token_rotate(stage: str = typer.Option(..., help="add | activate | retire"),
                 key: str = typer.Option("user", help="user | service"),
                 out_dir: Optional[Path] = typer.Option(None, help="Where to write the token_keyring secret."),
                 token: Optional[str] = TokenOpt):
    """Staged signing-key rotation with no downtime.

    add: new key verifies only (deploy, wait one reload) -> activate: new key signs, old still verifies
    (wait > token TTL) -> retire: old key removed; its tokens are refused.
    """
    if stage not in {"add", "activate", "retire"}:
        fail("stage must be add, activate or retire")
    op = operator(token, ["security-admin"])
    ring = load_keyring()
    if not ring:  # bootstrap from the single configured secret
        current = resolve_secret("token_signing_key") or os.environ.get("AIREADY_DEV_SIGNING_KEY")
        if not current:
            fail("no keyring and no token_signing_key to start from")
        ring = [{"kid": f"{key}-1", "use": key, "status": "active", "secret": current, "created_at": time.time()}]
    mine = [k for k in ring if k.get("use", "user") == key]
    if stage == "add":
        if any(k["status"] == "verify" for k in mine):
            fail("a new key is already staged; run --stage activate")
        n = 1 + max([int(str(k["kid"]).rsplit("-", 1)[-1]) for k in mine if str(k["kid"]).rsplit("-", 1)[-1].isdigit()] or [0])
        ring.append({"kid": f"{key}-{n}", "use": key, "status": "verify", "secret": secrets.token_urlsafe(48),
                     "created_at": time.time()})
    elif stage == "activate":
        staged = [k for k in mine if k["status"] == "verify"]
        if not staged:
            fail("nothing staged; run --stage add first")
        for k in mine:
            if k["status"] == "active":
                k["status"] = "previous"
        staged[-1]["status"] = "active"
    else:
        for k in mine:
            if k["status"] == "previous":
                k["status"] = "retired"
    path = _write_secret("token_keyring", json.dumps(ring, indent=2), out_dir)
    get_audit().for_principal(op, "token.rotate", action=stage, resource=f"keyring:{key}", decision="applied",
                              kids={k["kid"]: k["status"] for k in ring})
    table([{k_: v for k_, v in k.items() if k_ != "secret"} for k in ring], ["kid", "use", "status"], "token keyring")
    ok(f"wrote {path}. Publish it as the 'token_keyring' secret to every service (rolling restart or secret reload).")


# =========================================================================== audit


def _audit_cfg() -> dict[str, Any]:
    return get_config().get("audit") or {}


def _chain_files(all_chains: bool, chain: Path | None) -> list[Path]:
    if chain:
        return [chain]
    sinks = _audit_cfg().get("sinks") or [{"type": "jsonl", "path": "data/audit/audit.jsonl"}]
    files: list[Path] = []
    for s in sinks:
        if s.get("type") == "jsonl":
            p = Path(s.get("path", "data/audit/audit.jsonl"))
            files += [Path(f) for f in sorted(glob.glob(str(p.parent / "*.jsonl")))] if all_chains else [p]
    return files


def _log_for(path: Path) -> AuditLog:
    cfg = _audit_cfg()
    prev = cfg.get("previous_hmac_keys") or resolve_secret("audit_hmac_keys_previous") or []
    if isinstance(prev, str):
        prev = [p for p in prev.split(",") if p]
    return AuditLog([JsonlSink(str(path))], hmac_key=cfg.get("hmac_key") or resolve_secret("audit_hmac_key"),
                    previous_hmac_keys=prev)


@audit_app.command("verify")
def audit_verify(all_chains: bool = typer.Option(False, "--all-chains"), chain: Optional[Path] = None,
                 since: Optional[str] = typer.Option(None, help="Only report; verification always walks the full chain.")):
    """Walk every hash chain; non-zero exit if any record was modified, removed or re-signed."""
    files = _chain_files(all_chains, chain)
    if not files:
        sink = get_audit()
        good, msg = sink.verify()
        console.print(f"system of record: {msg}")
        raise typer.Exit(0 if good else 1)
    failed = 0
    for f in files:
        if not f.exists():
            console.print(f"[yellow]{f}: missing[/yellow]")
            continue
        good, msg = _log_for(f).verify()
        quarantined = ".broken-" in f.name
        colour = "green" if good else ("yellow" if quarantined else "red")
        console.print(f"[{colour}]{f}: {msg}{' (quarantined segment)' if quarantined else ''}[/{colour}]")
        failed += 0 if good or quarantined else 1
    raise typer.Exit(1 if failed else 0)


@audit_app.command("export")
def audit_export(since: Optional[str] = None, until: Optional[str] = None, fmt: str = typer.Option("jsonl", "--format"),
                 chain: Optional[Path] = None, verify: bool = typer.Option(False, "--verify"),
                 out: Optional[Path] = typer.Option(None, "--out")):
    """Export records in a time range (evidence for auditors / incident review)."""
    files = _chain_files(False, chain)
    log = _log_for(files[0]) if files else get_audit()
    if verify:
        good, msg = log.verify()
        console.print(f"chain: {msg}", stderr=True) if hasattr(console, "print") else None
        if not good:
            fail(f"chain verification failed: {msg}")
    t0, t1 = _parse_time(since), _parse_time(until)
    recs = [r for r in log.read(since=t0) if t1 is None or r.ts <= t1]
    lines = [r.model_dump_json() for r in recs] if fmt == "jsonl" else [json.dumps([r.model_dump() for r in recs], default=str)]
    text = "\n".join(lines) + "\n"
    if out:
        out.write_text(text)
        ok(f"{len(recs)} records -> {out} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    else:
        typer.echo(text, nl=False)


@audit_app.command("rotate-key")
def audit_rotate_key(out_dir: Optional[Path] = None, token: Optional[str] = TokenOpt):
    """Rotate the audit HMAC key: a record signed with the old key announces the new key's fingerprint."""
    op = operator(token, ["security-admin"])
    log = get_audit()
    new_key = secrets.token_urlsafe(48)
    fp = hashlib.sha256(new_key.encode()).hexdigest()[:16]
    log.for_principal(op, "audit.key_rotated", action="rotate", decision="applied", new_key_fingerprint=fp)
    old = log.hmac_key.decode() if log.hmac_key else None
    log.set_hmac_key(new_key)
    _write_secret("audit_hmac_key", new_key, out_dir)
    if old:
        prev = [old] + [k.decode() for k in log.previous_hmac_keys[1:]]
        _write_secret("audit_hmac_keys_previous", ",".join(prev), out_dir)
    ok(f"new audit key {fp} written; distribute 'audit_hmac_key' and 'audit_hmac_keys_previous' to every service")


@audit_app.command("reset-chain")
def audit_reset_chain(incident: str = typer.Option(...), chain: Optional[Path] = None, token: Optional[str] = TokenOpt):
    """After a confirmed break: quarantine the broken segment (kept as evidence) and start a new chain that
    references it by hash and incident id."""
    op = operator(token, ["security-admin"])
    files = _chain_files(False, chain)
    if not files or not files[0].exists():
        fail("no jsonl chain file to reset")
    f = files[0]
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    quarantined = f.with_name(f"{f.stem}.broken-{incident}-{int(time.time())}{f.suffix}")
    f.rename(quarantined)
    log = _log_for(f)
    log.record("audit.chain_reset", actor=op.sub, action="reset", decision="applied", incident=incident,
               quarantined_file=quarantined.name, quarantined_sha256=digest)
    ok(f"broken segment kept at {quarantined}; new chain started in {f}")


# =========================================================================== registry


@registry_model_app.command("register")
def registry_register(name: str, version: str, licence: str = typer.Option(...), context_length: int = typer.Option(...),
                      base: Optional[str] = typer.Option(None, "--base"), slot: str = "general",
                      availability: str = "private_gpu", type_: str = typer.Option("chat", "--type"),
                      parameters_b: Optional[float] = None, checksum: Optional[str] = None,
                      endpoint: Optional[str] = None, data_class: list[str] = typer.Option([], "--data-class"),
                      token: Optional[str] = TokenOpt):
    """Register a model version as draft (it is not routable until approved)."""
    op = operator(token, ["model-approver", "platform-admin"])
    rec: dict[str, Any] = {"name": name, "version": version, "licence": licence, "context_length": context_length,
                           "base_model": base, "slot": slot, "availability": availability, "type": type_,
                           "parameters_b": parameters_b, "weights_checksum": checksum, "endpoint": endpoint}
    if data_class:
        rec["data_classes_allowed"] = data_class
    emit(_governance().registry.register_model({k: v for k, v in rec.items() if v is not None}, op).public())


@registry_model_app.command("approve")
def registry_approve(name: str, version: str, note: str = "", token: Optional[str] = TokenOpt):
    """Approve a model version (four eyes: the registrant cannot approve)."""
    op = operator(token, ["model-approver"])
    emit(_governance().registry.approve_model(name, version, op, note=note).public())


@registry_model_app.command("deprecate")
def registry_deprecate(name: str, version: str, reason: str = "", token: Optional[str] = TokenOpt):
    op = operator(token, ["model-approver"])
    emit(_governance().registry.deprecate_model(name, version, op, reason=reason).public())


@registry_model_app.command("list")
def registry_list(status: Optional[str] = None):
    rows = [m.public() for m in _governance().registry.list_models(status=status)]
    table(rows, ["name", "version", "slot", "availability", "status", "licence", "context_length"], "models")


# runbook word order: `aiready registry approve model <name> <version>`
def _only_models(kind: str) -> None:
    if kind != "model":
        fail("only 'model' is supported here; prompts and adapters are managed through the governance API")


@registry_app.command("register")
def registry_register_verb(kind: str, name: str, version: str, licence: str = typer.Option(...),
                           context_length: int = typer.Option(...), base: Optional[str] = typer.Option(None, "--base"),
                           token: Optional[str] = TokenOpt):
    _only_models(kind)
    registry_register(name, version, licence=licence, context_length=context_length, base=base, slot="general",
                      availability="private_gpu", type_="chat", parameters_b=None, checksum=None, endpoint=None,
                      data_class=[], token=token)


@registry_app.command("approve")
def registry_approve_verb(kind: str, name: str, version: str, note: str = "", token: Optional[str] = TokenOpt):
    _only_models(kind)
    registry_approve(name, version, note=note, token=token)


@registry_app.command("deprecate")
def registry_deprecate_verb(kind: str, name: str, version: str, reason: str = "", token: Optional[str] = TokenOpt):
    _only_models(kind)
    registry_deprecate(name, version, reason=reason, token=token)


# =========================================================================== incident

NS_INCIDENTS = "incidents"


@incident_app.command("open")
def incident_open(title: str = typer.Option(...), severity: str = typer.Option("sev2"), token: Optional[str] = TokenOpt):
    op = operator(token, ["security-admin"])
    store = get_docstore()
    n = len(store.list(NS_INCIDENTS)) + 1
    inc = {"id": f"INC-{n}", "title": title, "severity": severity, "status": "open", "opened_by": op.sub,
           "opened_at": time.time(), "actions": []}
    store.put(NS_INCIDENTS, inc["id"], inc)
    get_audit().for_principal(op, "incident.opened", resource=inc["id"], decision="open", title=title, severity=severity)
    typer.echo(inc["id"])


def _incident_action(incident: str | None, action: dict[str, Any]) -> None:
    if not incident:
        return
    store = get_docstore()
    inc = store.get(NS_INCIDENTS, incident)
    if inc:
        inc["actions"].append({**action, "at": time.time()})
        store.put(NS_INCIDENTS, incident, inc)


@incident_app.command("close-route")
def incident_close_route(route: Optional[str] = None, model: Optional[str] = None,
                         egress: Optional[str] = typer.Option(None, help="Egress destination to close."),
                         reason: str = typer.Option(...), incident: Optional[str] = None, token: Optional[str] = TokenOpt):
    """Close a gateway route, a model or an egress destination immediately."""
    op = operator(token, ["security-admin"])
    res = _governance().incidents.close_route(op, route=route, model=model, egress_destination=egress, reason=reason,
                                              incident_id=incident or reason)
    _incident_action(incident, {"close_route": res})
    emit(res)


@incident_app.command("revoke")
def incident_revoke(subject: Optional[str] = None, jti: Optional[str] = None, reason: str = typer.Option(...),
                    incident: Optional[str] = None, token: Optional[str] = TokenOpt):
    op = operator(token, ["security-admin"])
    res = _governance().incidents.revoke(op, subject=subject, jti=jti, reason=reason, incident_id=incident or reason)
    _incident_action(incident, {"revoke": res})
    emit(res)


@incident_app.command("trace-impact")
def incident_trace_impact(incident: Optional[str] = None, since: str = typer.Option(...), until: Optional[str] = None,
                          subject: Optional[str] = None, model: Optional[str] = None,
                          key_id: Optional[str] = None, route: Optional[str] = None, limit: int = 200,
                          token: Optional[str] = TokenOpt):
    """Who and what was affected in a time window (from the audit log)."""
    op = operator(token, ["security-admin"])
    rep = _governance().incidents.trace_impact(op, start=_parse_time(since) or 0, end=_parse_time(until), model=model or route,
                                               user=subject, limit=limit, incident_id=incident)
    data = rep.model_dump()
    if key_id:  # API-key traffic is recorded with the key id in the gateway audit details
        data["samples"] = [s for s in data["samples"] if key_id in json.dumps(s, default=str)]
    _incident_action(incident, {"trace_impact": {k: data[k] for k in ("records", "users", "events")}})
    emit(data)


@incident_app.command("close")
def incident_close(incident: str = typer.Option(...), summary: str = typer.Option(...), token: Optional[str] = TokenOpt):
    op = operator(token, ["security-admin"])
    store = get_docstore()
    inc = store.get(NS_INCIDENTS, incident)
    if not inc:
        fail(f"unknown incident {incident}")
    inc.update(status="closed", closed_by=op.sub, closed_at=time.time(), summary=summary)
    store.put(NS_INCIDENTS, incident, inc)
    get_audit().for_principal(op, "incident.closed", resource=incident, decision="closed", summary=summary)
    ok(f"{incident} closed")


@incident_app.command("list")
def incident_list():
    rows = [v for _, v in get_docstore().list(NS_INCIDENTS)]
    table(rows, ["id", "severity", "status", "title", "opened_by"], "incidents")
