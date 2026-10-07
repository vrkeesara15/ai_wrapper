"""Shared CLI plumbing: operator identity, async runner, output helpers.

Operator identity. Every governed CLI action is performed *as a person* and audited under their name:
  * ``AIREADY_TOKEN`` (or ``--token``): a token from the identity bridge / IdP - the normal path;
  * break-glass: when no token is given but the token signing key is available in this environment
    (``AIREADY_SECRET_TOKEN_SIGNING_KEY`` / mounted secret), a short-lived operator token is minted for
    ``$USER`` with the roles passed in ``--as-role``. This is written to the audit log as ``cli.break_glass``
    so it is visible in access reviews.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import sys
from typing import Any, Awaitable, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from aiready.core.audit import get_audit
from aiready.core.errors import AIReadyError
from aiready.core.identity import Principal, get_token_service

T = TypeVar("T")
console = Console()
err_console = Console(stderr=True)

OPERATOR_ROLES = ["platform-admin", "security-admin", "model-approver", "ingestion-admin", "admin"]


def run(coro: Awaitable[T]) -> T:
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except AIReadyError as exc:
        fail(f"{exc.code}: {exc.message}" + (f" {json.dumps(exc.details, default=str)}" if exc.details else ""))
        raise  # unreachable


def fail(message: str, code: int = 1) -> None:
    err_console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def ok(message: str) -> None:
    console.print(f"[green]ok:[/green] {message}")


def emit(data: Any, *, as_json: bool = True) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    if isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        data = [d.model_dump(mode="json") for d in data]
    if as_json or not sys.stdout.isatty():
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        console.print(data)


def table(rows: list[dict[str, Any]], columns: list[str], title: str | None = None) -> None:
    t = Table(title=title)
    for c in columns:
        t.add_column(c)
    for r in rows:
        t.add_row(*[str(r.get(c, "")) for c in columns])
    console.print(t)


def operator(token: str | None = None, roles: list[str] | None = None) -> Principal:
    ts = get_token_service()
    tok = token or os.environ.get("AIREADY_TOKEN")
    if tok:
        return ts.verify(tok)
    user = os.environ.get("AIREADY_OPERATOR") or getpass.getuser()
    granted = roles or OPERATOR_ROLES
    try:
        minted = ts.issue(sub=user, roles=granted, actor="cli:break-glass", ttl=300)
    except Exception as exc:  # no signing key here: the operator must bring a token
        fail(f"no operator token (set AIREADY_TOKEN) and no signing key for break-glass: {exc}")
        raise
    p = ts.verify(minted)
    get_audit().for_principal(p, "cli.break_glass", action="operator.token", decision="issued", roles=granted,
                              argv=sys.argv[1:])
    return p
