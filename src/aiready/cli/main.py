"""``aiready`` - the operator CLI.

    aiready serve all --mock-models          # pilot: every service in one process, offline models
    aiready config validate                  # schema + policy compilation of the merged config
    aiready policy test                      # CI gate for policy changes
    aiready eval run claims --release 1.4.0  # offline evaluation; `eval gate` blocks a release that scores lower
    aiready audit verify --all-chains        # tamper-evidence check
    aiready token rotate --stage add|activate|retire
    aiready registry model approve <name> <version>
    aiready incident open|revoke|close-route|trace-impact|close
    aiready ingest run --all-sources | dlq list | dlq replay | verify
    aiready sizing worksheet | cost breakeven | bundle build|verify|import
"""

from __future__ import annotations

import os

import typer

from aiready import __version__
from aiready.cli import apps, governance, ops

os.environ.setdefault("AIREADY_SETUP_LOGGING", "0")
os.environ.setdefault("AIREADY_CONFIG_WATCH", "0")

app = typer.Typer(name="aiready", help="AI-Ready platform operator CLI.", no_args_is_help=True,
                  pretty_exceptions_show_locals=False)

app.command("serve")(ops.serve)
app.add_typer(ops.config_app, name="config")
app.add_typer(apps.app_app, name="app")
app.add_typer(governance.policy_app, name="policy")
app.add_typer(governance.token_app, name="token")
app.add_typer(governance.audit_app, name="audit")
app.add_typer(governance.registry_app, name="registry")
app.add_typer(governance.incident_app, name="incident")
app.add_typer(ops.ingest_app, name="ingest")
app.add_typer(ops.sizing_app, name="sizing")
app.add_typer(ops.cost_app, name="cost")
app.add_typer(ops.bundle_app, name="bundle")

try:
    from aiready.evals.cli import evals_app

    app.add_typer(evals_app, name="eval")
except ImportError:  # pragma: no cover
    pass


@app.command()
def version() -> None:
    """Print the platform version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
