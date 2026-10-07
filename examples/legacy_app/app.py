"""The sample "existing application": a small insurance claims system.

It represents the client's untouched line-of-business application, so it must never import ``aiready``.
It has what a typical legacy system has and nothing more:

* a session-cookie login (``POST /login`` sets ``LEGACYSESSID``) and a "who am I" endpoint (``GET /api/me``),
* JSON endpoints over business functions (claims, policies, notes) that enforce the app's own rules
  (adjusters only see claims of their region; fraud investigators see all; admins may delete),
* a SQLite database with ``updated_at`` columns, a tombstone table and an outbox table,
* a folder of policy documents (``docs/``), some with ``.acl.json`` sidecars.

Run it with::

    uvicorn examples.legacy_app.app:create_app --factory --port 8000
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

from examples.legacy_app.db import hash_password, init_db, write_outbox

HERE = Path(__file__).resolve().parent
COOKIE = "LEGACYSESSID"
SESSION_TTL = 8 * 3600


class LoginBody(BaseModel):
    login: str
    password: str


class NoteBody(BaseModel):
    text: str


class StatusBody(BaseModel):
    status: str


VALID_STATUSES = {"open", "under_review", "approved", "rejected", "closed"}


def create_app(db_path: str | None = None, docs_dir: str | None = None) -> FastAPI:
    """Build the legacy app. ``db_path`` defaults to ``$LEGACY_DB`` or ``examples/legacy_app/data/legacy.db``."""
    path = db_path or os.environ.get("LEGACY_DB") or str(HERE / "data" / "legacy.db")
    conn = init_db(path)
    lock = threading.Lock()
    app = FastAPI(title="Legacy Claims App")
    app.state.db_path = path
    app.state.docs_dir = docs_dir or str(HERE / "docs")
    app.state.conn = conn

    def _user(session_id: str | None) -> sqlite3.Row:
        if not session_id:
            raise HTTPException(401, "not logged in")
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.id = ? AND s.expires_at > ?",
            (session_id, time.time())).fetchone()
        if row is None:
            raise HTTPException(401, "session expired")
        return row

    def _roles(user: sqlite3.Row) -> list[str]:
        return list(json.loads(user["roles"]))

    def _can_see(user: sqlite3.Row, claim: sqlite3.Row) -> bool:
        roles = _roles(user)
        return "fraud_investigator" in roles or "administrator" in roles or claim["region"] == user["region"]

    def _claim_row(number: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM claims WHERE claim_number = ?", (number,)).fetchone()
        if row is None:
            raise HTTPException(404, "claim not found")
        return row

    @app.post("/login")
    def login(body: LoginBody, response: Response) -> dict[str, Any]:
        user = conn.execute("SELECT * FROM users WHERE login = ?", (body.login,)).fetchone()
        if user is None or hash_password(body.password, user["salt"]) != user["password_hash"]:
            raise HTTPException(401, "invalid credentials")
        sid = secrets.token_urlsafe(24)
        with lock:
            conn.execute("INSERT INTO sessions (id, user_id, created_at, expires_at) VALUES (?,?,?,?)",
                         (sid, user["id"], time.time(), time.time() + SESSION_TTL))
            conn.commit()
        response.set_cookie(COOKIE, sid, httponly=True, samesite="lax")
        return {"ok": True, "login": user["login"]}

    @app.post("/logout")
    def logout(response: Response, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        if LEGACYSESSID:
            with lock:
                conn.execute("DELETE FROM sessions WHERE id = ?", (LEGACYSESSID,))
                conn.commit()
        response.delete_cookie(COOKIE)
        return {"ok": True}

    @app.get("/api/me")
    def me(LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        """The app's own "who am I" endpoint - the identity bridge introspects sessions through it."""
        u = _user(LEGACYSESSID)
        return {
            "user": {"login": u["login"], "email": u["email"], "display_name": u["display_name"]},
            "roles": _roles(u),
            "department": u["department"],
            "region": u["region"],
            "company": "acme",
        }

    @app.get("/api/claims")
    def list_claims(status: str | None = None, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=500),
                    LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        u = _user(LEGACYSESSID)
        rows = conn.execute("SELECT * FROM claims ORDER BY id").fetchall()
        visible = [dict(r) for r in rows if _can_see(u, r) and (status is None or r["status"] == status)]
        start = (page - 1) * page_size
        items = visible[start:start + page_size]
        has_more = start + page_size < len(visible)
        return {"items": items, "page": page, "next_page": page + 1 if has_more else None, "total": len(visible)}

    @app.get("/api/claims/{number}")
    def get_claim(number: str, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        u = _user(LEGACYSESSID)
        row = _claim_row(number)
        if not _can_see(u, row):
            raise HTTPException(403, "not allowed to see this claim")
        notes = conn.execute("SELECT author, text, created_at FROM claim_notes WHERE claim_id = ? ORDER BY id",
                             (row["id"],)).fetchall()
        return {**dict(row), "notes": [dict(n) for n in notes]}

    @app.post("/api/claims/{number}/notes")
    def add_note(number: str, body: NoteBody, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        u = _user(LEGACYSESSID)
        row = _claim_row(number)
        if not _can_see(u, row):
            raise HTTPException(403, "not allowed to update this claim")
        now = time.time()
        with lock:
            cur = conn.execute("INSERT INTO claim_notes (claim_id, author, text, created_at) VALUES (?,?,?,?)",
                               (row["id"], u["login"], body.text, now))
            conn.execute("UPDATE claims SET updated_at = ? WHERE id = ?", (now, row["id"]))
            write_outbox(conn, "claim", row["id"], "note_added", {"claim_number": number, "author": u["login"]})
            conn.commit()
        return {"ok": True, "note_id": cur.lastrowid, "author": u["login"]}

    @app.put("/api/claims/{number}/status")
    def set_status(number: str, body: StatusBody, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        u = _user(LEGACYSESSID)
        if body.status not in VALID_STATUSES:
            raise HTTPException(422, f"status must be one of {sorted(VALID_STATUSES)}")
        row = _claim_row(number)
        if not _can_see(u, row):
            raise HTTPException(403, "not allowed to update this claim")
        with lock:
            conn.execute("UPDATE claims SET status = ?, updated_at = ? WHERE id = ?", (body.status, time.time(), row["id"]))
            write_outbox(conn, "claim", row["id"], "status_changed", {"claim_number": number, "status": body.status})
            conn.commit()
        return {"ok": True, "claim_number": number, "status": body.status}

    @app.delete("/api/claims/{number}")
    def delete_claim(number: str, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        u = _user(LEGACYSESSID)
        if "administrator" not in _roles(u):
            raise HTTPException(403, "administrators only")
        row = _claim_row(number)
        with lock:
            conn.execute("DELETE FROM claim_notes WHERE claim_id = ?", (row["id"],))
            conn.execute("DELETE FROM claims WHERE id = ?", (row["id"],))
            write_outbox(conn, "claim", row["id"], "deleted", {"claim_number": number})
            conn.commit()
        return {"ok": True}

    @app.get("/api/policies/{number}")
    def get_policy(number: str, LEGACYSESSID: str | None = Cookie(default=None)) -> dict[str, Any]:
        _user(LEGACYSESSID)
        row = conn.execute("SELECT * FROM policies WHERE number = ?", (number,)).fetchone()
        if row is None:
            raise HTTPException(404, "policy not found")
        return dict(row)

    return app
