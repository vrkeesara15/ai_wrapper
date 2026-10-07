"""Database of the sample legacy claims application.

This stands for a client's existing system of record. It deliberately knows nothing about AI-Ready:
the platform reads it through a replica (polling with ``updated_at``), a tombstone table filled by
triggers, and an outbox table - all of which are typical of real line-of-business databases.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    login TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    email TEXT,
    display_name TEXT,
    roles TEXT NOT NULL DEFAULT '[]',        -- JSON list of application roles
    department TEXT,
    region TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY,
    number TEXT UNIQUE NOT NULL,
    holder_name TEXT NOT NULL,
    product TEXT NOT NULL,
    premium REAL NOT NULL,
    status TEXT NOT NULL,
    region TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY,
    claim_number TEXT UNIQUE NOT NULL,
    policy_id INTEGER NOT NULL REFERENCES policies(id),
    claimant TEXT NOT NULL,
    status TEXT NOT NULL,
    amount REAL NOT NULL,
    description TEXT NOT NULL,
    region TEXT NOT NULL,
    handler TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claim_notes (
    id INTEGER PRIMARY KEY,
    claim_id INTEGER NOT NULL REFERENCES claims(id),
    author TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
-- Tombstones written by triggers so replicas can see deletes.
CREATE TABLE IF NOT EXISTS deleted_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    deleted_at REAL NOT NULL
);
-- Transactional outbox: the application records business events next to its writes.
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    aggregate TEXT NOT NULL,
    aggregate_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS claims_tombstone AFTER DELETE ON claims BEGIN
    INSERT INTO deleted_records (table_name, record_id, deleted_at) VALUES ('claims', OLD.id, strftime('%s','now'));
END;
CREATE TRIGGER IF NOT EXISTS policies_tombstone AFTER DELETE ON policies BEGIN
    INSERT INTO deleted_records (table_name, record_id, deleted_at) VALUES ('policies', OLD.id, strftime('%s','now'));
END;
"""

SEED_USERS = [
    # login, password, email, display name, roles, department, region
    ("alice", "alice-pw", "alice@example.com", "Alice Adjuster", ["adjuster"], "claims", "north"),
    ("bob", "bob-pw", "bob@example.com", "Bob Investigator", ["adjuster", "fraud_investigator"], "claims", "south"),
    ("carol", "carol-pw", "carol@example.com", "Carol Service", ["customer_service"], "service", "north"),
    ("dave", "dave-pw", "dave@example.com", "Dave Admin", ["administrator"], "it", "north"),
]

SEED_POLICIES = [
    ("POL-1001", "Jane Smith", "home", 820.0, "active", "north"),
    ("POL-1002", "Omar Haddad", "motor", 640.0, "active", "south"),
    ("POL-1003", "Lin Wei", "home", 910.0, "lapsed", "north"),
]

SEED_CLAIMS = [
    ("CLM-2001", 1, "Jane Smith", "open", 4200.0, "Water damage in the kitchen after a burst pipe.", "north", "alice"),
    ("CLM-2002", 2, "Omar Haddad", "under_review", 12800.0, "Rear-end collision on the ring road; vehicle towed.",
     "south", "bob"),
    ("CLM-2003", 3, "Lin Wei", "closed", 650.0, "Stolen bicycle from the garden shed.", "north", "alice"),
]


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str | os.PathLike[str], *, seed: bool = True) -> sqlite3.Connection:
    """Create the schema and (once) seed demo data."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    conn.executescript(SCHEMA)
    if seed and conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        now = time.time()
        for login, pw, email, name, roles, dept, region in SEED_USERS:
            salt = secrets.token_hex(8)
            conn.execute(
                "INSERT INTO users (login, password_hash, salt, email, display_name, roles, department, region) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (login, hash_password(pw, salt), salt, email, name, json.dumps(roles), dept, region),
            )
        for number, holder, product, premium, status, region in SEED_POLICIES:
            conn.execute(
                "INSERT INTO policies (number, holder_name, product, premium, status, region, updated_at) "
                "VALUES (?,?,?,?,?,?,?)", (number, holder, product, premium, status, region, now))
        for number, pid, claimant, status, amount, desc, region, handler in SEED_CLAIMS:
            cur = conn.execute(
                "INSERT INTO claims (claim_number, policy_id, claimant, status, amount, description, region, handler,"
                " updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (number, pid, claimant, status, amount, desc, region, handler, now))
            write_outbox(conn, "claim", int(cur.lastrowid or 0), "created", {"claim_number": number})
        conn.commit()
    return conn


def write_outbox(conn: sqlite3.Connection, aggregate: str, aggregate_id: int, event_type: str,
                 payload: dict) -> None:
    conn.execute("INSERT INTO outbox (aggregate, aggregate_id, event_type, payload, created_at) VALUES (?,?,?,?,?)",
                 (aggregate, aggregate_id, event_type, json.dumps(payload), time.time()))
