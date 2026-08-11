"""SQLite storage layer for the Memory Engine.

Owns the connection lifecycle, WAL-mode/busy-timeout setup, and schema DDL
for conversation messages, long-term facts, and per-turn recall
embeddings. Concurrency guarantees are layered on top in
``concurrency.py`` (in-process locking) and ``engine.py`` (transaction
discipline) -- this module only owns "how do I get a correctly configured
connection."

WAL mode plus ``busy_timeout`` is this package's substitute for
``db_engine``'s cross-process ``filelock``-based write lock: SQLite
serializes concurrent writers itself, blocking and retrying internally on
``SQLITE_BUSY`` up to ``busy_timeout``, rather than a hand-rolled file
lock.
"""

import os
import sqlite3
import threading
import time
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id             TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL DEFAULT 'default',
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    last_active_at           TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    turn_counter               INTEGER NOT NULL DEFAULT 0,
    turns_since_extraction      INTEGER NOT NULL DEFAULT 0,
    last_extracted_at            TEXT,
    rolling_summary                TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    message_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL REFERENCES sessions(session_id),
    turn_number           INTEGER,
    role                     TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
    content                    TEXT NOT NULL,
    metadata_json                 TEXT,
    created_at                       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_messages_session_turn ON messages(session_id, turn_number, message_id);

CREATE TABLE IF NOT EXISTS facts (
    user_id               TEXT NOT NULL,
    fact_key                 TEXT NOT NULL,
    fact_value                  TEXT NOT NULL,
    confidence                     REAL NOT NULL DEFAULT 1.0,
    source_session_id                 TEXT NOT NULL,
    source_turn_number                   INTEGER,
    created_at                              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at                                 TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (user_id, fact_key)
);

CREATE TABLE IF NOT EXISTS message_embeddings (
    session_id         TEXT NOT NULL REFERENCES sessions(session_id),
    turn_number            INTEGER NOT NULL,
    user_id                   TEXT NOT NULL,
    embedding                    BLOB NOT NULL,
    embedding_dim                    INTEGER NOT NULL,
    created_at                          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (session_id, turn_number)
);
CREATE INDEX IF NOT EXISTS idx_message_embeddings_user ON message_embeddings(user_id);
"""

_thread_local = threading.local()


def _set_wal_mode(conn: sqlite3.Connection, attempts: int = 3) -> None:
    """Switches a (possibly brand-new) database file into WAL mode.

    Two processes creating the file for the first time can transiently
    collide on this specific pragma, so retry a few times with a short
    backoff before giving up.

    Args:
        conn: The connection to switch into WAL mode.
        attempts: Number of attempts before re-raising the last error.

    Raises:
        sqlite3.OperationalError: If every attempt fails.
    """
    last_error: Optional[sqlite3.OperationalError] = None
    for attempt in range(attempts):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error


def _open_connection(cfg: dict) -> sqlite3.Connection:
    db_path = cfg["memory"]["db_path"]
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=True)
    conn.execute(f"PRAGMA busy_timeout={cfg['memory']['busy_timeout_ms']}")
    _set_wal_mode(conn)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA)
    return conn


def get_connection(cfg: dict) -> sqlite3.Connection:
    """Returns this thread's SQLite connection, opening and configuring it lazily.

    One connection is kept per thread (via ``threading.local``) and reused
    across calls rather than reopened every time.

    Args:
        cfg: The loaded config dict (see ``config.read_config``).

    Returns:
        A WAL-mode, busy-timeout-configured connection private to the
        calling thread, with the schema already applied.
    """
    if not hasattr(_thread_local, "conn"):
        _thread_local.conn = _open_connection(cfg)
    return _thread_local.conn


def checkpoint(conn: sqlite3.Connection) -> None:
    """Truncates the WAL file back into the main database file.

    Called opportunistically after fact-upsert cycles and on
    ``clear_session`` so the ``-wal`` sidecar doesn't grow unbounded over a
    long-running process.
    """
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def ensure_session(conn: sqlite3.Connection, session_id: str, cfg: dict) -> None:
    """Creates the session row if it doesn't already exist (idempotent).

    Must be called from inside an already-open write transaction.
    """
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (?, ?)",
        (session_id, cfg["memory"]["default_user_id"]),
    )


def resolve_user_id(conn: sqlite3.Connection, session_id: str, cfg: dict) -> str:
    """Resolves ``user_id`` for a write path, creating the session row if needed.

    Must be called from inside an already-open write transaction --
    unlike :func:`resolve_user_id_readonly`, this has the side effect of
    inserting a session row.
    """
    ensure_session(conn, session_id, cfg)
    return conn.execute(
        "SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()[0]


def resolve_user_id_readonly(conn: sqlite3.Connection, session_id: str, cfg: dict) -> str:
    """Resolves ``user_id`` for a read path without creating a session row.

    Falls back to ``cfg["memory"]["default_user_id"]`` if the session
    doesn't exist yet, so read paths never need a lock or a write.
    """
    row = conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    return row[0] if row else cfg["memory"]["default_user_id"]
