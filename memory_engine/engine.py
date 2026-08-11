"""Core engine for the Memory Engine: locking, transactions, and dispatch.

This is the only module in the package that acquires locks or opens
SQLite transactions directly -- ``store.py``, ``concurrency.py``,
``facts.py``, and ``recall.py`` are all called from here under that
discipline.

Concurrency summary (see the project's memory_engine plan for the full
reasoning): WAL mode + ``busy_timeout`` (store.py) handle cross-process
write serialization the way ``db_engine``'s ``filelock`` does; a
per-session in-process lock (``concurrency.py``) handles same-process
thread contention; every write opens with ``BEGIN IMMEDIATE`` to avoid the
classic SQLite deferred-reader-upgrade race. Reads take no lock at all.
Because this package makes no generative LLM calls of its own, it never
needs to hold a lock open across a slow external call -- a future agent
calling ``get_recent_context`` then an LLM then ``upsert_facts`` gets that
same safety for free from three independently-locked primitives, with no
special-casing needed here. The one exception is the per-turn recall
embedding (``recall.py``), which does still follow an
unlock/embed/relock shape, since that embedding call stays inside this
package.
"""

import json
import logging
import os
import random
import sqlite3
import threading
import time
from typing import Optional

from memory_engine.concurrency import session_lock
from memory_engine.facts import needs_extraction as facts_needs_extraction
from memory_engine.facts import upsert_facts as facts_upsert_facts
from memory_engine.recall import embed_text
from memory_engine.recall import recall_related as compute_recall_related
from memory_engine.store import checkpoint, ensure_session, get_connection, resolve_user_id_readonly

logger = logging.getLogger(__name__)

_VALID_ROLES = ("system", "user", "assistant")
_NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def _configure_logging(cfg: dict) -> None:
    """Attaches a file handler to the ``memory_engine`` logger hierarchy, once.

    Every submodule logs via ``logging.getLogger(__name__)`` (e.g.
    ``memory_engine.facts``), which propagates up to the ``memory_engine``
    logger by default -- so attaching one handler here at
    ``cfg["memory"]["log_file"]`` is enough to persist warnings from
    anywhere in the package (invalid fact_key rejections, failed
    embedding calls, etc.), instead of them only reaching Python's default
    stderr handler-of-last-resort and never being written anywhere.
    """
    root = logging.getLogger("memory_engine")
    if root.handlers:
        return
    log_path = cfg["memory"]["log_file"]
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.WARNING)


class MemoryEngine:
    """Owns a config and dispatches to the storage/recall/facts modules under lock."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        _configure_logging(cfg)

    def append_message(
        self, session_id: str, role: str, content: str, metadata: Optional[dict] = None
    ) -> None:
        """Appends one message to a session's transcript.

        Increments the session's turn bookkeeping on ``role='user'``, and
        triggers a best-effort recall embedding for the just-completed
        turn on ``role='assistant'``.

        Args:
            session_id: The conversation session to append to.
            role: One of ``'system'``, ``'user'``, ``'assistant'``.
            content: The message text.
            metadata: Optional small JSON-serializable dict stored alongside the message.

        Raises:
            ValueError: If role is not one of the three valid roles.
        """
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid role: {role!r}")
        metadata_json = json.dumps(metadata) if metadata else None

        def _write(conn: sqlite3.Connection):
            ensure_session(conn, session_id, self._cfg)
            if role == "user":
                conn.execute(
                    f"UPDATE sessions SET turn_counter = turn_counter + 1, "
                    f"turns_since_extraction = turns_since_extraction + 1, "
                    f"last_active_at = {_NOW_SQL} WHERE session_id = ?",
                    (session_id,),
                )
                turn_number = conn.execute(
                    "SELECT turn_counter FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            elif role == "assistant":
                conn.execute(
                    f"UPDATE sessions SET last_active_at = {_NOW_SQL} WHERE session_id = ?",
                    (session_id,),
                )
                turn_number = conn.execute(
                    "SELECT turn_counter FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            else:
                turn_number = None
            conn.execute(
                f"INSERT INTO messages (session_id, turn_number, role, content, metadata_json, created_at) "
                f"VALUES (?, ?, ?, ?, ?, {_NOW_SQL})",
                (session_id, turn_number, role, content, metadata_json),
            )
            return turn_number

        with session_lock(session_id):
            turn_number = self._run_write_transaction(_write)

        if role == "assistant" and turn_number is not None:
            self._maybe_embed_turn(session_id, turn_number)

    def get_recent_context(
        self, session_id: str, turns: Optional[int] = None, include_system: bool = False
    ) -> list[dict]:
        """Returns the most recent turns for a session, in chronological order.

        Replaces ``old/create_chatbot.py``'s buggy ``get_previous_messages``:
        turn pairing comes from the ``turn_number`` column via SQL, not
        list-slicing arithmetic, so there's no bound-check to get wrong.

        Args:
            session_id: The session to read from.
            turns: How many recent turns to include; defaults to
                ``cfg["memory"]["default_context_window_turns"]``.
            include_system: Whether to also include ``role='system'``
                messages (which aren't part of any turn).

        Returns:
            ``[{"turn_number": int|None, "role": str, "content": str,
            "created_at": str}, ...]`` in chronological order. Empty list
            if the session has no messages yet.
        """
        turns = turns or self._cfg["memory"]["default_context_window_turns"]
        conn = get_connection(self._cfg)
        turn_numbers = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT turn_number FROM messages "
                "WHERE session_id = ? AND turn_number IS NOT NULL "
                "ORDER BY turn_number DESC LIMIT ?",
                (session_id, turns),
            ).fetchall()
        ]
        clauses = []
        params: list = [session_id]
        if turn_numbers:
            clauses.append(f"turn_number IN ({','.join('?' * len(turn_numbers))})")
            params.extend(turn_numbers)
        if include_system:
            clauses.append("role = 'system'")
        if not clauses:
            return []
        where = " OR ".join(clauses)
        rows = conn.execute(
            f"SELECT turn_number, role, content, created_at FROM messages "
            f"WHERE session_id = ? AND ({where}) ORDER BY message_id",
            params,
        ).fetchall()
        return [{"turn_number": r[0], "role": r[1], "content": r[2], "created_at": r[3]} for r in rows]

    def get_profile_facts(self, session_id: str) -> dict[str, str]:
        """Returns this session's user's long-term facts as ``{key: value}``."""
        conn = get_connection(self._cfg)
        user_id = resolve_user_id_readonly(conn, session_id, self._cfg)
        rows = conn.execute("SELECT fact_key, fact_value FROM facts WHERE user_id = ?", (user_id,)).fetchall()
        return {key: value for key, value in rows}

    def recall_related(self, session_id: str, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Finds relevant turns from this user's OTHER sessions. See ``recall.py``."""
        conn = get_connection(self._cfg)
        return compute_recall_related(conn, session_id, query, top_k, self._cfg)

    def needs_extraction(self, session_id: str) -> bool:
        """True once enough user turns have accumulated since the last extraction ack."""
        conn = get_connection(self._cfg)
        return facts_needs_extraction(conn, session_id, self._cfg)

    def upsert_facts(self, session_id: str, facts: list[dict], turns_covered: int) -> dict:
        """Atomically upserts extracted facts and acknowledges the extraction attempt.

        Must be called once per extraction attempt (even with ``facts=[]``
        on failure) -- see the ack contract documented in ``facts.py``.
        """

        def _write(conn: sqlite3.Connection):
            return facts_upsert_facts(conn, session_id, facts, turns_covered, self._cfg)

        with session_lock(session_id):
            return self._run_write_transaction(_write)

    def clear_session(self, session_id: str) -> None:
        """Deletes this session's messages and embeddings; leaves facts untouched.

        Facts are user-scoped and meant to outlive an individual session
        clear -- only ``messages``, ``message_embeddings``, and this
        session's counters are reset.
        """

        def _write(conn: sqlite3.Connection):
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_embeddings WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET turn_counter = 0, turns_since_extraction = 0, "
                "last_extracted_at = NULL, rolling_summary = NULL WHERE session_id = ?",
                (session_id,),
            )

        with session_lock(session_id):
            self._run_write_transaction(_write)
        checkpoint(get_connection(self._cfg))

    def get_rolling_summary(self, session_id: str) -> Optional[str]:
        """Returns this session's rolling summary, or None if unset or the session doesn't exist.

        Read-only, no lock -- same pattern as ``get_profile_facts``.
        """
        conn = get_connection(self._cfg)
        row = conn.execute("SELECT rolling_summary FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else None

    def set_rolling_summary(self, session_id: str, summary: str) -> None:
        """Atomically overwrites this session's rolling summary.

        Intended for a future agent's own compression pass (e.g. folding
        older turns into a running summary once a session gets long) --
        this method only stores whatever text it's given; it never
        generates or compresses anything itself, same boundary as the
        rest of this package. Creates the session row if it doesn't exist
        yet.
        """

        def _write(conn: sqlite3.Connection):
            ensure_session(conn, session_id, self._cfg)
            conn.execute(
                "UPDATE sessions SET rolling_summary = ? WHERE session_id = ?", (summary, session_id)
            )

        with session_lock(session_id):
            self._run_write_transaction(_write)

    def _maybe_embed_turn(self, session_id: str, turn_number: int) -> None:
        """Best-effort: embeds a just-completed turn and stores it for recall.

        Runs the embedding call outside any lock (it's the slow part),
        then re-acquires the session lock only for the short atomic write.
        A failed embedding call is logged and skipped -- the turn's raw
        text is already safely committed in ``messages`` regardless.
        """
        conn = get_connection(self._cfg)
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND turn_number = ? ORDER BY message_id",
            (session_id, turn_number),
        ).fetchall()
        user_content = next((c for r, c in rows if r == "user"), None)
        assistant_content = next((c for r, c in rows if r == "assistant"), None)
        if user_content is None or assistant_content is None:
            return

        try:
            vector = embed_text(f"Q: {user_content}\nA: {assistant_content}", self._cfg)
        except Exception:
            logger.warning("recall embedding failed for session=%s turn=%s", session_id, turn_number)
            return

        def _write(conn: sqlite3.Connection):
            user_id = resolve_user_id_readonly(conn, session_id, self._cfg)
            conn.execute(
                f"INSERT OR REPLACE INTO message_embeddings "
                f"(session_id, turn_number, user_id, embedding, embedding_dim, created_at) "
                f"VALUES (?, ?, ?, ?, ?, {_NOW_SQL})",
                (session_id, turn_number, user_id, vector.tobytes(), vector.shape[0]),
            )

        with session_lock(session_id):
            self._run_write_transaction(_write)

    def _run_write_transaction(self, fn):
        """Runs ``fn(conn)`` inside a ``BEGIN IMMEDIATE`` transaction.

        Retries with jitter on lock contention
        (``cfg["memory"]["write_retry_attempts"]``) before surfacing
        ``sqlite3.OperationalError``. Any other exception raised by ``fn``
        rolls back and propagates immediately, without retrying.
        """
        conn = get_connection(self._cfg)
        attempts = self._cfg["memory"]["write_retry_attempts"]
        last_error: Optional[sqlite3.OperationalError] = None
        for attempt in range(attempts):
            try:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(conn)
                    conn.execute("COMMIT")
                    return result
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            except sqlite3.OperationalError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1) + random.uniform(0, 0.05))
        raise last_error


_engine_instance: Optional[MemoryEngine] = None
_engine_init_lock = threading.Lock()


def get_engine(cfg: Optional[dict] = None) -> MemoryEngine:
    """Returns the process-wide ``MemoryEngine`` singleton, creating it if needed.

    The first caller's ``cfg`` (or the default from ``config.read_config()``)
    wins for the process lifetime -- mirrors ``db_engine.engine.get_engine()``.
    ``MemoryEngine`` itself stays directly instantiable so tests can build
    isolated instances.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_init_lock:
            if _engine_instance is None:
                from memory_engine.config import read_config

                _engine_instance = MemoryEngine(cfg or read_config())
    return _engine_instance
