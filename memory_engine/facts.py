"""Fact bookkeeping and the extraction-ack primitive for the Memory Engine.

Owns the ``turns_since_extraction`` counter/threshold and the atomic
UPSERT into the ``facts`` table. This is the data-layer half of "insight
extraction" -- the reasoning half (prompt, LLM call, JSON parsing) belongs
to the future ``agents`` package's insight-extractor, never to this
module (see ``db_engine``'s own precedent of embedding for storage but
never generating text).

Ack contract: whoever attempts extraction must call :func:`upsert_facts`
exactly once when done, even with an empty ``facts`` list on failure.
``turns_since_extraction`` only goes down when this function runs -- not
when the LLM call happens to succeed -- the same way an at-least-once
queue consumer acks a message once it's handled, whether or not the
handling produced a useful result. Skipping the ack leaves
``needs_extraction`` stuck ``True`` on every subsequent turn, reintroducing
the per-turn-LLM-call cost bug this whole design exists to avoid.
"""

import logging
import re
import sqlite3

from memory_engine.store import resolve_user_id

logger = logging.getLogger(__name__)

_FACT_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_NOW_SQL = "strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def needs_extraction(conn: sqlite3.Connection, session_id: str, cfg: dict) -> bool:
    """True once ``turns_since_extraction`` has crossed the configured threshold.

    Args:
        conn: An open, unlocked read connection.
        session_id: The session to check.
        cfg: The loaded config dict; reads ``cfg["memory"]["extraction_interval_turns"]``.

    Returns:
        False if the session doesn't exist yet.
    """
    row = conn.execute(
        "SELECT turns_since_extraction FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return False
    return row[0] >= cfg["memory"]["extraction_interval_turns"]


def upsert_facts(
    conn: sqlite3.Connection, session_id: str, facts: list[dict], turns_covered: int, cfg: dict
) -> dict:
    """Atomically upserts facts and decrements the extraction counter.

    Must be called inside an already-open write transaction (see
    ``engine.py``'s write-transaction helper) -- this function issues no
    transaction control of its own.

    Args:
        conn: An open connection inside a ``BEGIN IMMEDIATE`` transaction.
        session_id: The session the extraction attempt was run against.
        facts: ``[{"key": str, "value": str, "confidence": float}, ...]``.
            May be empty -- an empty list is a valid, expected
            acknowledgment of a failed or fact-less extraction attempt.
        turns_covered: How many user turns this attempt's snapshot
            covered; subtracted from ``turns_since_extraction`` (never
            blindly reset to 0), so turns that arrived mid-attempt stay
            correctly pending for the next trigger.
        cfg: The loaded config dict.

    Returns:
        ``{"upserted": int}`` -- the number of facts actually written
        after ``fact_key`` validation.
    """
    user_id = resolve_user_id(conn, session_id, cfg)
    upserted = 0
    for fact in facts:
        key = fact.get("key", "")
        if not _FACT_KEY_RE.match(key):
            logger.warning("rejecting invalid fact_key=%r for session=%s", key, session_id)
            continue
        conn.execute(
            f"INSERT INTO facts (user_id, fact_key, fact_value, confidence, source_session_id, "
            f"source_turn_number, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, {_NOW_SQL}, {_NOW_SQL}) "
            f"ON CONFLICT(user_id, fact_key) DO UPDATE SET "
            f"fact_value = excluded.fact_value, confidence = excluded.confidence, "
            f"source_session_id = excluded.source_session_id, "
            f"source_turn_number = excluded.source_turn_number, updated_at = excluded.updated_at",
            (
                user_id,
                key,
                fact.get("value", ""),
                fact.get("confidence", 1.0),
                session_id,
                fact.get("turn_number"),
            ),
        )
        upserted += 1
    conn.execute(
        f"UPDATE sessions SET turns_since_extraction = MAX(0, turns_since_extraction - ?), "
        f"last_extracted_at = {_NOW_SQL} WHERE session_id = ?",
        (turns_covered, session_id),
    )
    return {"upserted": upserted}
