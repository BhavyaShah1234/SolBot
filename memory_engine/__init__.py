"""The Memory Engine: SolBot's conversation memory component.

Storage and retrieval for conversation memory: appending messages,
fetching recent same-session context, recalling relevant exchanges from a
user's other sessions, and persisting long-term personalization facts.

This package makes no generative LLM calls of its own -- contextualizing a
follow-up query and reasoning over a conversation to extract insights are
both deferred to a future ``agents`` package. This mirrors ``db_engine``'s
own shape (it embeds for storage but never generates text): ``get_recent_context``,
``needs_extraction``, and ``upsert_facts`` are the data-layer primitives a
future contextualizer and insight-extractor will call, the same way
``db_engine.fetch()`` is the primitive a future worker agent calls for
retrieval before generating an answer.
"""

from typing import Optional

from memory_engine.config import read_config
from memory_engine.engine import MemoryEngine, get_engine

__all__ = [
    "append_message",
    "get_recent_context",
    "get_profile_facts",
    "recall_related",
    "needs_extraction",
    "upsert_facts",
    "clear_session",
    "read_config",
    "MemoryEngine",
]


def append_message(
    session_id: str,
    role: str,
    content: str,
    cfg: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> None:
    """Appends a message to a session's transcript.

    Args:
        session_id: The conversation session to append to.
        role: One of ``'system'``, ``'user'``, ``'assistant'``.
        content: The message text.
        cfg: Optional config override; defaults to ``read_config()``.
        metadata: Optional small JSON-serializable dict stored alongside the message.

    Raises:
        ValueError: If role is not one of the three valid roles.
    """
    get_engine(cfg).append_message(session_id, role, content, metadata=metadata)


def get_recent_context(
    session_id: str,
    turns: Optional[int] = None,
    cfg: Optional[dict] = None,
    include_system: bool = False,
) -> list[dict]:
    """Returns the most recent turns for a session, in chronological order.

    Args:
        session_id: The session to read from.
        turns: How many recent turns to include; defaults to
            ``cfg["memory"]["default_context_window_turns"]``.
        cfg: Optional config override; defaults to ``read_config()``.
        include_system: Whether to also include ``role='system'`` messages.

    Returns:
        ``[{"turn_number": int|None, "role": str, "content": str,
        "created_at": str}, ...]``.
    """
    return get_engine(cfg).get_recent_context(session_id, turns=turns, include_system=include_system)


def get_profile_facts(session_id: str, cfg: Optional[dict] = None) -> dict[str, str]:
    """Returns this session's user's long-term facts as ``{key: value}``.

    Resolves ``session_id`` to a ``user_id`` internally, so facts survive
    across different sessions for the same user.
    """
    return get_engine(cfg).get_profile_facts(session_id)


def recall_related(
    session_id: str, query: str, top_k: Optional[int] = None, cfg: Optional[dict] = None
) -> list[dict]:
    """Finds relevant turns from this user's OTHER sessions.

    Args:
        session_id: The current session -- always excluded from results.
        query: The text to search for semantically similar past turns.
        top_k: Max results to return; defaults to ``cfg["memory"]["recall_top_k"]``.
        cfg: Optional config override; defaults to ``read_config()``.

    Returns:
        Turns scoring above ``cfg["memory"]["recall_score_threshold"]``,
        best first: ``[{"session_id", "turn_number", "user_content",
        "assistant_content", "score", "created_at"}, ...]``.
    """
    return get_engine(cfg).recall_related(session_id, query, top_k=top_k)


def needs_extraction(session_id: str, cfg: Optional[dict] = None) -> bool:
    """True once enough user turns have accumulated since the last extraction ack.

    A future orchestrator/agent polls this after each turn to decide
    whether to run the insight-extractor agent. This package never checks
    it itself and never calls an LLM as a result of it being true.
    """
    return get_engine(cfg).needs_extraction(session_id)


def upsert_facts(
    session_id: str, facts: list[dict], turns_covered: int, cfg: Optional[dict] = None
) -> dict:
    """Atomically upserts extracted facts and acknowledges an extraction attempt.

    Args:
        session_id: The session the extraction attempt was run against.
        facts: ``[{"key": str, "value": str, "confidence": float}, ...]``,
            produced by a future agent's LLM call -- never by this
            package. Pass ``[]`` if the attempt failed or found nothing.
        turns_covered: How many user turns the attempt's snapshot covered.
        cfg: Optional config override; defaults to ``read_config()``.

    Returns:
        ``{"upserted": int}``.
    """
    return get_engine(cfg).upsert_facts(session_id, facts, turns_covered)


def clear_session(session_id: str, cfg: Optional[dict] = None) -> None:
    """Deletes this session's messages and embeddings; leaves facts untouched."""
    get_engine(cfg).clear_session(session_id)
