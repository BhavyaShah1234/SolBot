"""Per-turn embeddings and cross-session recall for the Memory Engine.

Embeds each completed conversation turn once, using whatever embedding
model ``db_engine`` is currently configured with (``cfg["embedding"]["model"]``,
a project-wide, shared config key -- deliberately not a second copy under
``memory:``), and recalls relevant turns from a user's OTHER sessions via
brute-force cosine similarity. This is a representation/indexing concern,
not a generative one -- unlike contextualization/extraction, it stays
inside this package.

Because the embedding model is a live, changeable config value (e.g. a
deadline-driven swap to a smaller/faster model), stored vectors are not
assumed to be homogeneous: each row records its own ``embedding_dim``, and
:func:`recall_related` only compares rows whose dimension matches the
current query's, skipping (never crashing on) rows left behind by a
retired model -- see that function's docstring for details.

Brute-force is adequate here: this is one user's personal conversation
history (hundreds to low thousands of turns realistically), loaded into
memory and scored in milliseconds. Not a general-purpose ANN index -- if a
single user's turn count ever grows large enough for that to matter,
reusing ``db_engine``'s Chroma pattern (deliberately avoided here for
weight) is the right call, deferred rather than solved now.
"""

import logging
import sqlite3
from typing import Optional

import numpy as np
from langchain_ollama import OllamaEmbeddings

from memory_engine.store import resolve_user_id_readonly

logger = logging.getLogger(__name__)

_embeddings_clients: dict[tuple, OllamaEmbeddings] = {}


def _get_embeddings_client(cfg: dict) -> OllamaEmbeddings:
    """Returns a cached client for the given (model, base_url) pair.

    ``base_url`` is optional (``cfg["embedding"].get("base_url")``) -- when
    absent, ``OllamaEmbeddings`` falls back to the local Ollama instance,
    unchanged from before this option existed. The cache key includes
    ``base_url`` alongside ``model`` so pointing at a different host (e.g.
    a machine with more VRAM) doesn't silently keep returning a stale
    client bound to the wrong one.
    """
    model = cfg["embedding"]["model"]
    base_url = cfg["embedding"].get("base_url")
    key = (model, base_url)
    if key not in _embeddings_clients:
        _embeddings_clients[key] = OllamaEmbeddings(model=model, base_url=base_url)
    return _embeddings_clients[key]


def embed_text(text: str, cfg: dict) -> np.ndarray:
    """Embeds a single string, returning a float32 numpy vector.

    Args:
        text: The text to embed.
        cfg: The loaded config dict; reuses ``db_engine``'s
            ``cfg["embedding"]["model"]`` (and optional
            ``cfg["embedding"]["base_url"]``) keys rather than declaring a
            second copy, since it's the same model/host doing the same job.

    Returns:
        A 1-D float32 numpy array.

    Raises:
        Exception: Whatever the underlying Ollama call raises (connection
            errors, timeouts) -- callers are responsible for treating a
            failed embedding as a best-effort miss, not a hard failure.
    """
    client = _get_embeddings_client(cfg)
    vector = client.embed_query(text)
    return np.array(vector, dtype=np.float32)


def recall_related(
    conn: sqlite3.Connection, session_id: str, query: str, top_k: Optional[int], cfg: dict
) -> list[dict]:
    """Finds relevant turns from the session's user's OTHER sessions.

    Args:
        conn: An open, unlocked read connection.
        session_id: The current session -- always excluded from results,
            since ``get_recent_context`` already covers it.
        query: The text to search for semantically similar past turns.
        top_k: Max results to return; defaults to ``cfg["memory"]["recall_top_k"]``.
        cfg: The loaded config dict.

    Returns:
        Turns scoring above ``cfg["memory"]["recall_score_threshold"]``,
        best first: ``[{"session_id", "turn_number", "user_content",
        "assistant_content", "score", "created_at"}, ...]``. Returns ``[]``
        if there's nothing compatible to compare against or the query
        embedding call fails -- a flaky embedding call degrades to "no
        cross-session context found" rather than raising.
    """
    top_k = top_k or cfg["memory"]["recall_top_k"]
    threshold = cfg["memory"]["recall_score_threshold"]
    user_id = resolve_user_id_readonly(conn, session_id, cfg)

    rows = conn.execute(
        "SELECT session_id, turn_number, embedding, embedding_dim, created_at FROM message_embeddings "
        "WHERE user_id = ? AND session_id != ?",
        (user_id, session_id),
    ).fetchall()
    if not rows:
        return []

    try:
        query_vector = embed_text(query, cfg)
    except Exception:
        return []

    # Stored embeddings can span an embedding-model change (different models
    # produce different-length vectors) -- np.stack requires uniform shapes,
    # so mixing dimensions would raise, not just score poorly. Rows from a
    # retired model are permanently incomparable to the current query, not
    # an error condition: skip them and keep going rather than crash.
    query_dim = query_vector.shape[0]
    compatible_rows = [row for row in rows if row[3] == query_dim]
    skipped = len(rows) - len(compatible_rows)
    if skipped:
        logger.warning(
            "recall_related: skipped %d stored embedding(s) with mismatched dimension "
            "for user_id=%s (query embedded at dim=%d) -- likely an embedding-model "
            "change; those vectors are now permanently inert, not an error",
            skipped,
            user_id,
            query_dim,
        )
    if not compatible_rows:
        return []

    matrix = np.stack([np.frombuffer(blob, dtype=np.float32) for _, _, blob, _, _ in compatible_rows])
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores = matrix_norm @ query_norm

    ranked = sorted(zip(compatible_rows, scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for (r_session_id, r_turn_number, _blob, _dim, created_at), score in ranked:
        if score < threshold:
            break  # ranked descending -- everything after this also falls below threshold
        if len(results) >= top_k:
            break
        msg_rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? AND turn_number = ?",
            (r_session_id, r_turn_number),
        ).fetchall()
        user_content = next((c for role, c in msg_rows if role == "user"), None)
        assistant_content = next((c for role, c in msg_rows if role == "assistant"), None)
        results.append(
            {
                "session_id": r_session_id,
                "turn_number": r_turn_number,
                "user_content": user_content,
                "assistant_content": assistant_content,
                "score": float(score),
                "created_at": created_at,
            }
        )
    return results
