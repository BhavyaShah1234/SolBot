"""The DB Engine: SolBot's vector-database component.

This package is the "DB Engine" (K) from SolBot's architecture — the
component responsible for accepting fetch requests from other components
and agents, managing reads and writes to the vector database while
preventing race conditions, updating a specific page's information when it
changes upstream (on trigger), and resetting/rebuilding the whole vector
database (an admin operation). It knows nothing about Confluence-page
formatting beyond what's needed to chunk and store it, and it never
generates text — every function here returns raw retrieved context or a
sync-operation summary; producing a finding or answer from that context is
always the caller's responsibility.

Public API (thin wrappers over the process-wide singleton in
:mod:`db_engine.engine` — callers never need to construct a
:class:`~db_engine.engine.DBEngine` themselves in normal use):

* :func:`fetch` — raw retrieval for any caller.
* :func:`get_all_chunks` — the full corpus, embeddings included, for
  callers that need more than a similarity search (e.g. a Phase-2
  sparse/BM25 index).
* :func:`sync_page` — triggered update of one specific page.
* :func:`sync_all` — full manifest-diffed sync across the whole space.
* :func:`reset` — admin operation: wipe and rebuild from scratch.
* :func:`read_config` — re-exported from :mod:`db_engine.config` for
  convenience.
"""

from typing import Optional

from db_engine.config import read_config
from db_engine.engine import DBEngine, get_engine
from db_engine.store import Chunk

__all__ = [
    "fetch",
    "get_all_chunks",
    "sync_page",
    "sync_all",
    "reset",
    "read_config",
    "DBEngine",
    "Chunk",
]


def fetch(
    query: str,
    cfg: Optional[dict] = None,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[tuple]:
    """Retrieves raw context for a query. Never generates.

    Args:
        query: The natural-language query to embed and search with.
        cfg: Configuration dict. Only used to initialize the shared engine
            singleton on the first call in this process; ignored on
            subsequent calls (see :func:`db_engine.engine.get_engine`). If
            ``None`` on the first call, the default ``config.yaml`` is
            loaded automatically.
        top_k: Maximum number of chunks to consider before filtering by
            score. Defaults to ``cfg["retrieval"]["top_k"]``.
        score_threshold: Minimum cosine relevance score (``[0, 1]``) a
            chunk must meet to be returned. Defaults to
            ``cfg["retrieval"]["score_threshold"]``.

    Returns:
        A list of ``(langchain_core.documents.Document, score)`` tuples for
        every chunk meeting the threshold, most relevant first.
    """
    return get_engine(cfg).fetch(query, top_k, score_threshold)


def get_all_chunks(cfg: Optional[dict] = None) -> list[Chunk]:
    """Reads every chunk in the collection, embeddings included. Never generates.

    Args:
        cfg: Configuration dict; see :func:`fetch` for singleton-init
            semantics.

    Returns:
        One :class:`Chunk` per stored chunk, in no particular order.
    """
    return get_engine(cfg).get_all_chunks()


def sync_page(page_id: str, cfg: Optional[dict] = None) -> dict:
    """Fetches one Confluence page and updates the vector store if it changed.

    Args:
        page_id: The Confluence page id to check and, if stale, resync.
        cfg: Configuration dict; see :func:`fetch` for singleton-init
            semantics.

    Returns:
        A dict with keys ``page_id`` and ``status`` (one of ``"added"``,
        ``"updated"``, ``"unchanged"``), plus ``chunks_written`` when
        applicable.
    """
    return get_engine(cfg).sync_page(page_id)


def sync_all(cfg: Optional[dict] = None) -> dict:
    """Performs a full manifest-diffed sync across the whole Confluence space.

    Args:
        cfg: Configuration dict; see :func:`fetch` for singleton-init
            semantics.

    Returns:
        A summary dict with counts for ``added``, ``updated``,
        ``unchanged``, ``deleted``, and ``chunks_written``.
    """
    return get_engine(cfg).sync_all()


def reset(cfg: Optional[dict] = None) -> dict:
    """Wipes and rebuilds the entire vector database from scratch.

    Args:
        cfg: Configuration dict; see :func:`fetch` for singleton-init
            semantics.

    Returns:
        The summary dict from the fresh full sync performed after the
        reset (same shape as :func:`sync_all`'s return value).
    """
    return get_engine(cfg).reset()
