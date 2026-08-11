"""The DBEngine class and its process-wide singleton accessor.

``DBEngine`` caches the vector store client for one process and provides
the four core operations (``fetch``, ``sync_page``, ``sync_all``,
``reset``) with two-layer write concurrency control:

* Layer 1 — an in-process ``threading.Lock`` held by the instance itself,
  serializing writes from multiple threads/tasks within the same process
  (e.g. several Phase-2 workers running in a thread pool).
* Layer 2 — the cross-process file lock from
  :mod:`db_engine.concurrency`, serializing writes across separate OS
  processes (e.g. an admin CLI invocation racing a live server process).

Reads (``fetch``) take neither lock — an occasional read of mid-write
state is an acceptable tradeoff for a RAG system.

``get_engine()`` is a thin, thread-safe lazy singleton on top of
``DBEngine`` purely to avoid re-paying Chroma/Ollama client setup cost on
every call. ``DBEngine`` itself stays directly instantiable so isolated
tests can construct their own instance against a temp directory without
touching the shared singleton.
"""

import logging
import threading
from typing import Optional

from db_engine.concurrency import write_lock as _cross_process_write_lock
from db_engine.confluence_source import fetch_all_pages, fetch_page
from db_engine.manifest import diff_and_apply, load_manifest, save_manifest
from db_engine.store import Chunk, get_all_chunks, get_vector_store

logger = logging.getLogger(__name__)


class DBEngine:
    """Caches the vector store client for one process and serializes writes.

    Not shared across processes — see the module docstring for the
    two-layer locking strategy that handles that. Normally obtained via
    :func:`get_engine`, but directly instantiable for isolated tests.

    Attributes:
        Nothing public; ``_cfg``, ``_store``, and ``_in_process_lock`` are
        internal.
    """

    def __init__(self, cfg: dict):
        """Initializes the engine and constructs its cached vector store client.

        Args:
            cfg: The loaded configuration dict this engine will use for
                every subsequent call, for the lifetime of this instance.
        """
        self._cfg = cfg
        self._store = get_vector_store(cfg)
        self._in_process_lock = threading.Lock()

    def fetch(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[tuple]:
        """Retrieves raw context for a query. Never generates.

        Args:
            query: The natural-language query to embed and search with.
            top_k: Maximum number of chunks to consider before filtering
                by score. Defaults to ``cfg["retrieval"]["top_k"]``.
            score_threshold: Minimum cosine relevance score (``[0, 1]``) a
                chunk must meet to be returned. Defaults to
                ``cfg["retrieval"]["score_threshold"]``.

        Returns:
            A list of ``(langchain_core.documents.Document, score)`` tuples
            for every chunk meeting the threshold, most relevant first.
            The caller is responsible for any generation over this raw
            context — the engine never calls an LLM.
        """
        retrieval_cfg = self._cfg["retrieval"]
        resolved_top_k = top_k or retrieval_cfg["top_k"]
        resolved_threshold = (
            retrieval_cfg["score_threshold"] if score_threshold is None else score_threshold
        )
        results = self._store.similarity_search_with_relevance_scores(query, k=resolved_top_k)
        return [(doc, score) for doc, score in results if score >= resolved_threshold]

    def get_all_chunks(self) -> list[Chunk]:
        """Reads every chunk in the collection, embeddings included. Never generates.

        Read-only, no locking (see :meth:`fetch`'s docstring for why).
        Intended for callers that need the whole corpus rather than a
        similarity search against it — e.g. a Phase-2 sparse/BM25 index
        built alongside dense retrieval.

        Returns:
            One :class:`~db_engine.store.Chunk` per stored chunk.
        """
        return get_all_chunks(self._store)

    def sync_page(self, page_id: str) -> dict:
        """Fetches one Confluence page and updates the vector store if it changed.

        Args:
            page_id: The Confluence page id to check and, if stale, resync.

        Returns:
            A dict with keys ``page_id`` and ``status`` (one of
            ``"added"``, ``"updated"``, ``"unchanged"``), plus
            ``chunks_written`` when applicable.

        Raises:
            filelock.Timeout: If the cross-process write lock can't be
                acquired within its timeout.
            atlassian.errors.ApiError: If Confluence has no page with the
                given id, or the configured credentials lack permission to
                view it.
        """
        with self._in_process_lock, _cross_process_write_lock(self._cfg):
            manifest = load_manifest(self._cfg)
            page = fetch_page(page_id, self._cfg)
            result = diff_and_apply(page, manifest, self._store, self._cfg)
            save_manifest(self._cfg, manifest)
            logger.info("sync_page(%s): %s", page_id, result)
            return result

    def sync_all(self) -> dict:
        """Performs a full manifest-diffed sync across the whole Confluence space.

        Fetches every current page, upserts any that are new or changed,
        and deletes any pages present in the manifest but no longer
        present upstream.

        Returns:
            A summary dict with counts for ``added``, ``updated``,
            ``unchanged``, ``deleted``, and ``chunks_written``.

        Raises:
            filelock.Timeout: If the cross-process write lock can't be
                acquired within its timeout.
        """
        with self._in_process_lock, _cross_process_write_lock(self._cfg):
            manifest = load_manifest(self._cfg)
            pages = fetch_all_pages(self._cfg)
            logger.info("sync_all: fetched %d pages from Confluence", len(pages))

            results = []
            seen_ids = set()
            for page in pages:
                seen_ids.add(str(page["id"]))
                results.append(diff_and_apply(page, manifest, self._store, self._cfg))

            deleted_ids = [page_id for page_id in manifest if page_id not in seen_ids]
            for page_id in deleted_ids:
                self._store.delete(ids=manifest[page_id].get("chunk_ids", []))
                del manifest[page_id]

            save_manifest(self._cfg, manifest)
            summary = _summarize(results, deleted_ids)
            logger.info("sync_all: %s", summary)
            return summary

    def reset(self) -> dict:
        """Wipes and rebuilds the entire vector database from scratch.

        An admin operation: deletes and recreates the Chroma collection,
        clears the manifest, then performs a fresh full sync. Intended for
        situations where the vector database and Confluence have drifted
        enough that a targeted or full diff-sync isn't trusted to recover
        cleanly.

        Returns:
            The summary dict from the fresh :meth:`sync_all` call that
            follows the reset (same shape as its return value).

        Raises:
            filelock.Timeout: If the cross-process write lock can't be
                acquired within its timeout.
        """
        logger.info("reset: wiping collection and manifest")
        with self._in_process_lock, _cross_process_write_lock(self._cfg):
            self._store.reset_collection()
            save_manifest(self._cfg, {})
        return self.sync_all()  # re-acquires both locks for the rebuild


def _summarize(results: list[dict], deleted_ids: list[str]) -> dict:
    """Aggregates per-page diff results into a single sync summary.

    Args:
        results: The list of per-page dicts returned by
            ``db_engine.manifest.diff_and_apply``, one per page considered.
        deleted_ids: Page ids that were present in the manifest but absent
            from the current fetch, and have therefore been removed.

    Returns:
        A dict with counts for ``added``, ``updated``, ``unchanged``,
        ``deleted``, and total ``chunks_written``.
    """
    summary = {"added": 0, "updated": 0, "unchanged": 0, "deleted": len(deleted_ids), "chunks_written": 0}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        summary["chunks_written"] += result.get("chunks_written", 0)
    return summary


_engine_instance: Optional["DBEngine"] = None
_engine_init_lock = threading.Lock()


def get_engine(cfg: Optional[dict] = None) -> DBEngine:
    """Returns the process-wide cached :class:`DBEngine` instance, creating it if needed.

    Thread-safe lazy initialization via double-checked locking, so
    concurrent first-callers can't race to construct two instances. The
    first caller's ``cfg`` (or a freshly loaded one, if none is given)
    wins for the lifetime of the process — config is assumed stable while
    the process runs, matching how ``config.yaml`` is meant to be used
    here.

    Args:
        cfg: Configuration dict to initialize the engine with, if it
            hasn't been created yet. Ignored on subsequent calls once the
            singleton already exists. If ``None`` and the singleton
            doesn't exist yet, loads the default config via
            :func:`db_engine.config.read_config`.

    Returns:
        The shared :class:`DBEngine` instance for this process.
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_init_lock:
            if _engine_instance is None:
                from db_engine.config import read_config

                _engine_instance = DBEngine(cfg or read_config())
    return _engine_instance
