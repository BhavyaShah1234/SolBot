"""Freshness bookkeeping for the DB Engine.

Tracks, per Confluence page id, what's currently synced into the vector
store — its ``version``, a content ``hash``, and the exact chunk ids
written for it — in a JSON manifest file. This is what makes syncing
incremental and idempotent instead of a blind full rebuild every time:
:func:`diff_and_apply` is the single shared decision point both
``sync_page`` and ``sync_all`` call to compare a freshly-fetched page
against the manifest and update the vector store only if it actually
changed.

Change detection uses ``version`` alone, not ``hash``. A hash of the
rendered HTML body was tried as a secondary guard for content edits that
don't bump ``version``, but live testing against the real RC space showed
Confluence's ``body.view`` rendering is not deterministic across requests
— fetching the same, unedited page (``version`` unchanged) twice in a row
produced two different hashes, evidently from a dynamically-rendered
macro. Using the hash as a change trigger would therefore flag a random
subset of pages as "updated" on every single sync run, forever, with no
actual content change behind it. ``version`` is Confluence's own
authoritative signal for stored-content edits and was confirmed stable
across repeated fetches, so it's the sole trigger. ``hash`` is still
recorded in the manifest as an informational/debugging field, but nothing
here reads it to make a decision.
"""

import json
import os
import time
from hashlib import sha256
from typing import Any

from db_engine.chunking import chunk_page


def load_manifest(cfg: dict) -> dict:
    """Loads the sync manifest from disk.

    Args:
        cfg: The loaded configuration dict; reads
            ``cfg["vector_store"]["manifest_path"]``.

    Returns:
        The manifest as a dict keyed by page id. Returns an empty dict if
        no manifest file exists yet (the first-ever sync case).
    """
    path = cfg["vector_store"]["manifest_path"]
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_manifest(cfg: dict, manifest: dict) -> None:
    """Writes the sync manifest to disk atomically.

    Writes to a temporary file first and then renames it into place via
    ``os.replace``, so a crash mid-write can't leave a corrupted manifest.

    Args:
        cfg: The loaded configuration dict; reads
            ``cfg["vector_store"]["manifest_path"]``.
        manifest: The manifest dict to persist.
    """
    path = cfg["vector_store"]["manifest_path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, path)


def diff_and_apply(page: dict, manifest: dict, store: Any, cfg: dict) -> dict:
    """Compares one page against the manifest and upserts it if changed.

    Shared by ``sync_page`` (one page) and ``sync_all`` (every page) so the
    "skip if unchanged, re-chunk and upsert with stale-id cleanup if
    changed" logic lives in exactly one place. Mutates ``manifest`` in
    place on a change; callers are responsible for persisting it via
    :func:`save_manifest` afterward.

    Args:
        page: A page dict as produced by
            ``db_engine.confluence_source._to_page_dict``.
        manifest: The current manifest dict, checked and updated in place.
        store: A ``langchain_chroma.Chroma`` instance to upsert/delete
            chunks against.
        cfg: The loaded configuration dict, passed through to
            :func:`db_engine.chunking.chunk_page`.

    Returns:
        A dict with key ``page_id`` and ``status`` (one of ``"added"``,
        ``"updated"``, ``"unchanged"``), plus ``chunks_written`` when the
        page was added or updated.
    """
    page_id = str(page["id"])
    digest = sha256(page["html"].encode("utf-8")).hexdigest()
    record = manifest.get(page_id)

    # Version alone, not hash — see module docstring for why hash is unreliable here.
    if record and record["version"] == page["version"]:
        return {"page_id": page_id, "status": "unchanged"}

    ids, texts, metadatas = chunk_page(page, cfg)
    if record:
        store.delete(ids=record.get("chunk_ids", []))  # purge stale ids before re-upsert
    if ids:  # some pages (folders/index pages) have no body content and chunk to nothing
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    manifest[page_id] = {
        "version": page["version"],
        "hash": digest,
        "title": page["title"],
        "chunk_ids": ids,
        "synced_at": int(time.time()),
    }
    return {
        "page_id": page_id,
        "status": "updated" if record else "added",
        "chunks_written": len(ids),
    }
