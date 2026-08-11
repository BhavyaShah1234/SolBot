"""Cross-process write serialization for the DB Engine.

Provides the second of the engine's two locking layers: a filesystem-based
lock guarding write operations (``sync_page``, ``sync_all``, ``reset``)
across separate OS processes — e.g. an admin's CLI invocation running
concurrently with a live server process. This complements, and does not
replace, the in-process ``threading.Lock`` held by
``db_engine.engine.DBEngine`` (that lock has no visibility outside its own
process). Reads (``fetch``) never take this lock: an occasional read of
mid-write state is an acceptable tradeoff for a RAG system, and blocking
reads on writes would hurt retrieval latency for no real benefit.
"""

import os
from contextlib import contextmanager
from typing import Iterator

from filelock import FileLock


@contextmanager
def write_lock(cfg: dict, timeout: float = 30.0) -> Iterator[None]:
    """Acquires the cross-process write lock for the duration of the block.

    Creates the lock file's parent directory first if it doesn't exist yet
    (true on a first-ever sync, before ``./asu_rc/`` has been created).

    Args:
        cfg: The loaded configuration dict; reads
            ``cfg["vector_store"]["lock_path"]`` for the lock file location.
        timeout: Maximum seconds to wait for the lock before raising.

    Yields:
        Nothing; used purely for its ``with`` block scoping.

    Raises:
        filelock.Timeout: If the lock isn't acquired within ``timeout``
            seconds (e.g. another process is mid-write).
    """
    lock_path = cfg["vector_store"]["lock_path"]
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock = FileLock(lock_path, timeout=timeout)
    with lock:
        yield
