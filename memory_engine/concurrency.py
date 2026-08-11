"""In-process, per-session locking for the Memory Engine.

Cross-process write serialization is handled by SQLite itself (WAL mode +
``busy_timeout``, see ``store.py``) -- this module only guards against
concurrent threads within one process racing on the same session. Locks
are created lazily, one per ``session_id``, so unrelated sessions never
contend with each other.
"""

import threading
from contextlib import contextmanager
from typing import Iterator

_session_locks: dict[str, threading.Lock] = {}
_registry_meta_lock = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _registry_meta_lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


@contextmanager
def session_lock(session_id: str) -> Iterator[None]:
    """Acquires the in-process lock for ``session_id`` for the block's duration."""
    with _get_session_lock(session_id):
        yield
