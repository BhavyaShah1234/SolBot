"""Context-aware logging for the ``agents`` package.

Every log line — including from inside :func:`agents.worker.execute_plan`'s
concurrently-running worker threads — is traceable back to which turn
(``session_id``) and which DAG node (``node_id``) produced it, without
threading a logger object through every function signature by hand.
Every other module just does ``logger = logging.getLogger(__name__)`` and
logs normally; :class:`_ContextFilter` stamps ``session_id``/``node_id``
onto each ``LogRecord`` from whatever the current ``contextvars.Context``
happens to be.

Safe under :func:`agents.worker.execute_plan`'s ``ThreadPoolExecutor``, not
just single-threaded code — but this requires the caller to opt in
explicitly. Unlike ``asyncio`` (where a ``Task`` always runs in a copy of
the context that created it), plain ``concurrent.futures.Executor.submit()``
does **not** copy the calling context into the worker thread — verified
directly against this interpreter: a contextvar set before ``submit()``
reads back as its default inside the submitted callable. ``execute_plan``
therefore captures ``contextvars.copy_context()`` itself, once per node,
in the submitting thread right before dispatch, and submits
``ctx.run(worker.run, ...)`` rather than ``worker.run`` directly. Each
node's copied context is an independent snapshot: one task's
``node_id_var.set(...)`` at the top of ``Worker.run()`` (executing inside
its own ``ctx.run()`` call) never leaks into a sibling task's copy running
concurrently on a different pool thread, and reusing a pool thread for a
later, unrelated task doesn't leak the earlier task's ``node_id`` either.
"""

import contextvars
import logging
from contextlib import contextmanager
from typing import Optional

session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
node_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("node_id", default="-")


class _ContextFilter(logging.Filter):
    """Stamps the current session_id/node_id contextvars onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = session_id_var.get()
        record.node_id = node_id_var.get()
        return True


_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)s] "
    "[session=%(session_id)s node=%(node_id)s thread=%(threadName)s] "
    "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)

_configured = False


def configure_logging(cfg: dict) -> None:
    """Attaches a context-aware file handler to the ``"agents"`` logger tree.

    Idempotent — safe to call more than once (e.g. from both a script and
    a test harness importing it) without duplicating handlers.

    Args:
        cfg: The loaded config dict; reads ``cfg["app"]["log_file"]`` and
            ``cfg["app"].get("log_level", "INFO")``.
    """
    global _configured
    logger = logging.getLogger("agents")
    if _configured:
        return
    handler = logging.FileHandler(cfg["app"]["log_file"])
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(_ContextFilter())
    logger.setLevel(cfg["app"].get("log_level", "INFO"))
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


@contextmanager
def log_context(*, session_id: Optional[str] = None, node_id: Optional[str] = None):
    """Sets session_id/node_id for every log line emitted within the block.

    Only the contextvars named by non-``None`` arguments are touched;
    omitted ones keep whatever value (if any) an enclosing ``log_context``
    already set. Restores the prior value on exit, so nested contexts
    (e.g. a per-turn ``session_id`` context wrapping a per-node ``node_id``
    context) compose correctly.

    Args:
        session_id: The conversation session this block's logs belong to.
        node_id: The DAG node this block's logs belong to (``"-"`` for
            code running outside any specific node).
    """
    tokens = []
    if session_id is not None:
        tokens.append((session_id_var, session_id_var.set(session_id)))
    if node_id is not None:
        tokens.append((node_id_var, node_id_var.set(node_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
