"""Interactive CLI for the ``agents`` package.

Usage:
    python -m agents

Type a question and press enter. ``\\quit`` exits, ``\\clear`` resets this
session's conversation history (personalization facts persist -- see
``memory_engine.clear_session``). ``\\debug`` toggles a per-turn trace
(route decision, plan DAG, per-node confidence/tools used, verification
outcome -- the same fields ``Session.ask()`` already returns, see
``test/chat_check.py``'s ``_print_trace`` for the design this mirrors).
``\\thinking`` toggles showing the model's raw reasoning trace for every
LLM call made during a turn (requests Ollama's separated reasoning mode --
see ``agents/llm.py``'s ``show_thinking`` -- which costs extra generated
tokens/latency while on, hence off by default). Both toggle independently
and can be flipped mid-session. On a real terminal, debug lines print in
cyan and thinking lines in dim magenta, distinct from the plain-text
answer -- colors are skipped automatically when stdout isn't a tty (e.g.
piped/redirected output). Logs to ``cfg["app"]["log_file"]``
(default ``agents.log``), tagged per-turn/per-node -- see
``agents/logging_setup.py``.
"""

import sys
import uuid

from agents.config import read_config
from agents.logging_setup import configure_logging
from agents.orchestrator import Session

_RESET = "\033[0m"
_DEBUG_COLOR = "\033[36m"  # cyan -- structured trace info
_THINKING_COLOR = "\033[2;35m"  # dim magenta -- de-emphasized raw reasoning


def _colorize(text: str, code: str) -> str:
    """Wraps ``text`` in an ANSI color code, only when stdout is a real terminal.

    Piping/redirecting output (as in scripted testing) would otherwise
    dump raw escape sequences into the captured text -- ``isatty()`` keeps
    that output clean while still coloring a live interactive session.
    """
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def _print_debug_trace(result: dict) -> None:
    """Prints one turn's route/plan/verification trace, for ``\\debug`` mode.

    Mirrors ``test/chat_check.py``'s ``_print_trace`` but against a live
    result dict rather than a scripted scenario. ``plan``/``results``/
    ``verification`` are ``None`` for the ``chat``/``clarify`` short-circuit
    routes (see ``Session.ask()``'s docstring), so this only prints the
    route line for those.
    """
    print(_colorize(f"  [debug] route: {result['route']}", _DEBUG_COLOR))
    plan = result.get("plan")
    if plan is None:
        return
    print(_colorize(f"  [debug] plan: {len(plan.nodes)} node(s), layers={plan.layers()}", _DEBUG_COLOR))
    results = result.get("results") or {}
    for node in plan.nodes:
        node_result = results.get(node.id)
        print(_colorize(f"  [debug]   [{node.id}] {node.question!r} depends_on={node.depends_on}", _DEBUG_COLOR))
        if node_result is not None:
            print(
                _colorize(
                    f"  [debug]       -> ok={node_result.ok} confidence={node_result.confidence:.2f} "
                    f"tools_used={node_result.tools_used}",
                    _DEBUG_COLOR,
                )
            )
    verification = result.get("verification")
    if verification is not None:
        print(
            _colorize(
                f"  [debug] verification: overall_confidence={verification.overall_confidence:.2f} "
                f"grounded={sorted(verification.grounded_ids)} problems={verification.problems}",
                _DEBUG_COLOR,
            )
        )


def _print_thinking_log(llm) -> None:
    """Prints and clears any reasoning traces captured during the last turn, for ``\\thinking`` mode.

    Drains unconditionally after every turn (not just while the toggle is
    on) so a stray entry can never leak into a later turn -- in practice
    the log is only ever non-empty when ``show_thinking`` was on for the
    turn that just completed, since that's the only time ``agents.llm.LLM``
    populates it.
    """
    for role, thinking in llm.drain_thinking_log():
        print(_colorize(f"  [thinking:{role}] {thinking}", _THINKING_COLOR))


def main() -> None:
    cfg = read_config()
    configure_logging(cfg)

    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    session = Session(session_id=session_id, cfg=cfg)
    debug_mode = False

    print(
        "SolBot (Phase 2) -- type a question, \\quit to exit, \\clear to reset conversation "
        "history, \\debug to toggle route/plan/verification trace, \\thinking to toggle the "
        "model's raw reasoning trace (adds latency while on)."
    )
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query == "\\quit":
            break
        if query == "\\clear":
            session.reset()
            print("(conversation history cleared)")
            continue
        if query == "\\debug":
            debug_mode = not debug_mode
            print(f"(debug trace {'on' if debug_mode else 'off'})")
            continue
        if query == "\\thinking":
            session.llm.show_thinking = not session.llm.show_thinking
            state = "on -- adds latency" if session.llm.show_thinking else "off"
            print(f"(thinking trace {state})")
            continue

        result = session.ask(query)
        if session.llm.show_thinking:
            _print_thinking_log(session.llm)
        if debug_mode:
            _print_debug_trace(result)
        print(result["answer"])


if __name__ == "__main__":
    main()
