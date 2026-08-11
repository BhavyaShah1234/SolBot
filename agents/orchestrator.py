"""The orchestrator: a plain bounded Python loop, not a graph library.

Wires everything else in this package together around one conversation
session: ``contextualize -> route -> plan -> execute -> verify ->
(replan?) -> synthesize``. No LangGraph — CLAUDE.md's revamp plan
explicitly favors small, owned, understood control flow over the
DAG/LangGraph machinery that made ``new/`` unreviewable; a bounded
``for``/``while`` loop achieves the same cycle without a dependency.

Conversation state (messages, facts, rolling summary) lives entirely in
``memory_engine`` — ``Session`` itself only holds a ``session_id`` and the
process-lifetime objects (LLM client, retriever, tool registry, worker)
that are expensive to build and safe to reuse across turns.
"""

import logging
import re
import time
import uuid
from typing import Optional

from agents import contextualize, insight, planner, synthesis
from agents.config import read_config
from agents.fusion import Retriever
from agents.llm import LLM
from agents.logging_setup import log_context
from agents.tools import build_registry
from agents.worker import Worker, execute_plan

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+")


def _fetch_user_urls(query: str, cfg: dict) -> list:
    """Fetches any URL(s) the user pasted directly in their raw message.

    Deterministic and code-level -- not left to a worker's own judgment.
    ``fetch_url`` (agents/tools.py) is otherwise only ever invoked at a
    worker's own discretion on a *web_search* hit; it has no visibility
    into, or special handling for, a URL the user typed directly. Worse,
    ``contextualize()``'s rewrite doesn't guarantee preserving a literal
    URL string verbatim (observed directly: a user-pasted URL was
    paraphrased away into "the ASU Research Computing documentation page
    for Sol hardware" before it ever reached the planner), so even a
    worker willing to fetch it might never see it. This runs on the raw
    query, before any of that, so a user-provided link is always fetched
    regardless of what contextualize/planning later does with the text.

    Args:
        query: The user's raw message for this turn (not the
            contextualized/standalone rewrite).
        cfg: The loaded config dict; reads ``cfg["web"]``.

    Returns:
        One ``agents.fusion.Evidence`` per URL that fetched successfully,
        in message order, capped at 2 URLs per turn. A bad/unreachable
        link (blocked domain, extraction failure, network error, or
        ``web.enabled: false``) is logged and skipped, never raised --
        one broken link in the message must not break the turn.
    """
    urls = _URL_RE.findall(query)[:2]
    if not urls:
        return []

    import web_search

    from agents.fusion import Evidence

    pages = []
    for url in urls:
        url = url.rstrip(").,;:!?\"'")  # strip trailing punctuation a sentence commonly attaches
        try:
            page = web_search.fetch_page(url, cfg=cfg["web"])
        except Exception as exc:  # noqa: BLE001 - a bad user-provided link must never break the turn
            logger.warning("_fetch_user_urls: failed to fetch %r: %s: %s", url, type(exc).__name__, exc)
            continue
        pages.append(Evidence(text=page.text, source=page.url, title=page.title, origin="web", doc_id=page.url))
        logger.info("_fetch_user_urls: fetched %r (%d chars, truncated=%s)", url, len(page.text), page.truncated)
    return pages


class Session:
    """One conversation. Wraps a ``session_id``; all durable state lives in ``memory_engine``."""

    def __init__(self, session_id: Optional[str] = None, cfg: Optional[dict] = None):
        """Builds the process-lifetime objects this session's turns will reuse.

        Args:
            session_id: This conversation's id. Defaults to a fresh
                ``cli-<8 hex chars>`` id. User identity (and therefore
                fact/recall persistence across sessions) is resolved
                internally by ``memory_engine`` via
                ``cfg["memory"]["default_user_id"]`` — every session
                sharing that default is treated as the same local user.
            cfg: The loaded config dict. Defaults to
                :func:`agents.config.read_config`.
        """
        self.cfg = cfg if cfg is not None else read_config()
        self.session_id = session_id or f"cli-{uuid.uuid4().hex[:8]}"

        self.llm = LLM(self.cfg)
        self.retriever = Retriever(self.cfg, self.llm)
        self.retriever.refresh_indexes()
        self.registry = build_registry(self.cfg, self.retriever)
        self.worker = Worker(self.cfg, self.llm, self.registry)

        logger.info("Session: initialized session_id=%s", self.session_id)

    def ask(self, query: str) -> dict:
        """Answers one turn, end to end.

        See the "End-to-end turn workflow" section of this package's
        design plan for the full step-by-step trace this implements.

        Args:
            query: The user's raw message for this turn.

        Returns:
            ``{"answer": str, "route": str, "plan": Plan | None,
            "results": dict[str, NodeResult], "verification":
            Verification | None}``. ``plan``/``results``/``verification``
            are ``None`` for the ``"chat"``/``"clarify"`` short-circuit
            routes, which never build a plan.
        """
        import memory_engine

        with log_context(session_id=self.session_id, node_id="-"):
            start = time.monotonic()
            logger.info("ask: session=%s query=%r", self.session_id, query)

            memory_engine.append_message(self.session_id, "user", query, cfg=self.cfg)
            user_pages = _fetch_user_urls(query, self.cfg)

            standalone = contextualize.contextualize(self.llm, self.cfg, self.session_id, query)

            try:
                insight.maybe_run_extraction(self.llm, self.cfg, self.session_id)
            except Exception:  # noqa: BLE001 - insight extraction must never block the answer
                logger.warning("ask: session=%s insight extraction raised, continuing", self.session_id)

            profile = memory_engine.get_profile_facts(self.session_id, cfg=self.cfg)
            try:
                recalled = memory_engine.recall_related(self.session_id, standalone, cfg=self.cfg)
            except Exception:  # noqa: BLE001 - cross-session recall is a nice-to-have, never a blocker
                logger.warning("ask: session=%s recall_related raised, continuing without it", self.session_id)
                recalled = []

            route = planner.route(self.llm, self.cfg, standalone)

            if route in ("chat", "clarify"):
                answer = self._answer_without_plan(route, standalone, profile, user_pages)
                memory_engine.append_message(self.session_id, "assistant", answer, cfg=self.cfg)
                elapsed = time.monotonic() - start
                logger.info("ask: session=%s route=%s done in %.1fs", self.session_id, route, elapsed)
                return {"answer": answer, "route": route, "plan": None, "results": None, "verification": None}

            plan = planner.plan(self.llm, self.cfg, standalone)
            planner_cfg = self.cfg["planner"]
            results = execute_plan(plan, self.worker, max_parallel=self.cfg["worker"]["max_parallel_nodes"])
            verification = synthesis.verify(self.llm, self.cfg, standalone, results, user_pages=user_pages)

            replans_done = 0
            while (
                verification.needs_more_research
                and planner_cfg["allow_replanning"]
                and replans_done < planner_cfg["max_replans"]
            ):
                plan = planner.replan(self.llm, self.cfg, plan, verification.problems)
                results = execute_plan(plan, self.worker, max_parallel=self.cfg["worker"]["max_parallel_nodes"])
                verification = synthesis.verify(self.llm, self.cfg, standalone, results, user_pages=user_pages)
                replans_done += 1

            answer = synthesis.synthesise(
                self.llm,
                self.cfg,
                plan,
                results,
                verification,
                profile=profile,
                recalled=recalled,
                user_pages=user_pages,
            )
            memory_engine.append_message(self.session_id, "assistant", answer, cfg=self.cfg)

            elapsed = time.monotonic() - start
            logger.info(
                "ask: session=%s route=%s nodes=%d replans=%d confidence=%.2f done in %.1fs",
                self.session_id,
                route,
                len(plan.nodes),
                replans_done,
                verification.overall_confidence,
                elapsed,
            )
            return {"answer": answer, "route": route, "plan": plan, "results": results, "verification": verification}

    def _answer_without_plan(self, route: str, standalone: str, profile: dict, user_pages: list) -> str:
        """One lightweight generation call for the ``chat``/``clarify`` short-circuit routes.

        Includes recent conversation history in the prompt -- previously
        this call saw only ``standalone`` (the current turn's rewritten
        query) and profile facts, with **no memory of the conversation at
        all**. That caused a confident, hallucinated denial of history
        ("we don't have any previous conversation history in this chat
        session") on a real multi-turn session where memory_engine was
        storing everything correctly -- the bug was this call never
        reading it back, not a storage/recall failure. ``recent`` here
        includes the just-appended current user turn (``ask()`` calls
        ``memory_engine.append_message`` before this runs), so it overlaps
        slightly with ``standalone`` -- same shape as
        ``contextualize.contextualize()``'s own history rendering, which
        has the identical overlap and works fine; not worth the extra
        bookkeeping to trim one redundant line.
        """
        import memory_engine

        recent = memory_engine.get_recent_context(
            self.session_id, turns=self.cfg["memory"]["default_context_window_turns"], cfg=self.cfg
        )
        history_block = ""
        if recent:
            transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)
            history_block = f"\n\nConversation so far:\n{transcript}"

        facts_block = synthesis.render_facts_for_prompt(profile)
        if route == "chat":
            system = (
                "You are a friendly assistant for ASU Research Computing support. "
                "Respond briefly and warmly to this small talk / greeting."
            )
        else:
            system = (
                "The user's question is too vague to research as-is. Ask ONE brief, "
                "specific clarifying question to narrow it down (e.g. which cluster, "
                "what resource, what error message). Use the conversation so far below "
                "to avoid asking about something the user already told you -- if it "
                "already answers what you'd otherwise ask, don't ask it again."
            )
        if history_block:
            system += history_block
        if facts_block:
            system += f"\n\nWhat we know about this user:\n{facts_block}\nUse it naturally if relevant (e.g. address them by name)."
        if user_pages:
            pages_block = synthesis.render_user_pages(user_pages)
            system += (
                f"\n\nThe user included a URL in their message; here is that page's actual content, "
                f"fetched directly -- treat it as ground truth for anything it covers:\n{pages_block}"
            )
        return self.llm.text(f"answer_{route}", system, standalone)

    def reset(self) -> None:
        """Clears this session's conversation history (``\\clear``). Facts persist."""
        import memory_engine

        memory_engine.clear_session(self.session_id, cfg=self.cfg)
        logger.info("reset: session=%s cleared", self.session_id)
