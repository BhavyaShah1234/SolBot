"""Conversation memory and follow-up resolution.

Version 1 spent one large language model call per stored message per turn
deciding relatedness -- ten turns of history meant ten extra inferences before
retrieval even started, and the result was still just a filtered message list.

This module does the job in **one** call by inverting the problem. Instead of
asking "which past messages are relevant?", it asks the model to rewrite the
new message as a standalone question. "Can I run that for longer?" becomes
"Can I run a GPU job on Sol for longer than 4 hours?" -- and then every
downstream stage, planner and retriever alike, sees a fully specified query
with no history dependency at all.

A rolling summary keeps the window bounded on long sessions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Config
from .llm import LLM

logger = logging.getLogger(__name__)

CONTEXTUALIZE_SYSTEM = """You rewrite a follow-up message into a standalone question.

Return JSON only: {"standalone": "<rewritten question>", "changed": true|false}

Resolve every pronoun and elliptical reference ("that", "it", "there", "the
same thing", "longer") using the conversation history. If the message is
already self-contained, return it unchanged with changed=false. Never answer
the question. Never add information that is not in the history."""

SUMMARY_SYSTEM = """Compress a conversation into under 120 words.

Return JSON only: {"summary": "<text>"}

Keep entity names, cluster names, job identifiers, decisions reached, and any
constraint the user stated. Drop pleasantries."""


@dataclass
class Turn:
    """One exchange."""

    user: str
    assistant: str
    metadata: dict = field(default_factory=dict)


class Memory:
    """Bounded conversation store with contextualisation and summarisation."""

    def __init__(self, cfg: Config, llm: LLM) -> None:
        self.cfg, self.llm = cfg, llm
        self.turns: list[Turn] = []
        self.summary: str = ""
        self.max_turns = int(cfg.get("memory.max_turns_kept", 20))
        self.window = int(cfg.get("memory.contextualize_window", 4))
        self.summarize_after = int(cfg.get("memory.summarize_after_turns", 12))

    def add(self, user: str, assistant: str, metadata: dict | None = None) -> None:
        """Record a turn and compress the tail when the session gets long."""
        self.turns.append(Turn(user, assistant, metadata or {}))
        if len(self.turns) > self.summarize_after:
            self._compress()
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()
        self.summary = ""

    def recent_transcript(self, turns: int | None = None) -> str:
        """Render the last ``turns`` exchanges, prefixed by any rolling summary."""
        selected = self.turns[-(turns or self.window) :]
        lines = [f"Earlier summary: {self.summary}"] if self.summary else []
        lines += [f"User: {t.user}\nAssistant: {t.assistant}" for t in selected]
        return "\n\n".join(lines)

    def contextualize(self, query: str) -> str:
        """Rewrite ``query`` as a standalone question. One model call, or zero."""
        if not self.turns:
            return query

        payload = self.llm.json(
            "CONTEXTUALIZER",
            CONTEXTUALIZE_SYSTEM,
            f"Conversation so far:\n{self.recent_transcript()}\n\nFollow-up message: {query}",
            default=None,
        )
        if not payload:
            return query

        standalone = str(payload.get("standalone") or "").strip()
        if not standalone:
            return query
        # A rewrite that collapses the question to almost nothing is a model
        # failure, not a legitimate compression.
        if len(standalone.split()) < max(2, len(query.split()) // 4):
            return query
        if standalone != query:
            logger.debug("contextualized %r -> %r", query, standalone)
        return standalone

    def _compress(self) -> None:
        """Fold the oldest half of the buffer into the rolling summary."""
        cutoff = len(self.turns) // 2
        older, self.turns = self.turns[:cutoff], self.turns[cutoff:]
        transcript = "\n\n".join(f"User: {t.user}\nAssistant: {t.assistant}" for t in older)
        payload = self.llm.json(
            "SUMMARIZER",
            SUMMARY_SYSTEM,
            f"Existing summary: {self.summary or 'none'}\n\nNew material:\n{transcript}",
            default=None,
        )
        if payload and payload.get("summary"):
            self.summary = str(payload["summary"]).strip()
