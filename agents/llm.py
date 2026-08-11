"""Ollama chat/embedding client wrapper for the ``agents`` package.

Wraps ``langchain_ollama.ChatOllama``/``OllamaEmbeddings`` rather than a
hand-rolled HTTP client (unlike ``new/llm.py``'s raw ``urllib`` approach,
which is design-reference-only and not reused) — ``test/generation_check.py``
already established ``ChatOllama`` as this repo's proven path for
generation, so this module builds directly on that precedent instead of
reinventing it.
"""

import json
import logging
import math
import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLM:
    """One process-wide chat + embeddings client pair, reused across calls."""

    def __init__(self, cfg: dict):
        """Builds the underlying Ollama clients.

        Args:
            cfg: The loaded config dict; reads ``cfg["generation"]``
                (model, base_url, temperature, num_ctx,
                request_timeout_seconds) and ``cfg["embedding"]``
                (model, base_url — shared with ``db_engine`` and
                ``memory_engine``, same embedding model doing the same
                job, not redeclared here). ``base_url`` is optional in
                both sections; omitted or ``None`` falls back to
                ``langchain_ollama``'s own default (local Ollama).
        """
        gen_cfg = cfg["generation"]
        self._json_temperature = gen_cfg["json_temperature"]
        self._chat = ChatOllama(
            model=gen_cfg["model"],
            base_url=gen_cfg.get("base_url"),
            temperature=gen_cfg["temperature"],
            num_ctx=gen_cfg["num_ctx"],
            timeout=gen_cfg["request_timeout_seconds"],
        )
        self._embeddings = OllamaEmbeddings(
            model=cfg["embedding"]["model"], base_url=cfg["embedding"].get("base_url")
        )

        # Toggled by the CLI's `\thinking` command (see agents/__main__.py). When
        # True, every text()/json() call below requests Ollama's separated
        # reasoning mode (`reasoning=True`) and appends (role, reasoning_content)
        # to thinking_log rather than discarding it -- costs extra generated
        # tokens/latency while on, hence off by default and gated per-call
        # rather than baked into the constructor.
        self.show_thinking = False
        self.thinking_log: list[tuple[str, str]] = []

    def text(
        self, role: str, system: str, user: str, temperature: Optional[float] = None
    ) -> str:
        """Makes one freeform chat completion call.

        Args:
            role: A short label for what's calling this (e.g.
                ``"contextualize"``, ``"synthesis"``) — logged, not sent to
                the model.
            system: The system prompt.
            user: The user-turn content.
            temperature: Overrides ``cfg["generation"]["temperature"]`` for
                this call only.

        Returns:
            The model's response text.
        """
        llm = self._with_temperature(temperature)
        messages = [SystemMessage(system), HumanMessage(user)]
        logger.debug("llm.text role=%s chars=%d", role, len(user))
        response = llm.invoke(messages, reasoning=True) if self.show_thinking else llm.invoke(messages)
        self._record_thinking(role, response)
        return response.content

    def json(self, role: str, system: str, user: str, default: Any = None) -> Any:
        """Makes one chat completion call and parses the response as JSON.

        Retries once with a stricter follow-up instruction if the first
        response doesn't parse. Never raises on malformed output — callers
        in this package are built around graceful degradation (e.g.
        ``planner.plan()`` falling back to a single-node plan), so a
        parse failure here returns ``default`` rather than propagating.

        Args:
            role: A short label for what's calling this — logged only.
            system: The system prompt. Should already instruct the model
                to respond with JSON only.
            user: The user-turn content.
            default: Returned if both the initial call and the repair
                retry fail to produce parseable JSON.

        Returns:
            The parsed JSON value, or ``default`` on failure.
        """
        llm = self._with_temperature(self._json_temperature)
        messages = [SystemMessage(system), HumanMessage(user)]
        logger.debug("llm.json role=%s chars=%d", role, len(user))
        response = llm.invoke(messages, reasoning=True) if self.show_thinking else llm.invoke(messages)
        self._record_thinking(role, response)
        raw = response.content
        try:
            return extract_json(raw)
        except ValueError:
            logger.warning("llm.json role=%s unparseable, retrying once", role)

        repair_messages = messages + [
            HumanMessage(
                "That was not valid JSON. Respond with ONLY a single valid JSON "
                "value — no prose, no markdown fences, no explanation."
            )
        ]
        raw_retry = llm.invoke(repair_messages).content
        try:
            return extract_json(raw_retry)
        except ValueError:
            logger.warning("llm.json role=%s still unparseable after retry, using default", role)
            return default

    def _record_thinking(self, role: str, response) -> None:
        """Appends a response's reasoning trace to ``thinking_log``, if one was captured.

        No-op if ``show_thinking`` was off for this call (the invoke above
        never requested ``reasoning=True``, so ``reasoning_content`` won't
        be present) or if the model produced no reasoning content even
        with it on.
        """
        thinking = response.additional_kwargs.get("reasoning_content")
        if thinking:
            self.thinking_log.append((role, thinking))

    def drain_thinking_log(self) -> list[tuple[str, str]]:
        """Returns every ``(role, reasoning_content)`` entry captured since the last drain, clearing it.

        Intended to be called once per turn (e.g. by the CLI's ``\\thinking``
        mode after ``Session.ask()`` returns) so entries never leak into a
        later, unrelated turn.
        """
        entries, self.thinking_log = self.thinking_log, []
        return entries

    def _with_temperature(self, temperature: Optional[float]) -> ChatOllama:
        """Returns a chat client variant with a different temperature, if requested.

        Uses ``model_copy(update=...)`` rather than ``.bind(temperature=...)``:
        ``ChatOllama.bind()`` passes extra kwargs straight through to the
        underlying ``ollama.Client.chat()`` call instead of routing them
        through its own typed fields, which raises
        ``TypeError: Client.chat() got an unexpected keyword argument
        'temperature'`` — confirmed against the installed
        ``langchain-ollama`` version. ``model_copy`` sets the field
        properly on a new pydantic-model instance instead.
        """
        return self._chat if temperature is None else self._chat.model_copy(update={"temperature": temperature})

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of strings with the shared embedding model.

        Args:
            texts: Strings to embed.

        Returns:
            One embedding vector per input string, same order.
        """
        return self._embeddings.embed_documents(texts)


def extract_json(text: str) -> Any:
    """Extracts and parses a JSON value from raw LLM output.

    Tries, in order: the whole string as-is; the contents of a fenced
    ```json``` or ``` ``` block; the substring between the first ``{``/``[``
    and its matching last ``}``/``]``. Local reasoning models routinely wrap
    JSON in prose or markdown fences despite being told not to — this is
    the single place that tolerates it, so every call site in this package
    can assume ``llm.json()`` either returns real data or ``default``.

    Args:
        text: Raw model output.

    Returns:
        The parsed JSON value.

    Raises:
        ValueError: If no parseable JSON value can be found.
    """
    candidates = [text.strip()]

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    first_brace = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if first_brace != -1:
        opener = text[first_brace]
        closer = "}" if opener == "{" else "]"
        last_close = text.rfind(closer)
        if last_close > first_brace:
            candidates.append(text[first_brace : last_close + 1].strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no parseable JSON found in: {text[:200]!r}")


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns:
        A value in ``[-1, 1]``, or ``0.0`` if either vector is all-zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
