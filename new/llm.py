"""Language model and embedding access.

Two backends implement the same tiny interface:

``OllamaBackend``  Talks to a local Ollama server over its Hypertext Transfer
                   Protocol (HTTP) application programming interface (API)
                   using only the standard library. No LangChain wrapper is
                   involved, which keeps the code stable across LangChain
                   releases and gives direct access to Ollama's ``format:
                   json`` structured-output mode.

``StubBackend``    Returns canned, deterministic responses selected by a marker
                   embedded in the system prompt. This lets the whole pipeline
                   -- planner, workers, verifier, synthesiser -- run inside a
                   unit test with no model server and no network.

Every prompt sent through :meth:`LLM.json` carries a ``[[ROLE:<NAME>]]`` marker
as its first line. The marker is what the stub keys off, and it also makes
production traces trivially greppable by stage.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import Config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def extract_json(text: str) -> Any:
    """Pull the first complete JSON value out of a model response.

    Small local models wrap JSON in markdown fences, prepend "Sure, here is
    the plan:", or emit trailing commentary. This function strips fences, then
    brace-matches forward from the first ``{`` or ``[`` so that trailing prose
    cannot break the parse.

    Raises ``ValueError`` if no parseable JSON value is present.
    """
    cleaned = _FENCE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = min(
        (i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError(f"no JSON found in model output: {text[:200]!r}")

    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    depth, in_string, escaped = 0, False, False

    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : index + 1])

    raise ValueError(f"unbalanced JSON in model output: {text[:200]!r}")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class OllamaBackend:
    """Minimal Ollama client built on ``urllib``.

    ``chat`` maps to ``POST /api/chat`` and ``embed`` to ``POST /api/embed``.
    Streaming is disabled because every call site in SolBot needs the complete
    response before it can act on it.
    """

    def __init__(self, cfg: Config) -> None:
        self.host = cfg.get("llm.host", "http://localhost:11434").rstrip("/")
        self.chat_model = cfg.get("llm.chat_model", "gemma3")
        self.embedding_model = cfg.get("llm.embedding_model", "nomic-embed-text")
        self.temperature = float(cfg.get("llm.temperature", 0.1))
        self.json_temperature = float(cfg.get("llm.json_temperature", 0.0))
        self.num_ctx = int(cfg.get("llm.num_ctx", 8192))
        self.timeout = int(cfg.get("llm.request_timeout_seconds", 180))

    def _post(self, path: str, payload: dict) -> dict:
        """Send one JSON request and return the decoded JSON response."""
        request = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"Ollama request to {path} failed: {error}") from error

    def chat(self, messages: list[dict], *, force_json: bool, temperature: float | None) -> str:
        options = {
            "temperature": self.json_temperature if force_json else (
                self.temperature if temperature is None else temperature
            ),
            "num_ctx": self.num_ctx,
        }
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if force_json:
            # Ollama constrains decoding to valid JSON when this is set.
            payload["format"] = "json"
        return self._post("/api/chat", payload)["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.embedding_model, "input": texts}
        return self._post("/api/embed", payload)["embeddings"]


class StubBackend:
    """Deterministic backend used by the test suite.

    Responses are chosen by the ``[[ROLE:NAME]]`` marker in the system prompt.
    Handlers can be overridden per test via :meth:`register`.
    """

    def __init__(self, cfg: Config) -> None:
        self._handlers: dict[str, Callable[[str], str]] = {}
        self.calls: list[tuple[str, str]] = []  # (role, user prompt) for assertions

    def register(self, role: str, handler: Callable[[str], str]) -> None:
        """Bind a callable that receives the user prompt and returns raw text."""
        self._handlers[role.upper()] = handler

    @staticmethod
    def _role_of(messages: list[dict]) -> str:
        for message in messages:
            match = re.search(r"\[\[ROLE:([A-Z_]+)\]\]", message.get("content", ""))
            if match:
                return match.group(1)
        return "UNKNOWN"

    def chat(self, messages: list[dict], *, force_json: bool, temperature: float | None) -> str:
        role = self._role_of(messages)
        user_prompt = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        self.calls.append((role, user_prompt))
        handler = self._handlers.get(role)
        if handler is None:
            return "{}" if force_json else ""
        return handler(user_prompt)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Hash-based pseudo-embeddings: stable, cheap, and non-degenerate."""
        vectors = []
        for text in texts:
            vector = [0.0] * 16
            for token in text.lower().split():
                vector[hash(token) % 16] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #


class LLM:
    """The object every other module depends on.

    Exposes three methods:

    ``text``  free-form completion
    ``json``  completion constrained and parsed into a Python object
    ``embed`` batch embeddings
    """

    def __init__(self, cfg: Config) -> None:
        backend_name = cfg.get("llm.backend", "ollama")
        self.backend = StubBackend(cfg) if backend_name == "stub" else OllamaBackend(cfg)
        self.cfg = cfg

    @staticmethod
    def _messages(role: str, system: str, user: str) -> list[dict]:
        """Build the message list, stamping the role marker onto the system turn."""
        return [
            {"role": "system", "content": f"[[ROLE:{role.upper()}]]\n{system}"},
            {"role": "user", "content": user},
        ]

    def text(self, role: str, system: str, user: str, temperature: float | None = None) -> str:
        return self.backend.chat(
            self._messages(role, system, user), force_json=False, temperature=temperature
        )

    def json(self, role: str, system: str, user: str, default: Any = None) -> Any:
        """Constrained completion, parsed.

        On a parse failure the call is retried once with an explicit repair
        instruction. If that also fails, ``default`` is returned so that a
        malformed plan degrades into a fallback path instead of a stack trace.
        """
        raw = self.backend.chat(
            self._messages(role, system, user), force_json=True, temperature=None
        )
        try:
            return extract_json(raw)
        except ValueError:
            logger.warning("%s produced unparseable JSON, retrying once", role)

        repair = f"{user}\n\nYour previous reply was not valid JSON. Reply with JSON only."
        raw = self.backend.chat(
            self._messages(role, system, repair), force_json=True, temperature=None
        )
        try:
            return extract_json(raw)
        except ValueError:
            logger.error("%s failed JSON parse twice; using default", role)
            return default

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed(texts)
