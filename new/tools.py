"""Tools available to worker agents.

A tool is a named, described, JSON-callable capability. The registry renders
its own documentation into the worker prompt, so adding a tool here makes it
immediately reachable by the agent -- there is no second place to update. That
is the structural difference from version 1, where each capability needed its
own ``answer_*`` function and its own branch in a hand-written ``if`` chain.

Tools currently registered
--------------------------
``vector_search``    Authoritative Research Computing documentation.
``web_search``       Open web, for upstream or general facts.
``fetch_url``        Full text of one page, optionally written through into the
                     vector store so the next question benefits from it.
``refresh_knowledge`` Trigger an incremental sync when docs look stale.
``current_time``     Grounds relative dates. Small models are otherwise happy
                     to invent "as of last month".
"""

from __future__ import annotations

import datetime
import logging
import re
import zoneinfo
from dataclasses import dataclass, field
from typing import Callable

from .config import Config
from .retrieval import Evidence, Retriever

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform return type so the worker never special-cases a tool."""

    ok: bool
    content: str
    evidence: list[Evidence] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Tool:
    """Name, human-readable contract, and the callable itself."""

    name: str
    description: str
    arguments: str
    run: Callable[..., ToolResult]

    def spec(self) -> str:
        return f"- {self.name}({self.arguments}): {self.description}"


class ToolRegistry:
    """Holds the tools and renders the prompt fragment describing them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def render_specs(self) -> str:
        return "\n".join(self._tools[name].spec() for name in sorted(self._tools))

    def call(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch with defensive error handling.

        A tool raising must never kill the request: the failure comes back as
        an observation the agent can reason about and route around.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"unknown tool {name!r}; available: {self.names()}")
        try:
            return tool.run(**arguments)
        except TypeError as error:
            return ToolResult(False, f"bad arguments for {name}: {error}")
        except Exception as error:  # noqa: BLE001
            logger.exception("tool %s failed", name)
            return ToolResult(False, f"{name} failed: {error}")


# --------------------------------------------------------------------------- #
# Individual tools
# --------------------------------------------------------------------------- #


def strip_html(html: str) -> str:
    """Crude tag stripper used when no markdown converter is installed."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_vector_search(retriever: Retriever) -> Tool:
    def run(query: str, k: int | None = None) -> ToolResult:
        result = retriever.retrieve(query)
        if not result.evidence:
            return ToolResult(False, f"no documentation matched: {result.reason}")
        rendered = "\n\n".join(
            f"[{i + 1}] ({item.title}) {item.text[:900]}"
            for i, item in enumerate(result.evidence[: k or len(result.evidence)])
        )
        return ToolResult(
            True,
            rendered,
            result.evidence,
            {"sufficient": result.is_sufficient, "top_score": result.top_score},
        )

    return Tool(
        "vector_search",
        "Search the ASU Research Computing documentation. Use this first for "
        "anything about Sol, Phoenix, Slurm on ASU clusters, storage, or RC policy.",
        'query: str, k: int optional',
        run,
    )


def make_web_search(cfg: Config) -> Tool:
    blocked = set(cfg.get("web.blocked_domains", []) or [])
    max_results = int(cfg.get("web.max_results", 5))

    def run(query: str, max_results_override: int | None = None) -> ToolResult:
        if not cfg.get("web.enabled", True):
            return ToolResult(False, "web search is disabled in configuration")
        from ddgs import DDGS  # noqa: PLC0415 - lazy, keeps the import optional

        limit = max_results_override or max_results
        hits = []
        with DDGS() as engine:
            for hit in engine.text(query, max_results=limit * 2):
                url = hit.get("href", "")
                if any(domain in url for domain in blocked):
                    continue
                hits.append(hit)
                if len(hits) >= limit:
                    break

        if not hits:
            return ToolResult(False, f"no web results for {query!r}")

        evidence = [
            Evidence(
                text=hit.get("body", ""),
                source=hit.get("href", ""),
                title=hit.get("title", ""),
                score=0.5,
                origin="web",
            )
            for hit in hits
        ]
        rendered = "\n\n".join(f"[{h.get('title')}] {h.get('body')}\n{h.get('href')}" for h in hits)
        return ToolResult(True, rendered, evidence)

    return Tool(
        "web_search",
        "Search the public web. Use for upstream Slurm or NVIDIA documentation, "
        "or anything not covered by ASU's own pages.",
        "query: str",
        run,
    )


def make_fetch_url(cfg: Config, knowledge_base=None) -> Tool:
    timeout = int(cfg.get("web.fetch_timeout_seconds", 20))
    max_chars = int(cfg.get("web.max_fetch_chars", 12000))
    write_through = bool(cfg.get("web.write_through_to_vector_store", True))

    def run(url: str) -> ToolResult:
        import urllib.request  # noqa: PLC0415

        request = urllib.request.Request(url, headers={"User-Agent": "SolBot/2.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")

        try:
            import markdownify  # noqa: PLC0415

            text = markdownify.markdownify(raw, heading_style="ATX")
        except ImportError:
            text = strip_html(raw)

        text = text[:max_chars]
        title = (re.search(r"<title[^>]*>(.*?)</title>", raw, re.S | re.I) or [None, url])[1]
        title = str(title).strip()

        if write_through and knowledge_base is not None:
            try:
                knowledge_base.ingest_web_page(url, title, text)
            except Exception:  # noqa: BLE001 - caching is best-effort
                logger.warning("write-through ingestion failed for %s", url)

        return ToolResult(
            True, text, [Evidence(text=text[:2000], source=url, title=title, score=0.6, origin="web")]
        )

    return Tool(
        "fetch_url",
        "Retrieve the full text of one web page. Use after web_search when a "
        "result snippet is promising but incomplete.",
        "url: str",
        run,
    )


def make_refresh_knowledge(knowledge_base) -> Tool:
    def run() -> ToolResult:
        report = knowledge_base.sync()
        return ToolResult(True, f"knowledge base synced: {report.summary()}", meta={"report": report})

    return Tool(
        "refresh_knowledge",
        "Re-sync the documentation index from Confluence. Use only when the "
        "documentation appears out of date or a page is expected but missing.",
        "no arguments",
        run,
    )


def make_current_time(cfg: Config) -> Tool:
    zone = cfg.get("app.timezone", "America/Phoenix")

    def run() -> ToolResult:
        now = datetime.datetime.now(zoneinfo.ZoneInfo(zone))
        return ToolResult(True, now.strftime("%A, %d %B %Y %H:%M %Z"))

    return Tool("current_time", "Current date and time in the deployment timezone.", "no arguments", run)


def build_registry(cfg: Config, retriever: Retriever, knowledge_base=None) -> ToolRegistry:
    """Assemble the registry the workers will see."""
    registry = ToolRegistry()
    registry.register(make_vector_search(retriever))
    registry.register(make_current_time(cfg))
    if cfg.get("web.enabled", True):
        registry.register(make_web_search(cfg))
        registry.register(make_fetch_url(cfg, knowledge_base))
    if knowledge_base is not None:
        registry.register(make_refresh_knowledge(knowledge_base))
    return registry
