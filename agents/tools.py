"""The tool boundary between a ``Worker`` and everything it can call.

Every tool returns a uniform :class:`ToolResult` so the worker's ReAct
loop never special-cases a tool by name, and :meth:`ToolRegistry.call`
never raises — a failing tool becomes an ``ok=False`` observation the
worker's own reasoning can react to (retry differently, try another tool,
give up on this angle), not a crash. Design mined from ``new/tools.py``
(reference only, not reused as code).

``web_search``'s own functions expect the ``web`` *section* of config as
their ``cfg`` argument (``cfg["enabled"]``, not ``cfg["web"]["enabled"]``)
— every wrapper here is careful to pass ``cfg["web"]``, not the full
``agents`` config dict, when calling into that package.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import web_search
from agents.fusion import Evidence, Retriever

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform return shape for every tool call."""

    ok: bool
    content: str
    evidence: list[Evidence] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Tool:
    """One registered tool."""

    name: str
    description: str
    arguments: str  # human-readable description of expected arguments, for the worker's prompt
    run: Callable[[dict], ToolResult]

    def spec(self) -> str:
        """Renders this tool as one line for a system prompt's tool listing."""
        return f"- {self.name}({self.arguments}): {self.description}"


class ToolRegistry:
    """Holds a set of tools and dispatches calls to them, never raising."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def render_specs(self) -> str:
        """Renders every registered tool as a newline-joined listing.

        Adding a tool is registering it here — there's no second place to
        keep this listing in sync.
        """
        return "\n".join(tool.spec() for tool in self._tools.values())

    def call(self, name: str, arguments: dict) -> ToolResult:
        """Calls a tool by name. Never raises.

        Args:
            name: A registered tool's name.
            arguments: Arguments to pass to the tool's ``run`` function.

        Returns:
            The tool's :class:`ToolResult`, or an ``ok=False`` result if
            the tool doesn't exist or its ``run`` function raised.
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("call: unknown tool %r", name)
            return ToolResult(ok=False, content=f"no such tool: {name!r}")
        try:
            result = tool.run(arguments)
            logger.info("call: tool=%s ok=%s evidence=%d", name, result.ok, len(result.evidence))
            if not result.ok:
                logger.warning("call: tool=%s failed: %s", name, result.content[:200])
            return result
        except Exception as exc:  # noqa: BLE001 - a failing tool must never crash the worker
            logger.warning("call: tool=%s raised %s: %s", name, type(exc).__name__, exc)
            return ToolResult(ok=False, content=f"{type(exc).__name__}: {exc}")


def make_vector_search(retriever: Retriever) -> Tool:
    """Builds the ``vector_search`` tool: fused dense+sparse retrieval over the RC knowledge base."""

    def run(arguments: dict) -> ToolResult:
        query = arguments.get("query", "")
        if not query:
            return ToolResult(ok=False, content="vector_search requires a 'query' argument")
        result = retriever.retrieve(query)
        if not result.evidence:
            return ToolResult(ok=False, content="no matching documentation found", meta={"sufficient": False})
        content = "\n\n".join(f"[{e.title}]\n{e.text}" for e in result.evidence)
        return ToolResult(
            ok=True,
            content=content,
            evidence=result.evidence,
            meta={"sufficient": result.is_sufficient, "top_score": result.top_score},
        )

    return Tool(
        name="vector_search",
        description="Search ASU Research Computing's documentation (Sol/Agave clusters, Slurm, storage, policies).",
        arguments="query: str",
        run=run,
    )


def make_web_search(cfg: dict) -> Tool:
    """Builds the ``web_search`` tool: open-web search, snippets only, no page fetch."""
    web_cfg = cfg["web"]

    def run(arguments: dict) -> ToolResult:
        query = arguments.get("query", "")
        if not query:
            return ToolResult(ok=False, content="web_search requires a 'query' argument")
        hits = web_search.search(query, cfg=web_cfg)
        if not hits:
            return ToolResult(ok=False, content="no web results found")
        content = "\n".join(f"- {h.title} ({h.url}): {h.snippet}" for h in hits)
        evidence = [
            Evidence(text=h.snippet, source=h.url, title=h.title, origin="web", doc_id=h.url) for h in hits
        ]
        return ToolResult(ok=True, content=content, evidence=evidence, meta={"count": len(hits)})

    return Tool(
        name="web_search",
        description="Search the open web. Use when RC documentation doesn't cover the question (e.g. upstream Slurm/NVIDIA docs, live status).",
        arguments="query: str",
        run=run,
    )


def make_fetch_url(cfg: dict) -> Tool:
    """Builds the ``fetch_url`` tool: fetch and clean one page's main content."""
    web_cfg = cfg["web"]

    def run(arguments: dict) -> ToolResult:
        url = arguments.get("url", "")
        if not url:
            return ToolResult(ok=False, content="fetch_url requires a 'url' argument")
        try:
            page = web_search.fetch_page(url, cfg=web_cfg)
        except web_search.ExtractionFailed as exc:
            return ToolResult(ok=False, content=str(exc))
        evidence = [Evidence(text=page.text, source=page.url, title=page.title, origin="web", doc_id=page.url)]
        return ToolResult(ok=True, content=page.text, evidence=evidence, meta={"truncated": page.truncated})

    return Tool(
        name="fetch_url",
        description="Fetch one URL's cleaned page content. Use on a promising web_search result, not blindly.",
        arguments="url: str",
        run=run,
    )


def make_current_time(cfg: dict) -> Tool:
    """Builds the ``current_time`` tool: the current time in the configured app timezone."""
    tz = ZoneInfo(cfg["app"]["timezone"])

    def run(arguments: dict) -> ToolResult:  # noqa: ARG001 - no arguments needed
        now = datetime.now(tz)
        return ToolResult(ok=True, content=now.strftime("%A, %Y-%m-%d %H:%M %Z"))

    return Tool(
        name="current_time",
        description="Get the current date/time in the configured local timezone.",
        arguments="(no arguments)",
        run=run,
    )


def build_registry(cfg: dict, retriever: Retriever) -> ToolRegistry:
    """Builds a registry with all four standard tools registered."""
    registry = ToolRegistry()
    registry.register(make_vector_search(retriever))
    registry.register(make_web_search(cfg))
    registry.register(make_fetch_url(cfg))
    registry.register(make_current_time(cfg))
    return registry
