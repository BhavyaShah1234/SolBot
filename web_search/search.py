"""Cheap web search: query -> titles/URLs/snippets, no page fetching.

Uses ``ddgs`` (DuckDuckGo Search) -- free, no API key, no formal quota.
"""

from dataclasses import dataclass
from typing import Optional

from web_search.config import read_config


@dataclass
class SearchResult:
    """One search hit. Deliberately thin -- snippet only, no fetched content."""

    title: str
    url: str
    snippet: str


def _is_blocked(url: str, blocked_domains: list[str]) -> bool:
    return any(domain in url for domain in blocked_domains)


def search(
    query: str,
    cfg: Optional[dict] = None,
    max_results: Optional[int] = None,
) -> list[SearchResult]:
    """Runs a web search and returns lightweight results.

    Args:
        query: The search query string.
        cfg: The ``web`` section of the config (as returned by
            :func:`web_search.config.read_config`). If ``None``, the
            default ``config.yaml`` is loaded.
        max_results: Overrides ``cfg["max_results"]`` for this call.

    Returns:
        A list of :class:`SearchResult`, most relevant first, with any hit
        matching ``cfg["blocked_domains"]`` filtered out.

    Raises:
        RuntimeError: If ``cfg["enabled"]`` is ``False``.
        NotImplementedError: If ``cfg["search_provider"]`` isn't ``"ddgs"``.
    """
    cfg = cfg or read_config()

    if not cfg["enabled"]:
        raise RuntimeError("web search is disabled in config")

    provider = cfg["search_provider"]
    if provider != "ddgs":
        raise NotImplementedError(f"unsupported web.search_provider: {provider!r}")

    limit = max_results or cfg["max_results"]
    blocked_domains = cfg["blocked_domains"]
    timeout = cfg["search_timeout_seconds"]

    from ddgs import DDGS

    results: list[SearchResult] = []
    with DDGS(timeout=timeout) as engine:
        for hit in engine.text(query, max_results=limit * 2):
            url = hit.get("href", "")
            if _is_blocked(url, blocked_domains):
                continue
            results.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=url,
                    snippet=hit.get("body", ""),
                )
            )
            if len(results) >= limit:
                break

    return results
