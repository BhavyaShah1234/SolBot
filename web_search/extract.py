"""Full-page fetch + content cleaning.

This is the hard part of the tool: raw fetched HTML is full of navigation,
ads, cookie banners, related-article rails, and footers. ``trafilatura``
locates the actual main-content region via DOM heuristics and discards the
rest, instead of keeping everything the way a blanket HTML-to-markdown
conversion would.
"""

from dataclasses import dataclass
from typing import Optional

from web_search.config import read_config

_USER_AGENT = "SolBot/1.0"


class ExtractionFailed(RuntimeError):
    """Raised when a fetched page yields no usable main-content text.

    Distinct from network/HTTP errors (which propagate as whatever
    ``requests`` raises) -- this means "the page loaded fine but there was
    nothing worth keeping," an expected, catchable outcome so a caller can
    move on to the next search result instead of treating it as a hard
    failure.
    """


@dataclass
class ExtractedPage:
    """Cleaned main content of one fetched page."""

    title: str
    url: str
    text: str
    truncated: bool


def _is_blocked(url: str, blocked_domains: list[str]) -> bool:
    return any(domain in url for domain in blocked_domains)


def fetch_page(url: str, cfg: Optional[dict] = None) -> ExtractedPage:
    """Fetches one URL and extracts its cleaned main-content text.

    Args:
        url: The page to fetch.
        cfg: The ``web`` section of the config (as returned by
            :func:`web_search.config.read_config`). If ``None``, the
            default ``config.yaml`` is loaded.

    Returns:
        The cleaned page content, truncated to ``cfg["max_fetch_chars"]``
        if longer (``truncated=True`` in that case).

    Raises:
        ValueError: If ``url``'s domain is in ``cfg["blocked_domains"]`` --
            checked here too, not just in :func:`web_search.search.search`,
            so a caller can't bypass the blocklist by fetching a URL
            directly.
        NotImplementedError: If ``cfg["extraction_backend"]`` isn't
            ``"trafilatura"``.
        ExtractionFailed: If no usable main-content text could be extracted
            (page is too short, JS-only shell, extractor found nothing).
        requests.RequestException: On network failure, timeout, or non-2xx
            response -- left to propagate; this module doesn't decide how a
            caller should react to a failed fetch.
    """
    cfg = cfg or read_config()

    blocked_domains = cfg["blocked_domains"]
    if _is_blocked(url, blocked_domains):
        raise ValueError(f"refusing to fetch blocked domain: {url}")

    backend = cfg["extraction_backend"]
    if backend != "trafilatura":
        raise NotImplementedError(f"unsupported web.extraction_backend: {backend!r}")

    import requests

    response = requests.get(
        url,
        timeout=cfg["fetch_timeout_seconds"],
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()

    import trafilatura

    text = trafilatura.extract(
        response.text,
        url=url,
        include_comments=False,
        include_tables=True,
        output_format="markdown",
    )

    min_chars = cfg["min_extracted_chars"]
    if not text or len(text) < min_chars:
        raise ExtractionFailed(f"no usable content extracted from {url}")

    max_chars = cfg["max_fetch_chars"]
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    metadata = trafilatura.extract_metadata(response.text)
    title = metadata.title if metadata and metadata.title else url

    return ExtractedPage(title=title, url=url, text=text, truncated=truncated)
