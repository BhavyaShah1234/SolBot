"""Confluence access for the DB Engine.

Wraps ``atlassian.Confluence`` to fetch pages from the RC space, either all
at once (for a full sync) or one at a time by id (for a targeted update).
Deliberately avoids ``Confluence.get_all_pages_from_space()``: its own
built-in pagination loop stops after exactly one page whenever the space is
larger than one page-size (its terminator condition, ``len(results) <=
limit``, is true on the very first full batch). ``get_all_pages_from_space_raw``
is called directly instead, with the pagination loop driven here.
"""

from atlassian import Confluence

from db_engine.config import require_confluence_credentials


def _client(cfg: dict) -> Confluence:
    """Builds an authenticated Confluence client from config and environment.

    Args:
        cfg: The loaded configuration dict; only ``cfg["confluence"]["url"]``
            is used here.

    Returns:
        An authenticated ``atlassian.Confluence`` client.

    Raises:
        RuntimeError: If Confluence credentials are missing from the
            environment (propagated from ``require_confluence_credentials``).
    """
    username, api_token = require_confluence_credentials()
    return Confluence(url=cfg["confluence"]["url"], username=username, password=api_token, cloud=True)


def fetch_all_pages(cfg: dict) -> list[dict]:
    """Fetches every current page in the configured Confluence space.

    Drives its own ``start``/``limit`` pagination loop rather than relying
    on the client library's built-in one (see module docstring for why).

    Args:
        cfg: The loaded configuration dict; reads ``cfg["confluence"]``.

    Returns:
        A list of page dicts, one per page, each shaped by
        :func:`_to_page_dict`.

    Raises:
        RuntimeError: If Confluence credentials are missing.
        requests.HTTPError: If the Confluence API request fails.
    """
    client = _client(cfg)
    conf = cfg["confluence"]
    pages: list[dict] = []
    start, limit = 0, conf["page_fetch_page_size"]
    while True:
        response = client.get_all_pages_from_space_raw(
            space=conf["space"],
            start=start,
            limit=limit,
            expand="body.view,version",
            status="current",
            content_type="page",
        )
        batch = response.get("results", [])
        pages.extend(batch)
        if len(batch) < limit:
            break
        start += limit
    return [_to_page_dict(p, conf["url"]) for p in pages]


def fetch_page(page_id: str, cfg: dict) -> dict:
    """Fetches a single Confluence page by id.

    Args:
        page_id: The Confluence page id to fetch.
        cfg: The loaded configuration dict; reads ``cfg["confluence"]``.

    Returns:
        A page dict shaped by :func:`_to_page_dict`.

    Raises:
        RuntimeError: If Confluence credentials are missing.
        atlassian.errors.ApiError: If no page exists with the given id, or
            the configured credentials lack permission to view it.
    """
    client = _client(cfg)
    conf = cfg["confluence"]
    page = client.get_page_by_id(page_id, expand="body.view,version")
    return _to_page_dict(page, conf["url"])


def _to_page_dict(page: dict, base_url: str) -> dict:
    """Normalizes a raw Confluence API page payload into the shape used downstream.

    Args:
        page: A single page's raw JSON payload from the Confluence API,
            expected to have been fetched with ``expand="body.view,version"``.
        base_url: The Confluence base URL, prefixed onto the page's web UI
            link to build a full source URL.

    Returns:
        A dict with keys ``id``, ``title``, ``version``, ``source``, and
        ``html`` (the raw HTML body).
    """
    return {
        "id": page["id"],
        "title": page.get("title", ""),
        "version": page.get("version", {}).get("number"),
        "source": base_url + page.get("_links", {}).get("webui", ""),
        "html": page.get("body", {}).get("view", {}).get("value", ""),
    }
