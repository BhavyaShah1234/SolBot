"""Confluence access for the DB Engine.

Wraps ``atlassian.Confluence`` to fetch pages from the RC space, either all
at once (for a full sync) or one at a time by id (for a targeted update).
Uses the Cloud v2 API (``api_version=2``) rather than the library's default
legacy v1-compatible client: Confluence Cloud has retired the v1 ``content``
REST endpoints that the legacy client's ``get_all_pages_from_space``/
``get_all_pages_from_space_raw`` were built on (confirmed directly against a
live instance -- the legacy client's calls silently returned 0 results,
having received the site's HTML shell back instead of JSON). The v2 list
endpoint (``get_pages``) also can't return rendered ``view``-format bodies
itself (``body-format`` there is restricted to storage/atlas_doc_format/
markdown) the way the old one-shot bulk fetch could, so a full sync here is
a list-then-fetch-each-page's-view-body two-step instead of one bulk call.
"""

from atlassian import Confluence

from db_engine.config import require_confluence_credentials


def _client(cfg: dict) -> Confluence:
    """Builds an authenticated Confluence Cloud v2 client from config and environment.

    Args:
        cfg: The loaded configuration dict; only ``cfg["confluence"]["url"]``
            is used here.

    Returns:
        An authenticated ``atlassian.Confluence`` client backed by the
        Cloud v2 API (``api_version=2`` -- see module docstring for why).

    Raises:
        RuntimeError: If Confluence credentials are missing from the
            environment (propagated from ``require_confluence_credentials``).
    """
    username, api_token = require_confluence_credentials()
    return Confluence(
        url=cfg["confluence"]["url"], username=username, password=api_token, cloud=True, api_version=2
    )


def fetch_all_pages(cfg: dict) -> list[dict]:
    """Fetches every current page in the configured Confluence space.

    Lists every current page id in the space (cheap, no bodies), then fetches
    each one's rendered body individually -- see module docstring for why a
    single bulk call can't do both on the v2 API.

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
    space = client.get_space_by_key(conf["space"])
    # body_format is passed explicitly (rather than left at get_pages()'s own
    # default) because that default requests "body-format=none", which the
    # live API rejects outright (400: "none" isn't a valid
    # PrimaryBodyRepresentation) -- confirmed directly against this Confluence
    # Cloud instance. The listing's own body isn't used below (each page's
    # rendered view body is fetched individually instead), so the format
    # chosen here doesn't matter beyond being one the API will accept.
    listing = client.get_pages(
        space_id=space["id"], status="current", body_format="storage", limit=conf["page_fetch_page_size"]
    )
    return [
        _to_page_dict(client.get_page_by_id(page["id"], body_format="view"), conf["url"]) for page in listing
    ]


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
    page = client.get_page_by_id(page_id, body_format="view")
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
