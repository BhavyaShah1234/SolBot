"""The web_search tool: search the open web and fetch cleaned page content.

Pure read-only -- nothing here ever mutates shared state, so unlike
``db_engine`` (which serializes writes with two layers of locking), this
package needs no locking, no singleton, and no shared mutable state at all.
Every call is independent; any number of agents can call it concurrently
from separate threads or processes with zero coordination.

Two capabilities, kept separate so a caller controls when it pays the cost
of a full page fetch:

* :func:`search` -- cheap: query -> titles/URLs/snippets only.
* :func:`fetch_page` -- the hard part: fetch a page and extract its cleaned
  main-content text, discarding navigation/ads/boilerplate.

Standalone by design: no dependency on ``db_engine`` or any orchestrator.
"""

from web_search.config import read_config
from web_search.extract import ExtractedPage, ExtractionFailed, fetch_page
from web_search.search import SearchResult, search

__all__ = [
    "search",
    "fetch_page",
    "SearchResult",
    "ExtractedPage",
    "ExtractionFailed",
    "read_config",
]
