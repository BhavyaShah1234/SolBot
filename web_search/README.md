# `web_search`

Open-web search and page-content extraction for SolBot's future worker agents.

Two capabilities, kept as separate calls so a caller controls when it pays
the cost of a full page fetch:

- **`search(query)`** — cheap: query the web, get back titles/URLs/snippets.
- **`fetch_page(url)`** — the hard part: fetch a page and extract its
  cleaned main-content text, discarding navigation, ads, cookie banners,
  and boilerplate.

Standalone by design: no dependency on `db_engine`, no dependency on an
orchestrator or worker (SolBot's revamp doesn't have one yet). Pure
read-only — no writes, no locking, no shared mutable state — so any number
of agents can call it concurrently, from separate threads or separate
processes, with zero coordination.

## Why this exists

SolBot's revamp plan scopes web search to the end of its Phase 2
(orchestrator + worker agents), triggered when a worker's RAG evidence
turns up thin. Phase 2 doesn't exist yet. This package was built ahead of
that sequencing anyway, as a self-contained tool a future worker will call
— not something that depends on Phase 2 machinery itself.

## Installation

```bash
pip install ddgs trafilatura requests
```

(Also listed in the repo's root `requirements.txt`.) No API key or
credential is required for either `ddgs` (search) or `trafilatura`
(extraction).

## Configuration

Reads the `web:` section of `config.yaml` (repo root). Every key has a
built-in default, so the section — or the whole file — can be omitted
entirely for default behavior.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | If `false`, `search()` raises `RuntimeError` rather than returning `[]` — an empty list must always mean "no hits," never "misconfigured" |
| `search_provider` | `ddgs` | Only `"ddgs"` is supported today; anything else raises `NotImplementedError` |
| `extraction_backend` | `trafilatura` | Only `"trafilatura"` is supported today; anything else raises `NotImplementedError` |
| `max_results` | `5` | Default cap on `search()` hits, overridable per call via `max_results=` |
| `search_timeout_seconds` | `10` | `ddgs` query timeout |
| `fetch_timeout_seconds` | `15` | `requests.get` timeout in `fetch_page()` |
| `max_fetch_chars` | `12000` | Extracted text is truncated to this length; `ExtractedPage.truncated` is `True` if it was cut |
| `min_extracted_chars` | `200` | Below this, extraction is treated as failed (`ExtractionFailed`) rather than returned as thin, low-value content |
| `blocked_domains` | `[]` | Checked against every `search()` hit's URL *and* every `fetch_page()` URL — a blocked domain can't be fetched directly even if it didn't come from `search()` |

Example:

```yaml
web:
  enabled: true
  search_provider: ddgs
  extraction_backend: trafilatura
  max_results: 5
  search_timeout_seconds: 10
  fetch_timeout_seconds: 15
  max_fetch_chars: 12000
  min_extracted_chars: 200
  blocked_domains:
    - pinterest.com
    - quora.com
```

## Usage

### Python API

```python
from web_search import search, fetch_page, ExtractionFailed

# Cheap: titles, URLs, snippets only.
hits = search("What GPUs does the Sol cluster have?")
for hit in hits:
    print(hit.title, hit.url, hit.snippet)

# Expensive but high-value: fetch + clean the actual page content.
try:
    page = fetch_page(hits[0].url)
    print(page.title)
    print(page.text)          # cleaned main content, not raw HTML
    print(page.truncated)     # True if cut at max_fetch_chars
except ExtractionFailed:
    ...  # this URL didn't pan out (thin/JS-only page) — try the next hit
```

`search()` and `fetch_page()` both accept an optional `cfg` dict (the
`web:` section, as returned by `web_search.read_config()`) if you don't
want the default `config.yaml` loaded on every call — useful for tests or
for reusing one loaded config across many calls.

### CLI

Manual verification harness — no LLM call involved, just the raw tool
output for a human to eyeball:

```bash
python -m web_search search "What GPUs does the Sol cluster have?"
python -m web_search fetch "https://cores.research.asu.edu/research-computing/capabilities"
```

## Public API

| Name | Kind | Description |
|---|---|---|
| `search(query, cfg=None, max_results=None)` | function | Returns `list[SearchResult]` |
| `fetch_page(url, cfg=None)` | function | Returns `ExtractedPage` |
| `SearchResult` | dataclass | `title: str`, `url: str`, `snippet: str` |
| `ExtractedPage` | dataclass | `title: str`, `url: str`, `text: str`, `truncated: bool` |
| `ExtractionFailed` | exception (`RuntimeError` subclass) | Raised when a fetched page yields no usable content |
| `read_config(path="config.yaml")` | function | Loads and returns the `web:` section, merged over defaults |

## Exceptions

No `try`/`except` wraps the network call itself — failures propagate so
the caller decides how to react, rather than this module silently
degrading:

| Exception | Raised when |
|---|---|
| `RuntimeError` | `web.enabled` is `False` |
| `NotImplementedError` | `web.search_provider` or `web.extraction_backend` is set to an unsupported value |
| `ValueError` | `fetch_page()` is called on a URL matching `blocked_domains` |
| `ExtractionFailed` | Page fetched successfully but no usable main content was found (or it's shorter than `min_extracted_chars`) — typically a JS-only shell page |
| `requests.RequestException` (and subclasses) | Network failure, timeout, or non-2xx response in `fetch_page()` |

## Design notes

- **Why `search()` and `fetch_page()` are separate calls, not one
  auto-fetching function:** a calling agent controls when it pays the
  latency cost of a full page fetch — search cheaply first, then fetch
  only the promising results.
- **Why no locking, no singleton:** unlike `db_engine` (which serializes
  writes to a shared vector store with a two-layer lock), this package
  never mutates shared state. Every call is independent, so there's
  nothing to coordinate.
- **Why `ddgs` for search:** free, open-source, no API key, no formal
  quota. The realistic caveat: DuckDuckGo's unofficial backend can
  soft-rate-limit under heavy scraping volume — there's no contractual
  cap, but it isn't bulletproof either. A self-hosted SearXNG instance was
  considered and rejected as disproportionate infra for a single tool
  module at this stage.
- **Why `trafilatura` for extraction:** it's purpose-built for locating a
  page's actual main-content region and discarding navigation, ads,
  cookie banners, and related-link rails — the problem a blanket
  HTML-to-markdown conversion doesn't solve. Preferred over
  `readability-lxml` (older heuristics, less actively maintained).
- **Why `provider`/`backend` config keys exist even with one supported
  value each:** so adding a second provider later is a config change, not
  a rewrite — without building unused abstraction now.

## Verified

- Standalone sequential calls: `search()`, `fetch_page()`, and every
  exception path above.
- Concurrent calls from multiple threads in one process — both `search()`
  alone and mixed `search()`/`fetch_page()` calls.
- Concurrent calls from four separate OS processes simultaneously (the
  realistic shape of independent worker agents) — all completed correctly
  with no shared-state issues.
- Content-cleaning quality: manually checked against real ASU Research
  Computing pages — extracted text was the actual article/doc body, no
  nav/footer/ad content.

## Not yet wired into anything

No Phase 2 worker exists yet to call this package. When one does, the
expected pattern is: call `search()` first, then selectively `fetch_page()`
on promising results, catching `ExtractionFailed` to move on to the next
candidate.
