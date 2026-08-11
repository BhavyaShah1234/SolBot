# db_engine

The "DB Engine" component of SolBot: owns the Chroma vector store backing ASU Research Computing's RAG chatbot. It accepts fetch requests from any caller (a script today, an `agents`/worker package eventually), keeps the store in sync with the source Confluence space, and prevents races between concurrent readers/writers. **It never generates text** — every function here returns raw retrieved context or a sync-operation summary; producing an answer or finding from that context is always the caller's job.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN
python -m db_engine          # first run: full build (real embedding calls, can take a while)
python -m db_engine          # second run: near-instant, reports all-unchanged
```

```bash
python -m db_engine sync-page 1852637228   # targeted update of one page by Confluence page id
python -m db_engine reset                   # admin: wipe and rebuild the entire store from scratch
```

```python
from db_engine import fetch

results = fetch("What GPUs does the Sol cluster have?")
for doc, score in results:
    print(round(score, 3), doc.metadata["title"], doc.metadata["source"])
```

That's enough to go from a fresh checkout to a working call. Everything below is context for when you need more than the happy path.

## Public API

All five are re-exported from `db_engine/__init__.py` as thin wrappers over a process-wide cached engine (see "Concurrency model" below) — you never construct anything yourself in normal use.

- **`fetch(query, cfg=None, top_k=None, score_threshold=None) -> list[tuple]`**
  Embeds `query`, runs a cosine similarity search, and returns `(Document, score)` tuples for every chunk scoring at or above `score_threshold` (default from `config.yaml`'s `retrieval.score_threshold`), most relevant first. `Document.metadata` carries `id` (the real Confluence page id), `title`, `source` (a full wiki URL), `chunk_index`, and `chunk_id` (see below).

- **`get_all_chunks(cfg=None) -> list[Chunk]`**
  Reads every chunk currently in the collection, including its embedding vector. `Chunk` is a dataclass: `id`, `text`, `metadata`, `embedding: list[float]`. Unlike `fetch`, this isn't a query-time search — it's the whole corpus, for callers that need more than dense similarity search alone (e.g. building a sparse/BM25 index on top, then fusing with `fetch()`'s dense results). Read-only, no locking, same reasoning as `fetch`.

  **Matching a `fetch()` result to a `get_all_chunks()` entry**: both carry a `chunk_id` field (in `Document.metadata["chunk_id"]` and `Chunk.metadata["chunk_id"]` respectively — also equal to `Chunk.id` itself) with the same value for the same underlying chunk. Use that field directly rather than reconstructing it from `id`/`chunk_index` — the reconstruction (`f"{metadata['id']}::{metadata['chunk_index']}"`) happens to match `chunking.py`'s internal id-formatting convention today, but relying on that convention from outside this package is an avoidable coupling; the explicit `chunk_id` field is the supported way to do this.

- **`sync_page(page_id, cfg=None) -> dict`**
  Fetches one Confluence page by id and updates the vector store if it changed. Returns `{"page_id": ..., "status": "added"|"updated"|"unchanged", "chunks_written": ...}`.

- **`sync_all(cfg=None) -> dict`**
  Full manifest-diffed sync across the whole Confluence space: fetches every current page, upserts new/changed ones, deletes anything removed upstream. Returns `{"added", "updated", "unchanged", "deleted", "chunks_written"}` counts. This is what `python -m db_engine` runs by default.

- **`reset(cfg=None) -> dict`**
  Admin operation: deletes and recreates the collection, clears the sync manifest, then runs a fresh `sync_all()`. For when the store and Confluence have drifted enough that a diff-sync isn't trusted to recover cleanly. Returns the same shape as `sync_all()`.

`read_config(path="config.yaml") -> dict` and the `DBEngine` class itself are also exported — see below.

## CLI

```
python -m db_engine               # sync-all (default)
python -m db_engine sync-all      # same, explicit
python -m db_engine sync-page ID  # targeted update of one page
python -m db_engine reset         # wipe and rebuild everything
```

Running via the CLI configures file logging to `config.yaml`'s `logging.log_file` (`db_engine.log` by default) — sync/reset operation summaries are logged there at INFO level. This is CLI-only: importing `db_engine` as a library elsewhere never configures logging handlers on your behalf (standard library-vs-application logging practice) — plug in your own handlers via `logging.getLogger("db_engine")` if you want its log records elsewhere.

## Concurrency model

Reads (`fetch`, `get_all_chunks`) take no lock at all — an occasional read of mid-write state is an acceptable tradeoff for a RAG system, and blocking reads on writes would hurt retrieval latency for no real benefit.

Writes (`sync_page`, `sync_all`, `reset`) serialize through **two layers**:

1. **In-process `threading.Lock`**, held by the `DBEngine` instance — serializes writers within the same process (e.g. several worker threads/tasks sharing one process).
2. **Cross-process `filelock`** (`db_engine/concurrency.py`) — serializes writers across separate OS processes (e.g. an admin CLI run racing a live server process). `threading.Lock` alone can't coordinate across process boundaries, so this layer can't be dropped.

`get_engine(cfg=None) -> DBEngine` is a thread-safe lazy singleton: the first caller in a process initializes it (loading `config.yaml` if no `cfg` is passed), every later call in that process reuses the same cached instance — avoiding rebuilding the Chroma/Ollama clients on every call. `DBEngine` itself stays directly instantiable outside the singleton (`DBEngine(cfg)`), so isolated tests can build their own instance against a temp directory without touching shared state.

## Config

Reads these `config.yaml` sections: `confluence` (source URL/space/credentials-adjacent settings), `chunking` (chunk size/overlap), `embedding` (model name), `vector_store` (persist directory, collection name, distance metric, manifest path, lock path), `logging`, and `retrieval` — though only `retrieval.top_k` and `retrieval.score_threshold` are read by this package; any other keys under `retrieval` belong to a different package's own retrieval logic built on top of `get_all_chunks()`.

Credentials (`CONFLUENCE_USERNAME`, `CONFLUENCE_API_TOKEN`) come from `.env`, never from `config.yaml`.

## What this package deliberately does not do

No generation, no chat loop, no query decomposition, no web-search fallback, no conversation memory. It's a data layer — retrieval and freshness only. All of the above are the responsibility of other packages (`web_search`, `memory_engine`, and whatever orchestrates them).

## Verified behavior

Against the real Confluence RC space (164 pages, 1241 chunks), not just in isolation: a full sync, a truly idempotent no-op re-run, a targeted single-page update, deletion handling for pages removed upstream, a full reset/rebuild, and the two-layer lock holding under concurrent reads/writes — both multiple in-process threads and multiple separate OS processes — with no corruption or lost writes. `get_all_chunks()` verified to return the full 1241-chunk corpus with real embedding vectors attached.
