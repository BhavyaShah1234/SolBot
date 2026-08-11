# memory_engine

SolBot's conversation memory component: SQLite-backed storage and retrieval for chat history, long-term user facts, and cross-session recall — built for [SolBot](../CLAUDE.md)'s Phase 2 revamp, sibling to [`db_engine`](../db_engine) (the vector-DB/retrieval layer).

## What it does

- **Stores conversation turns** as a session appends messages, one row per `(user, assistant)` exchange plus optional system messages.
- **Fetches recent context** for a session — the recency window a future contextualizer would rewrite a follow-up question against.
- **Recalls relevant exchanges from a user's *other* sessions** via embedding similarity, so a past conversation about the same topic can surface even if it happened days ago in a different session.
- **Persists long-term facts about a user** (name, timezone, preferences, anything personalization-relevant) that outlive any single session, with an idempotent upsert and a periodic, batched extraction-trigger contract instead of a per-turn cost.
- **Guarantees no lost writes under concurrency** — the hard requirement this package was built around — verified under both concurrent threads and concurrent separate OS processes.

## What it deliberately does *not* do

`memory_engine` makes **zero generative LLM calls**. Turning a follow-up question into a standalone query, and reasoning over a transcript to extract facts, both require an LLM call — and both are left to a future `agents` package. This mirrors `db_engine`'s own shape: `db_engine.fetch()` embeds and matches but never generates an answer; `memory_engine` stores and retrieves but never reasons. Two direct benefits of that split:

- Insight extraction can be toggled on or off purely by whether an orchestrator ever calls it — no flag needed inside this package.
- This package is fully testable and useful with zero dependency on any particular LLM provider or prompt design.

The one exception is *embedding* (not generation) — per-turn recall embeddings are computed here, the same way `db_engine` embeds Confluence chunks for storage. Representation is this package's job; reasoning isn't.

## Public API

```python
import memory_engine

memory_engine.append_message(session_id, role, content, metadata=None)
# role is one of "system", "user", "assistant". Increments turn bookkeeping on "user";
# triggers a best-effort recall embedding of the completed turn on "assistant".

memory_engine.get_recent_context(session_id, turns=None, include_system=False)
# -> [{"turn_number", "role", "content", "created_at"}, ...], chronological.

memory_engine.get_profile_facts(session_id)
# -> {fact_key: fact_value}, resolved by user_id -- survives across that user's sessions.

memory_engine.recall_related(session_id, query, top_k=None)
# -> relevant turns from this user's OTHER sessions, best first:
#    [{"session_id", "turn_number", "user_content", "assistant_content", "score", "created_at"}, ...]

memory_engine.needs_extraction(session_id)
# -> bool. True once enough user turns have accumulated since the last extraction ack.

memory_engine.upsert_facts(session_id, facts, turns_covered)
# facts: [{"key", "value", "confidence"}, ...], produced by a future agent's LLM call.
# Must be called once per extraction attempt, even with facts=[] on failure -- see
# "Extraction ack contract" below.

memory_engine.clear_session(session_id)
# Deletes this session's messages and embeddings. Facts are NOT deleted -- they're
# user-scoped and meant to outlive an individual session clear.
```

All functions accept an optional `cfg` override; otherwise they use `memory_engine.read_config()` (backed by `config.yaml`).

## Extraction ack contract

`needs_extraction` only goes back to `False` once `upsert_facts` is called — not when an LLM call happens to succeed. Whoever attempts extraction (a future agent) must call `upsert_facts(session_id, facts, turns_covered)` when done, passing `facts=[]` if the attempt failed or found nothing. This is the same idea as a queue consumer acking a message once it's *handled*, regardless of whether handling it produced anything useful — skipping the ack leaves extraction re-attempted on every subsequent turn, reintroducing the exact per-turn-LLM-call cost bug (`old/create_chatbot.py`'s `filter_messages_based_on_similarity`) this package was built to avoid.

## Concurrency model

Two layers, chosen to guarantee no lost writes under concurrent access without a hand-rolled cross-process lock:

1. **SQLite WAL mode + `busy_timeout`** — SQLite serializes concurrent writers itself, blocking and retrying internally rather than needing `db_engine`'s `filelock`-based approach. Every write opens with `BEGIN IMMEDIATE` (not deferred `BEGIN`) to avoid the classic SQLite reader-upgrade race, with jittered retry on contention.
2. **Per-session in-process locking** (`concurrency.py`) — a lazily-created `threading.Lock` per `session_id`, so unrelated sessions never contend with each other within one process.

Reads take no lock at all. Because this package makes no generative LLM calls, it never needs to hold a lock open across a slow external call — a future agent calling `get_recent_context` → an LLM → `upsert_facts` gets that safety for free from three independently-locked primitives. The one place that still follows an unlock/call/relock shape is the per-turn recall embedding, since that embedding call does stay inside this package.

Verified in `test/memory_check.py`: zero lost writes under 8 concurrent threads and under 4 concurrent separate OS processes appending to the same session.

## Storage

SQLite at `cfg["memory"]["db_path"]` (default `./memory_store/memory.db`, gitignored — regenerable local data). Four tables: `sessions` (per-session/user bookkeeping and counters), `messages` (the transcript, paired into turns via a `turn_number` column — not list-index arithmetic, which is what made `old/create_chatbot.py`'s `previous_messages[i+1]` unsafe), `facts` (`user_id`-scoped, idempotent upsert), and `message_embeddings` (one row per completed turn, brute-force cosine-matched for cross-session recall).

## Configuration

```yaml
memory:
  db_path: "./memory_store/memory.db"
  default_session_id: "default"
  default_user_id: "default"
  default_context_window_turns: 6
  extraction_interval_turns: 8
  busy_timeout_ms: 5000
  write_retry_attempts: 3
  recall_top_k: 5
  recall_score_threshold: 0.7
  log_file: "memory_engine.log"
```

Reuses `db_engine`'s existing top-level `embedding` key (`qwen3-embedding:latest`) for recall — no chat model is configured anywhere in this package, since it never calls one.

## CLI

```bash
python -m memory_engine show SESSION_ID [--turns N]
python -m memory_engine facts SESSION_ID
python -m memory_engine needs-extraction SESSION_ID
python -m memory_engine upsert-fact SESSION_ID KEY VALUE [--confidence F] [--turns-covered N]
python -m memory_engine clear SESSION_ID
```

## Testing

`test/memory_check.py` — no orchestrator or `agents` package exists yet to exercise this against, so it drives `memory_engine` directly: a multi-turn simulation, an in-process and cross-process concurrency stress test, a fact-primitive test, and a cross-session recall test. Two of its checks make their own throwaway `ChatOllama` calls as explicit stand-ins for the future `agents` package's contextualizer and insight-extractor — that reasoning doesn't belong inside `memory_engine` itself. Run with:

```bash
python test/memory_check.py
```

Requires Ollama running locally with `qwen3:latest` and `qwen3-embedding:latest` pulled for the LLM-dependent checks; the storage and concurrency checks run regardless.

## What's next

This package exposes the data-layer primitives (`get_recent_context`, `needs_extraction`, `upsert_facts`) a future `agents` package's contextualizer and insight-extractor will call, and that a future orchestrator will wire into the actual per-turn chat flow. See [`CLAUDE.md`](../CLAUDE.md)'s Phase 2 plan for the surrounding architecture.
