# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SolBot is a RAG-based CLI chatbot for ASU Research Computing. It answers questions about ASU supercomputers (Sol, Agave), HPC clusters, and RC administrative policies by retrieving relevant documentation chunks from a local Chroma vector store.

**The project is currently mid-revamp.** See "Revamp plan" below for the authoritative, in-progress design — it supersedes the architecture described later in this file wherever the two disagree. Read the revamp plan first before making changes.

## Repository layout

- `old/create_vector_db.py`, `old/create_chatbot.py` — the original, working v1 implementation. Kept as a reference for what to fix, not as code to build on. See "Known issues in v1" below.
- `new/` — an AI-generated (Claude web), unreviewed rewrite attempt (`config.py`, `dag.py`, `ingest.py`, `orchestrator.py`, `planner.py`, `worker.py`, etc.). It is **non-functional** — `main.py` and `test_solbot.py` import `from solbot import ...` but no `solbot` package exists anywhere on disk, and `new/__init__.py` itself has a broken self-referential import. Left untouched on disk for now (not deleted, not archived) but **not being built on** — none of its code is to be reused or imported. Its module boundaries (config/store/retrieval/memory/llm/tools/orchestrator/worker) and a few specific mechanisms (manifest-diffed incremental sync, query-rewriting for memory, RRF+MMR retrieval fusion, evidence-sufficiency gating for web search) are useful *design reference* for Phase 2 of the revamp plan below.
- Root level — where the revamp's Phase 1 files live once written (`config.py`, `ingest.py`, `retrieval_check.py`, `.env.example`, `requirements.txt`).

## Running the project (current v1, in `old/`)

**Prerequisites:** Ollama must be running locally. `old/` uses `nomic-embed-text` (embeddings) and `gemma3` (chat) — **note:** the revamp instead standardizes on `qwen3-embedding:latest` (embeddings) and `qwen3:latest` (chat), the models actually pulled locally; `old/`'s model names are being phased out, not the target going forward.

**Step 1 — Build the vector database** (run once, or to refresh docs):
```bash
python old/create_vector_db.py
```
Scrapes all pages from the `RC` Confluence space at `asurc.atlassian.net`, converts HTML to Markdown, splits by headers then by character, embeds, and persists to `./asu_rc` (Chroma).

**Step 2 — Run the chatbot:**
```bash
python old/create_chatbot.py
```
Interactive CLI loop. Type `\quit` to exit, `\clear` to reset conversation history. Logs written to `rough.log`.

### Known issues in v1 (driving the revamp)

1. **Semantic averaging** — embedding a compound query (e.g. "Which cluster has A100 GPUs and how do I request one for 4 hours?") produces one centroid vector between two unrelated topic regions, so top-k similarity search returns mediocre chunks from neither.
2. **Taxonomy is not a plan** — `classify_query` (`old/create_chatbot.py:24-40`) forces every query into exactly one of 6 mutually exclusive labels; real queries are multi-intent.
3. **Frozen knowledge** — `create_vector_db.py` does a one-shot `Chroma.from_documents` rebuild (`old/create_vector_db.py:40`) with no incremental sync or freshness signal. Also: hard `limit=500` with no pagination loop (`old/create_vector_db.py:20`), and a metadata bug where `'id': page['title']` uses the page title instead of the actual Confluence page id (`old/create_vector_db.py:28`).
4. **Closed world** — no access to anything outside the RC Confluence space (live cluster status, upstream Slurm/NVIDIA docs, web).
5. **Cost bug + latent bugs in memory** — `filter_messages_based_on_similarity` (`old/create_chatbot.py:59-66`) issues one LLM call per historical message per turn, and indexes `previous_messages[i + 1]` with no bounds check (IndexError risk). `get_previous_messages` (`old/create_chatbot.py:42-46`) has an always-true guard condition (dead code): `max_previous_turns = -2 * turns` is always negative, so `len(...) >= max_previous_turns` is always true. `SCORE_THRESHOLD = 0.7` (`old/create_chatbot.py:16`) is declared and never applied. Confluence `USERNAME`/`API_KEY` are hardcoded placeholders in the script (`old/create_vector_db.py:12-13`) rather than env vars. Also: inconsistent dict keys returned across the six `answer_*` handlers (`previous_message_referred` vs `previous_messages_referred`), `int(classify_query(...))` with no try/except (crashes on non-numeric LLM output), log filename `rough.log` is a leftover debug name.

---

## Revamp plan

### Context

A prior rewrite attempt (`new/`) produced an ambitious orchestrator/worker/DAG architecture but the code is broken and unreviewed (see "Repository layout" above). The plan is to rewrite from scratch, in phases, so every file is understood and owned by the author — the way `old/` was — while fixing the five known issues above.

**Scope clarification (important, narrower than "vector DB then chatbot" might suggest):** Phase 1 is vector-database-only — creation plus manual, trigger-based freshness updates. It explicitly does **not** include LLM generation or a working chat loop. Phase 1's "testing" means validating retrieval/augmentation quality with simple canned queries (simulating what a future worker agent would be handed), with no generation involved. The chatbot itself — decomposition, generation, memory, multi-intent handling — is Phase 2, built as an **orchestrator + worker agents** architecture (the orchestrator decomposes a compound query into sub-tasks; each worker researches its assigned sub-task against the vector store; results are synthesized into one reply). This is deliberately similar in shape to what `new/`'s `dag.py`/`orchestrator.py`/`planner.py`/`worker.py` were reaching for — useful as design reference, though none of that code is being reused. Web search (fixing "closed world") is explicitly deferred to the **end of Phase 2**, after the orchestrator/worker RAG-only pipeline is tested end-to-end; in Phase 2 it's triggered by worker agents when their assigned research turns up thin evidence, and that same evidence gap is what will (eventually) trigger vector-DB updates automatically — in Phase 1, the update trigger is manual only.

Models: `qwen3:latest` for generation (Phase 2 onward), `qwen3-embedding:latest` for embeddings (Phase 1 onward) — both confirmed as the models actually pulled in the local Ollama install, replacing `old/`'s `gemma3`/`nomic-embed-text`.

### Phase 1 — Vector Database: Creation, Freshness, Retrieval Validation

Scope: build and maintain the Chroma knowledge base, and prove retrieval quality on its own. No LLM chat loop, no generation.

**Target files** (root of repo, small and few by design — mirrors `old/`'s one-file-per-concern simplicity, scoped to ingestion only):
```
config.py            # small constants module + .env loader (no config.yaml yet — only add if constants sprawl)
ingest.py             # build/sync the vector DB (replaces old/create_vector_db.py)
retrieval_check.py    # manual validation harness: canned queries -> retrieved chunks + scores, no LLM call
.env.example          # CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN
requirements.txt
```

**Step 1 — `ingest.py`: correct the ingestion pipeline, one-shot build.** Carry over `old/create_vector_db.py`'s working parts (Confluence → markdownify → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` → `OllamaEmbeddings(qwen3-embedding:latest)` → Chroma), fixing: paginated `get_all_pages_from_space` (loop on `start` instead of a single `limit=500` call), correct `'id': page['id']` metadata (not title), credentials from `.env` via `config.py`, a sensible log filename. Verify: run `python ingest.py`, confirm page/chunk counts logged, collection exists at `./asu_rc`.

**Step 2 — `ingest.py`: manifest-diffed incremental sync (manual trigger).** JSON manifest (e.g. `./asu_rc/manifest.json`) mapping `page_id -> {version, hash}`. Each run: skip pages whose version+hash match the manifest; for new/changed pages, re-chunk with deterministic chunk ids (`f"{page_id}::{chunk_index}"`) and `upsert` into Chroma (idempotent); for pages removed upstream, delete their chunk ids and drop from the manifest. Needs direct `chromadb` collection access for id-based upsert/delete (`langchain_chroma.Chroma.from_documents` doesn't expose this cleanly — check whether `Chroma._collection` is sufficient or drop to the raw `chromadb` client). Trigger is manual: `python ingest.py` always does a full diff-sync (empty manifest = first full build, same code path). Verify: run twice back-to-back with no upstream changes — second run reports 0 changed/added/deleted, no embedding calls issued. Then hand-edit the manifest to simulate a changed page and confirm the diff picks it up.

**Step 3 — `retrieval_check.py`: retrieval/augmentation validation harness.** Standalone script, **no LLM generation call**: a handful of canned test queries standing in for what a future worker agent would be handed as an already-decomposed sub-task (e.g. "What GPUs does the Sol cluster have?", "How do I request a GPU allocation on Sol?", "What is ASU RC's appointment scheduling policy?"); run similarity search with `TOP_K` and (finally, actually) `SCORE_THRESHOLD` applied to filter weak matches; print/log each query's retrieved chunks with scores and source metadata for manual review. This is the practical proof that per-sub-question retrieval (the atomic case decomposition will produce in Phase 2) returns good chunks — the fix for issue #1 (semantic averaging), without yet needing an orchestrator to do the decomposing. Verify: run `python retrieval_check.py`, manually confirm each canned query's top chunks are topically correct.

**Open item:** check `langchain_chroma.Chroma`'s upsert/delete-by-id support against the installed version early in Step 2; drop to the raw `chromadb` client for the storage layer if it's too restrictive (keep `langchain_text_splitters` for chunking either way).

### Phase 2 — Orchestrator + Worker Agents (Generation, decomposition, memory)

Direction-level (to be re-planned in detail once Phase 1 is reviewed):

- **Orchestrator**: decomposes a compound query into sub-tasks (fixes #1 semantic averaging and #2 taxonomy-is-not-a-plan — intent becomes a per-sub-task tag, not one exclusive label for the whole message), dispatches to worker(s), synthesizes their findings into one reply.
- **Worker agent(s)**: each takes one sub-task, retrieves from the vector store (reusing/formalizing Phase 1's retrieval logic), and — RAG-only at first — produces a finding using `qwen3`. Web search is explicitly *not* in the first pass.
- **Memory**: single contextualization call per turn (rewrite query using recent history) replacing the O(turns) `filter_messages_based_on_similarity` cost bug; fix the dead-condition bug in `get_previous_messages` outright.
- **End-of-Phase-2 steps** (only after the orchestrator/worker RAG pipeline is tested end-to-end): add a `web_search` tool workers can call when their research turns up thin evidence (fixes #4 closed world); wire that same "thin evidence" signal to trigger `ingest.py`'s sync programmatically, closing the freshness loop (upgrading Phase 1's manual-only trigger).
- **Structure**: build this properly separated from the start (config, store, retrieval, memory, llm client, orchestrator, worker, tools, cli) rather than monolith-then-refactor — `new/`'s module boundaries are a reasonable reference for *what* to separate, though its DAG/LangGraph machinery and code are not reused. Apply DRY/SOLID/KISS as this is built.
- Also fix while rebuilding the chat side: inconsistent handler dict keys, unguarded `int(classify_query(...))` parsing, `rough.log` filename (see "Known issues in v1" above).

### Phase 3 — Adversarial testing (direction only)

Once Phase 2's modules exist to test in isolation: malformed/non-JSON LLM output on every structured call site; Ollama unreachable/timeout; Confluence API errors/rate limits/pagination edge cases; empty retrieval results; empty/short conversation history (regression test for the exact `previous_messages[i+1]` IndexError shape); concurrent ingest runs; web search timeouts/failures; prompt injection via retrieved or fetched content; unicode/markdown edge cases from `markdownify`; worker-failure and partial-synthesis cases specific to the orchestrator/worker design.

### Phase 4 — Web deployment (direction only)

Thin API layer over the orchestrator (one `/chat` endpoint + health check); session/history storage decision for multi-worker deployment (in-memory won't survive restarts — needs Redis/SQLite/etc., deferred); containerization (Ollama likely stays a separate service); scheduled ingest sync (cron/systemd timer/APScheduler, deferred); auth/rate-limiting once public-facing.

### Verification approach across Phase 1

- `python ingest.py` run twice consecutively → second run reports zero changes, zero embedding calls.
- Manual manifest edit (simulate a Confluence page change) → next run correctly re-embeds only that page.
- `python retrieval_check.py` → manually confirm each canned query's top-`TOP_K` chunks (post `SCORE_THRESHOLD` filter) are topically on-target.
- No LLM generation is exercised in Phase 1 by design — nothing to verify there yet.

### Credentials

Confluence username + API token must be moved to environment variables (`.env`, gitignored) before any ingestion runs — never hardcoded, unlike `old/create_vector_db.py`'s placeholders.
