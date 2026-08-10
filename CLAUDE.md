# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SolBot is a RAG-based CLI chatbot for ASU Research Computing. It answers questions about ASU supercomputers (Sol, Agave), HPC clusters, and RC administrative policies by retrieving relevant documentation chunks from a local Chroma vector store.

**The project is currently mid-revamp.** See "Revamp plan" below for the authoritative, in-progress design — it supersedes the architecture described later in this file wherever the two disagree. Read the revamp plan first before making changes.

## Repository layout

- `old/create_vector_db.py`, `old/create_chatbot.py` — the original, working v1 implementation. Kept as a reference for what to fix, not as code to build on. See "Known issues in v1" below.
- `new/` — an AI-generated (Claude web), unreviewed rewrite attempt (`config.py`, `dag.py`, `ingest.py`, `orchestrator.py`, `planner.py`, `worker.py`, etc.). It is **non-functional** — `main.py` and `test_solbot.py` import `from solbot import ...` but no `solbot` package exists anywhere on disk, and `new/__init__.py` itself has a broken self-referential import. Left untouched on disk for now (not deleted, not archived) but **not being built on** — none of its code is to be reused or imported. Its module boundaries (config/store/retrieval/memory/llm/tools/orchestrator/worker) are useful *design reference* for Phase 2's refactor below. A few of its specific mechanisms — manifest-diffed incremental sync, query-rewriting for memory, evidence-sufficiency gating for web search — turned out useful enough to pull into Phase 1 directly (in simplified, single-process form); RRF+MMR retrieval fusion remains a possible Phase 2+ enhancement, not yet adopted.
- Root level — where the revamp's Phase 1 files live once written (`config.py`, `config.yaml`, `ingest.py`, `chatbot.py`, `.env.example`, `requirements.txt`).

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

**Phase 1 covers both the vector database and the chatbot, built as a monolith.** Earlier drafts of this plan scoped Phase 1 as vector-database-only with the chat loop deferred entirely to a Phase 2 orchestrator/worker rebuild. That's been revised: Phase 1 now builds a single-process, few-files chatbot right alongside the vector database — fixing all five known issues immediately — and Phase 2 becomes a structural *refactor* of that monolith rather than the first time generation, decomposition, and memory get built. The reasoning: keeping every fix in a handful of files the author can read end-to-end and own, mirroring how `old/` was written, before introducing any orchestrator/worker/DAG machinery. Query decomposition — tagging sub-queries with non-exclusive intents, each retrieved and handled independently, then synthesized into one reply — replaces both the old 6-way classifier and the need for a separate orchestrator/worker/DAG system in Phase 1; it's a lightweight, single-process version of what `new/`'s `dag.py`/`orchestrator.py`/`planner.py`/`worker.py` were reaching for, without adopting their machinery. Web search (fixing "closed world") is still built last within Phase 1 — behind a `web.enabled` config flag, off by default — rather than deferred to a later phase, since Phase 2 no longer has a "first pass at generation" milestone to attach it to.

Models: `qwen3:latest` for generation, `qwen3-embedding:latest` for embeddings — both confirmed as the models actually pulled in the local Ollama install (`ollama list`), replacing `old/`'s `gemma3`/`nomic-embed-text`.

### Phase 1 — Vector Database & Chatbot: Monolithic Implementation

Scope: build and maintain the Chroma knowledge base, then build a single-process chat loop that fixes all five known issues — all in a small number of files, vector DB first, then chatbot. No orchestrator/worker/DAG architecture yet; that's Phase 2's refactor target, not new capability to build here.

**Target files** (root of repo, small and few by design — mirrors `old/`'s one-file-per-concern simplicity):
```
config.py            # constants + .env loader + small dotted-key reader over config.yaml
config.yaml           # tunable values only (model names, chunk sizes, TOP_K, SCORE_THRESHOLD, web.enabled flag) — much smaller than new/config.yaml's 90 lines; nothing about planners/DAGs/verification, since Phase 1 doesn't have those
ingest.py             # build/sync the vector DB (replaces old/create_vector_db.py)
chatbot.py            # chat loop: decomposition, retrieval, memory, web fallback (replaces old/create_chatbot.py)
.env.example          # CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN
requirements.txt
```

**Step 1 — `ingest.py`: correct the ingestion pipeline, one-shot build.** Carry over `old/create_vector_db.py`'s working parts (Confluence → markdownify → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` → `OllamaEmbeddings(qwen3-embedding:latest)` → Chroma), fixing: paginated `get_all_pages_from_space` (loop on `start` instead of a single `limit=500` call), correct `'id': page['id']` metadata (not title), credentials from `.env` via `config.py`, a sensible log filename. Verify: run `python ingest.py`, confirm page/chunk counts logged, collection exists at `./asu_rc`, and a manual similarity query returns sensible chunks.

**Step 2 — `ingest.py`: manifest-diffed incremental sync (manual trigger).** JSON manifest (e.g. `./asu_rc/manifest.json`) mapping `page_id -> {version, hash}`. Each run: skip pages whose version+hash match the manifest; for new/changed pages, re-chunk with deterministic chunk ids (`f"{page_id}::{chunk_index}"`) and `upsert` into Chroma (idempotent); for pages removed upstream, delete their chunk ids and drop from the manifest. Needs direct `chromadb` collection access for id-based upsert/delete (`langchain_chroma.Chroma.from_documents` doesn't expose this cleanly — drop to the raw `chromadb` client for the storage layer, per `new/store.py`'s reasoning, while keeping `langchain_text_splitters` for chunking). Trigger is manual: `python ingest.py` always does a full diff-sync (empty manifest = first full build, same code path). Verify: run twice back-to-back with no upstream changes — second run reports 0 changed/added/deleted, no embedding calls issued. Then hand-edit the manifest (or a real page) to simulate a changed page and confirm the diff picks it up.

**Step 3 — `chatbot.py`: monolithic chat loop fixing all five known issues.** Each fix maps to `chatbot.py` as follows:
- *Semantic averaging* → `decompose_query(query, history) -> list[SubQuery]`, one LLM call per turn breaking a message into 1-N standalone sub-questions each tagged with a non-exclusive intent (`greeting`, `doc_question`, `action_request`, `clarification`, `irrelevant`, `other`). A simple greeting still comes back as one sub-query. Each sub-query needing docs is retrieved independently (`TOP_K` per sub-query, not per whole message), results merged/deduped by chunk id before the final answer is composed. Replaces `classify_query` + the single whole-message retrieval call.
- *Taxonomy is not a plan* → retires the exclusive single-label `int(classify_query(...))` branch entirely (`old/create_chatbot.py:24-40,192-204`). Intents become non-exclusive tags on sub-queries from the point above. The six `answer_*` handlers collapse into `handle_greeting`, `handle_doc_question`, `handle_action_request`, `handle_clarification`, `handle_off_topic` (merging irrelevant+other). The main loop runs each tagged sub-query through its handler and a final LLM call synthesizes one coherent reply — so "Is Sol down, and if so how do I check my queued jobs?" produces an `action_request` + a `doc_question` result, combined, instead of being forced into one bucket.
- *Closed world* → `web_search(query) -> list[SearchResult]` behind `config.yaml`'s `web.enabled` flag (**off by default**, built as the last step of Phase 1 rather than the first — keeps "vector db first, then chatbot" clean and avoids a new external dependency before the core loop works). An evidence-sufficiency check after doc retrieval — if the best chunk's score is below `SCORE_THRESHOLD` (finally applying that constant) or too few chunks came back — triggers `web_search` and folds results into that handler's context. `handle_action_request` (e.g. "is Sol down") is the other natural consumer once enabled, replacing the current canned "I don't have access" reply (`old/create_chatbot.py:138-151`).
- *Memory cost bug + latent bugs* → replace `filter_messages_based_on_similarity` (one LLM call per historical message, unguarded `previous_messages[i+1]` IndexError risk) with a single `contextualize_query(query, recent_turns) -> str` call per turn that rewrites the current query into a standalone form using recent history — feeds directly into decomposition/retrieval above. Fix `get_previous_messages`'s dead-condition bug (`max_previous_turns = -2*turns` compared with `>=` is always true) and the off-by-one that drops the most recent message, by tracking history as `(human, ai)` pairs and slicing `pairs[-turns:]` directly — likely merges into one `get_recent_turns(n)` used by the contextualizer. Actually apply `SCORE_THRESHOLD` in retrieval (no longer a dead constant). Standardize on `previous_messages_referred` everywhere (fixes the `previous_message_referred` typo in 3 of 6 old handlers). Wrap the decomposition call's structured-output parsing in try/except with a safe fallback (treat the whole message as one `doc_question` sub-query) instead of crashing on malformed LLM output. Rename `rough.log` to `solbot.log`, path configurable via `config.yaml`.

Verify: manual pass through a greeting, a single-topic doc question, the compound A100/Slurm example (issue #1), the "is Sol down + queued jobs" example (issue #2), a clarification follow-up, an irrelevant question, `\clear`, and (if `web.enabled`) a query that should trigger web fallback.

**Open item:** check `langchain_chroma.Chroma`'s upsert/delete-by-id support against the installed version early in Step 2; drop to the raw `chromadb` client for the storage layer if it's too restrictive (keep `langchain_text_splitters` for chunking either way).

### Phase 2 — Refactor into a properly separated codebase

Direction-level (to be re-planned in detail once Phase 1 is reviewed): once Phase 1's monolithic `ingest.py`/`chatbot.py` work and are understood, split along the module boundaries `new/` was reaching for — adapted to what Phase 1 actually built, not a copy of `new/`'s DAG/planner/worker machinery:

- `config.py` — carries over mostly unchanged.
- `ingest.py` / `sync.py` — manifest-diff logic, Confluence client, chunking.
- `store.py` — thin Chroma wrapper (query/upsert/delete by id) extracted out of `ingest.py`, shared by ingestion and retrieval instead of each reimplementing Chroma access.
- `retrieval.py` — embedding + similarity search + evidence-sufficiency scoring, used by both `chatbot.py`'s handlers and any future tool.
- `memory.py` — conversation history, contextualization/query-rewriting.
- `llm.py` — thin Ollama client wrapper (chat + structured/JSON-mode calls with retry/parse-guard) so call sites aren't each hand-rolling `llm.invoke(...)` and ad hoc parsing.
- `tools.py` — web search (and future tools) behind a uniform `ToolResult`-style interface, the way `new/tools.py` sketched (a registry with a uniform return type so callers never special-case a tool is worth keeping even though the surrounding orchestrator/DAG is not).
- `orchestrator.py` (or folded into a `cli.py`-facing module) — decomposition, per-sub-query dispatch, synthesis; kept intentionally simpler than `new/orchestrator.py`/`dag.py`/`planner.py`/`worker.py`'s multi-file agentic-graph design unless Phase 1 experience shows real need for it.
- `cli.py` / `main.py` — the interactive loop and `\quit`/`\clear` handling, thin over `orchestrator`.
- `test_*.py` — see Phase 3.

Key milestone: a working `import solbot` package (fixing the exact defect that made `new/` non-functional — no `solbot` package existed anywhere on disk), with parity testing against Phase 1's behavior before Phase 1's flat files are retired.

### Phase 3 — Adversarial testing (direction only)

Once Phase 2's modules exist to test in isolation: malformed/non-JSON LLM output on every structured call site (decomposition, contextualization, evidence scoring); Ollama unreachable/timeout; Confluence API errors/rate limits/pagination edge cases; empty retrieval results; empty/short conversation history (regression test for the exact `previous_messages[i+1]` IndexError shape); concurrent ingest runs; web search timeouts/failures; prompt injection via retrieved or fetched content; unicode/markdown edge cases from `markdownify`; worker-failure and partial-synthesis cases specific to the orchestrator/worker design.

### Phase 4 — Web deployment (direction only)

Thin API layer over the orchestrator (one `/chat` endpoint + health check); session/history storage decision for multi-worker deployment (in-memory won't survive restarts — needs Redis/SQLite/etc., deferred); containerization (Ollama likely stays a separate service); scheduled ingest sync (cron/systemd timer/APScheduler, deferred); auth/rate-limiting once public-facing.

### Verification approach across Phase 1

- `python ingest.py` run twice consecutively → second run reports zero changes, zero embedding calls.
- Manual manifest edit (simulate a Confluence page change) → next run correctly re-embeds only that page.
- `python chatbot.py` manual pass through: a greeting, a single-topic doc question, the compound A100/Slurm example, the "is Sol down + queued jobs" multi-intent example, a clarification follow-up, an irrelevant question, `\clear`, and (if `web.enabled`) a query that should trigger web fallback.
- No automated test suite yet by design — Phase 1 is monolithic and manually verified; Phase 3 is where systematic testing is introduced.

### Credentials

Confluence username + API token must be moved to environment variables (`.env`, gitignored) before any ingestion runs — never hardcoded, unlike `old/create_vector_db.py`'s placeholders.
