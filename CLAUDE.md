# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SolBot is a RAG-based CLI chatbot for ASU Research Computing. It answers questions about ASU supercomputers (Sol, Agave), HPC clusters, and RC administrative policies by retrieving relevant documentation chunks from a local Chroma vector store, decomposing compound questions into sub-tasks, and synthesizing a grounded, cited reply — with a fallback to open-web search when RC's own documentation doesn't cover something.

The revamp described below (Phases 1-2) is **complete and in active use**. Phase 3 (systematic adversarial testing) and Phase 4 (web deployment) are not started.

## Repository layout

- **`db_engine/`** — owns the Chroma vector store: builds it from Confluence, keeps it in sync via a manifest-diffed incremental sync (add/update/delete by page, idempotent re-runs), and serves `fetch()`/`get_all_chunks()`. Never generates text. See `db_engine/README.md` for the full API, CLI, and concurrency model.
- **`memory_engine/`** — SQLite-backed conversation memory: message history, per-turn recall via embedding similarity across a user's sessions, and long-term user facts (name, timezone, preferences). Never calls an LLM to reason, only to embed. See `memory_engine/README.md`.
- **`web_search/`** — standalone, read-only open-web search + page-content extraction (`search()` via `ddgs`, `fetch_page()` via `requests` + `trafilatura`). Zero dependency on any other package here. See `web_search/README.md`.
- **`agents/`** — the orchestrator/worker chatbot itself: `python -m agents` for the interactive CLI. Wires everything above together as a plain bounded loop (`contextualize -> route -> plan -> execute -> verify -> replan? -> synthesize`), not a graph library — see `agents/orchestrator.py`'s module docstring. Decomposes compound queries into a DAG of sub-questions (`agents/dag.py`, `agents/planner.py`), answers each via a bounded ReAct loop over tools (`agents/worker.py`, `agents/tools.py`: `vector_search`, `web_search`, `fetch_url`, `current_time`), fuses dense+BM25 retrieval with MMR reranking (`agents/fusion.py`), and audits findings for groundedness before writing a final reply (`agents/synthesis.py`). `\debug` (route/plan/verification trace) and `\thinking` (raw model reasoning) are togglable at the CLI prompt.
- **`old/`** — the original, hand-written v1 (`create_vector_db.py`, `create_chatbot.py`). Kept as historical reference for the specific bugs that motivated this rewrite — see "Known issues in v1" below, which cites exact line numbers here. Not built on, not run.
- **`test/`** — manual, human-read validation harnesses (`retrieval_check.py`, `generation_check.py`, `chat_check.py`) predating any real pytest suite. Not automated regression tests yet (that's Phase 3).
- **`config.yaml`** — single source of truth for all packages' tunables (Confluence, chunking, embedding/generation model + `base_url`, retrieval fusion weights, planner/worker/verification thresholds, memory, web search). `.env` (gitignored) holds `CONFLUENCE_USERNAME`/`CONFLUENCE_API_TOKEN` only.
- **Gitignored, regenerable local data** (not committed): `asu_rc/` (the Chroma vector store + sync manifest), `memory_store/` (the SQLite conversation DB), `*.log` (per-package log files).

## Running the project

**Prerequisites:** Ollama reachable (locally or via `config.yaml`'s `base_url` overrides) with the configured chat and embedding models pulled.

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN

python -m db_engine          # build/sync the vector store (first run: full build; reruns: near-instant diff-sync)
python -m agents              # interactive chatbot: \quit, \clear, \debug, \thinking
```

Each package's own README (`db_engine/README.md`, `memory_engine/README.md`, `web_search/README.md`) has its full public API, CLI, and config reference — not duplicated here.

### Known issues in v1 (`old/`) — what this rewrite fixed

1. **Semantic averaging** — embedding a compound query produced one centroid vector between unrelated topic regions, so top-k similarity returned mediocre chunks from neither. Fixed by `agents/planner.py` decomposing compound queries into a DAG of atomic sub-questions, each retrieved independently.
2. **Taxonomy is not a plan** — `classify_query` (`old/create_chatbot.py:24-40`) forced every query into one of 6 exclusive labels. Fixed by the same DAG decomposition: intent is per-sub-task, not a single label for the whole message.
3. **Frozen knowledge** — `create_vector_db.py` did a one-shot rebuild with no incremental sync (`old/create_vector_db.py:40`), a hard `limit=500` with no pagination (`old/create_vector_db.py:20`), and a metadata bug using the page title as its id (`old/create_vector_db.py:28`). Fixed by `db_engine`'s manifest-diffed sync with proper pagination and real Confluence page ids.
4. **Closed world** — no access outside the RC Confluence space. Fixed by `agents/tools.py`'s `web_search`/`fetch_url` tools, escalated when vector-search evidence is thin (`agents/fusion.py`'s sufficiency check, `agents/worker.py`'s confidence-based escalation).
5. **Memory cost bug + latent bugs** — `filter_messages_based_on_similarity` (`old/create_chatbot.py:59-66`) issued one LLM call per historical message per turn, with an unguarded `previous_messages[i+1]` IndexError risk; `get_previous_messages` (`old/create_chatbot.py:42-46`) had a dead/always-true guard; `SCORE_THRESHOLD` was declared but never applied; Confluence credentials were hardcoded (`old/create_vector_db.py:12-13`). Fixed by `agents/contextualize.py` (one rewrite call per turn, not per message), `memory_engine`'s `turn_number`-based pairing (no list-index arithmetic), `db_engine`/`agents` actually applying `score_threshold`, and `.env`-sourced credentials throughout.

### Models

`qwen3:4b` for generation, `qwen3-embedding:4b` for embeddings, both reachable via `config.yaml`'s `base_url` (may point at a networked Ollama instance rather than localhost — check current `config.yaml` for where). Chosen after real-hardware benchmarking showed structured-reasoning calls (contextualize/route/plan/verify/synthesize, not bare completions) cost ~35-50s each regardless of local vs. remote on CPU-bound inference — smaller models were the effective lever, not network placement.

## What's left

- **Phase 3 — adversarial testing** (not started): malformed/non-JSON LLM output at every structured call site beyond what's already guarded, Ollama/network unreachable mid-session (currently uncaught — crashes the CLI, see `agents/__main__.py`), Confluence API errors/rate limits, empty retrieval results, concurrent ingest runs, web search timeouts, prompt injection via retrieved/fetched content, worker-failure and partial-synthesis edge cases. `test/*.py`'s manual harnesses are the starting point, not a substitute for this.
- **Phase 4 — web deployment** (not started): thin API layer over `agents.Session`, session/history storage decision for multi-worker deployment, containerization, scheduled ingest sync, auth/rate-limiting.
- **Known open issue**: retrieval sometimes ranks a narrower page (e.g. a specific accelerator type) above a more complete hardware-overview page for broad "what hardware does X have" questions, and synthesis can overstate a negative claim not fully supported by the top-ranked chunk alone. Partially mitigated (user-provided URLs are now fetched directly and treated as authoritative ground truth — see `agents/orchestrator.py`'s `_fetch_user_urls`), but the underlying ranking behavior itself hasn't been changed.
