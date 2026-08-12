# SolBot

A RAG-based CLI chatbot for **ASU Research Computing** (RC). SolBot answers questions about ASU's supercomputers (Sol, Agave), HPC clusters, Slurm usage, storage, and RC administrative policies by retrieving relevant documentation chunks from a local vector store built from RC's Confluence space — decomposing compound questions into sub-tasks, researching each one (falling back to the open web when RC's own docs don't cover something), auditing its own findings for groundedness, and streaming a cited, grounded reply live to the terminal.

The current implementation is a full rewrite of an earlier, simpler version — see [Project history](#project-history) below for what changed and why.

## Features

- **Query decomposition, not classification.** Compound questions ("Which cluster has A100 GPUs and how do I request one for four hours?") are broken into a small DAG of atomic sub-questions instead of being forced into one intent label or one embedding — each sub-question is researched and retrieved independently.
- **Hybrid retrieval.** Dense (embedding) and sparse (BM25) search are fused via Reciprocal Rank Fusion and reranked with Maximal Marginal Relevance, so results aren't just semantically similar but also diverse and literal-term-aware.
- **Self-auditing answers.** Before anything is shown to the user, a verification pass checks whether each sub-question's answer is actually grounded in the evidence it cites — unverified findings get hedged in the final reply rather than stated as flat fact, and a plan can be extended (replanned) once if real gaps are found.
- **Open-web fallback.** When RC's own documentation doesn't cover a topic, SolBot escalates to a real web search and page-fetch, cleaning the fetched HTML down to substantive content before using it as evidence.
- **User-provided links are actually read.** If you paste a URL in your message, SolBot fetches that exact page and treats it as authoritative ground truth — it won't just re-search its own corpus and hope for the best.
- **Freshness, not a frozen snapshot.** The vector store is kept in sync with Confluence via a manifest-diffed incremental sync (add/update/delete by page), not a one-shot rebuild — a second sync run with no upstream changes is a near-instant no-op.
- **Real conversation memory.** Message history, per-turn recall across a user's *other* sessions (via embedding similarity), and durable personal facts (name, timezone, preferences) persist in a SQLite store — a follow-up question like "how long can I run it for?" resolves "it" correctly using recent context.
- **Streaming, live answers.** A real turn costs tens of seconds of local LLM inference; the reply streams token-by-token as it's generated instead of leaving the terminal blank.
- **Debug and thinking traces.** Toggle `\debug` to see the route/plan/verification trace behind an answer, or `\thinking` to see the model's raw reasoning, live, in a distinct color — both off by default.
- **Hardened against bad input.** Adversarially fuzz-tested against 23 scenarios (empty/malformed/huge/gibberish/prompt-injection-shaped input) with a top-level backstop so a failure degrades to an error message, not a crash.

## Architecture

```
Confluence (RC space)
        │
        ▼
   db_engine  ──────────────►  Chroma vector store (asu_rc/)
 (sync + fetch)                       │
                                       ▼
                                  agents (orchestrator)
                                  contextualize → route → plan
                                       │
                              ┌────────┴────────┐
                              ▼                 ▼
                          worker(s)         memory_engine
                      (ReAct loop over    (history, facts,
                    vector_search /web_    cross-session recall)
                    search/fetch_url/
                       current_time)
                              │
                              ▼
                       verify → synthesize
                              │
                              ▼
                     streamed reply (CLI)
```

Four independent packages, each with a single clear responsibility, wired together by `agents/`:

| Package | Responsibility |
|---|---|
| [`db_engine/`](db_engine/README.md) | Owns the Chroma vector store. Builds it from Confluence, keeps it in sync via manifest-diffed incremental sync, serves `fetch()`/`get_all_chunks()`. Never generates text. |
| [`memory_engine/`](memory_engine/README.md) | SQLite-backed conversation memory: message history, cross-session recall, long-term user facts. Never calls an LLM to reason, only to embed. |
| [`web_search/`](web_search/README.md) | Standalone, read-only open-web search + page-content extraction. Zero dependency on any other package here. |
| `agents/` | The chatbot itself — wires everything above together as a plain bounded loop (`contextualize → route → plan → execute → verify → replan? → synthesize`), not a graph library. `python -m agents` is the interactive entry point. |

Each package's own README has its full public API, CLI, and config reference.

## Getting started

**Prerequisites:** Python 3.12+, and an [Ollama](https://ollama.com) instance reachable either locally or over the network (`config.yaml`'s `base_url` settings), with the configured chat and embedding models pulled.

```bash
git clone <this repo>
cd SolBot
pip install -r requirements.txt

cp .env.example .env
# fill in CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN

python -m db_engine    # build/sync the vector store — first run does a full build,
                        # reruns are a near-instant diff-sync
python -m agents        # start the interactive chatbot
```

## Usage

```
$ python -m agents
SolBot -- type a question, \quit to exit, \clear to reset conversation history,
\debug to toggle route/plan/verification trace, \thinking to toggle the
model's raw reasoning trace (adds latency while on).
> What GPUs does the Sol cluster have?
...
```

| Command | Effect |
|---|---|
| `\quit` | Exit |
| `\clear` | Reset this session's conversation history (personal facts persist) |
| `\debug` | Toggle a per-turn trace: route decision, plan DAG, per-node confidence/tools used, verification outcome |
| `\thinking` | Toggle showing the model's raw reasoning trace live (costs extra latency while on) |

Pasting a URL directly in your message makes SolBot fetch and read that exact page as part of its answer.

## Configuration

`config.yaml` is the single source of truth for every package's tunables: Confluence source, chunking, embedding/generation model + `base_url`, retrieval fusion weights, planner/worker/verification thresholds, memory, and web search settings. `.env` (gitignored) holds only `CONFLUENCE_USERNAME`/`CONFLUENCE_API_TOKEN` — never committed, never hardcoded.

Currently configured for `qwen3:4b` (generation) and `qwen3-embedding:4b` (embeddings) via Ollama, chosen after real-hardware benchmarking showed structured-reasoning calls cost ~35-90s each on CPU-bound inference regardless of local vs. networked placement — smaller models were the effective lever.

## Project structure

```
agents/          orchestrator + worker chatbot — python -m agents
db_engine/       Chroma vector store, Confluence sync
memory_engine/   conversation history, facts, cross-session recall
web_search/      open-web search + page extraction
test/            manual validation harnesses + an adversarial fuzz-input suite
config.yaml      all tunables
.env             Confluence credentials (gitignored, not committed)
```

Regenerable local data (gitignored, not committed): `asu_rc/` (the vector store), `memory_store/` (the conversation database), `*.log` (per-package logs).

## Testing

`test/` holds manual, human-read validation harnesses rather than an automated pytest suite (that's still ahead — see [Roadmap](#roadmap)):

- `retrieval_check.py` — retrieval quality against canned queries, no LLM call
- `generation_check.py` — generation sanity check
- `chat_check.py` — scripted multi-turn conversations exercising decomposition, memory, and web escalation, with a full trace printed for human review
- `fuzz_check.py` — adversarial-input stress test across 23 scenarios (empty/malformed/huge/gibberish/mixed-language/prompt-injection-shaped input), reporting crashed vs. soft-failed vs. clean; never claims success unconditionally

## Roadmap

- **Adversarial testing** — the input-robustness slice (`fuzz_check.py`) is done; still ahead: Confluence API errors/rate limits, empty retrieval results as their own scenario, concurrent ingest runs, web search timeouts, worker-failure/partial-synthesis edge cases.
- **Web deployment** — not started: a thin API layer over the chatbot, session/history storage for multi-worker deployment, containerization, scheduled ingest sync, auth/rate-limiting.

### Known open issues

- **Retrieval ranking**: a narrower page can occasionally outrank a more complete overview page for broad "what hardware does X have" questions, and synthesis can overstate a negative claim not fully supported by the top-ranked chunk alone. Partially mitigated — a URL you paste directly is always fetched and treated as authoritative — but the underlying ranking behavior itself is unchanged.
- **Prompt injection is mitigated, not solved.** Every prompt that sees a raw user query treats it explicitly as content, not instructions, and every prompt that interpolates retrieved/fetched text wraps it in untrusted-content markers. This measurably reduces how often an embedded instruction in a document or web page gets followed, but the underlying model has no architectural separation between trusted instructions and untrusted data — nothing here closes that surface, it only reduces how often it opens.

## Project history

SolBot began as a single-pass v1 with a 6-way intent classifier and one-shot vector store build. Five concrete problems drove the rewrite into what exists today:

1. **Semantic averaging** — embedding a compound query produced one centroid vector between unrelated topic regions, returning mediocre results for both. Fixed by decomposing into a DAG of atomic sub-questions, each retrieved independently.
2. **Taxonomy is not a plan** — a single exclusive intent label can't represent a compound, multi-part question. Fixed by the same decomposition: intent is per-sub-task now.
3. **Frozen knowledge** — the original vector store was a one-shot build with no incremental sync, a hard page-fetch limit with no pagination, and a metadata bug using page titles as ids. Fixed by `db_engine`'s manifest-diffed sync with real Confluence page ids.
4. **Closed world** — no access to anything outside RC's own Confluence space. Fixed by the web-search/fetch-url escalation path.
5. **A memory cost bug and several latent bugs** — the original history-filtering logic issued one LLM call *per historical message per turn*, had an always-true dead-code guard, an unguarded index risk, a declared-but-unused score threshold, and hardcoded credentials. Fixed by a single per-turn contextualization call, `memory_engine`'s `turn_number`-based pairing, an actually-applied score threshold, and `.env`-sourced credentials throughout.

