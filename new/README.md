# SolBot v2 — agentic ASU Research Computing assistant

Rebuild of [SolBot](https://github.com/BhavyaShah1234/SolBot) from a scripted
intent classifier into a planning agent that decomposes compound questions into
a Directed Acyclic Graph (DAG), answers each node with a Reason-and-Act (ReAct)
worker, verifies grounding, and synthesises a cited reply.

## What changed

| Concern | v1 | v2 |
|---|---|---|
| Query handling | 6-way intent classifier, one branch each | DAG planner, adaptive replanning |
| Compound queries | one embedding for the whole question | one node per sub-question, run in parallel layers |
| Retrieval | dense `similarity`, `k=3` | dense + BM25, Reciprocal Rank Fusion, MMR, LLM relevance filter |
| Knowledge base | one-shot build, never refreshed | manifest-diffed incremental sync + web write-through |
| Outside knowledge | none | web search and fetch, escalated automatically on thin evidence |
| Follow-ups | one LLM call *per stored message per turn* | one contextualisation call per turn |
| Grounding | none | pre-answer verification, replan on failure |
| Configuration | module-level constants | `config.yaml` |
| Tests | none | 38 offline tests, no model server required |

## Module map

| File | Responsibility |
|---|---|
| `config.py` | Loads `config.yaml`, expands `${ENV_VAR}`, dotted lookup |
| `llm.py` | Ollama HTTP client, JSON-mode helper, brace-matching JSON extractor, deterministic stub backend |
| `dag.py` | `PlanNode` / `Plan` / `NodeResult`, Kahn's algorithm, layering, validation, placeholder resolution |
| `store.py` | `VectorStore` protocol; Chroma and in-memory backends |
| `retrieval.py` | BM25, Reciprocal Rank Fusion, Maximal Marginal Relevance, LLM rerank, evidence sufficiency |
| `ingest.py` | Markdown splitting, incremental Confluence sync via version manifest, web write-through |
| `tools.py` | Tool registry; vector search, web search, URL fetch, refresh, clock |
| `planner.py` | Router gate, DAG generation prompt, validation with graceful degradation, replanning |
| `worker.py` | ReAct loop per node, web escalation policy, layered parallel executor |
| `synthesis.py` | Grounding verifier, final composition, escalation fallback |
| `memory.py` | Standalone-query rewriting, rolling summary |
| `orchestrator.py` | LangGraph state machine with the replan cycle |
| `main.py` | `sync`, `plan`, `chat` subcommands |

## Migration, in order

1. **Config first.** Move the v1 constants into `config.yaml`. Nothing else
   changes yet, and you get a reversible checkpoint.
2. **Swap ingestion.** Run `python main.py sync`. First run rebuilds and writes
   `manifest.json`; subsequent runs touch only changed pages. Verify with a
   second immediate run — it should report `unchanged=N chunks=0`.
3. **Swap retrieval.** Point the old chatbot's lookup at `Retriever.retrieve`.
   Compare answers on your existing questions before going further; this alone
   usually fixes a third of the failures.
4. **Add the planner.** Use `python main.py plan "<query>"` to inspect
   decompositions without executing them. Tune `max_nodes` and the planner
   prompt here, where iteration is cheap.
5. **Add workers and tools.** Start with `vector_search` only, then enable
   `web.enabled` once node-level answers look right.
6. **Add verification and replanning last.** They are the most expensive stages
   and the least useful if the layers beneath them are wrong.

## Operating notes

- **Sync cadence.** `sync_interval_minutes` in config; drive it with cron,
  systemd timer, or a GitHub Actions schedule. It is idempotent, so overlapping
  runs are safe apart from wasted embedding calls.
- **Model sizing.** The planner and verifier need reliable JSON. If `gemma3`
  produces malformed plans, the fallback keeps answers flowing but you lose
  decomposition — check `solbot.log` for `rejected plan` before blaming
  retrieval.
- **Latency budget.** A three-node plan costs roughly: 1 contextualise + 1 route
  + 1 plan + (2–4 per node × 3, parallel across layers) + 1 verify + 1
  synthesise. Two layers means about three sequential worker rounds, not nine.
- **Testing.** `pytest tests/ -q`. Everything runs against `StubBackend` and
  `MemoryStore`, so continuous integration needs neither Ollama nor Chroma.

## Known extension points

- Replace the `status` tool hint with a real cluster-status endpoint.
- Add a human-in-the-loop interrupt on the `plan` node (LangGraph
  `interrupt_before`) for high-cost actions.
- Swap the LLM reranker for a cross-encoder (`bge-reranker-base`) when
  throughput matters more than flexibility.
- Persist traces to JSONL and score them with RAGAS-style faithfulness and
  context-precision metrics over a golden set of compound questions.
