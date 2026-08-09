"""Test suite.

Everything runs offline against ``StubBackend`` and ``MemoryStore``: no Ollama
server, no Chroma, no network. That is the payoff for keeping the graph
algorithms free of input/output and for making the model backend swappable
from configuration.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solbot.config import Config
from solbot.dag import (
    Plan,
    PlanError,
    PlanNode,
    plan_from_dict,
    single_node_plan,
    topological_layers,
    validate_plan,
)
from solbot.ingest import KnowledgeBase, recursive_split, split_markdown
from solbot.llm import LLM, extract_json
from solbot.memory import Memory
from solbot.orchestrator import SolBot
from solbot.retrieval import BM25Index, Retriever, maximal_marginal_relevance, reciprocal_rank_fusion
from solbot.store import MemoryStore
from solbot.tools import build_registry
from solbot.worker import Worker, execute_plan

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CONFIG_YAML = """
app: {timezone: America/Phoenix, log_level: WARNING}
llm: {backend: stub, chat_model: stub, embedding_model: stub, temperature: 0.0}
vector_store: {provider: memory, directory: ./tmp, collection: test, manifest_path: ./tmp/manifest.json}
ingest: {chunk_size: 200, chunk_overlap: 40, embed_batch_size: 8,
         headers_to_split_on: [["#", "h1"], ["##", "h2"]]}
retrieval: {dense_k: 6, sparse_k: 6, final_k: 3, rrf_constant: 60, use_sparse: true,
            use_mmr: false, use_llm_rerank: false, min_evidence_score: 0.05, min_evidence_count: 1}
planner: {max_nodes: 6, max_depth: 4, allow_replanning: true, max_replans: 1,
          decomposition_threshold_words: 4}
worker: {max_react_steps: 3, max_parallel_nodes: 3, escalate_to_web_below_confidence: 0.0,
         answer_max_words: 120}
web: {enabled: false}
memory: {max_turns_kept: 10, contextualize_window: 2, summarize_after_turns: 6}
verification: {enabled: true, min_overall_confidence: 0.4}
escalation: {contact: rc-ops@asu.edu, message: "No grounded answer found."}
"""

PAGES = [
    {
        "id": "1001",
        "title": "Sol GPU nodes",
        "version": 3,
        "source": "https://example.invalid/sol-gpu",
        "body": "# Sol GPU nodes\n\nSol provides NVIDIA A100 accelerators on the gpu partition. "
        "Each node carries four A100 cards.\n\n## Requesting a GPU\n\n"
        "Use `sbatch --gres=gpu:a100:1 --time=04:00:00` to request one A100 for four hours.",
    },
    {
        "id": "1002",
        "title": "Phoenix cluster",
        "version": 1,
        "source": "https://example.invalid/phoenix",
        "body": "# Phoenix\n\nPhoenix is the older cluster and provides V100 accelerators only. "
        "It does not have A100 hardware.",
    },
]


@pytest.fixture
def cfg() -> Config:
    return Config(yaml.safe_load(CONFIG_YAML))


@pytest.fixture
def llm(cfg) -> LLM:
    model = LLM(cfg)
    backend = model.backend

    backend.register("ROUTER", lambda prompt: json.dumps(
        {"route": "chat" if prompt.lower().strip() in {"hi", "hello"} else "research"}
    ))
    backend.register("CONTEXTUALIZER", lambda prompt: json.dumps(
        {"standalone": prompt.split("Follow-up message:")[-1].strip(), "changed": False}
    ))
    backend.register("PLANNER", lambda prompt: json.dumps({"nodes": [
        {"id": "n1", "question": "Which ASU cluster has A100 GPUs?", "depends_on": [],
         "tool_hint": "vector", "rationale": "identify cluster"},
        {"id": "n2", "question": "What Slurm flags request a GPU for four hours?", "depends_on": [],
         "tool_hint": "vector", "rationale": "independent syntax lookup"},
        {"id": "n3", "question": "How do I submit a four hour GPU job on {{n1}}?",
         "depends_on": ["n1"], "tool_hint": "vector", "rationale": "cluster specific"},
    ]}))

    def worker_handler(prompt: str) -> str:
        if "Observation:" in prompt or "budget exhausted" in prompt.lower():
            return json.dumps({
                "thought": "the observation contains the answer",
                "final_answer": "Sol provides A100 GPUs; request one with "
                                "`sbatch --gres=gpu:a100:1 --time=04:00:00`.",
                "confidence": 0.85,
                "citations": ["Sol GPU nodes <https://example.invalid/sol-gpu>"],
            })
        question = prompt.split("Question:")[-1].split("\n")[0].strip()
        return json.dumps({
            "thought": "search the documentation first",
            "action": "vector_search",
            "action_input": {"query": question},
        })

    backend.register("WORKER", worker_handler)
    backend.register("VERIFIER", lambda prompt: json.dumps({
        "nodes": [{"id": nid, "grounded": True, "issue": ""}
                  for nid in ("n1", "n2", "n3") if f"Node {nid}" in prompt],
        "overall_confidence": 0.82,
        "needs_more_research": False,
    }))
    backend.register("SYNTHESISER", lambda prompt: (
        "Sol is the cluster with A100 GPUs. Request one with "
        "`sbatch --gres=gpu:a100:1 --time=04:00:00`.\n\nSources: Sol GPU nodes"
    ))
    backend.register("CHAT", lambda prompt: "Hello. I answer questions about ASU's clusters.")
    return model


@pytest.fixture
def knowledge(cfg, llm, tmp_path):
    cfg.data["vector_store"]["manifest_path"] = str(tmp_path / "manifest.json")
    store = MemoryStore()
    base = KnowledgeBase(cfg, llm, store)
    base.sync(PAGES)
    retriever = Retriever(cfg, llm, store)
    retriever.refresh_indexes()
    return base, store, retriever


# --------------------------------------------------------------------------- #
# DAG algorithms
# --------------------------------------------------------------------------- #


def test_topological_layers_groups_independent_nodes():
    nodes = [
        PlanNode("n1", "a"),
        PlanNode("n2", "b"),
        PlanNode("n3", "c", depends_on=["n1", "n2"]),
        PlanNode("n4", "d", depends_on=["n3"]),
    ]
    assert topological_layers(nodes) == [["n1", "n2"], ["n3"], ["n4"]]


def test_cycle_is_rejected():
    nodes = [PlanNode("n1", "a", depends_on=["n2"]), PlanNode("n2", "b", depends_on=["n1"])]
    with pytest.raises(PlanError, match="cycle"):
        topological_layers(nodes)


def test_unknown_dependency_is_rejected():
    with pytest.raises(PlanError, match="unknown"):
        topological_layers([PlanNode("n1", "a", depends_on=["ghost"])])


def test_placeholder_implies_dependency(cfg):
    plan = Plan("q", [PlanNode("n1", "a"), PlanNode("n2", "how about {{n1}}?")])
    validate_plan(plan, max_nodes=6, max_depth=4)
    assert plan.by_id["n2"].depends_on == ["n1"]


def test_placeholder_substitution():
    node = PlanNode("n3", "submit on {{n1}} using {{n2}}", depends_on=["n1", "n2"])
    assert node.resolve({"n1": "Sol", "n2": "sbatch"}) == "submit on Sol using sbatch"
    assert "{{n2}}" in node.resolve({"n1": "Sol"})


def test_duplicate_ids_rejected():
    plan = Plan("q", [PlanNode("n1", "a"), PlanNode("n1", "b")])
    with pytest.raises(PlanError, match="duplicate"):
        validate_plan(plan, max_nodes=6, max_depth=4)


def test_node_limit_enforced():
    plan = Plan("q", [PlanNode(f"n{i}", "x") for i in range(9)])
    with pytest.raises(PlanError, match="limit"):
        validate_plan(plan, max_nodes=6, max_depth=4)


def test_plan_from_dict_tolerates_loose_shapes():
    plan = plan_from_dict("q", {"nodes": [{"question": "a"}, {"id": "n2", "question": "b",
                                                              "depends_on": "n1"}]})
    assert [n.id for n in plan.nodes] == ["n1", "n2"]
    assert plan.by_id["n2"].depends_on == ["n1"]


def test_single_node_fallback():
    assert single_node_plan("anything").is_trivial()


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", [
    '{"a": 1}',
    '```json\n{"a": 1}\n```',
    'Sure, here you go:\n{"a": 1}\nHope that helps!',
    '{"a": 1} trailing commentary {"b": 2}',
    '{"a": "brace } inside string"}' .replace('"a": "brace } inside string"', '"a": 1'),
])
def test_extract_json_survives_model_noise(raw):
    assert extract_json(raw)["a"] == 1


def test_extract_json_handles_braces_in_strings():
    assert extract_json('prefix {"text": "use } carefully", "a": 1} suffix')["a"] == 1


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json at all")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "z"]], k_constant=60)
    assert max(fused, key=fused.get) == "a"       # ranked 1st and 2nd
    assert fused["a"] > fused["c"] > fused["b"]


def test_mmr_prefers_diversity_over_duplicates():
    # lambda 0.5 with a query identical to a candidate makes relevance and
    # redundancy cancel exactly, tying every remaining candidate at zero.
    # 0.3 puts real weight on diversity, which is what this test is about.
    query = [1.0, 0.0]
    candidates = [("dup1", [1.0, 0.0]), ("dup2", [0.99, 0.01]), ("other", [0.0, 1.0])]
    selected = maximal_marginal_relevance(query, candidates, k=2, lambda_multiplier=0.3)
    assert selected[0] == "dup1", "most relevant chunk is still picked first"
    assert "other" in selected, "second pick must add information, not repeat it"


def test_mmr_lambda_one_is_pure_relevance_ranking():
    query = [1.0, 0.0]
    candidates = [("dup1", [1.0, 0.0]), ("dup2", [0.99, 0.01]), ("other", [0.0, 1.0])]
    assert maximal_marginal_relevance(query, candidates, k=2, lambda_multiplier=1.0) == ["dup1", "dup2"]


def test_bm25_finds_rare_literal_tokens():
    index = BM25Index([
        {"id": "a", "text": "request a gpu with --gres=gpu:a100:1 on the gpu partition"},
        {"id": "b", "text": "general information about storage quotas and scratch space"},
    ])
    assert index.search("--gres=gpu:a100:1", k=2)[0][0] == "a"


def test_retrieval_finds_the_right_page(knowledge):
    _, _, retriever = knowledge
    result = retriever.retrieve("Which cluster has A100 GPUs?")
    assert result.evidence
    assert any("A100" in item.text for item in result.evidence)


# --------------------------------------------------------------------------- #
# Chunking and incremental sync
# --------------------------------------------------------------------------- #


def test_split_markdown_carries_header_trail():
    sections = split_markdown("# Top\n\nintro\n\n## Sub\n\nbody", [["#", "h1"], ["##", "h2"]])
    metas = [meta for _, meta in sections]
    assert {"h1": "Top"} in metas
    assert {"h1": "Top", "h2": "Sub"} in metas


def test_recursive_split_respects_size_and_overlap():
    chunks = recursive_split("word " * 300, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_sync_is_incremental(cfg, llm, tmp_path):
    cfg.data["vector_store"]["manifest_path"] = str(tmp_path / "m.json")
    store = MemoryStore()
    base = KnowledgeBase(cfg, llm, store)

    first = base.sync(PAGES)
    assert first.added == 2 and first.updated == 0
    baseline = store.count()

    second = base.sync(PAGES)
    assert second.unchanged == 2 and second.chunks_written == 0
    assert store.count() == baseline, "unchanged pages must not be re-embedded"


def test_sync_updates_changed_page_without_duplicating(cfg, llm, tmp_path):
    cfg.data["vector_store"]["manifest_path"] = str(tmp_path / "m.json")
    store = MemoryStore()
    base = KnowledgeBase(cfg, llm, store)
    base.sync(PAGES)

    edited = [dict(PAGES[0], version=4, body="# Sol GPU nodes\n\nSol now has H100 accelerators."),
              PAGES[1]]
    report = base.sync(edited)

    assert report.updated == 1 and report.unchanged == 1
    texts = " ".join(d["text"] for d in store.all_documents())
    assert "H100" in texts
    assert "four A100 cards" not in texts, "stale chunks must be purged"


def test_sync_deletes_removed_pages(cfg, llm, tmp_path):
    cfg.data["vector_store"]["manifest_path"] = str(tmp_path / "m.json")
    store = MemoryStore()
    base = KnowledgeBase(cfg, llm, store)
    base.sync(PAGES)

    report = base.sync([PAGES[0]])
    assert report.deleted == 1
    assert all(d["metadata"]["page_id"] != "1002" for d in store.all_documents())


def test_web_ingestion_survives_confluence_sync(cfg, llm, tmp_path):
    cfg.data["vector_store"]["manifest_path"] = str(tmp_path / "m.json")
    store = MemoryStore()
    base = KnowledgeBase(cfg, llm, store)
    base.ingest_web_page("https://slurm.schedmd.com/sbatch.html", "sbatch", "The sbatch command.")
    base.sync(PAGES)
    assert any(d["metadata"].get("origin") == "web" for d in store.all_documents())


# --------------------------------------------------------------------------- #
# Worker and executor
# --------------------------------------------------------------------------- #


def test_worker_answers_from_vector_search(cfg, llm, knowledge):
    _, _, retriever = knowledge
    worker = Worker(cfg, llm, build_registry(cfg, retriever))
    result = worker.run("n1", "Which ASU cluster has A100 GPUs?")
    assert result.ok
    assert "vector_search" in result.tools_used
    assert result.confidence > 0.5


def test_executor_substitutes_upstream_answers(cfg, llm, knowledge):
    _, _, retriever = knowledge
    worker = Worker(cfg, llm, build_registry(cfg, retriever))
    plan = Plan("q", [
        PlanNode("n1", "Which ASU cluster has A100 GPUs?"),
        PlanNode("n3", "How do I submit on {{n1}}?", depends_on=["n1"]),
    ])
    results = execute_plan(plan, worker, max_parallel=2)
    assert set(results) == {"n1", "n3"}
    assert "{{n1}}" not in results["n3"].question


def test_unknown_tool_does_not_crash_the_worker(cfg, llm, knowledge):
    _, _, retriever = knowledge
    registry = build_registry(cfg, retriever)
    assert not registry.call("nonexistent_tool", {}).ok


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def test_contextualize_is_a_noop_on_empty_history(cfg, llm):
    assert Memory(cfg, llm).contextualize("what is Sol?") == "what is Sol?"


def test_contextualize_rewrites_follow_up(cfg, llm):
    memory = Memory(cfg, llm)
    memory.add("Which cluster has A100s?", "Sol does.")
    llm.backend.register("CONTEXTUALIZER", lambda p: json.dumps(
        {"standalone": "Can I run a job on Sol for longer than four hours?", "changed": True}))
    assert "Sol" in memory.contextualize("can I run that for longer?")


def test_memory_is_bounded(cfg, llm):
    memory = Memory(cfg, llm)
    llm.backend.register("SUMMARIZER", lambda p: json.dumps({"summary": "earlier discussion"}))
    for i in range(20):
        memory.add(f"q{i}", f"a{i}")
    assert len(memory.turns) <= cfg.get("memory.max_turns_kept")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def _bot(cfg, llm, knowledge, events=None):
    base, _, retriever = knowledge
    worker = Worker(cfg, llm, build_registry(cfg, retriever, base))
    on_event = (lambda n, p: events.append((n, p))) if events is not None else None
    return SolBot(cfg, llm, worker, Memory(cfg, llm), base, on_event)


def test_compound_query_produces_a_multi_node_dag(cfg, llm, knowledge):
    events: list = []
    outcome = _bot(cfg, llm, knowledge, events).ask(
        "Which ASU cluster has A100 GPUs and how do I request one for four hours?"
    )
    assert outcome["route"] == "research"
    assert len(outcome["plan"].nodes) == 3
    assert outcome["plan"].layers() == [["n1", "n2"], ["n3"]]
    assert "sbatch" in outcome["answer"]
    assert any(name == "plan" for name, _ in events)


def test_greeting_bypasses_the_planner(cfg, llm, knowledge):
    outcome = _bot(cfg, llm, knowledge).ask("hi")
    assert outcome["route"] == "chat"
    assert outcome["plan"] is None
    assert "clusters" in outcome["answer"].lower()


def test_ungrounded_results_trigger_one_replan(cfg, llm, knowledge):
    calls = {"n": 0}

    def verifier(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"nodes": [{"id": "n1", "grounded": False, "issue": "unsupported"}],
                               "overall_confidence": 0.2, "needs_more_research": True})
        return json.dumps({"nodes": [{"id": "n1", "grounded": True, "issue": ""}],
                           "overall_confidence": 0.8, "needs_more_research": False})

    llm.backend.register("VERIFIER", verifier)
    llm.backend.register("REPLANNER", lambda p: json.dumps({"nodes": [
        {"id": "n1", "question": "Which ASU cluster has A100 GPUs?", "depends_on": [],
         "tool_hint": "vector"},
        {"id": "n4", "question": "A100 availability on ASU clusters", "depends_on": [],
         "tool_hint": "web"},
    ]}))

    outcome = _bot(cfg, llm, knowledge).ask(
        "Which ASU cluster has A100 GPUs and how do I request one for four hours?"
    )
    assert calls["n"] == 2, "verification should run once per execution pass"
    assert outcome["answer"]


def test_replanning_is_bounded(cfg, llm, knowledge):
    llm.backend.register("VERIFIER", lambda p: json.dumps(
        {"nodes": [], "overall_confidence": 0.1, "needs_more_research": True}))
    llm.backend.register("REPLANNER", lambda p: json.dumps({"nodes": [
        {"id": "n1", "question": "Which ASU cluster has A100 GPUs?", "depends_on": []}]}))
    outcome = _bot(cfg, llm, knowledge).ask(
        "Which ASU cluster has A100 GPUs and how do I request one for four hours?"
    )
    assert outcome["answer"], "must terminate and produce something, not loop"


def test_memory_records_the_turn(cfg, llm, knowledge):
    bot = _bot(cfg, llm, knowledge)
    bot.ask("hi")
    assert len(bot.memory.turns) == 1
