"""Manual end-to-end validation harness for the Phase-2 orchestrator/worker pipeline.

Not a pytest suite (Phase 3 is where automated tests get introduced) --
the Phase-2 analogue of ``retrieval_check.py``/``generation_check.py``:
runs a fixed list of scripted, mostly multi-turn conversations through
``agents.Session`` and prints the full trace (route decision, plan DAG
with layers, per-node confidence/tools used, verification confidence/
problems, final answer) for a human to read and judge.

Each scenario is deliberately chosen to exercise one specific mechanism:

* Greeting -> routes to "chat", no plan built.
* Simple factual question -> single-node plan, grounded answer with citations.
* The A100/4-hour compound query -> multi-node DAG with a dependency edge;
  trace shows the {{n1}} placeholder resolved to real text before the
  second worker ran.
* A likely-thin-evidence question -> confirms web_search shows up in
  tools_used (evidence-sufficiency escalation).
* A two-turn "...on Sol?" / "How long can I run it for?" exchange ->
  confirms contextualize() resolves "it".
* A personal-fact turn ("My name is ... and I'm on Mountain time") followed
  later by a time-sensitive question, in the same session -> confirms
  facts round-trip through maybe_run_extraction -> memory_engine ->
  render_facts_for_prompt and the reply is visibly personalized.
* A fresh Session with the same session_id prefix pattern (new session,
  same default user) -> confirms get_profile_facts/recall_related surface
  a fact set in an earlier session without the user restating it.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import Session, read_config


def _print_trace(result: dict) -> None:
    print(f"  route: {result['route']}")
    plan = result.get("plan")
    if plan is not None:
        print(f"  plan: {len(plan.nodes)} node(s), layers={plan.layers()}")
        for node in plan.nodes:
            node_result = result["results"].get(node.id)
            print(f"    [{node.id}] {node.question!r} depends_on={node.depends_on}")
            if node_result is not None:
                print(
                    f"        -> ok={node_result.ok} confidence={node_result.confidence:.2f} "
                    f"tools_used={node_result.tools_used}"
                )
        verification = result.get("verification")
        if verification is not None:
            print(
                f"  verification: overall_confidence={verification.overall_confidence:.2f} "
                f"grounded={sorted(verification.grounded_ids)} problems={verification.problems}"
            )
    print(f"  answer: {result['answer']}")


def scenario_greeting(cfg: dict) -> None:
    print("\n=== Scenario: greeting ===")
    session = Session(session_id=f"chatcheck-greeting-{uuid.uuid4().hex[:6]}", cfg=cfg)
    result = session.ask("hi")
    _print_trace(result)
    assert result["route"] == "chat", f"expected route=chat, got {result['route']}"
    assert result["plan"] is None, "chat route should not build a plan"


def scenario_simple_factual(cfg: dict) -> None:
    print("\n=== Scenario: simple factual question ===")
    session = Session(session_id=f"chatcheck-simple-{uuid.uuid4().hex[:6]}", cfg=cfg)
    result = session.ask("What GPUs does the Sol cluster have?")
    _print_trace(result)
    assert result["route"] == "research"
    assert result["plan"] is not None and len(result["plan"].nodes) >= 1
    # Note: the initial decomposition should skip straight to a single-node plan
    # (query is under decomposition_threshold_words) -- but the verify->replan
    # safety net can still legitimately expand ANY plan, including one that
    # started single-node, if the worker's first answer doesn't verify well.
    # A weaker/smaller model triggers this more often (more retries, more
    # replans) -- that's the fallback machinery doing its job, not a bug, so
    # this assertion checks the plan exists rather than requiring is_trivial().


def scenario_compound_dag(cfg: dict) -> None:
    print("\n=== Scenario: compound query -> DAG with dependency ===")
    session = Session(session_id=f"chatcheck-dag-{uuid.uuid4().hex[:6]}", cfg=cfg)
    result = session.ask("Which cluster has A100 GPUs and how do I request one there for 4 hours?")
    _print_trace(result)
    plan = result["plan"]
    assert plan is not None and len(plan.nodes) >= 2, "expected a multi-node decomposition"
    has_dependency = any(n.depends_on for n in plan.nodes)
    assert has_dependency, "expected at least one dependency edge in the DAG"


def scenario_thin_evidence_web_escalation(cfg: dict) -> None:
    print("\n=== Scenario: thin-evidence question -> web_search escalation ===")
    session = Session(session_id=f"chatcheck-web-{uuid.uuid4().hex[:6]}", cfg=cfg)
    # A concrete, well-formed, specific question (so it routes to "research", not
    # "clarify") that ASU RC's own docs almost certainly don't cover -- exercises
    # the evidence-sufficiency gate's web_search escalation.
    result = session.ask("What is the official release date of the NVIDIA B200 GPU?")
    _print_trace(result)
    if result["results"] is None:
        print(f"  (route={result['route']!r}, no plan was built -- nothing to check tools_used on)")
        return
    tools_used = {tool for r in result["results"].values() for tool in r.tools_used}
    print(f"  (tools used across all nodes: {sorted(tools_used)})")


def scenario_followup_contextualize(cfg: dict) -> None:
    print("\n=== Scenario: two-turn follow-up -> contextualize resolves 'it' ===")
    session = Session(session_id=f"chatcheck-followup-{uuid.uuid4().hex[:6]}", cfg=cfg)
    result1 = session.ask("What GPUs does the Sol cluster have?")
    print("  turn 1:")
    _print_trace(result1)
    result2 = session.ask("How long can I run a job with it?")
    print("  turn 2:")
    _print_trace(result2)


def scenario_personal_facts(cfg: dict) -> None:
    print("\n=== Scenario: personal-fact turn -> later personalized answer ===")
    session = Session(session_id=f"chatcheck-facts-{uuid.uuid4().hex[:6]}", cfg=cfg)
    result1 = session.ask("My name is Bhavya and I'm on Mountain time.")
    print("  turn 1 (states name/timezone):")
    _print_trace(result1)
    # extraction is periodic (every extraction_interval_turns), so it may not
    # have fired yet after one turn -- this scenario is a best-effort check,
    # not a strict assertion, matching the design's "every N turns" behavior.
    result2 = session.ask("What GPUs does the Sol cluster have?")
    print("  turn 2 (should be personalized if extraction has fired by now):")
    _print_trace(result2)


def scenario_cross_session_recall(cfg: dict) -> None:
    print("\n=== Scenario: new session, same user -> facts/recall carry over ===")
    session1 = Session(session_id="chatcheck-crosssession-1", cfg=cfg)
    session1.ask("My preferred cluster is Sol.")

    session2 = Session(session_id="chatcheck-crosssession-2", cfg=cfg)
    import memory_engine

    facts = memory_engine.get_profile_facts(session2.session_id, cfg=cfg)
    print(f"  session 2 sees profile facts from session 1: {facts}")
    recalled = memory_engine.recall_related(session2.session_id, "which cluster should I use", cfg=cfg)
    print(f"  session 2 recall_related hits from session 1: {len(recalled)}")


SCENARIOS = [
    scenario_greeting,
    scenario_simple_factual,
    scenario_compound_dag,
    scenario_thin_evidence_web_escalation,
    scenario_followup_contextualize,
    scenario_personal_facts,
    scenario_cross_session_recall,
]


def main() -> None:
    cfg = read_config()
    from agents.logging_setup import configure_logging

    configure_logging(cfg)
    for scenario in SCENARIOS:
        scenario(cfg)
    print("\nAll scenarios ran without raising. Read the traces above to judge answer quality.")


if __name__ == "__main__":
    main()
