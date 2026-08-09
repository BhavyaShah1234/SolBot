"""Query decomposition into a Directed Acyclic Graph (DAG).

This module replaces version 1's six-way intent classifier. The distinction
matters and is worth stating precisely:

* A **classifier** answers "which of my six scripts should run?" It is a
  closed set, single-label, and the answer format is baked in.
* A **planner** answers "what would I need to look up in order to answer
  this?" It is open-ended, multi-step, and the shape of the answer is
  discovered rather than pre-declared.

Three functions here, in the order the orchestrator calls them:

``route``    A cheap gate. Greetings and clarification requests never reach the
             planner at all, so the expensive path stays reserved for questions
             that need it.
``plan``     Produces and validates the DAG, degrading to a single-node plan on
             any failure rather than erroring out.
``replan``   Adaptive extension. When a node comes back under-evidenced, the
             planner is shown what failed and may add nodes. This is the
             difference between a static Plan-and-Execute agent and one that
             adapts mid-flight.
"""

from __future__ import annotations

import logging

from .config import Config
from .dag import Plan, PlanError, plan_from_dict, single_node_plan, validate_plan
from .llm import LLM

logger = logging.getLogger(__name__)

ROUTE_SYSTEM = """You route a user message for an ASU Research Computing assistant.

Return JSON only:
{"route": "<chat|clarify|research>", "reason": "<short>"}

chat      Greetings, thanks, small talk, or questions about your own abilities.
clarify   The message is too ambiguous to research. It refers to something
          unspecified, or could mean several different things.
research  Anything requiring factual lookup about clusters, jobs, storage,
          software, allocations, policy, or general computing.

When uncertain, choose research."""

PLAN_SYSTEM = """You decompose a question into a Directed Acyclic Graph of atomic
sub-questions for an ASU Research Computing assistant.

Return JSON only:
{"nodes": [
  {"id": "n1",
   "question": "<one self-contained factual question>",
   "depends_on": [],
   "tool_hint": "vector|web|status|none",
   "rationale": "<why this node exists>"}
]}

Rules:
1. Each node asks exactly ONE thing. If a question contains "and", "then",
   "also", or compares two items, it must be split.
2. A node that needs an earlier node's answer lists that node in depends_on and
   embeds it as {{n1}} inside its question text.
3. Independent nodes MUST have empty depends_on so they can run in parallel.
   Do not invent dependencies to force an ordering.
4. Never create a cycle.
5. Prefer tool_hint "vector" for anything ASU-specific (Sol, Phoenix, ASU
   Slurm policy, ASU storage, ASU allocations). Prefer "web" for upstream
   software documentation, hardware specifications, or general knowledge.
6. Emit the SMALLEST graph that answers the question. A simple question is one
   node. Never exceed the stated node limit.

Example
Question: "Which ASU cluster has A100 GPUs, how do I request one for 4 hours
there, and is that cluster up right now?"
{"nodes": [
 {"id":"n1","question":"Which ASU cluster has NVIDIA A100 GPUs?","depends_on":[],"tool_hint":"vector","rationale":"identifies the target cluster"},
 {"id":"n2","question":"What Slurm flags request a GPU with a 4 hour wall time?","depends_on":[],"tool_hint":"vector","rationale":"independent of which cluster"},
 {"id":"n3","question":"How do I submit a 4 hour GPU job on {{n1}}?","depends_on":["n1"],"tool_hint":"vector","rationale":"cluster-specific submission details"},
 {"id":"n4","question":"What is the current operational status of {{n1}}?","depends_on":["n1"],"tool_hint":"status","rationale":"live status of the identified cluster"}
]}"""

REPLAN_SYSTEM = """You extend an existing plan that came back under-evidenced.

Return JSON only: {"nodes": [...]} using the same node schema.

Include EVERY node from the existing plan unchanged, then add at most two new
nodes that attack the gap from a different angle: a broader phrasing, a
different vocabulary, or tool_hint "web" where "vector" failed. Do not renumber
existing nodes and do not create cycles."""


def route(llm: LLM, cfg: Config, query: str) -> str:
    """Return ``"chat"``, ``"clarify"`` or ``"research"``.

    Short messages skip the model entirely, which removes an inference from
    the most common interactions ("hi", "thanks").
    """
    threshold = int(cfg.get("planner.decomposition_threshold_words", 8))
    words = query.split()
    if len(words) <= 3 and not query.rstrip().endswith("?"):
        lowered = query.lower().strip(" .!?")
        if lowered in {"hi", "hello", "hey", "thanks", "thank you", "bye", "yo", "sup"}:
            return "chat"

    payload = llm.json("ROUTER", ROUTE_SYSTEM, query, default={"route": "research"})
    decision = str((payload or {}).get("route", "research")).lower()
    if decision not in {"chat", "clarify", "research"}:
        decision = "research"
    logger.debug("route(%r) -> %s", query[:60], decision)

    # A long, specific question is research regardless of a jittery small model.
    if decision == "chat" and len(words) > threshold:
        decision = "research"
    return decision


def plan(llm: LLM, cfg: Config, query: str) -> Plan:
    """Build a validated DAG for ``query``.

    Degradation ladder, worst case first: unparseable JSON -> single-node
    plan; structurally invalid graph -> single-node plan; valid graph ->
    returned as is. The system therefore never fails to answer because the
    planner misbehaved; it merely answers less cleverly.
    """
    max_nodes = int(cfg.get("planner.max_nodes", 6))
    max_depth = int(cfg.get("planner.max_depth", 4))

    if len(query.split()) < int(cfg.get("planner.decomposition_threshold_words", 8)):
        return single_node_plan(query)

    payload = llm.json(
        "PLANNER",
        PLAN_SYSTEM,
        f"Node limit: {max_nodes}. Depth limit: {max_depth}.\n\nQuestion: {query}",
        default=None,
    )
    if not payload:
        logger.warning("planner returned nothing; falling back to single node")
        return single_node_plan(query)

    try:
        candidate = plan_from_dict(query, payload)
        return validate_plan(candidate, max_nodes=max_nodes, max_depth=max_depth)
    except PlanError as error:
        logger.warning("rejected plan (%s); falling back to single node", error)
        return single_node_plan(query)


def replan(llm: LLM, cfg: Config, current: Plan, gaps: list[str]) -> Plan:
    """Extend ``current`` to address ``gaps``.

    ``gaps`` is a list of human-readable descriptions of what went wrong, e.g.
    ``["n2: no documentation matched", "n3: confidence 0.21"]``. If the model
    returns anything invalid, the original plan is returned untouched so the
    pipeline continues with what it already has.
    """
    if not cfg.get("planner.allow_replanning", True):
        return current

    existing = {
        "nodes": [
            {
                "id": node.id,
                "question": node.question,
                "depends_on": node.depends_on,
                "tool_hint": node.tool_hint,
                "rationale": node.rationale,
            }
            for node in current.nodes
        ]
    }
    payload = llm.json(
        "REPLANNER",
        REPLAN_SYSTEM,
        f"Original question: {current.query}\n\nExisting plan: {existing}\n\nGaps: {gaps}",
        default=None,
    )
    if not payload:
        return current

    try:
        candidate = plan_from_dict(current.query, payload)
        return validate_plan(
            candidate,
            max_nodes=int(cfg.get("planner.max_nodes", 6)) + 2,
            max_depth=int(cfg.get("planner.max_depth", 4)) + 1,
        )
    except PlanError as error:
        logger.warning("rejected replan (%s); keeping original", error)
        return current
