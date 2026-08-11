"""Query routing and decomposition for the orchestrator.

Three functions, cheap-to-expensive: :func:`route` picks a coarse path
(often for free, no LLM call), :func:`plan` decomposes a research query
into a DAG, :func:`replan` patches a plan that turned out to have gaps.
Every failure mode degrades gracefully to something that still answers
the user rather than raising — mirrors ``new/planner.py``'s design
(reference only, not reused as code): unparseable JSON, a too-short
query, or a structurally invalid graph all fall back to
:func:`agents.dag.single_node_plan`, never a hard failure.
"""

import logging
import re

from agents.dag import Plan, PlanError, PlanNode, plan_from_dict, single_node_plan, validate_plan
from agents.llm import LLM

logger = logging.getLogger(__name__)

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|great|bye|goodbye|yo|sup)[\s!.,]*$", re.IGNORECASE
)

# Domain-grounding, prepended to prompts observed to be at risk of entity
# confusion on a small model: "Sol" and "Agave" are ASU Research Computing's
# supercomputer cluster names, not the more prominent training-data entities
# a weak model might default to (Solana the cryptocurrency, agave the plant).
# Added after a live failure where replan() generated a node asking whether
# "Solana provides public SSH access to its clusters" in response to a
# question about ASU's Sol cluster -- the replan prompt had no domain
# context at all, just the bare query text and a gap string.
_DOMAIN_GROUNDING = (
    "Context: this is ASU Research Computing support. \"Sol\" and \"Agave\" refer ONLY to "
    "ASU's supercomputer clusters -- never confuse them with unrelated same-named things "
    "(e.g. the Solana cryptocurrency/blockchain, the agave plant). If a question mentions "
    "\"Sol\" or \"Agave\", it means the ASU cluster.\n\n"
)

_ROUTE_SYSTEM = """You classify a user's message for a Research Computing support assistant into exactly one route.
Respond with ONLY JSON: {"route": "chat" | "clarify" | "research"}
- "chat": a greeting, thanks, or small talk with no question to research.
- "clarify": a question too vague/ambiguous to research as-is (missing which cluster, what resource, etc.).
- "research": a real question about ASU Research Computing (clusters, Slurm, storage, policies, GPUs, etc.), even if compound."""

_PLAN_SYSTEM = _DOMAIN_GROUNDING + """You decompose a Research Computing support question into a small graph of atomic sub-questions.
Respond with ONLY JSON: {"nodes": [{"id": str, "question": str, "depends_on": [str], "tool_hint": "vector"|"web"|"none", "rationale": str}]}
Rules:
- Each node's "question" must be self-contained and atomic (answerable by one focused search).
- If a node's question needs another node's answer, its question text MUST contain a placeholder of the EXACT form {{<that node's id>}} -- the placeholder text must be character-for-character identical to the other node's own "id" field, never a descriptive name. That id MUST also appear in this node's "depends_on" list.
- Prefer independent nodes (no dependencies) when sub-questions don't actually depend on each other -- only use "depends_on" for genuine chains.
- tool_hint is advisory: "vector" for RC documentation questions, "web" for anything outside RC's own docs, "none" if no search is needed.
- Keep the graph small: as few nodes as the question genuinely requires.

Example -- query "Which cluster has A100 GPUs and how do I request one there for 4 hours?":
{"nodes": [
  {"id": "n1", "question": "Which ASU Research Computing cluster has A100 GPUs?", "depends_on": [], "tool_hint": "vector", "rationale": "identify the cluster first"},
  {"id": "n2", "question": "How do I request a GPU allocation for 4 hours on {{n1}}?", "depends_on": ["n1"], "tool_hint": "vector", "rationale": "needs n1's answer to know which cluster's docs to search"}
]}
Note how n2's question contains the literal text "{{n1}}", matching n1's "id" exactly, and "n1" is listed in n2's "depends_on"."""

_REPLAN_SYSTEM = _DOMAIN_GROUNDING + """A research plan's execution left some sub-questions poorly answered. You may add AT MOST 2 new nodes to attack the gaps from a different angle.
Respond with ONLY JSON: {"new_nodes": [{"id": str, "question": str, "depends_on": [str], "tool_hint": "vector"|"web"|"none", "rationale": str}]}
Do not repeat existing node ids. New node ids must be new. Stay within the ASU Research Computing domain -- do not introduce unrelated entities or topics not implied by the original query. If you cannot think of a useful new angle, respond with {"new_nodes": []}."""


def route(llm: LLM, cfg: dict, query: str) -> str:
    """Classifies a query into ``"chat"``, ``"clarify"``, or ``"research"``.

    Short-circuits to ``"chat"`` for obvious greetings/thanks without an
    LLM call at all; otherwise makes one classification call.

    Args:
        llm: The shared LLM client.
        cfg: The loaded config dict (unused today, accepted for a
            consistent signature with :func:`plan`/:func:`replan`).
        query: The (ideally already-standalone) query to classify.

    Returns:
        One of ``"chat"``, ``"clarify"``, ``"research"`` — always
        ``"research"`` if classification fails, since that's the safest
        default (worst case: an unnecessary research pass, not a dropped
        question).
    """
    stripped = query.strip()
    if len(stripped.split()) <= 4 and _GREETING_RE.match(stripped):
        logger.info("route: short-circuited to chat (greeting pattern), query=%r", query)
        return "chat"

    result = llm.json("route", _ROUTE_SYSTEM, query, default={"route": "research"})
    route_value = result.get("route", "research") if isinstance(result, dict) else "research"
    if route_value not in ("chat", "clarify", "research"):
        route_value = "research"
    logger.info("route: query=%r -> %s", query, route_value)
    return route_value


def plan(llm: LLM, cfg: dict, query: str) -> Plan:
    """Decomposes a query into a validated DAG, degrading gracefully on any failure.

    Degradation ladder: a query shorter than
    ``cfg["planner"]["decomposition_threshold_words"]`` skips decomposition
    entirely; unparseable LLM output or a structurally invalid graph both
    fall back to :func:`agents.dag.single_node_plan` rather than raising.

    Args:
        llm: The shared LLM client.
        cfg: The loaded config dict; reads ``cfg["planner"]``.
        query: The (ideally already-standalone) query to decompose.

    Returns:
        A validated :class:`Plan` — never raises.
    """
    planner_cfg = cfg["planner"]

    if len(query.split()) < planner_cfg["decomposition_threshold_words"]:
        logger.info("plan: query too short to decompose, using single_node_plan: %r", query)
        return single_node_plan(query)

    payload = llm.json("plan", _PLAN_SYSTEM, query, default=None)
    if payload is None:
        logger.warning("plan: unparseable planner output, falling back to single_node_plan")
        return single_node_plan(query)

    try:
        candidate = plan_from_dict(query, payload)
        validated = validate_plan(
            candidate, max_nodes=planner_cfg["max_nodes"], max_depth=planner_cfg["max_depth"]
        )
    except PlanError as exc:
        logger.warning("plan: invalid plan (%s), falling back to single_node_plan", exc)
        return single_node_plan(query)

    logger.info("plan: query=%r -> %d nodes, depth=%d", query, len(validated.nodes), validated.depth())
    return validated


def replan(llm: LLM, cfg: dict, current: Plan, gaps: list[str]) -> Plan:
    """Patches a plan with up to 2 new nodes attacking reported gaps.

    Fail-safe: keeps all of ``current``'s nodes verbatim and only ever
    appends; any invalid or empty response leaves ``current`` unchanged
    rather than raising or degrading it.

    Args:
        llm: The shared LLM client.
        cfg: The loaded config dict; reads ``cfg["planner"]``.
        current: The plan whose execution revealed gaps.
        gaps: Human-readable gap descriptions (e.g.
            ``"n2: no documentation matched"``).

    Returns:
        ``current`` with up to 2 new nodes appended, still validated
        against the same ``max_nodes``/``max_depth`` limits — or
        ``current`` unchanged if the LLM's response was empty, invalid,
        or made the plan too large/deep.
    """
    planner_cfg = cfg["planner"]
    existing_ids = {node.id for node in current.nodes}
    gap_text = "\n".join(f"- {gap}" for gap in gaps)
    user = f"Original query: {current.query}\n\nGaps found:\n{gap_text}"

    payload = llm.json("replan", _REPLAN_SYSTEM, user, default=None)
    new_entries = payload.get("new_nodes") if isinstance(payload, dict) else None
    if not new_entries:
        logger.info("replan: no usable new nodes, keeping current plan unchanged")
        return current

    new_nodes = []
    for entry in new_entries[:2]:
        if not isinstance(entry, dict) or "id" not in entry or "question" not in entry:
            continue
        node_id = str(entry["id"])
        if node_id in existing_ids:
            continue
        new_nodes.append(
            PlanNode(
                id=node_id,
                question=str(entry["question"]),
                depends_on=[str(d) for d in entry.get("depends_on", [])],
                tool_hint=entry.get("tool_hint", "vector"),
                rationale=entry.get("rationale", ""),
            )
        )
        existing_ids.add(node_id)

    if not new_nodes:
        logger.info("replan: new nodes were all invalid/duplicate ids, keeping current plan unchanged")
        return current

    candidate = Plan(query=current.query, nodes=[*current.nodes, *new_nodes])
    try:
        validated = validate_plan(
            candidate, max_nodes=planner_cfg["max_nodes"], max_depth=planner_cfg["max_depth"]
        )
    except PlanError as exc:
        logger.warning("replan: patched plan invalid (%s), keeping current plan unchanged", exc)
        return current

    logger.info("replan: added %d new node(s), plan now has %d nodes", len(new_nodes), len(validated.nodes))
    return validated
