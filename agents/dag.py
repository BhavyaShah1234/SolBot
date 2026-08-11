"""DAG data structures for the orchestrator's decomposition plans.

Pure data structures — no LLM, no store, no network — which is
deliberate: this is the cheapest part of the pipeline to unit test
exhaustively (hand-built graphs, including deliberately bad ones). Shape
mined from ``new/dag.py`` (reference only, not reused as code); freshly
written here.

A ``Plan`` is a real dependency graph, not a flat fan-out list: nodes can
depend on other nodes' answers via ``depends_on`` and reference them in
their own question text with a ``{{node_id}}`` placeholder (e.g. "how do I
request one there for 4 hours" after a node that already identified
*where*). :func:`topological_layers` groups nodes into parallel-executable
waves via Kahn's algorithm, so independent sub-questions still run
concurrently even though the plan overall supports chaining.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class PlanError(ValueError):
    """Raised when a plan is structurally invalid: a cycle, or a reference to a nonexistent node.

    Callers (``planner.plan()``/``planner.replan()``) are expected to
    catch this and degrade to :func:`single_node_plan` rather than let it
    propagate — the system should never fail to answer just because a
    plan came back malformed.
    """


@dataclass
class PlanNode:
    """One sub-task in a decomposition plan.

    Attributes:
        id: Short id, unique within a plan (e.g. ``"n1"``).
        question: A question, ideally self-contained; may reference
            another node's answer via a ``{{node_id}}`` placeholder.
        depends_on: Ids of nodes whose answers this one's question
            references. Must be a superset of :meth:`referenced_ids`
            after :func:`validate_plan` runs.
        tool_hint: Advisory only — ``"vector"``, ``"web"``, or ``"none"``.
            The worker is free to override based on what it actually
            finds.
        rationale: Why the planner created this node. Trace-only, never
            shown to the end user.
    """

    id: str
    question: str
    depends_on: list[str] = field(default_factory=list)
    tool_hint: str = "vector"
    rationale: str = ""

    def resolve(self, answers: dict[str, str]) -> str:
        """Substitutes ``{{node_id}}`` placeholders with prior nodes' answers.

        Args:
            answers: ``{node_id: answer_text}`` for already-executed nodes.

        Returns:
            The question text with every placeholder whose id is present
            in ``answers`` substituted. A placeholder whose id is missing
            from ``answers`` is left verbatim (not blanked) so a failure
            to resolve stays visible in the resulting text rather than
            silently vanishing.
        """

        def _sub(match: re.Match) -> str:
            node_id = match.group(1)
            return answers.get(node_id, match.group(0))

        return _PLACEHOLDER_RE.sub(_sub, self.question)

    def referenced_ids(self) -> set[str]:
        """Ids referenced via ``{{node_id}}`` placeholders in this node's question."""
        return set(_PLACEHOLDER_RE.findall(self.question))


@dataclass
class NodeResult:
    """A worker's outcome for one plan node."""

    node_id: str
    question: str
    answer: str
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Plan:
    """A query's decomposition into a DAG of :class:`PlanNode`."""

    query: str
    nodes: list[PlanNode]

    @property
    def by_id(self) -> dict[str, PlanNode]:
        return {node.id: node for node in self.nodes}

    def layers(self) -> list[list[str]]:
        """Groups node ids into parallel-executable waves via Kahn's algorithm.

        Raises:
            PlanError: If the dependency graph contains a cycle.
        """
        return topological_layers(self.nodes)

    def depth(self) -> int:
        """Number of sequential layers this plan requires."""
        return len(self.layers())

    def is_trivial(self) -> bool:
        """True if this plan has at most one node (nothing to decompose)."""
        return len(self.nodes) <= 1


def topological_layers(nodes: Iterable[PlanNode]) -> list[list[str]]:
    """Groups node ids into parallel-executable waves via Kahn's algorithm.

    Layer 0 has no dependencies; layer *i* depends only on nodes in
    layers ``< i``. Cycle detection is a free byproduct: if the queue
    empties before every node is placed, whatever remains is part of a
    cycle.

    Args:
        nodes: The plan's nodes, in any order.

    Returns:
        A list of layers, each a list of node ids, in execution order.

    Raises:
        PlanError: If the dependency graph contains a cycle.
    """
    nodes = list(nodes)
    by_id = {node.id: node for node in nodes}
    remaining_deps = {node.id: set(node.depends_on) & by_id.keys() for node in nodes}
    placed: set[str] = set()
    layers: list[list[str]] = []

    while len(placed) < len(nodes):
        ready = sorted(
            node_id for node_id, deps in remaining_deps.items() if node_id not in placed and deps <= placed
        )
        if not ready:
            stuck = sorted(set(remaining_deps) - placed)
            raise PlanError(f"cycle detected among nodes: {stuck}")
        layers.append(ready)
        placed.update(ready)

    return layers


def validate_plan(plan: Plan, *, max_nodes: int, max_depth: int) -> Plan:
    """Validates and auto-repairs a plan before it's trusted for execution.

    Repairs (never rejects) a placeholder referenced in a node's question
    but missing from that node's ``depends_on`` — the planner LLM
    forgetting to declare a dependency it clearly used is a common,
    harmless mistake, fixed in place rather than treated as fatal.

    Args:
        plan: The plan to validate.
        max_nodes: Rejects plans with more nodes than this.
        max_depth: Rejects plans whose dependency chain is deeper than this.

    Returns:
        ``plan``, with any missing ``depends_on`` entries filled in.

    Raises:
        PlanError: If the plan is empty, has more than ``max_nodes``
            nodes, references a node id that doesn't exist anywhere in
            the plan, contains a cycle, or is deeper than ``max_depth``.
    """
    if not plan.nodes:
        raise PlanError("plan has no nodes")
    if len(plan.nodes) > max_nodes:
        raise PlanError(f"plan has {len(plan.nodes)} nodes, exceeds max_nodes={max_nodes}")

    valid_ids = {node.id for node in plan.nodes}
    for node in plan.nodes:
        referenced = node.referenced_ids()
        unknown = referenced - valid_ids
        if unknown:
            raise PlanError(f"node {node.id!r} references unknown node ids: {sorted(unknown)}")
        missing_deps = referenced - set(node.depends_on)
        if missing_deps:
            node.depends_on = sorted(set(node.depends_on) | missing_deps)

    depth = plan.depth()  # raises PlanError on a cycle
    if depth > max_depth:
        raise PlanError(f"plan depth {depth} exceeds max_depth={max_depth}")

    return plan


def plan_from_dict(query: str, payload: dict) -> Plan:
    """Builds a :class:`Plan` from a planner LLM's parsed JSON output.

    Args:
        query: The (standalone) query this plan answers.
        payload: Expected shape: ``{"nodes": [{"id", "question",
            "depends_on"?, "tool_hint"?, "rationale"?}, ...]}``.

    Returns:
        The constructed plan, unvalidated — callers should run it through
        :func:`validate_plan` before trusting it.

    Raises:
        PlanError: If ``payload`` isn't a dict with a non-empty ``nodes``
            list, or any node entry is missing ``id``/``question``.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list) or not payload["nodes"]:
        raise PlanError(f"malformed plan payload: {payload!r}")

    nodes = []
    for entry in payload["nodes"]:
        if not isinstance(entry, dict) or "id" not in entry or "question" not in entry:
            raise PlanError(f"malformed plan node: {entry!r}")
        nodes.append(
            PlanNode(
                id=str(entry["id"]),
                question=str(entry["question"]),
                depends_on=[str(d) for d in entry.get("depends_on", [])],
                tool_hint=entry.get("tool_hint", "vector"),
                rationale=entry.get("rationale", ""),
            )
        )
    return Plan(query=query, nodes=nodes)


def single_node_plan(query: str, tool_hint: str = "vector") -> Plan:
    """Builds the trivial one-node plan: ask the whole query as-is.

    The universal fallback for the planner's degradation ladder — too
    short to decompose, unparseable LLM output, or a structurally invalid
    graph all land here, so the system never fails to answer at all.
    """
    return Plan(query=query, nodes=[PlanNode(id="n1", question=query, tool_hint=tool_hint, rationale="single-node fallback")])
