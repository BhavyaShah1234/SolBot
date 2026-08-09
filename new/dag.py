"""Directed Acyclic Graph (DAG) primitives for query decomposition.

This module is deliberately free of any language model, network or vector
store dependency. It is pure data structure and graph algorithm, which makes
it the easiest part of the system to test exhaustively and the part least
likely to break when a library version changes.

Core ideas
----------
* A :class:`PlanNode` is one atomic sub-question.
* ``depends_on`` lists the node identifiers whose answers must be known first.
* A dependent question embeds ``{{n1}}`` placeholders that are substituted with
  the upstream answer at execution time.
* :func:`topological_layers` groups nodes into execution waves using Kahn's
  algorithm. Everything within a wave is mutually independent and therefore
  safe to run in parallel. The same traversal detects cycles for free: if any
  node is left over when the queue empties, the graph is cyclic.

Note on the project convention: configuration lives in YAML, never in a
``@dataclass``. The dataclasses below hold *runtime* values produced during a
single request, which is a different concern entirely.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


class PlanError(ValueError):
    """Raised when a planner-produced graph is structurally invalid."""


@dataclass
class PlanNode:
    """One atomic sub-question in the plan.

    Attributes
    ----------
    id
        Short unique identifier, e.g. ``"n1"``. Used in ``depends_on`` and in
        ``{{...}}`` placeholders.
    question
        A self-contained question. May contain placeholders.
    depends_on
        Identifiers of nodes that must complete first.
    tool_hint
        Advisory only: ``vector``, ``web``, ``status`` or ``none``. The worker
        may override it if the hinted tool yields nothing.
    rationale
        Why the planner created this node. Surfaced in traces, never shown to
        the end user.
    """

    id: str
    question: str
    depends_on: list[str] = field(default_factory=list)
    tool_hint: str = "vector"
    rationale: str = ""

    def resolve(self, answers: dict[str, str]) -> str:
        """Return ``question`` with every ``{{id}}`` replaced by that node's answer.

        An unresolved placeholder is left verbatim rather than blanked, so a
        downstream failure is visible in the trace instead of silently
        producing a truncated question.
        """
        return PLACEHOLDER.sub(
            lambda m: answers.get(m.group(1), m.group(0)), self.question
        )

    def referenced_ids(self) -> set[str]:
        """Node identifiers mentioned in placeholders inside ``question``."""
        return set(PLACEHOLDER.findall(self.question))


@dataclass
class NodeResult:
    """What a worker produced for one node."""

    node_id: str
    question: str
    answer: str
    confidence: float = 0.0
    citations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.answer.strip())


@dataclass
class Plan:
    """A validated DAG plus the query it decomposes."""

    query: str
    nodes: list[PlanNode]

    @property
    def by_id(self) -> dict[str, PlanNode]:
        return {node.id: node for node in self.nodes}

    def layers(self) -> list[list[str]]:
        return topological_layers(self.nodes)

    def depth(self) -> int:
        return len(self.layers())

    def is_trivial(self) -> bool:
        return len(self.nodes) <= 1


# --------------------------------------------------------------------------- #
# Graph algorithms
# --------------------------------------------------------------------------- #


def topological_layers(nodes: Iterable[PlanNode]) -> list[list[str]]:
    """Group node identifiers into parallel execution waves (Kahn's algorithm).

    Returns a list of lists: index 0 holds every node with no dependencies,
    index 1 holds nodes whose dependencies are all satisfied by layer 0, and
    so on.

    Raises
    ------
    PlanError
        If the graph contains a cycle or references an unknown identifier.
    """
    nodes = list(nodes)
    known = {node.id for node in nodes}

    indegree: dict[str, int] = {}
    dependents: dict[str, list[str]] = {node.id: [] for node in nodes}

    for node in nodes:
        unknown = [dep for dep in node.depends_on if dep not in known]
        if unknown:
            raise PlanError(f"node {node.id!r} depends on unknown nodes {unknown}")
        indegree[node.id] = len(set(node.depends_on))
        for dep in set(node.depends_on):
            dependents[dep].append(node.id)

    queue = deque(sorted(nid for nid, deg in indegree.items() if deg == 0))
    layers: list[list[str]] = []
    settled = 0

    while queue:
        layer = sorted(queue)
        layers.append(layer)
        queue.clear()
        for node_id in layer:
            settled += 1
            for child in dependents[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

    if settled != len(nodes):
        stuck = sorted(nid for nid, deg in indegree.items() if deg > 0)
        raise PlanError(f"plan contains a cycle involving {stuck}")

    return layers


def validate_plan(plan: Plan, *, max_nodes: int, max_depth: int) -> Plan:
    """Reject structurally unusable plans.

    Checks, in order: node count, unique identifiers, self-dependency,
    non-empty questions, placeholder/dependency consistency, acyclicity, and
    depth. Placeholder consistency matters because a node that interpolates
    ``{{n1}}`` without declaring ``n1`` as a dependency would run before ``n1``
    finished and interpolate nothing.
    """
    if not plan.nodes:
        raise PlanError("plan has no nodes")
    if len(plan.nodes) > max_nodes:
        raise PlanError(f"plan has {len(plan.nodes)} nodes, limit is {max_nodes}")

    ids = [node.id for node in plan.nodes]
    if len(set(ids)) != len(ids):
        raise PlanError(f"duplicate node identifiers in {ids}")

    for node in plan.nodes:
        if not node.question.strip():
            raise PlanError(f"node {node.id!r} has an empty question")
        if node.id in node.depends_on:
            raise PlanError(f"node {node.id!r} depends on itself")
        undeclared = node.referenced_ids() - set(node.depends_on)
        if undeclared:
            # Repair rather than reject: the intent is unambiguous.
            node.depends_on = sorted(set(node.depends_on) | undeclared)

    layers = plan.layers()  # raises PlanError on cycles or unknown references
    if len(layers) > max_depth:
        raise PlanError(f"plan depth {len(layers)} exceeds limit {max_depth}")

    return plan


def plan_from_dict(query: str, payload: dict[str, Any]) -> Plan:
    """Build a :class:`Plan` from the planner's raw JSON output.

    Tolerates the shapes small models actually emit: a bare list of nodes, a
    ``{"nodes": [...]}`` wrapper, string-or-list ``depends_on``, and missing
    optional fields.
    """
    raw_nodes = payload if isinstance(payload, list) else payload.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raise PlanError(f"expected a list of nodes, got {type(raw_nodes).__name__}")

    nodes: list[PlanNode] = []
    for index, item in enumerate(raw_nodes, start=1):
        if not isinstance(item, dict):
            raise PlanError(f"node at position {index} is not an object")
        depends = item.get("depends_on") or []
        if isinstance(depends, str):
            depends = [depends]
        nodes.append(
            PlanNode(
                id=str(item.get("id") or f"n{index}"),
                question=str(item.get("question", "")).strip(),
                depends_on=[str(d) for d in depends],
                tool_hint=str(item.get("tool_hint", "vector")).lower(),
                rationale=str(item.get("rationale", "")),
            )
        )
    return Plan(query=query, nodes=nodes)


def single_node_plan(query: str, tool_hint: str = "vector") -> Plan:
    """Fallback plan: treat the whole query as one node.

    Used when the planner fails, when the query is short enough that
    decomposition is wasted effort, or when a validation error forces a
    graceful degradation.
    """
    return Plan(
        query=query,
        nodes=[
            PlanNode(
                id="n1",
                question=query,
                depends_on=[],
                tool_hint=tool_hint,
                rationale="fallback: query treated as a single atomic question",
            )
        ],
    )
