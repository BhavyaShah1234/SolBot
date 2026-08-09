"""The LangGraph state machine that ties every stage together.

Why a state machine rather than a straight function call chain: the replan edge
is a **cycle**. ``verify`` can send control backwards to ``plan``, bounded by
``max_replans``. Expressing that as nested ``if`` statements inside one long
function is how version 1's control flow became unreadable; expressing it as
labelled nodes with conditional edges keeps the topology inspectable and lets
you attach a checkpointer, a human-in-the-loop interrupt, or a tracer at any
node without touching the surrounding logic.

Graph shape::

    START -> contextualize -> route -+-> chat ------------------> END
                                     +-> clarify ---------------> END
                                     +-> plan -> execute -> verify -+-> synthesize -> END
                                            ^                       |
                                            +----- replan ----------+
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Config
from .dag import NodeResult, Plan
from .llm import LLM
from .memory import Memory
from .planner import plan as build_plan
from .planner import replan as extend_plan
from .planner import route as route_query
from .synthesis import Verification, synthesise, verify
from .worker import Worker, execute_plan

logger = logging.getLogger(__name__)

CHAT_SYSTEM = """You are SolBot, an assistant for ASU Research Computing.
Reply to this greeting or small talk in one or two sentences. Mention that you
answer questions about ASU's clusters, job scheduling, storage and software.
Do not invent facts about the clusters."""

CLARIFY_SYSTEM = """You are SolBot, an assistant for ASU Research Computing.
The user's message is too ambiguous to research. Ask exactly one specific
clarifying question. Do not ask more than one. Do not guess an answer."""


def _keep_last(_current: Any, incoming: Any) -> Any:
    """Reducer: last write wins. Explicit beats LangGraph's default."""
    return incoming


class SolBotState(TypedDict, total=False):
    """State carried between graph nodes."""

    query: Annotated[str, _keep_last]
    standalone_query: Annotated[str, _keep_last]
    route: Annotated[str, _keep_last]
    plan: Annotated[Plan | None, _keep_last]
    results: Annotated[dict[str, NodeResult], _keep_last]
    verification: Annotated[Verification | None, _keep_last]
    replan_count: Annotated[int, _keep_last]
    answer: Annotated[str, _keep_last]
    trace: Annotated[list[str], _keep_last]


class SolBot:
    """Assembles and runs the graph."""

    def __init__(
        self,
        cfg: Config,
        llm: LLM,
        worker: Worker,
        memory: Memory,
        knowledge_base=None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.cfg, self.llm, self.worker, self.memory = cfg, llm, worker, memory
        self.knowledge_base = knowledge_base
        self.on_event = on_event or (lambda _name, _payload: None)
        self.max_replans = int(cfg.get("planner.max_replans", 1))
        self.max_parallel = int(cfg.get("worker.max_parallel_nodes", 3))
        self.graph = self._build()

    # -- graph nodes ---------------------------------------------------------

    def _contextualize(self, state: SolBotState) -> dict:
        standalone = self.memory.contextualize(state["query"])
        self.on_event("contextualize", {"standalone": standalone})
        return {
            "standalone_query": standalone,
            "replan_count": 0,
            "results": {},
            "trace": [f"standalone: {standalone}"],
        }

    def _route(self, state: SolBotState) -> dict:
        decision = route_query(self.llm, self.cfg, state["standalone_query"])
        self.on_event("route", {"route": decision})
        return {"route": decision, "trace": state.get("trace", []) + [f"route: {decision}"]}

    def _chat(self, state: SolBotState) -> dict:
        return {"answer": self.llm.text("CHAT", CHAT_SYSTEM, state["query"]).strip()}

    def _clarify(self, state: SolBotState) -> dict:
        prompt = f"History:\n{self.memory.recent_transcript()}\n\nMessage: {state['query']}"
        return {"answer": self.llm.text("CLARIFIER", CLARIFY_SYSTEM, prompt).strip()}

    def _plan(self, state: SolBotState) -> dict:
        existing = state.get("plan")
        if existing is not None and state.get("verification") is not None:
            plan = extend_plan(
                self.llm, self.cfg, existing, state["verification"].problems
            )
        else:
            plan = build_plan(self.llm, self.cfg, state["standalone_query"])

        self.on_event(
            "plan",
            {"nodes": [(n.id, n.question, n.depends_on) for n in plan.nodes],
             "layers": plan.layers()},
        )
        return {
            "plan": plan,
            "trace": state.get("trace", []) + [f"plan: {len(plan.nodes)} nodes, depth {plan.depth()}"],
        }

    def _execute(self, state: SolBotState) -> dict:
        plan = state["plan"]
        assert plan is not None
        done = state.get("results", {})

        # On a replan, keep the nodes that already produced grounded answers.
        grounded = (
            state["verification"].grounded_ids if state.get("verification") else set()
        )
        reusable = {nid: res for nid, res in done.items() if nid in grounded}

        pending = Plan(
            query=plan.query, nodes=[n for n in plan.nodes if n.id not in reusable]
        )
        fresh = (
            execute_plan(
                pending,
                self.worker,
                self.max_parallel,
                on_node=lambda r: self.on_event(
                    "node", {"id": r.node_id, "confidence": r.confidence, "ok": r.ok}
                ),
            )
            if pending.nodes
            else {}
        )
        return {"results": {**reusable, **fresh}}

    def _verify(self, state: SolBotState) -> dict:
        result = verify(self.llm, self.cfg, state.get("results", {}))
        self.on_event(
            "verify",
            {"confidence": result.overall_confidence, "problems": result.problems},
        )
        return {
            "verification": result,
            "trace": state.get("trace", [])
            + [f"verify: confidence {result.overall_confidence:.2f}, {len(result.problems)} problems"],
        }

    def _synthesize(self, state: SolBotState) -> dict:
        freshness = self.knowledge_base.freshness() if self.knowledge_base else None
        answer = synthesise(
            self.llm,
            self.cfg,
            state["plan"],
            state.get("results", {}),
            state["verification"],
            freshness,
        )
        return {"answer": answer}

    # -- edges ---------------------------------------------------------------

    @staticmethod
    def _after_route(state: SolBotState) -> str:
        return {"chat": "chat", "clarify": "clarify"}.get(state.get("route", "research"), "plan")

    def _after_verify(self, state: SolBotState) -> str:
        verification = state.get("verification")
        if verification is None:
            return "synthesize"
        budget_left = state.get("replan_count", 0) < self.max_replans
        if verification.needs_more_research and budget_left and self.cfg.get("planner.allow_replanning", True):
            return "replan"
        return "synthesize"

    def _bump_replan(self, state: SolBotState) -> dict:
        count = state.get("replan_count", 0) + 1
        logger.info("replanning (attempt %d)", count)
        self.on_event("replan", {"attempt": count})
        return {"replan_count": count}

    def _build(self):
        graph = StateGraph(SolBotState)
        graph.add_node("contextualize", self._contextualize)
        graph.add_node("route", self._route)
        graph.add_node("chat", self._chat)
        graph.add_node("clarify", self._clarify)
        graph.add_node("plan", self._plan)
        graph.add_node("execute", self._execute)
        graph.add_node("verify", self._verify)
        graph.add_node("replan", self._bump_replan)
        graph.add_node("synthesize", self._synthesize)

        graph.add_edge(START, "contextualize")
        graph.add_edge("contextualize", "route")
        graph.add_conditional_edges(
            "route", self._after_route, {"chat": "chat", "clarify": "clarify", "plan": "plan"}
        )
        graph.add_edge("chat", END)
        graph.add_edge("clarify", END)
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "verify")
        graph.add_conditional_edges(
            "verify", self._after_verify, {"replan": "replan", "synthesize": "synthesize"}
        )
        graph.add_edge("replan", "plan")
        graph.add_edge("synthesize", END)
        return graph.compile()

    # -- public API ----------------------------------------------------------

    def ask(self, query: str) -> dict:
        """Answer one user message and record the turn in memory."""
        final = self.graph.invoke({"query": query})
        answer = final.get("answer", "").strip()
        self.memory.add(query, answer, {"route": final.get("route")})
        return {
            "answer": answer,
            "route": final.get("route"),
            "plan": final.get("plan"),
            "results": final.get("results", {}),
            "verification": final.get("verification"),
            "trace": final.get("trace", []),
        }
