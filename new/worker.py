"""Node-level worker agents and the layered DAG executor.

Each node of the plan is handed to a small Reason-and-Act (ReAct) loop:

    thought -> action -> observation -> thought -> ... -> final answer

The loop is capped by ``worker.max_react_steps``. A budget is not a nicety
here: an uncapped ReAct loop on a 4-billion-parameter local model will happily
call ``vector_search`` with the same query fifteen times.

Two policies sit on top of the raw loop:

**Vector first, web second.** ASU's own documentation is authoritative for ASU
facts, so a worker that answers from the web when the vector store had the
answer is a regression, not an improvement.

**Automatic escalation.** If the worker finishes below
``escalate_to_web_below_confidence`` and never touched the web, one forced web
search runs and the node is re-answered. This is what makes fallback a property
of the evidence rather than a branch in a script.

Parallelism: :func:`execute_plan` walks the topological layers and runs every
node in a layer concurrently on a thread pool. Threads rather than asyncio
because the bottleneck is a blocking Hypertext Transfer Protocol call to
Ollama, and threads keep the tool implementations plain synchronous code.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .config import Config
from .dag import NodeResult, Plan
from .llm import LLM
from .retrieval import Evidence
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

WORKER_SYSTEM = """You answer ONE narrow factual question for an ASU Research
Computing assistant, using tools.

Available tools:
{tool_specs}

Reply with JSON only, in exactly one of two shapes.

To use a tool:
{{"thought": "<why this tool, one sentence>", "action": "<tool name>", "action_input": {{...}}}}

To finish:
{{"thought": "<one sentence>", "final_answer": "<answer grounded ONLY in observations>",
  "confidence": <0.0-1.0>, "citations": ["<source url or page title>", ...]}}

Rules:
1. Call vector_search before web_search for anything ASU-specific.
2. Never state a fact that does not appear in an observation. If the
   observations do not contain the answer, say so plainly and set confidence
   below 0.3.
3. Do not repeat a tool call with an argument you already tried.
4. Keep final_answer under {max_words} words. No preamble, no pleasantries.
5. Every citation must correspond to something you actually observed."""


class Worker:
    """Runs the ReAct loop for a single node."""

    def __init__(self, cfg: Config, llm: LLM, registry: ToolRegistry) -> None:
        self.cfg, self.llm, self.registry = cfg, llm, registry
        self.max_steps = int(cfg.get("worker.max_react_steps", 4))
        self.max_words = int(cfg.get("worker.answer_max_words", 180))
        self.escalate_below = float(cfg.get("worker.escalate_to_web_below_confidence", 0.55))

    def _system_prompt(self) -> str:
        return WORKER_SYSTEM.format(
            tool_specs=self.registry.render_specs(), max_words=self.max_words
        )

    def run(self, node_id: str, question: str, tool_hint: str = "vector") -> NodeResult:
        """Answer ``question`` and return a :class:`NodeResult`."""
        transcript: list[str] = [f"Question: {question}", f"Suggested first tool: {tool_hint}"]
        evidence: list[Evidence] = []
        tools_used: list[str] = []
        attempted: set[tuple[str, str]] = set()

        for step in range(self.max_steps):
            decision = self.llm.json(
                "WORKER", self._system_prompt(), "\n\n".join(transcript), default={}
            ) or {}

            if "final_answer" in decision:
                return self._finalize(node_id, question, decision, evidence, tools_used, transcript)

            action = str(decision.get("action", "")).strip()
            arguments = decision.get("action_input") or {}
            if not isinstance(arguments, dict):
                arguments = {"query": str(arguments)}

            if not action:
                transcript.append("Observation: no action given. Provide final_answer now.")
                continue

            signature = (action, str(sorted(arguments.items())))
            if signature in attempted:
                transcript.append(
                    f"Observation: {action} was already called with those arguments. "
                    "Try a different tool or produce final_answer."
                )
                continue
            attempted.add(signature)

            result = self.registry.call(action, arguments)
            tools_used.append(action)
            evidence.extend(result.evidence)
            transcript.append(
                f"Thought: {decision.get('thought', '')}\n"
                f"Action: {action} {arguments}\n"
                f"Observation: {result.content[:2500] if result.ok else 'FAILED: ' + result.content}"
            )
            logger.debug("node %s step %d: %s ok=%s", node_id, step, action, result.ok)

        # Budget exhausted: force a grounded summary from what we have.
        transcript.append("Step budget exhausted. Produce final_answer now from the observations above.")
        decision = self.llm.json("WORKER", self._system_prompt(), "\n\n".join(transcript), default={}) or {}
        return self._finalize(node_id, question, decision, evidence, tools_used, transcript)

    def _finalize(
        self,
        node_id: str,
        question: str,
        decision: dict,
        evidence: list[Evidence],
        tools_used: list[str],
        transcript: list[str],
    ) -> NodeResult:
        """Build the result and apply the web escalation policy."""
        answer = str(decision.get("final_answer", "")).strip()
        try:
            confidence = float(decision.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        citations = [str(c) for c in (decision.get("citations") or [])]

        needs_web = (
            confidence < self.escalate_below
            and "web_search" not in tools_used
            and self.registry.get("web_search") is not None
        )
        if needs_web:
            logger.info("node %s confidence %.2f: escalating to web", node_id, confidence)
            result = self.registry.call("web_search", {"query": question})
            tools_used.append("web_search")
            if result.ok:
                evidence.extend(result.evidence)
                transcript.append(f"Observation (escalated web search): {result.content[:2500]}")
                retry = self.llm.json(
                    "WORKER", self._system_prompt(), "\n\n".join(transcript), default={}
                ) or {}
                if retry.get("final_answer"):
                    answer = str(retry["final_answer"]).strip()
                    try:
                        confidence = max(confidence, float(retry.get("confidence", confidence)))
                    except (TypeError, ValueError):
                        pass
                    citations = [str(c) for c in (retry.get("citations") or citations)]

        if not answer:
            return NodeResult(
                node_id=node_id,
                question=question,
                answer="",
                confidence=0.0,
                tools_used=tools_used,
                error="worker produced no answer",
            )

        return NodeResult(
            node_id=node_id,
            question=question,
            answer=answer,
            confidence=max(0.0, min(1.0, confidence)),
            citations=citations or [e.cite() for e in evidence[:3] if e.cite()],
            evidence=[e.text[:400] for e in evidence[:6]],
            tools_used=tools_used,
        )


def execute_plan(
    plan: Plan,
    worker: Worker,
    max_parallel: int = 3,
    on_node: Callable[[NodeResult], None] | None = None,
) -> dict[str, NodeResult]:
    """Run every node, layer by layer, with parallelism inside each layer.

    Placeholders are substituted immediately before a node runs, using the
    answers accumulated from previous layers. ``on_node`` is an optional
    callback for streaming progress to a user interface.
    """
    results: dict[str, NodeResult] = {}
    answers: dict[str, str] = {}
    node_map = plan.by_id

    for depth, layer in enumerate(plan.layers()):
        logger.info("executing layer %d with nodes %s", depth, layer)
        jobs = [(node_id, node_map[node_id].resolve(answers), node_map[node_id].tool_hint) for node_id in layer]

        if len(jobs) == 1:
            layer_results = [worker.run(*jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(max_parallel, len(jobs))) as pool:
                layer_results = list(pool.map(lambda job: worker.run(*job), jobs))

        for result in layer_results:
            results[result.node_id] = result
            answers[result.node_id] = result.answer if result.ok else "[unresolved]"
            if on_node:
                on_node(result)

    return results
