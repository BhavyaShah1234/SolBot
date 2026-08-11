"""Worker agents: the bounded ReAct loop that answers one DAG node, and the layer executor.

A worker never touches the vector store or web directly — every action
goes through a :class:`~agents.tools.ToolRegistry`, so the worker's own
code never special-cases a tool (see ``agents/tools.py``). Design mined
from ``new/worker.py`` (reference only, not reused as code): a bounded
``thought -> action -> observation`` loop, a loop-guard against repeated
identical tool calls, and a two-layer web-escalation policy (a soft nudge
via the tool observation, a hard fallback if the worker's own confidence
comes in low).
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Callable, Optional

from agents.dag import NodeResult, Plan
from agents.insight import distill_evidence
from agents.llm import LLM
from agents.logging_setup import log_context
from agents.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Domain-grounding, same text as agents/planner.py's _DOMAIN_GROUNDING (duplicated
# rather than imported -- these two modules don't otherwise depend on each other, and
# it's one string, not logic). Added after a live failure where a worker answering
# "How do I connect to Sol over SSH?" called web_search, got back Solana-blockchain
# infrastructure content for the ambiguous bare term "Sol", and blended it into the
# final answer ("Solana clusters do not provide public SSH access... consider Arcium").
_DOMAIN_GROUNDING = (
    "Context: this is ASU Research Computing support. \"Sol\" and \"Agave\" refer ONLY to "
    "ASU's supercomputer clusters -- never confuse them with unrelated same-named things "
    "(e.g. the Solana cryptocurrency/blockchain, the agave plant). If a question mentions "
    "\"Sol\" or \"Agave\", it means the ASU cluster. When calling web_search, include "
    "\"ASU\" or \"asu.edu\" in your query to avoid pulling in unrelated same-named entities, "
    "and if a tool result is clearly about something else (e.g. cryptocurrency, botany) "
    "despite a name match, discard it rather than using it.\n\n"
)

_REACT_SYSTEM_TEMPLATE = _DOMAIN_GROUNDING + """You are a research worker answering one focused question for ASU Research Computing support.

Available tools:
{tool_specs}

At each step, respond with ONLY JSON in ONE of these two shapes:
1. To call a tool: {{"thought": str, "action": {{"tool": str, "arguments": {{...}}}}}}
2. To finish: {{"thought": str, "final_answer": {{"answer": str, "confidence": float, "citations": [str]}}}}

Rules:
- Prefer vector_search for ASU Research Computing-specific facts before web_search.
- You MUST call at least one tool and use its results before giving a final_answer -- never answer purely from your own memory/training, even if you feel confident. If your tool calls find nothing useful, say so honestly in your final_answer (and set a low "confidence") rather than guessing.
- "confidence" is your own calibrated estimate in [0, 1] of how well-grounded your answer is in the evidence you gathered -- confidence with no supporting tool evidence must be low, never high.
- "citations" should list the source URLs/titles of evidence your answer actually relies on. An answer with no tool calls behind it should have empty citations and low confidence, not fabricated ones.
- Give your final_answer as soon as you have enough evidence -- don't call tools needlessly."""


class Worker:
    """Answers one DAG node's question via a bounded ReAct loop over tools."""

    def __init__(self, cfg: dict, llm: LLM, registry: ToolRegistry):
        self._cfg = cfg
        self._llm = llm
        self._registry = registry

    def run(self, node_id: str, question: str, tool_hint: str = "vector") -> NodeResult:
        """Answers one question, calling tools as needed, bounded by ``max_react_steps``.

        Sets the logging context's ``node_id`` as its very first action, so
        every log line from this call (tool calls, evidence distillation,
        the ReAct loop's own LLM calls) is tagged with the right node —
        including when this runs inside a worker thread dispatched by
        :func:`execute_plan`.

        Args:
            node_id: This plan node's id (for logging/tracing only).
            question: The question to answer (already placeholder-resolved
                by the caller).
            tool_hint: Advisory starting hint; the worker is free to use
                any registered tool regardless.

        Returns:
            A :class:`NodeResult`. Never raises — a failure becomes a
            ``NodeResult`` with ``error`` set instead.
        """
        with log_context(node_id=node_id):
            return self._run(node_id, question, tool_hint)

    def _run(self, node_id: str, question: str, tool_hint: str) -> NodeResult:
        worker_cfg = self._cfg["worker"]
        system = _REACT_SYSTEM_TEMPLATE.format(tool_specs=self._registry.render_specs())
        transcript: list[str] = [f"Question: {question}", f"(tool hint: {tool_hint})"]
        tools_used: list[str] = []
        all_evidence_sources: list[str] = []
        seen_calls: set[str] = set()

        logger.info("run: node=%s question=%r", node_id, question)

        for step in range(worker_cfg["max_react_steps"]):
            user = "\n\n".join(transcript)
            step_payload = self._llm.json(f"worker_step_{step}", system, user, default=None)

            if not isinstance(step_payload, dict):
                transcript.append("Observation: your last response was not valid JSON. Try again.")
                continue

            if "final_answer" in step_payload:
                final = step_payload["final_answer"]
                if not isinstance(final, dict) or "answer" not in final:
                    transcript.append("Observation: final_answer must be an object with an 'answer' field. Try again.")
                    continue
                reported_confidence = float(final.get("confidence", 0.5))
                if not tools_used:
                    # Code-level backstop for the system prompt's "must call a tool"
                    # rule: an 8B local model doesn't reliably follow that instruction
                    # on its own, and an ungrounded high-confidence answer is exactly
                    # the failure mode this design can't tolerate for a support bot.
                    # Clamping below the escalation threshold guarantees
                    # _maybe_escalate_to_web forces at least one real tool call
                    # (web_search) before this node can complete, regardless of what
                    # confidence the model claimed.
                    logger.warning(
                        "run: node=%s gave a final_answer with zero tool calls (reported confidence %.2f) "
                        "-- clamping confidence to force evidence-gathering before accepting it",
                        node_id,
                        reported_confidence,
                    )
                    reported_confidence = min(reported_confidence, 0.0)
                result = NodeResult(
                    node_id=node_id,
                    question=question,
                    answer=str(final["answer"]),
                    confidence=reported_confidence,
                    citations=[str(c) for c in final.get("citations", [])],
                    evidence=all_evidence_sources,
                    tools_used=tools_used,
                )
                return self._maybe_escalate_to_web(result, transcript, system)

            action = step_payload.get("action")
            if not isinstance(action, dict) or "tool" not in action:
                transcript.append("Observation: expected an 'action' with a 'tool' field, or a 'final_answer'. Try again.")
                continue

            tool_name = str(action["tool"])
            arguments = action.get("arguments", {}) if isinstance(action.get("arguments"), dict) else {}
            call_signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True, default=str)}"
            if call_signature in seen_calls:
                transcript.append(
                    f"Observation: you already called {tool_name} with these exact arguments. "
                    "Try different arguments, a different tool, or give your final_answer."
                )
                continue
            seen_calls.add(call_signature)

            tool_result = self._registry.call(tool_name, arguments)
            tools_used.append(tool_name)

            if tool_result.evidence:
                distilled = distill_evidence(self._llm, self._cfg, question, tool_result.evidence)
                all_evidence_sources.extend(f.source for f in distilled if f.source)
                observation_text = "\n".join(f"- {f.claim} (source: {f.source})" for f in distilled) or tool_result.content
            else:
                observation_text = tool_result.content

            sufficiency_note = ""
            if tool_result.meta.get("sufficient") is False:
                logger.warning("run: node=%s tool=%s returned thin evidence", node_id, tool_name)
                sufficiency_note = " (Note: this evidence looks thin -- consider web_search if this doesn't answer the question.)"

            transcript.append(f"Action: {tool_name}({arguments})")
            transcript.append(f"Observation: {observation_text}{sufficiency_note}")

        logger.warning("run: node=%s exhausted max_react_steps without a final_answer", node_id)
        return NodeResult(
            node_id=node_id,
            question=question,
            answer="",
            confidence=0.0,
            evidence=all_evidence_sources,
            tools_used=tools_used,
            error="exhausted max_react_steps without a final_answer",
        )

    def _maybe_escalate_to_web(self, result: NodeResult, transcript: list[str], system: str) -> NodeResult:
        """Hard fallback: forces one web_search + re-answer if confidence came in low.

        Only fires if ``web_search`` wasn't already called this run — the
        soft nudge in :meth:`_run` (via the tool observation's sufficiency
        note) is the first line of defense; this is the second,
        independent one, judged on the worker's own calibrated confidence
        rather than the retrieval-time sufficiency signal.
        """
        worker_cfg = self._cfg["worker"]
        if result.confidence >= worker_cfg["escalate_to_web_below_confidence"] or "web_search" in result.tools_used:
            return result

        logger.warning(
            "run: node=%s confidence %.2f below threshold, forcing web_search escalation",
            result.node_id,
            result.confidence,
        )
        web_result = self._registry.call("web_search", {"query": result.question})
        transcript.append(f"Action: web_search({{'query': {result.question!r}}})")
        transcript.append(f"Observation: {web_result.content}")
        transcript.append(
            "Your confidence was too low. Reconsider your answer using this new evidence "
            "and respond with a final_answer."
        )

        retry_payload = self._llm.json("worker_web_escalation", system, "\n\n".join(transcript), default=None)
        final = retry_payload.get("final_answer") if isinstance(retry_payload, dict) else None
        if not isinstance(final, dict) or "answer" not in final:
            result.tools_used = [*result.tools_used, "web_search"]
            return result

        return NodeResult(
            node_id=result.node_id,
            question=result.question,
            answer=str(final["answer"]),
            confidence=float(final.get("confidence", result.confidence)),
            citations=[str(c) for c in final.get("citations", result.citations)],
            evidence=result.evidence,
            tools_used=[*result.tools_used, "web_search"],
        )


def execute_plan(
    plan: Plan,
    worker: Worker,
    max_parallel: int = 3,
    on_node: Optional[Callable[[NodeResult], None]] = None,
) -> dict[str, NodeResult]:
    """Executes a plan layer by layer, running each layer's nodes concurrently.

    Nodes within one layer have no dependency on each other (by
    construction of :meth:`agents.dag.Plan.layers`), so they run
    concurrently on a thread pool — the bottleneck is blocking HTTP calls
    to a local LLM server, not CPU, so threads are the right tool.
    ``contextvars.copy_context()`` is captured per node right before
    dispatch and the task is submitted as ``ctx.run(worker.run, ...)``
    rather than ``worker.run`` directly — plain
    ``concurrent.futures.Executor.submit()`` does not itself propagate the
    calling context into the worker thread (verified against this
    interpreter; that guarantee is ``asyncio``-specific), so this explicit
    step is what makes each node's ``log_context(node_id=...)`` actually
    isolated per node. See ``agents/logging_setup.py`` for the full
    reasoning.

    Args:
        plan: The plan to execute.
        worker: The worker to dispatch each node to.
        max_parallel: Max concurrent nodes within one layer.
        on_node: Optional callback invoked with each :class:`NodeResult`
            as soon as it completes (e.g. for incremental logging/UI).

    Returns:
        ``{node_id: NodeResult}`` for every node in the plan.
    """
    results: dict[str, NodeResult] = {}

    for layer in plan.layers():
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for node_id in layer:
                node = plan.by_id[node_id]
                question = node.resolve({nid: results[nid].answer for nid in results})
                ctx = copy_context()
                future = executor.submit(ctx.run, worker.run, node_id, question, node.tool_hint)
                futures[future] = node_id

            for future in futures:
                node_id = futures[future]
                result = future.result()
                results[node_id] = result
                if on_node is not None:
                    on_node(result)

    return results
