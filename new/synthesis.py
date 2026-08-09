"""Verification and final answer composition.

Two stages, in this order, because the order is the point:

``verify``     Inspects the node results *before* anything is written for the
               user. It reports which nodes are under-evidenced, which drives
               the orchestrator's replan decision. Verifying after writing
               would only let you apologise; verifying before lets you fix.

``synthesise`` Merges node answers in topological order into one response with
               citations. It is explicitly instructed that it may not add
               facts -- its input is the node answers, and everything it writes
               must be traceable to one of them.

The escalation path replaces version 1's ``answer_other``: instead of a
category the classifier fell through to, it is a calibrated confidence falling
below a configured floor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Config
from .dag import NodeResult, Plan
from .llm import LLM

logger = logging.getLogger(__name__)

VERIFY_SYSTEM = """You audit draft sub-answers for grounding.

Return JSON only:
{"nodes": [{"id": "<node id>", "grounded": true|false, "issue": "<short or empty>"}],
 "overall_confidence": <0.0-1.0>,
 "needs_more_research": true|false}

A sub-answer is grounded only if it is supported by its evidence excerpts. Mark
grounded=false when the answer asserts specifics (numbers, flag names, partition
names, limits) that the evidence does not contain, or when the answer admits it
could not find the information."""

SYNTHESIS_SYSTEM = """You write the final reply for an ASU Research Computing
assistant.

You are given the user's original question and a set of verified sub-answers.
Compose one coherent reply.

Rules:
1. Use ONLY the sub-answers. Do not add facts, commands, flags or numbers that
   do not appear in them.
2. Answer every part of the original question. If a part could not be
   answered, say so explicitly rather than quietly omitting it.
3. Put commands and flags in backticks.
4. Write plainly. No preamble, no "great question", no restating the question.
5. Prefer short paragraphs. Use a numbered list only for ordered steps.
6. End with a "Sources:" line listing the distinct citations provided."""


@dataclass
class Verification:
    """Outcome of the grounding audit."""

    grounded_ids: set[str] = field(default_factory=set)
    problems: list[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    needs_more_research: bool = False


def verify(llm: LLM, cfg: Config, results: dict[str, NodeResult]) -> Verification:
    """Audit node results for grounding.

    Nodes that errored or came back empty are marked as problems without
    consulting the model -- there is nothing to audit. Only substantive
    answers are sent for grading, which keeps the audit cheap.
    """
    if not cfg.get("verification.enabled", True):
        return Verification({r.node_id for r in results.values() if r.ok}, [], 1.0, False)

    mechanical_problems = [
        f"{r.node_id}: {r.error or 'empty answer'}" for r in results.values() if not r.ok
    ]
    answerable = {nid: r for nid, r in results.items() if r.ok}
    if not answerable:
        return Verification(set(), mechanical_problems, 0.0, True)

    rendered = "\n\n".join(
        f"Node {r.node_id}\nQuestion: {r.question}\nAnswer: {r.answer}\n"
        f"Confidence: {r.confidence:.2f}\nEvidence: {' || '.join(r.evidence)[:1500]}"
        for r in answerable.values()
    )
    payload = llm.json("VERIFIER", VERIFY_SYSTEM, rendered, default=None)

    if not payload:
        # Fail open on the audit, but fall back to the workers' own confidence.
        average = sum(r.confidence for r in answerable.values()) / len(answerable)
        return Verification(
            set(answerable),
            mechanical_problems,
            average,
            average < float(cfg.get("verification.min_overall_confidence", 0.45)),
        )

    grounded: set[str] = set()
    problems = list(mechanical_problems)
    for entry in payload.get("nodes", []) or []:
        node_id = str(entry.get("id", ""))
        if entry.get("grounded"):
            grounded.add(node_id)
        else:
            problems.append(f"{node_id}: {entry.get('issue') or 'not grounded in evidence'}")

    try:
        confidence = float(payload.get("overall_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Verification(
        grounded_ids=grounded,
        problems=problems,
        overall_confidence=confidence,
        needs_more_research=bool(payload.get("needs_more_research")) or bool(problems),
    )


def collect_citations(results: dict[str, NodeResult]) -> list[str]:
    """De-duplicated citations in first-seen order."""
    seen: list[str] = []
    for result in results.values():
        for citation in result.citations:
            if citation and citation not in seen:
                seen.append(citation)
    return seen


def synthesise(
    llm: LLM,
    cfg: Config,
    plan: Plan,
    results: dict[str, NodeResult],
    verification: Verification,
    freshness: dict | None = None,
) -> str:
    """Compose the user-facing reply.

    Sub-answers are presented in topological order so the model reads
    prerequisites before the conclusions that depend on them.
    """
    floor = float(cfg.get("verification.min_overall_confidence", 0.45))
    if verification.overall_confidence < floor and not verification.grounded_ids:
        contact = cfg.get("escalation.contact", "the Research Computing team")
        message = str(cfg.get("escalation.message", "")).strip()
        return f"{message}\n\nContact: {contact}"

    ordered: list[NodeResult] = []
    for layer in plan.layers():
        for node_id in layer:
            result = results.get(node_id)
            if result and result.ok:
                ordered.append(result)

    blocks = []
    for result in ordered:
        flag = "" if result.node_id in verification.grounded_ids else " [UNVERIFIED — hedge this]"
        blocks.append(
            f"Sub-question: {result.question}\nSub-answer{flag}: {result.answer}\n"
            f"Citations: {', '.join(result.citations) or 'none'}"
        )

    unanswered = [
        plan.by_id[nid].question
        for nid in plan.by_id
        if nid not in results or not results[nid].ok
    ]
    if unanswered:
        blocks.append("Could not be answered: " + "; ".join(unanswered))

    answer = llm.text(
        "SYNTHESISER",
        SYNTHESIS_SYSTEM,
        f"Original question: {plan.query}\n\n" + "\n\n".join(blocks),
    ).strip()

    if freshness and freshness.get("age_hours") is not None:
        answer += f"\n\n_Documentation index last refreshed {freshness['age_hours']} hours ago._"
    return answer
