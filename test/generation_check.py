"""Manual generation-quality validation harness, simulating a single Phase-2
worker's retrieve-then-generate flow in isolation: no chat loop, no
conversation history, no query decomposition. Calls ``db_engine.fetch()``
for raw context (the engine itself never generates — see
``db_engine``'s package docstring) and does its own single LLM call per
canned query. Proves the retrieve-then-generate wiring works and gives a
first read on answer groundedness for straightforward questions; not a
preview of Phase 2's generation quality on compound queries.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from db_engine import fetch, read_config
from retrieval_check import CANNED_QUERIES

SYSTEM_PROMPT = (
    "You are a helpful assistant for ASU Research Computing. "
    "Answer the user's question using only the provided context. "
    "If the context does not contain the answer, say so explicitly."
)


def generate_answer(llm: ChatOllama, query: str, results: list[tuple]) -> str:
    """Builds a context-grounded prompt from raw retrieved chunks and generates an answer.

    Args:
        llm: The chat model to generate with.
        query: The user's question.
        results: ``(Document, score)`` tuples as returned by
            ``db_engine.fetch()`` — the caller's raw context; this function
            is where generation actually happens, not the engine.

    Returns:
        The model's answer text.
    """
    context = "\n\n".join(f"[{doc.metadata.get('title')}]\n{doc.page_content}" for doc, _score in results)
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(f"Context:\n{context}\n\nQuestion: {query}")]
    return llm.invoke(messages).content


def main() -> None:
    """Runs every canned query through fetch-then-generate and prints each answer."""
    cfg = read_config()
    llm = ChatOllama(model=cfg["generation"]["model"], temperature=0.1)
    for query in CANNED_QUERIES:
        results = fetch(query, cfg)
        if results:
            answer = generate_answer(llm, query, results)
        else:
            answer = "(no chunks passed SCORE_THRESHOLD -- skipping generation)"
        print(f"\n=== {query!r} ===\n{answer}")


if __name__ == "__main__":
    main()
