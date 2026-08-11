"""Manual validation harness for memory_engine, mirroring retrieval_check.py
and generation_check.py's role: no orchestrator or agents package exists
yet to exercise this against, so this script drives it directly -- a
multi-turn conversation simulation, a concurrency stress test (the primary
proof of the "avoid race conditions" requirement), a fact-primitive test,
and a cross-session recall test.

Two checks below make their own throwaway, script-level ChatOllama calls,
explicitly as stand-ins for the future agents package's contextualizer and
insight-extractor -- that reasoning does not belong inside memory_engine
itself (memory_engine makes zero generative LLM calls by design; see its
package docstring). Both stand-ins reuse cfg["generation"]["model"] and the
SystemMessage/HumanMessage style already established in
generation_check.py, for consistency.

If Ollama is unreachable, the storage/concurrency checks still run in
full; the LLM stand-ins are skipped with a printed warning rather than
failing the whole script.
"""

import json
import multiprocessing
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

import memory_engine
from memory_engine.config import read_config
from memory_engine.store import get_connection

CFG = read_config()


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _ollama_available() -> bool:
    try:
        ChatOllama(model=CFG["generation"]["model"], temperature=0.1).invoke(
            [HumanMessage("ping")]
        )
        return True
    except Exception as exc:
        print(f"(Ollama unreachable, skipping LLM stand-in checks: {exc})")
        return False


def check_multi_turn_simulation(ollama_available: bool) -> None:
    _print_header("1. Multi-turn simulation")
    session_id = f"check-multiturn-{uuid.uuid4().hex[:8]}"

    memory_engine.append_message(session_id, "user", "What GPUs does the Sol cluster have?")
    memory_engine.append_message(session_id, "assistant", "Sol nodes offer NVIDIA A100 and A30 GPUs.")

    context = memory_engine.get_recent_context(session_id)
    print("get_recent_context ->")
    print(json.dumps(context, indent=2))

    if ollama_available:
        # Stand-in for the future agents package's contextualizer -- NOT part of memory_engine.
        raw_query = "Can I run that for longer?"
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in context)
        llm = ChatOllama(model=CFG["generation"]["model"], temperature=0.1)
        messages = [
            SystemMessage(
                "Rewrite the user's final message as a standalone question, using "
                "the conversation history for context. Reply with only the "
                "rewritten question, nothing else."
            ),
            HumanMessage(f"History:\n{transcript}\n\nFinal message: {raw_query}"),
        ]
        rewrite = llm.invoke(messages).content
        print(f"\nraw query:       {raw_query!r}")
        print(f"standalone form: {rewrite!r}")

    memory_engine.clear_session(session_id)


def check_concurrency_stress() -> None:
    _print_header("2. Concurrency stress test (in-process threads)")
    session_id = f"check-stress-{uuid.uuid4().hex[:8]}"
    n_threads = 8
    msgs_per_thread = 5

    def _worker(worker_id: int) -> None:
        for i in range(msgs_per_thread):
            memory_engine.append_message(session_id, "user", f"worker {worker_id} msg {i}")
            memory_engine.append_message(session_id, "assistant", f"reply to worker {worker_id} msg {i}")

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(_worker, range(n_threads)))

    conn = get_connection(CFG)
    total = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    expected = n_threads * msgs_per_thread * 2
    turn_numbers = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT turn_number FROM messages WHERE session_id = ? "
            "AND turn_number IS NOT NULL ORDER BY turn_number",
            (session_id,),
        ).fetchall()
    ]
    max_turn = conn.execute("SELECT turn_counter FROM sessions WHERE session_id = ?", (session_id,)).fetchone()[0]

    print(f"expected messages: {expected}, actual: {total}")
    assert total == expected, "lost writes under concurrent append!"
    assert turn_numbers == list(range(1, max_turn + 1)), "turn_number gaps/collisions detected!"
    assert max_turn == len(turn_numbers), "turn_counter doesn't match distinct turn count!"
    print("no lost writes, turn numbers contiguous, turn_counter consistent -- PASS")

    memory_engine.clear_session(session_id)


def _cross_process_worker(session_id: str, worker_id: int, msgs: int) -> None:
    import memory_engine as m  # re-imported in the child process

    for i in range(msgs):
        m.append_message(session_id, "user", f"proc {worker_id} msg {i}")
        m.append_message(session_id, "assistant", f"reply to proc {worker_id} msg {i}")


def check_cross_process_stress() -> None:
    _print_header("2b. Concurrency stress test (separate processes)")
    session_id = f"check-mpstress-{uuid.uuid4().hex[:8]}"
    n_procs = 4
    msgs_per_proc = 5

    procs = [
        multiprocessing.Process(target=_cross_process_worker, args=(session_id, i, msgs_per_proc))
        for i in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    conn = get_connection(CFG)
    total = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0]
    expected = n_procs * msgs_per_proc * 2
    print(f"expected messages: {expected}, actual: {total}")
    assert total == expected, "lost writes under cross-process concurrent append!"
    print("no lost writes across processes -- PASS")

    memory_engine.clear_session(session_id)


def check_fact_primitive(ollama_available: bool) -> None:
    _print_header("3. Fact primitive test")
    session_id = f"check-facts-{uuid.uuid4().hex[:8]}"
    memory_engine.append_message(session_id, "user", "hello")

    result1 = memory_engine.upsert_facts(
        session_id, [{"key": "user_name", "value": "Bhavya", "confidence": 1.0}], turns_covered=1
    )
    print(f"first upsert:  {result1}")
    result2 = memory_engine.upsert_facts(
        session_id, [{"key": "user_name", "value": "Bhavya", "confidence": 1.0}], turns_covered=0
    )
    print(f"second upsert (same key, proves UPSERT not INSERT-only): {result2}")

    facts = memory_engine.get_profile_facts(session_id)
    print(f"get_profile_facts -> {facts}")
    assert facts.get("user_name") == "Bhavya"

    threshold = CFG["memory"]["extraction_interval_turns"]
    for i in range(threshold):
        memory_engine.append_message(session_id, "user", f"filler {i}")
    print(f"needs_extraction after {threshold} user turns -> {memory_engine.needs_extraction(session_id)}")
    assert memory_engine.needs_extraction(session_id) is True

    memory_engine.upsert_facts(session_id, [], turns_covered=threshold)
    print(f"needs_extraction after ack (facts=[]) -> {memory_engine.needs_extraction(session_id)}")
    assert memory_engine.needs_extraction(session_id) is False

    memory_engine.clear_session(session_id)

    if ollama_available:
        # Stand-in for the future agents package's insight-extractor -- NOT part of memory_engine.
        session_id2 = f"check-extract-{uuid.uuid4().hex[:8]}"
        memory_engine.append_message(session_id2, "user", "My name is Bhavya and I'm in Arizona time.")
        memory_engine.append_message(session_id2, "assistant", "Got it, nice to meet you!")
        context = memory_engine.get_recent_context(session_id2)
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in context)
        llm = ChatOllama(model=CFG["generation"]["model"], temperature=0.1)
        messages = [
            SystemMessage(
                "Extract durable, personalization-relevant facts about the USER "
                "from this conversation excerpt (name, timezone, preferences, "
                "etc). Never invent facts not stated or clearly implied. Return "
                'JSON only: {"facts": [{"key": "<snake_case>", "value": '
                '"<short string>", "confidence": <0.0-1.0>}, ...]}. If nothing '
                'new, return {"facts": []}.'
            ),
            HumanMessage(transcript),
        ]
        try:
            raw = llm.invoke(messages).content
            start, end = raw.find("{"), raw.rfind("}")
            extracted_facts = json.loads(raw[start : end + 1]).get("facts", [])
        except Exception as exc:
            print(f"extraction stand-in call failed (parsing/LLM issue, not a memory_engine bug): {exc}")
            extracted_facts = []
        print(f"extracted facts: {extracted_facts}")
        memory_engine.upsert_facts(session_id2, extracted_facts, turns_covered=1)
        print(f"get_profile_facts for session_id2's user -> {memory_engine.get_profile_facts(session_id2)}")
        memory_engine.clear_session(session_id2)


def check_cross_session_recall(ollama_available: bool) -> None:
    _print_header("4. Cross-session recall")
    if not ollama_available:
        print("(skipped -- requires the embedding model, which needs Ollama)")
        return

    gpu_session = f"check-recall-gpu-{uuid.uuid4().hex[:8]}"
    storage_session = f"check-recall-storage-{uuid.uuid4().hex[:8]}"
    query_session = f"check-recall-query-{uuid.uuid4().hex[:8]}"

    memory_engine.append_message(gpu_session, "user", "How do I request an A100 GPU allocation on Sol?")
    memory_engine.append_message(gpu_session, "assistant", "Use the 'sol_gpu' partition in your Slurm script.")

    memory_engine.append_message(storage_session, "user", "What's my storage quota on scratch?")
    memory_engine.append_message(storage_session, "assistant", "Scratch storage defaults to 10TB per user.")

    memory_engine.append_message(query_session, "user", "placeholder turn so this session has a user_id")

    results = memory_engine.recall_related(query_session, "requesting a GPU for a job")
    print(json.dumps(results, indent=2))
    assert any(r["session_id"] == gpu_session for r in results), "expected the GPU session to be recalled"
    assert all(r["session_id"] != query_session for r in results), "recall must never echo the querying session"

    own_results = memory_engine.recall_related(gpu_session, "A100 GPU allocation")
    assert all(
        r["session_id"] != gpu_session for r in own_results
    ), "recall must never return the session's own turns"

    print("cross-session recall behaves as expected -- PASS")

    memory_engine.clear_session(gpu_session)
    memory_engine.clear_session(storage_session)
    memory_engine.clear_session(query_session)


def main() -> None:
    ollama_available = _ollama_available()
    check_multi_turn_simulation(ollama_available)
    check_concurrency_stress()
    check_cross_process_stress()
    check_fact_primitive(ollama_available)
    check_cross_session_recall(ollama_available)
    print("\nAll checks completed.")


if __name__ == "__main__":
    main()
