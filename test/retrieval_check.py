"""Manual retrieval-quality validation harness. Not a pytest suite (Phase 3
is where automated tests get introduced) — this is a canned-query script a
human reads the output of, standing in for what a future Phase-2 worker
agent would be handed as an already-decomposed sub-task. Makes zero LLM
generation calls; it only proves ``db_engine.fetch()`` returns topically
correct, appropriately scored chunks.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_engine import fetch, read_config

CANNED_QUERIES = [
    "What GPUs does the Sol cluster have?",
    "How do I request a GPU allocation on Sol?",
    "What is ASU RC's appointment scheduling policy?",
    "How do I submit a Slurm job?",
    "What is the difference between Sol and Agave?",
    "How do I check my storage quota?",
]


def main() -> None:
    """Runs every canned query through ``db_engine.fetch()`` and prints the results.

    For each query, prints the score, title, chunk index, source URL, and a
    short preview for every chunk that passed ``SCORE_THRESHOLD`` — for
    manual review of topical correctness.
    """
    cfg = read_config()
    for query in CANNED_QUERIES:
        results = fetch(query, cfg)
        print(f"\n=== {query!r} ===")
        if not results:
            print("  (nothing passed SCORE_THRESHOLD)")
        for doc, score in results:
            title = doc.metadata.get("title")
            chunk_index = doc.metadata.get("chunk_index")
            source = doc.metadata.get("source")
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  [{round(score, 4)}] {title} (chunk {chunk_index}) - {source}")
            print(f"      {preview}...")


if __name__ == "__main__":
    main()
