"""CLI for the Memory Engine.

    python -m memory_engine show SESSION_ID [--turns N]
    python -m memory_engine facts SESSION_ID
    python -m memory_engine needs-extraction SESSION_ID
    python -m memory_engine upsert-fact SESSION_ID KEY VALUE [--confidence F] [--turns-covered N]
    python -m memory_engine clear SESSION_ID

Manual inspection/testing tool, mirroring ``db_engine/__main__.py``'s shape.
"""

import argparse
import json

from memory_engine import (
    clear_session,
    get_profile_facts,
    get_recent_context,
    needs_extraction,
    upsert_facts,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m memory_engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="Print recent context for a session")
    show_parser.add_argument("session_id")
    show_parser.add_argument("--turns", type=int, default=None)

    facts_parser = subparsers.add_parser("facts", help="Print long-term facts for a session's user")
    facts_parser.add_argument("session_id")

    needs_parser = subparsers.add_parser("needs-extraction", help="Check if a session is due for extraction")
    needs_parser.add_argument("session_id")

    upsert_parser = subparsers.add_parser("upsert-fact", help="Manually upsert one fact (no LLM involved)")
    upsert_parser.add_argument("session_id")
    upsert_parser.add_argument("key")
    upsert_parser.add_argument("value")
    upsert_parser.add_argument("--confidence", type=float, default=1.0)
    upsert_parser.add_argument("--turns-covered", type=int, default=0)

    clear_parser = subparsers.add_parser("clear", help="Clear a session's messages (keeps facts)")
    clear_parser.add_argument("session_id")

    args = parser.parse_args()

    if args.command == "show":
        print(json.dumps(get_recent_context(args.session_id, turns=args.turns), indent=2))
    elif args.command == "facts":
        print(json.dumps(get_profile_facts(args.session_id), indent=2))
    elif args.command == "needs-extraction":
        print(needs_extraction(args.session_id))
    elif args.command == "upsert-fact":
        fact = {"key": args.key, "value": args.value, "confidence": args.confidence}
        result = upsert_facts(args.session_id, [fact], args.turns_covered)
        print(json.dumps(result, indent=2))
    elif args.command == "clear":
        clear_session(args.session_id)
        print(f"cleared session {args.session_id}")


if __name__ == "__main__":
    main()
