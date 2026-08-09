"""SolBot v2 command line interface.

Subcommands
-----------
``sync``   Run one incremental knowledge base refresh and print the delta.
``plan``   Show the DAG for a query without executing it. The fastest way to
           tell whether a decomposition problem is in the planner or in
           retrieval.
``chat``   Interactive session. ``\\quit`` exits, ``\\clear`` resets memory,
           ``\\trace`` toggles stage-by-stage output, ``\\sync`` refreshes.
"""

from __future__ import annotations

import argparse
import logging
import sys

from solbot import build_app, load_config
from solbot.llm import LLM
from solbot.planner import plan as build_plan


def configure_logging(cfg) -> None:
    logging.basicConfig(
        filename=cfg.get("app.log_file", "solbot.log"),
        filemode="a",
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        level=getattr(logging, str(cfg.get("app.log_level", "INFO")).upper(), logging.INFO),
    )


def command_sync(args) -> int:
    from solbot.ingest import KnowledgeBase
    from solbot.store import build_store

    cfg = load_config(args.config)
    configure_logging(cfg)
    llm = LLM(cfg)
    knowledge_base = KnowledgeBase(cfg, llm, build_store(cfg))
    print(knowledge_base.sync().summary())
    return 0


def command_plan(args) -> int:
    cfg = load_config(args.config)
    configure_logging(cfg)
    plan = build_plan(LLM(cfg), cfg, args.query)

    print(f"query: {plan.query}")
    print(f"nodes: {len(plan.nodes)}   depth: {plan.depth()}\n")
    for depth, layer in enumerate(plan.layers()):
        print(f"layer {depth} (parallel):")
        for node_id in layer:
            node = plan.by_id[node_id]
            deps = ", ".join(node.depends_on) or "-"
            print(f"  {node.id:<4} [{node.tool_hint:<6}] deps({deps}) {node.question}")
    return 0


def command_chat(args) -> int:
    cfg = load_config(args.config)
    configure_logging(cfg)

    show_trace = {"on": args.trace}

    def on_event(name: str, payload: dict) -> None:
        if not show_trace["on"]:
            return
        if name == "plan":
            print(f"  · plan: {len(payload['nodes'])} nodes, layers {payload['layers']}", file=sys.stderr)
        elif name == "node":
            mark = "ok" if payload["ok"] else "FAILED"
            print(f"  · node {payload['id']}: {mark} conf={payload['confidence']:.2f}", file=sys.stderr)
        else:
            print(f"  · {name}: {payload}", file=sys.stderr)

    bot = build_app(args.config, on_event)
    print("SolBot v2. \\quit  \\clear  \\trace  \\sync")

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message == "\\quit":
            return 0
        if message == "\\clear":
            bot.memory.clear()
            print("memory cleared")
            continue
        if message == "\\trace":
            show_trace["on"] = not show_trace["on"]
            print(f"trace {'on' if show_trace['on'] else 'off'}")
            continue
        if message == "\\sync":
            print(bot.knowledge_base.sync().summary())
            continue

        print(f"\nSolBot: {bot.ask(message)['answer']}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="solbot")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("sync").set_defaults(func=command_sync)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("query")
    plan_parser.set_defaults(func=command_plan)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument("--trace", action="store_true")
    chat_parser.set_defaults(func=command_chat)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
