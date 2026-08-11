"""Command-line entry point for the DB Engine.

Usage:
    python -m db_engine               # full manifest-diffed sync (default)
    python -m db_engine sync-all      # same as above, explicit
    python -m db_engine sync-page ID  # targeted update of one page
    python -m db_engine reset         # admin: wipe and rebuild from scratch
"""

import argparse
import logging

from db_engine import read_config, reset, sync_all, sync_page


def main() -> None:
    """Parses CLI arguments, dispatches to the requested engine operation, and prints the result.

    Configures file logging for this standalone run via ``logging.basicConfig``
    (writing to ``cfg["logging"]["log_file"]``) — this only happens here, in
    the CLI entry point, not on package import, so importing ``db_engine`` as
    a library elsewhere never hijacks the importing application's own
    logging configuration.
    """
    parser = argparse.ArgumentParser(prog="db_engine")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("sync-all", help="Full manifest-diffed sync across the whole space (default).")
    sync_page_parser = subparsers.add_parser("sync-page", help="Update one specific page by id.")
    sync_page_parser.add_argument("page_id")
    subparsers.add_parser("reset", help="Wipe and rebuild the entire vector database from scratch.")

    args = parser.parse_args()
    cfg = read_config()

    logging.basicConfig(
        filename=cfg["logging"]["log_file"],
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command in (None, "sync-all"):
        print(sync_all(cfg))
    elif args.command == "sync-page":
        print(sync_page(args.page_id, cfg))
    elif args.command == "reset":
        print(reset(cfg))


if __name__ == "__main__":
    main()
