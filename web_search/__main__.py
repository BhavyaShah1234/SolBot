"""Command-line entry point for the web_search tool.

Usage:
    python -m web_search search "query"   # titles/URLs/snippets only
    python -m web_search fetch "url"      # fetch + cleaned page content
"""

import argparse

from web_search import ExtractionFailed, fetch_page, search


def main() -> None:
    """Parses CLI arguments, dispatches to search or fetch, and prints the result."""
    parser = argparse.ArgumentParser(prog="web_search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search the web for a query.")
    search_parser.add_argument("query")

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and clean one page's content.")
    fetch_parser.add_argument("url")

    args = parser.parse_args()

    if args.command == "search":
        results = search(args.query)
        if not results:
            print("no results")
        for i, result in enumerate(results, 1):
            print(f"[{i}] {result.title}\n    {result.url}\n    {result.snippet}\n")
    elif args.command == "fetch":
        try:
            page = fetch_page(args.url)
        except ExtractionFailed as error:
            print(f"extraction failed: {error}")
            return
        marker = " (truncated)" if page.truncated else ""
        print(f"# {page.title}\n{page.url}{marker}\n\n{page.text}")


if __name__ == "__main__":
    main()
