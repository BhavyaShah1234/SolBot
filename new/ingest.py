"""Dynamic knowledge base construction and refresh.

Version 1 rebuilt the whole store from scratch every time, which meant it was
never rebuilt at all and the answers slowly went stale. This module replaces
that with a diff.

The manifest
------------
A JSON file maps each Confluence page identifier to the version number it was
last indexed at, a content hash, and the list of chunk identifiers it produced.
On each sync the live page list is compared against the manifest:

* **unchanged** -- version matches, skip entirely (no embedding calls)
* **changed**   -- delete the old chunk identifiers, re-chunk, re-embed, upsert
* **new**       -- chunk, embed, insert
* **deleted**   -- page is gone upstream, purge its chunk identifiers

Confluence's ``version.number`` is the primary change signal; the SHA-256 of
the rendered body is a secondary guard for the case where a page is touched
without its content actually changing (a label edit bumps the version).

Chunk identifiers are deterministic -- ``{page_id}::{index}`` -- so an upsert
overwrites in place instead of accumulating duplicates.

The second meaning of "dynamic"
-------------------------------
:func:`ingest_web_page` lets the running agent add a web page it just fetched
into the same store. Retrieval for the next question sees it immediately. This
is a write-through cache over the open web, and it is why the knowledge base
grows in the directions users actually ask about.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from .config import Config
from .llm import LLM

logger = logging.getLogger(__name__)


@dataclass
class SyncReport:
    """Counters returned by :meth:`KnowledgeBase.sync`."""

    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    chunks_written: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"added={self.added} updated={self.updated} deleted={self.deleted} "
            f"unchanged={self.unchanged} chunks={self.chunks_written} "
            f"in {self.duration_seconds:.1f}s"
        )


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def split_markdown(text: str, headers: list[list[str]]) -> list[tuple[str, dict]]:
    """Split markdown on headers, carrying the header trail as metadata.

    Returns ``(section_text, {"h1": ..., "h2": ...})`` pairs. Keeping the
    header trail matters because a chunk reading "Use ``--gres=gpu:1``" is
    ambiguous until you know it sits under "Sol > GPU jobs".
    """
    levels = {marker: name for marker, name in (tuple(h) for h in headers)}
    pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    sections: list[tuple[str, dict]] = []
    trail: dict[str, str] = {}
    cursor = 0
    current_meta: dict[str, str] = {}

    for match in pattern.finditer(text):
        body = text[cursor : match.start()].strip()
        if body:
            sections.append((body, dict(current_meta)))
        marker, title = match.group(1), match.group(2).strip()
        if marker in levels:
            depth = len(marker)
            trail = {k: v for k, v in trail.items() if len(k) <= 1 or int(k[1:]) < depth}
            trail[levels[marker]] = title
            current_meta = dict(trail)
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        sections.append((tail, dict(current_meta)))
    return sections or [(text.strip(), {})]


def recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character splitter that prefers natural boundaries.

    Tries paragraph breaks, then line breaks, then sentence ends, then raw
    character position -- the same strategy as
    ``RecursiveCharacterTextSplitter``, reimplemented here to avoid a
    dependency for forty lines of logic.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max((window.rfind(sep) for sep in separators), default=-1)
            if cut > chunk_size // 2:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #


class KnowledgeBase:
    """Owns the manifest and performs incremental synchronisation."""

    def __init__(self, cfg: Config, llm: LLM, store) -> None:
        self.cfg, self.llm, self.store = cfg, llm, store
        self.manifest_path = cfg.get("vector_store.manifest_path", "./manifest.json")
        self.manifest: dict[str, dict] = self._load_manifest()

    # -- manifest persistence ------------------------------------------------

    def _load_manifest(self) -> dict[str, dict]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _save_manifest(self) -> None:
        os.makedirs(os.path.dirname(self.manifest_path) or ".", exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)

    # -- chunking and writing ------------------------------------------------

    def _chunk_page(self, page_id: str, title: str, source: str, body: str) -> tuple[list[str], list[str], list[dict]]:
        """Turn one page into deterministic ``(ids, texts, metadatas)``."""
        headers = self.cfg.get("ingest.headers_to_split_on", [["#", "h1"], ["##", "h2"]])
        chunk_size = int(self.cfg.get("ingest.chunk_size", 1000))
        overlap = int(self.cfg.get("ingest.chunk_overlap", 200))

        ids, texts, metadatas = [], [], []
        index = 0
        for section_text, header_meta in split_markdown(body, headers):
            for piece in recursive_split(section_text, chunk_size, overlap):
                ids.append(f"{page_id}::{index}")
                texts.append(piece)
                metadatas.append(
                    {
                        "page_id": page_id,
                        "title": title,
                        "source": source,
                        "chunk_index": index,
                        "indexed_at": int(time.time()),
                        **{k: v for k, v in header_meta.items() if v},
                    }
                )
                index += 1
        return ids, texts, metadatas

    def _write(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> int:
        """Embed in batches and upsert. Returns the number of chunks written."""
        batch_size = int(self.cfg.get("ingest.embed_batch_size", 32))
        written = 0
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            vectors = self.llm.embed(texts[start:stop])
            self.store.upsert(ids[start:stop], texts[start:stop], metadatas[start:stop], vectors)
            written += stop - start if stop <= len(ids) else len(ids) - start
        return min(written, len(ids))

    # -- public operations ---------------------------------------------------

    def sync(self, pages: list[dict] | None = None) -> SyncReport:
        """Diff the source against the manifest and apply only the delta.

        ``pages`` is a list of dictionaries with keys ``id``, ``title``,
        ``version``, ``source`` and ``body`` (markdown). When ``None``, pages
        are fetched from Confluence using :func:`fetch_confluence_pages`.
        Passing them explicitly is what makes this method testable.
        """
        started = time.time()
        report = SyncReport()

        if pages is None:
            pages = fetch_confluence_pages(self.cfg)

        seen: set[str] = set()

        for page in pages:
            page_id = str(page["id"])
            seen.add(page_id)
            body = page.get("body", "") or ""
            digest = sha256(body)
            record = self.manifest.get(page_id)

            if record and record.get("version") == page.get("version") and record.get("hash") == digest:
                report.unchanged += 1
                continue

            try:
                if record:
                    self.store.delete(record.get("chunk_ids", []))
                ids, texts, metadatas = self._chunk_page(
                    page_id, page.get("title", ""), page.get("source", ""), body
                )
                report.chunks_written += self._write(ids, texts, metadatas)
                self.manifest[page_id] = {
                    "version": page.get("version"),
                    "hash": digest,
                    "title": page.get("title", ""),
                    "chunk_ids": ids,
                    "synced_at": int(time.time()),
                }
                if record:
                    report.updated += 1
                else:
                    report.added += 1
            except Exception as error:  # noqa: BLE001 - one bad page must not abort the sync
                report.errors.append(f"{page_id}: {error}")
                logger.exception("failed to index page %s", page_id)

        for page_id in [pid for pid in self.manifest if pid not in seen and not pid.startswith("web:")]:
            self.store.delete(self.manifest[page_id].get("chunk_ids", []))
            del self.manifest[page_id]
            report.deleted += 1

        self._save_manifest()
        report.duration_seconds = time.time() - started
        logger.info("sync complete: %s", report.summary())
        return report

    def ingest_web_page(self, url: str, title: str, body: str) -> int:
        """Add a fetched web page to the store immediately.

        Identifiers are namespaced with ``web:`` so the Confluence deletion
        pass never purges them.
        """
        page_id = f"web:{sha256(url)[:16]}"
        record = self.manifest.get(page_id)
        digest = sha256(body)
        if record and record.get("hash") == digest:
            return 0

        if record:
            self.store.delete(record.get("chunk_ids", []))
        ids, texts, metadatas = self._chunk_page(page_id, title, url, body)
        for meta in metadatas:
            meta["origin"] = "web"
        written = self._write(ids, texts, metadatas)
        self.manifest[page_id] = {
            "version": None,
            "hash": digest,
            "title": title,
            "chunk_ids": ids,
            "synced_at": int(time.time()),
        }
        self._save_manifest()
        return written

    def freshness(self) -> dict:
        """Age of the index, for the "documentation last synced" footer."""
        stamps = [rec.get("synced_at", 0) for rec in self.manifest.values()]
        newest = max(stamps, default=0)
        return {
            "pages": len(self.manifest),
            "chunks": self.store.count(),
            "last_sync_epoch": newest,
            "age_hours": round((time.time() - newest) / 3600, 1) if newest else None,
        }


def fetch_confluence_pages(cfg: Config) -> list[dict]:
    """Pull the configured Confluence space and normalise it to markdown.

    ``atlassian-python-api`` and ``markdownify`` are imported lazily so that
    the rest of SolBot runs without Confluence credentials installed.
    """
    from atlassian import Confluence  # noqa: PLC0415
    import markdownify  # noqa: PLC0415

    base_url = cfg.get("ingest.confluence_url")
    client = Confluence(
        url=base_url,
        username=cfg.get("ingest.username") or None,
        password=cfg.get("ingest.api_token") or None,
        cloud=True,
    )
    raw_pages = client.get_all_pages_from_space(
        space=cfg.get("ingest.space_key", "RC"),
        start=0,
        limit=int(cfg.get("ingest.page_limit", 500)),
        expand="body.view,version",
        status="current",
        content_type="page",
    )

    pages = []
    for page in raw_pages:
        html = page.get("body", {}).get("view", {}).get("value", "")
        pages.append(
            {
                "id": page["id"],
                "title": page.get("title", ""),
                "version": page.get("version", {}).get("number"),
                "source": base_url + page.get("_links", {}).get("webui", ""),
                "body": markdownify.markdownify(html, heading_style="ATX", code_language="bash"),
            }
        )
    logger.info("fetched %d pages from Confluence", len(pages))
    return pages
