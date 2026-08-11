"""Markdown conversion and chunking for the DB Engine.

Converts a Confluence page's HTML body to Markdown, splits it first by
header (so section boundaries aren't cut mid-thought) and then by
character count, and assigns each resulting chunk a deterministic id so
re-syncing a page overwrites its existing chunks in place instead of
duplicating them.
"""

from markdownify import markdownify
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


def chunk_page(page: dict, cfg: dict) -> tuple[list[str], list[str], list[dict]]:
    """Converts and splits one Confluence page into chunk ids, texts, and metadata.

    Args:
        page: A page dict as produced by
            ``db_engine.confluence_source._to_page_dict`` — must have
            ``id``, ``title``, ``source``, and ``html`` keys.
        cfg: The loaded configuration dict; reads ``cfg["chunking"]`` for
            ``chunk_size`` and ``chunk_overlap``.

    Returns:
        A 3-tuple ``(ids, texts, metadatas)`` of equal-length parallel
        lists, one entry per chunk:

        * ``ids``: deterministic chunk ids of the form
          ``f"{page_id}::{chunk_index}"``.
        * ``texts``: the chunk's text content.
        * ``metadatas``: dicts with ``id`` (the real Confluence page id,
          never the title), ``title``, ``source``, ``chunk_index``, and
          ``chunk_id`` (the same value as the corresponding entry in
          ``ids`` — included so callers can match a ``fetch()`` result
          back to a ``get_all_chunks()`` entry without having to know or
          reconstruct this module's id-formatting convention themselves).
    """
    markdown = markdownify(page["html"], heading_style="ATX", code_language="bash")
    header_docs = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON).split_text(markdown)

    base_meta = {"id": page["id"], "title": page["title"], "source": page["source"]}
    for doc in header_docs:
        doc.metadata.update(base_meta)

    chunking = cfg["chunking"]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunking["chunk_size"], chunk_overlap=chunking["chunk_overlap"]
    ).split_documents(header_docs)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for index, chunk in enumerate(chunks):
        chunk_id = f"{page['id']}::{index}"
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(chunk.metadata)
    return ids, texts, metadatas
