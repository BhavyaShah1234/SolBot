"""Vector store access layer for the DB Engine.

Wraps ``langchain_chroma.Chroma`` construction so the rest of the package
never has to know the embedding model or collection configuration details.
Verified directly against the installed ``langchain-chroma==1.1.0``:
``add_texts(..., ids=[...])`` is a true upsert, ``delete(ids=[...])`` is a
true delete-by-id (and safely no-ops on nonexistent ids), and
``similarity_search_with_relevance_scores()`` returns a normalized
``[0, 1]`` cosine score when the collection is configured with a cosine
HNSW space — no need to drop to the raw ``chromadb`` client.
"""

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


@dataclass
class Chunk:
    """One stored chunk's full record: text, metadata, and its embedding vector.

    Distinct from the ``(Document, score)`` tuples :func:`fetch` returns —
    those are query-time search results; a ``Chunk`` is a corpus-wide
    record with no associated query or score, used by callers (e.g. a
    Phase-2 sparse/BM25 index) that need the whole collection rather than
    a similarity search against it.
    """

    id: str
    text: str
    metadata: dict
    embedding: list[float]


def get_all_chunks(store: Chroma) -> list[Chunk]:
    """Reads every chunk currently in the collection, embeddings included.

    Read-only, no locking — same reasoning as :meth:`DBEngine.fetch`: an
    occasional read of mid-write state is an acceptable tradeoff for a RAG
    system, and this never mutates the store.

    Args:
        store: The vector store client, as returned by :func:`get_vector_store`.

    Returns:
        One :class:`Chunk` per stored chunk. Embedding vectors are reused
        directly from Chroma rather than re-embedded, avoiding a second
        Ollama round-trip for callers (e.g. MMR reranking) that need them.
    """
    raw = store.get(include=["documents", "metadatas", "embeddings"])
    return [
        Chunk(id=chunk_id, text=text, metadata=metadata, embedding=list(embedding))
        for chunk_id, text, metadata, embedding in zip(
            raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]
        )
    ]


def get_vector_store(cfg: dict) -> Chroma:
    """Builds a Chroma vector store client configured for cosine similarity.

    Args:
        cfg: The loaded configuration dict; reads ``cfg["vector_store"]``
            and ``cfg["embedding"]`` (``model``, and optional ``base_url``
            for a remote Ollama host — omitted or ``None`` falls back to
            ``langchain_ollama``'s own default of local Ollama).

    Returns:
        A ``Chroma`` instance bound to the configured persist directory and
        collection name, with cosine distance so
        ``similarity_search_with_relevance_scores`` returns normalized
        ``[0, 1]`` scores.
    """
    vector_store_cfg = cfg["vector_store"]
    embedding_model = OllamaEmbeddings(
        model=cfg["embedding"]["model"], base_url=cfg["embedding"].get("base_url")
    )
    return Chroma(
        collection_name=vector_store_cfg["collection_name"],
        embedding_function=embedding_model,
        persist_directory=vector_store_cfg["db_directory"],
        collection_configuration={"hnsw": {"space": vector_store_cfg["distance_metric"]}},
    )
