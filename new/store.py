"""Vector store abstraction.

SolBot talks to Chroma through ``chromadb`` directly rather than through the
``langchain_chroma`` wrapper. The reason is incremental sync: the wrapper hides
document identifiers and metadata-scoped deletes, and both are exactly what a
diff-based refresh needs. Working one level down costs about thirty lines and
buys precise upsert and purge semantics.

``MemoryStore`` implements the same interface with plain Python lists so the
retrieval, planning and orchestration layers can be unit tested without
installing Chroma or running an embedding server.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from .config import Config

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Minimum surface the rest of SolBot depends on."""

    def upsert(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def query(
        self, embedding: list[float], k: int, where: dict | None = None
    ) -> list[dict]: ...

    def all_documents(self) -> list[dict]: ...

    def count(self) -> int: ...


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity, guarding against zero-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class MemoryStore:
    """In-process store. Used by tests and by the ``--dry-run`` mode."""

    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def upsert(self, ids, texts, metadatas, embeddings) -> None:
        for doc_id, text, meta, vector in zip(ids, texts, metadatas, embeddings):
            self._docs[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": dict(meta),
                "embedding": list(vector),
            }

    def delete(self, ids) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)

    def query(self, embedding, k, where=None):
        candidates = [
            doc
            for doc in self._docs.values()
            if not where or all(doc["metadata"].get(key) == value for key, value in where.items())
        ]
        scored = [
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": cosine(embedding, doc["embedding"]),
            }
            for doc in candidates
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def all_documents(self):
        return [
            {"id": d["id"], "text": d["text"], "metadata": d["metadata"]}
            for d in self._docs.values()
        ]

    def count(self) -> int:
        return len(self._docs)


class ChromaStore:
    """Persistent Chroma-backed store.

    ``chromadb`` is imported lazily so that importing :mod:`solbot` in an
    environment without it (continuous integration, a lightweight worker)
    still succeeds.
    """

    def __init__(self, cfg: Config) -> None:
        import chromadb  # noqa: PLC0415 - deliberate lazy import

        directory = cfg.get("vector_store.directory", "./asu_rc")
        collection = cfg.get("vector_store.collection", "SolBot")
        self._client = chromadb.PersistentClient(path=directory)
        # Distance is cosine so that similarity == 1 - distance, which keeps
        # the score threshold in config.yaml interpretable.
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids, texts, metadatas, embeddings) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def delete(self, ids) -> None:
        if ids:
            self._collection.delete(ids=ids)

    def query(self, embedding, k, where=None):
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(meta or {}),
                    "score": 1.0 - float(distance),
                }
            )
        return hits

    def all_documents(self):
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            {"id": i, "text": t, "metadata": dict(m or {})}
            for i, t, m in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def count(self) -> int:
        return int(self._collection.count())


def build_store(cfg: Config) -> Any:
    """Instantiate the store named in ``config.yaml``."""
    provider = cfg.get("vector_store.provider", "chroma")
    if provider == "memory":
        return MemoryStore()
    return ChromaStore(cfg)
