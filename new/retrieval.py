"""Retrieval pipeline.

SolBot v1 issued a single ``similarity`` search with ``k=3``. This module
replaces that with a five-stage pipeline, each stage individually switchable
from ``config.yaml``:

1. **Dense retrieval** -- embedding nearest neighbours. Strong on paraphrase,
   weak on rare literal tokens such as ``--gres=gpu:a100:1`` or ``sbatch``.
2. **Sparse retrieval** -- Best Matching 25 (BM25) lexical scoring. The exact
   inverse: excellent on rare literal tokens, blind to paraphrase. High
   performance computing documentation is full of flags and module names, so
   this stage earns its keep here more than in most domains.
3. **Reciprocal Rank Fusion (RRF)** -- merges the two ranked lists using
   ``1 / (rrf_constant + rank)``. Rank-based rather than score-based, so the
   two systems' incomparable score scales never need calibrating.
4. **Maximal Marginal Relevance (MMR)** -- greedy reranking that trades a
   little relevance for diversity, preventing five near-identical chunks from
   the same page crowding out the one chunk that answers the other half of
   the question.
5. **Large language model (LLM) relevance filter** -- a cheap per-chunk yes/no
   grade. The embedder decides *similarity*; this stage decides *usefulness*.

The pipeline also returns an ``is_sufficient`` verdict. That single boolean is
what turns the web-search fallback from a scripted branch into an emergent
behaviour: the worker escalates because evidence was thin, not because an
intent classifier guessed a label.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import Config
from .llm import LLM
from .store import cosine

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that keeps hyphens and underscores.

    ``--gres=gpu:a100:1`` must not shatter into meaningless fragments, and
    ``module_load`` must stay one token.
    """
    return _TOKEN.findall(text.lower())


@dataclass
class Evidence:
    """One retrieved chunk with its provenance."""

    text: str
    source: str
    title: str = ""
    score: float = 0.0
    origin: str = "vector"  # vector | web
    doc_id: str = ""

    def cite(self) -> str:
        return f"{self.title or 'source'} <{self.source}>" if self.source else self.title


@dataclass
class RetrievalResult:
    """Everything a worker needs to decide whether to escalate."""

    evidence: list[Evidence] = field(default_factory=list)
    is_sufficient: bool = False
    top_score: float = 0.0
    reason: str = ""


# --------------------------------------------------------------------------- #
# Fusion and diversification
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k_constant: int = 60
) -> dict[str, float]:
    """Fuse several ranked identifier lists into one score map.

    Each list contributes ``1 / (k_constant + rank)`` per document, with rank
    starting at 1. A document ranked highly by both retrievers accumulates the
    most, so agreement between systems is rewarded without either system's raw
    scores needing to be on the same scale.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_constant + rank)
    return fused


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: list[tuple[str, list[float]]],
    k: int,
    lambda_multiplier: float,
) -> list[str]:
    """Greedy MMR selection.

    At each step pick the candidate maximising::

        lambda * sim(query, doc) - (1 - lambda) * max sim(doc, already_selected)

    ``lambda_multiplier`` of 1.0 collapses to plain relevance ranking; 0.0
    maximises diversity regardless of relevance.
    """
    if not candidates:
        return []

    lookup = dict(candidates)
    remaining = dict(candidates)
    selected: list[str] = []

    while remaining and len(selected) < k:
        # Ties are broken by insertion order, which is fusion rank order, so
        # an exact tie resolves in favour of the better-ranked candidate.
        best_id, best_score = None, float("-inf")
        for doc_id, vector in remaining.items():
            relevance = cosine(query_vector, vector)
            redundancy = max(
                (cosine(vector, lookup[chosen]) for chosen in selected),
                default=0.0,
            )
            score = lambda_multiplier * relevance - (1 - lambda_multiplier) * redundancy
            if score > best_score:
                best_id, best_score = doc_id, score
        selected.append(best_id)  # type: ignore[arg-type]
        remaining.pop(best_id)  # type: ignore[arg-type]

    return selected


class BM25Index:
    """Okapi BM25 over the store's documents.

    Rebuilt whenever the knowledge base is synced. Held in memory because the
    Research Computing space is a few thousand chunks -- small enough that an
    external inverted index would be premature.
    """

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = [doc["id"] for doc in documents]
        self.corpus = [tokenize(doc["text"]) for doc in documents]
        self.k1, self.b = k1, b
        self.doc_lengths = [len(tokens) for tokens in self.corpus]
        self.avg_length = (sum(self.doc_lengths) / len(self.corpus)) if self.corpus else 0.0

        self.frequencies: list[dict[str, int]] = []
        self.document_frequency: dict[str, int] = {}
        for tokens in self.corpus:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.frequencies.append(counts)
            for token in counts:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return the top ``k`` ``(document_id, score)`` pairs."""
        if not self.corpus:
            return []
        import math

        total = len(self.corpus)
        query_tokens = tokenize(query)
        scores: list[float] = [0.0] * total

        for token in query_tokens:
            df = self.document_frequency.get(token)
            if not df:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for index, counts in enumerate(self.frequencies):
                freq = counts.get(token, 0)
                if freq == 0:
                    continue
                length_norm = 1 - self.b + self.b * (
                    self.doc_lengths[index] / (self.avg_length or 1.0)
                )
                scores[index] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)

        ranked = sorted(zip(self.ids, scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked[:k] if pair[1] > 0]


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #

_RERANK_SYSTEM = """You grade whether a documentation excerpt helps answer a question.
Return JSON only: {"scores": [<float 0.0-1.0>, ...]} with exactly one score per
excerpt, in the order given. 1.0 means the excerpt directly answers the
question; 0.0 means it is unrelated."""


class Retriever:
    """Owns the vector store, the sparse index and the fusion policy."""

    def __init__(self, cfg: Config, llm: LLM, store) -> None:
        self.cfg, self.llm, self.store = cfg, llm, store
        self._bm25: BM25Index | None = None
        self._doc_cache: dict[str, dict] = {}

    def refresh_indexes(self) -> None:
        """Rebuild the BM25 index and document cache from the vector store.

        Called once at start-up and again after every knowledge base sync.
        """
        documents = self.store.all_documents()
        self._doc_cache = {doc["id"]: doc for doc in documents}
        self._bm25 = BM25Index(documents) if self.cfg.get("retrieval.use_sparse", True) else None
        logger.info("indexes refreshed over %d chunks", len(documents))

    def retrieve(self, question: str, where: dict | None = None) -> RetrievalResult:
        """Run the full pipeline for one sub-question."""
        cfg = self.cfg
        if not self._doc_cache:
            self.refresh_indexes()
        if not self._doc_cache:
            return RetrievalResult(reason="knowledge base is empty")

        query_vector = self.llm.embed([question])[0]

        dense_hits = self.store.query(
            query_vector, int(cfg.get("retrieval.dense_k", 12)), where
        )
        dense_scores = {hit["id"]: hit["score"] for hit in dense_hits}
        ranked_lists = [[hit["id"] for hit in dense_hits]]

        if self._bm25 is not None:
            sparse = self._bm25.search(question, int(cfg.get("retrieval.sparse_k", 12)))
            ranked_lists.append([doc_id for doc_id, _ in sparse])

        fused = reciprocal_rank_fusion(
            ranked_lists, int(cfg.get("retrieval.rrf_constant", 60))
        )
        ordered = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)

        final_k = int(cfg.get("retrieval.final_k", 5))
        if cfg.get("retrieval.use_mmr", True) and len(ordered) > final_k:
            pool = ordered[: final_k * 3]
            vectors = self.llm.embed([self._doc_cache[i]["text"] for i in pool])
            ordered = maximal_marginal_relevance(
                query_vector,
                list(zip(pool, vectors)),
                final_k,
                float(cfg.get("retrieval.mmr_lambda", 0.6)),
            )

        evidence = [
            Evidence(
                text=self._doc_cache[doc_id]["text"],
                source=self._doc_cache[doc_id]["metadata"].get("source", ""),
                title=self._doc_cache[doc_id]["metadata"].get("title", ""),
                score=dense_scores.get(doc_id, 0.0),
                origin="vector",
                doc_id=doc_id,
            )
            for doc_id in ordered[:final_k]
            if doc_id in self._doc_cache
        ]

        if cfg.get("retrieval.use_llm_rerank", True) and evidence:
            evidence = self._llm_rerank(question, evidence)

        return self._assess(evidence)

    def _llm_rerank(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        """Score each chunk for usefulness and drop the ones below threshold.

        One batched call rather than one call per chunk. If the model returns
        the wrong number of scores the filter is skipped rather than dropping
        evidence arbitrarily.
        """
        excerpts = "\n\n".join(
            f"[{index}] {item.text[:600]}" for index, item in enumerate(evidence)
        )
        payload = self.llm.json(
            "RERANKER",
            _RERANK_SYSTEM,
            f"Question: {question}\n\nExcerpts:\n{excerpts}",
            default={},
        )
        scores = (payload or {}).get("scores")
        if not isinstance(scores, list) or len(scores) != len(evidence):
            logger.debug("reranker returned %r, skipping filter", scores)
            return evidence

        threshold = float(self.cfg.get("retrieval.llm_rerank_threshold", 0.45))
        kept = []
        for item, score in zip(evidence, scores):
            try:
                value = float(score)
            except (TypeError, ValueError):
                value = 0.0
            item.score = max(item.score, value)
            if value >= threshold:
                kept.append(item)
        return kept or evidence[:1]

    def _assess(self, evidence: list[Evidence]) -> RetrievalResult:
        """Decide whether the retrieved evidence is enough to answer on."""
        top = max((item.score for item in evidence), default=0.0)
        min_score = float(self.cfg.get("retrieval.min_evidence_score", 0.30))
        min_count = int(self.cfg.get("retrieval.min_evidence_count", 2))

        if not evidence:
            return RetrievalResult([], False, 0.0, "no chunks retrieved")
        if top < min_score:
            return RetrievalResult(evidence, False, top, f"top score {top:.2f} below {min_score}")
        if len(evidence) < min_count:
            return RetrievalResult(evidence, False, top, "too few surviving chunks")
        return RetrievalResult(evidence, True, top, "sufficient")
