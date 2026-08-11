"""RRF+MMR retrieval fusion for the ``agents`` package.

Fuses dense search (``db_engine.fetch``, embedding-based, good at
paraphrase) with a sparse BM25 index built over the full corpus (good at
literal tokens like exact flag names or error strings) via reciprocal
rank fusion, then reranks the fused candidates with maximal marginal
relevance so near-duplicate chunks from the same page don't crowd out
coverage of other parts of a compound question. Design mined from
``new/retrieval.py`` (reference only, not reused as code) — see CLAUDE.md's
revamp plan for the full rationale.

Dense and sparse results are fused on a shared chunk id: ``db_engine``
stamps ``metadata["chunk_id"]`` on every chunk (equal to ``Chunk.id`` from
``db_engine.get_all_chunks``), so ``fetch()``'s ``Document.metadata`` and
``get_all_chunks()``'s ``Chunk`` objects share one id space directly —
no reconstruction needed on this side.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import db_engine
from agents.llm import LLM, cosine

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Evidence:
    """One piece of retrieved evidence, dense- or web-sourced."""

    text: str
    source: str
    title: str = ""
    score: float = 0.0
    origin: str = "vector"  # "vector" | "web"
    doc_id: str = ""


@dataclass
class RetrievalResult:
    """The outcome of one ``Retriever.retrieve()`` call."""

    evidence: list[Evidence] = field(default_factory=list)
    is_sufficient: bool = False
    top_score: float = 0.0
    reason: str = ""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k_constant: int) -> dict[str, float]:
    """Fuses several ranked id lists into one score per id via RRF.

    Rank-based fusion (rather than combining raw scores) sidesteps the
    problem that cosine similarity and BM25 scores live on incomparable,
    uncalibrated scales — a document ranked highly by both retrievers
    accumulates the most credit purely from agreement on rank.

    Args:
        ranked_lists: Each inner list is ids in best-first rank order.
        k_constant: RRF's smoothing constant; larger values flatten the
            influence of exact rank position.

    Returns:
        ``{id: fused_score}``, unsorted.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (k_constant + rank)
    return dict(scores)


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: list[tuple[str, list[float]]],
    k: int,
    lambda_multiplier: float,
) -> list[str]:
    """Greedily reranks candidates to trade a little relevance for diversity.

    At each step picks ``argmax[ λ·sim(query, doc) − (1−λ)·max sim(doc, selected) ]``
    so near-duplicate chunks don't crowd out coverage of other parts of a
    compound question. ``lambda_multiplier=1.0`` degenerates to plain
    relevance ranking; ``lambda_multiplier=0.0`` maximizes diversity only.

    Args:
        query_vector: The query's embedding.
        candidates: ``(id, embedding)`` pairs to select from, in any order.
        k: Number of ids to select.
        lambda_multiplier: Relevance/diversity tradeoff in ``[0, 1]``.

    Returns:
        Up to ``k`` selected ids, in selection order (most relevant/diverse
        first).
    """
    remaining = list(candidates)
    selected: list[tuple[str, list[float]]] = []
    selected_ids: list[str] = []

    while remaining and len(selected) < k:
        best_index = None
        best_score = float("-inf")
        for i, (item_id, embedding) in enumerate(remaining):
            relevance = cosine(query_vector, embedding)
            if selected:
                diversity_penalty = max(cosine(embedding, s_emb) for _, s_emb in selected)
            else:
                diversity_penalty = 0.0
            mmr_score = lambda_multiplier * relevance - (1 - lambda_multiplier) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = i
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        selected_ids.append(chosen[0])

    return selected_ids


class BM25Index:
    """Okapi BM25 sparse index over a corpus of ``db_engine.Chunk`` objects."""

    def __init__(self, chunks: list["db_engine.Chunk"], k1: float, b: float):
        """Builds the index.

        Args:
            chunks: The full corpus to index.
            k1: BM25 term-frequency saturation constant.
            b: BM25 length-normalization constant.
        """
        self._k1 = k1
        self._b = b
        self._chunk_ids = [c.id for c in chunks]
        self._doc_tokens = [_tokenize(c.text) for c in chunks]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_length = (sum(self._doc_lengths) / len(self._doc_lengths)) if chunks else 0.0
        self._term_freqs = [Counter(tokens) for tokens in self._doc_tokens]

        doc_freq: Counter = Counter()
        for tokens in self._doc_tokens:
            doc_freq.update(set(tokens))
        n_docs = len(chunks)
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Scores every indexed chunk against a query and returns the top-k.

        Args:
            query: The query text.
            k: Max number of results.

        Returns:
            ``(chunk_id, score)`` pairs, best first. Score is unnormalized
            BM25 (only relative order matters for fusion).
        """
        query_terms = _tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, chunk_id in enumerate(self._chunk_ids):
            doc_length = self._doc_lengths[i]
            term_freqs = self._term_freqs[i]
            score = 0.0
            for term in query_terms:
                if term not in term_freqs:
                    continue
                idf = self._idf.get(term, 0.0)
                tf = term_freqs[term]
                denom = tf + self._k1 * (1 - self._b + self._b * doc_length / max(self._avg_doc_length, 1e-9))
                score += idf * (tf * (self._k1 + 1)) / max(denom, 1e-9)
            if score > 0:
                scores.append((chunk_id, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:k]


class Retriever:
    """RRF+MMR fusion retrieval on top of ``db_engine``."""

    def __init__(self, cfg: dict, llm: LLM):
        self._cfg = cfg
        self._llm = llm
        self._bm25: Optional[BM25Index] = None
        self._chunks_by_id: dict[str, "db_engine.Chunk"] = {}

    def refresh_indexes(self) -> None:
        """Pulls the full corpus and (re)builds the in-memory BM25 index.

        Expensive relative to a single ``retrieve()`` call — intended to
        run once per process (or on an explicit refresh trigger), not per
        query.
        """
        retrieval_cfg = self._cfg["retrieval"]
        chunks = db_engine.get_all_chunks(self._cfg)
        self._chunks_by_id = {c.id: c for c in chunks}
        self._bm25 = BM25Index(chunks, k1=retrieval_cfg["bm25_k1"], b=retrieval_cfg["bm25_b"])
        logger.info("refresh_indexes: indexed %d chunks", len(chunks))

    def retrieve(self, question: str) -> RetrievalResult:
        """Runs dense+sparse fusion retrieval for one sub-question.

        Args:
            question: The (already atomic/standalone) question to retrieve
                evidence for.

        Returns:
            A :class:`RetrievalResult` with fused, MMR-reranked evidence
            and a sufficiency verdict.
        """
        if self._bm25 is None:
            self.refresh_indexes()
        retrieval_cfg = self._cfg["retrieval"]

        dense_results = db_engine.fetch(
            question, self._cfg, top_k=retrieval_cfg["dense_k"], score_threshold=0.0
        )
        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        for doc, score in dense_results:
            chunk_id = doc.metadata["chunk_id"]
            dense_ids.append(chunk_id)
            dense_scores[chunk_id] = score
            self._chunks_by_id.setdefault(
                chunk_id,
                db_engine.Chunk(id=chunk_id, text=doc.page_content, metadata=doc.metadata, embedding=[]),
            )

        sparse_results = self._bm25.search(question, k=retrieval_cfg["sparse_k"])
        sparse_ids = [chunk_id for chunk_id, _score in sparse_results]

        fused_scores = reciprocal_rank_fusion([dense_ids, sparse_ids], retrieval_cfg["rrf_k_constant"])
        if not fused_scores:
            logger.warning("retrieve: no candidates for question=%r", question)
            return RetrievalResult(evidence=[], is_sufficient=False, top_score=0.0, reason="no_candidates")

        fused_ranked = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)
        pool_size = max(retrieval_cfg["final_k"] * 3, retrieval_cfg["final_k"])
        pool_ids = fused_ranked[:pool_size]

        if retrieval_cfg["use_mmr"]:
            query_vector = self._llm.embed([question])[0]
            mmr_candidates = [
                (cid, self._chunks_by_id[cid].embedding)
                for cid in pool_ids
                if cid in self._chunks_by_id and self._chunks_by_id[cid].embedding
            ]
            selected_ids = maximal_marginal_relevance(
                query_vector, mmr_candidates, k=retrieval_cfg["final_k"], lambda_multiplier=retrieval_cfg["mmr_lambda"]
            )
            if not selected_ids:
                selected_ids = pool_ids[: retrieval_cfg["final_k"]]
        else:
            selected_ids = pool_ids[: retrieval_cfg["final_k"]]

        evidence = []
        for cid in selected_ids:
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                continue
            evidence.append(
                Evidence(
                    text=chunk.text,
                    source=chunk.metadata.get("source", ""),
                    title=chunk.metadata.get("title", ""),
                    score=dense_scores.get(cid, 0.0),
                    origin="vector",
                    doc_id=cid,
                )
            )

        result = self._assess(evidence)
        logger.info(
            "retrieve: question=%r dense=%d sparse=%d fused=%d selected=%d sufficient=%s top_score=%.3f",
            question,
            len(dense_ids),
            len(sparse_ids),
            len(fused_scores),
            len(evidence),
            result.is_sufficient,
            result.top_score,
        )
        return result

    def _assess(self, evidence: list[Evidence]) -> RetrievalResult:
        """Judges whether retrieved evidence is strong enough to answer from.

        Applied *after* fusion/MMR, on the final selected evidence — not a
        raw top-1 dense score — so the threshold reflects what the worker
        will actually see.
        """
        retrieval_cfg = self._cfg["retrieval"]
        if not evidence:
            return RetrievalResult(evidence=[], is_sufficient=False, top_score=0.0, reason="no_evidence")

        top_score = max(e.score for e in evidence)
        strong_count = sum(1 for e in evidence if e.score >= retrieval_cfg["min_evidence_score"])

        if top_score < retrieval_cfg["min_evidence_score"]:
            return RetrievalResult(evidence=evidence, is_sufficient=False, top_score=top_score, reason="top_score_below_threshold")
        if strong_count < retrieval_cfg["min_evidence_count"]:
            return RetrievalResult(evidence=evidence, is_sufficient=False, top_score=top_score, reason="too_few_strong_chunks")
        return RetrievalResult(evidence=evidence, is_sufficient=True, top_score=top_score, reason="")
