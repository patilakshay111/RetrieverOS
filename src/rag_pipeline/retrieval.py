# retrieval.py: combines FAISS dense search, BM25 keyword search, RRF, and cross-encoder ranking

from collections.abc import Sequence
import re
import threading

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingProvider
from rag_pipeline.models import RetrievedChunk
from rag_pipeline.storage import IndexSnapshot, KnowledgeBase


# Tokenize technical identifiers while keeping punctuation-bearing names searchable.
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", text.casefold())


# Calculate reciprocal-rank fusion scores without conflating raw dense and BM25 scales.
def reciprocal_rank_fusion(
    dense_ranks: Sequence[str],
    sparse_ranks: Sequence[str],
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for weight, ranked_ids in ((dense_weight, dense_ranks), (sparse_weight, sparse_ranks)):
        if weight == 0:
            continue
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (rrf_k + rank)
    return scores


# Lazily load the configured local cross-encoder and rank fused query-candidate pairs.
class CrossEncoderReranker:
    # Store model configuration and guard concurrent first-load operations.
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()

    # Rerank candidate chunks against the original question using one cross-encoder batch.
    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder

                        self._model = CrossEncoder(self.model_name)
                    except Exception as error:
                        raise RuntimeError(
                            f"Unable to load reranker model {self.model_name!r}."
                        ) from error
        pairs = [(query, item.chunk.text) for item in candidates]
        try:
            scores = np.asarray(self._model.predict(pairs), dtype=np.float32).reshape(-1)
        except Exception as error:
            raise RuntimeError("Cross-encoder reranking failed.") from error
        if scores.shape[0] != len(candidates):
            raise RuntimeError("Cross-encoder returned an unexpected number of scores.")
        ranked = [
            item.model_copy(update={"rerank_score": float(score)})
            for item, score in zip(candidates, scores, strict=True)
        ]
        return sorted(ranked, key=lambda item: item.rerank_score or 0.0, reverse=True)


# Search one immutable corpus snapshot and keep its BM25 corpus cached by generation.
class HybridRetriever:
    # Configure the vector, keyword, fusion, and reranking stages.
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embeddings: EmbeddingProvider,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.knowledge_base = knowledge_base
        self.embeddings = embeddings
        self.settings = settings
        self.reranker = reranker
        self._bm25_generation: str | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_lock = threading.Lock()

    # Return the cached or freshly built BM25 model for the selected snapshot.
    def _bm25_for(self, snapshot: IndexSnapshot) -> BM25Okapi | None:
        if not snapshot.chunks:
            return None
        with self._bm25_lock:
            if self._bm25_generation != snapshot.generation:
                self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in snapshot.chunks])
                self._bm25_generation = snapshot.generation
            return self._bm25

    # Retrieve, fuse, optionally rerank, and calibrate evidence for one user question.
    def search(self, query: str, mode: str = "hybrid") -> tuple[list[RetrievedChunk], float]:
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        snapshot = self.knowledge_base.snapshot()
        if not snapshot.chunks:
            return [], 0.0
        query_vector = self.embeddings.embed_query(query).astype(np.float32).reshape(1, -1)
        if snapshot.vectors.shape[1] != query_vector.shape[1]:
            raise RuntimeError("Query and index embedding dimensions do not match.")
        faiss.normalize_L2(query_vector)
        dense_count = min(self.settings.dense_top_k, len(snapshot.chunks))
        dense_scores, dense_positions = snapshot.dense_index.search(query_vector, dense_count)
        dense_ranked = [
            (snapshot.chunks[int(position)].chunk_id, float(score))
            for score, position in zip(dense_scores[0], dense_positions[0], strict=True)
            if position >= 0
        ]
        query_tokens = set(tokenize(query))
        sparse_model = self._bm25_for(snapshot) if mode != "dense" else None
        sparse_values = (
            sparse_model.get_scores(list(query_tokens))
            if sparse_model and query_tokens
            else np.zeros(len(snapshot.chunks), dtype=np.float32)
        )
        sparse_positions = np.argsort(sparse_values)[::-1]
        sparse_ranked: list[tuple[str, float]] = []
        for position in sparse_positions:
            chunk = snapshot.chunks[int(position)]
            score = float(sparse_values[position])
            has_exact_term = bool(query_tokens.intersection(tokenize(chunk.text)))
            if score > 0 or has_exact_term:
                sparse_ranked.append((chunk.chunk_id, score))
            if len(sparse_ranked) >= self.settings.sparse_top_k:
                break
        dense_by_id = dict(dense_ranked)
        sparse_by_id = dict(sparse_ranked)
        if mode == "dense":
            fusion_scores = {key: 1.0 / rank for rank, (key, _) in enumerate(dense_ranked, 1)}
        elif mode == "sparse":
            fusion_scores = {key: 1.0 / rank for rank, (key, _) in enumerate(sparse_ranked, 1)}
        else:
            fusion_scores = reciprocal_rank_fusion(
                [key for key, _ in dense_ranked],
                [key for key, _ in sparse_ranked],
                self.settings.dense_weight,
                self.settings.sparse_weight,
                self.settings.rrf_k,
            )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in snapshot.chunks}
        ranked_ids = sorted(fusion_scores, key=fusion_scores.__getitem__, reverse=True)
        candidates = [
            RetrievedChunk(
                chunk=chunks_by_id[chunk_id],
                dense_score=dense_by_id.get(chunk_id),
                bm25_score=sparse_by_id.get(chunk_id),
                rrf_score=fusion_scores[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ranked_ids[: self.settings.rerank_top_k], start=1)
        ]
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)
        candidates = [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(candidates[: self.settings.final_top_k], start=1)
        ]
        best_dense = max((score for _, score in dense_ranked), default=-1.0)
        dense_confidence = max(0.0, min(1.0, (best_dense - 0.35) / 0.45))
        best_sparse = max((score for _, score in sparse_ranked), default=0.0)
        lexical_coverage = max(
            (
                len(query_tokens.intersection(tokenize(item.chunk.text))) / len(query_tokens)
                for item in candidates
            ),
            default=0.0,
        )
        positive_bm25_confidence = (
            best_sparse / (best_sparse + 8.0) if best_sparse > 0 else 0.0
        )
        sparse_confidence = max(lexical_coverage, positive_bm25_confidence)
        if mode == "dense":
            confidence = dense_confidence
        elif mode == "sparse":
            confidence = sparse_confidence
        else:
            total_weight = self.settings.dense_weight + self.settings.sparse_weight
            confidence = (
                dense_confidence * self.settings.dense_weight
                + sparse_confidence * self.settings.sparse_weight
            ) / total_weight
        return candidates, confidence
