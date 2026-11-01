# retrieval.py: combines FAISS dense search, BM25 keyword search, RRF, and cross-encoder ranking

from collections import OrderedDict
from collections.abc import Sequence
import re
import threading

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from rag_pipeline.config import Settings
from rag_pipeline.embeddings import EmbeddingProvider
from rag_pipeline.models import AccessContext, RetrievedChunk
from rag_pipeline.lexical_store import LexicalStore
from rag_pipeline.storage import IndexSnapshot, KnowledgeBase
from rag_pipeline.vector_store import VectorStore


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
        vector_store: VectorStore | None = None,
        lexical_store: LexicalStore | None = None,
    ):
        if (vector_store is None) != (lexical_store is None):
            raise ValueError("Qdrant and lexical stores must be configured together.")
        self.knowledge_base = knowledge_base
        self.embeddings = embeddings
        self.settings = settings
        self.reranker = reranker
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self._bm25_generation: str | None = None
        self._bm25_cache: OrderedDict[
            tuple[str, str, tuple[str, ...], bool],
            tuple[BM25Okapi | None, list[int]],
        ] = OrderedDict()
        self._bm25_lock = threading.Lock()

    # Build sparse ranking only from chunks visible to the requesting access scope.
    def _bm25_for(
        self, snapshot: IndexSnapshot, access_context: AccessContext | None
    ) -> tuple[BM25Okapi | None, list[int]]:
        if not snapshot.chunks:
            return None, []
        if access_context is None:
            scope = ("*", "*", (), True)
        else:
            scope = (
                access_context.workspace_id,
                access_context.user_id,
                tuple(sorted(access_context.roles)),
                access_context.platform_admin,
            )
        with self._bm25_lock:
            if self._bm25_generation != snapshot.generation:
                self._bm25_cache.clear()
                self._bm25_generation = snapshot.generation
            cached = self._bm25_cache.get(scope)
            if cached is not None:
                self._bm25_cache.move_to_end(scope)
                return cached
            positions = [
                index
                for index, chunk in enumerate(snapshot.chunks)
                if access_context is None
                or access_context.permits(
                    chunk.workspace_id, chunk.owner_id, chunk.required_roles
                )
            ]
            model = (
                BM25Okapi([tokenize(snapshot.chunks[index].text) for index in positions])
                if positions
                else None
            )
            result = (model, positions)
            self._bm25_cache[scope] = result
            if len(self._bm25_cache) > 128:
                self._bm25_cache.popitem(last=False)
            return result

    # Retrieve, fuse, optionally rerank, and calibrate evidence for one user question.
    def search(
        self,
        query: str,
        mode: str = "hybrid",
        access_context: AccessContext | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        if mode not in {"hybrid", "dense", "sparse"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        durable = self.vector_store is not None and self.lexical_store is not None
        if durable and access_context is None:
            raise ValueError("Durable retrieval requires an authenticated access scope.")
        snapshot = None if durable else self.knowledge_base.snapshot()
        if snapshot is not None and not snapshot.chunks:
            return [], 0.0
        query_vector = self.embeddings.embed_query(query).astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        if durable:
            assert self.vector_store is not None
            assert self.lexical_store is not None
            dense_results = (
                self.vector_store.search(
                    query_vector.reshape(-1),
                    access_context,
                    self.settings.dense_top_k * 4,
                )
                if mode != "sparse"
                else []
            )
            current_chunks = self.lexical_store.get_chunks_by_ids(
                [item.chunk.chunk_id for item in dense_results],
                access_context,
            )
            chunks_by_id = {chunk.chunk_id: chunk for chunk in current_chunks}
            dense_ranked = [
                (item.chunk.chunk_id, float(item.dense_score))
                for item in dense_results
                if item.chunk.chunk_id in chunks_by_id
                and item.dense_score is not None
            ]
            sparse_results = (
                self.lexical_store.search(
                    query,
                    access_context,
                    self.settings.sparse_top_k,
                )
                if mode != "dense"
                else []
            )
            sparse_ranked = [
                (item.chunk.chunk_id, float(item.bm25_score or 0.0))
                for item in sparse_results
            ]
            chunks_by_id.update(
                {item.chunk.chunk_id: item.chunk for item in sparse_results}
            )
        else:
            assert snapshot is not None
            if snapshot.vectors.shape[1] != query_vector.shape[1]:
                raise RuntimeError("Query and index embedding dimensions do not match.")
            dense_count = min(self.settings.dense_top_k * 4, len(snapshot.chunks))
            dense_scores, dense_positions = snapshot.dense_index.search(query_vector, dense_count)
            dense_ranked = [
                (snapshot.chunks[int(position)].chunk_id, float(score))
                for score, position in zip(dense_scores[0], dense_positions[0], strict=True)
                if position >= 0
                and (
                    access_context is None
                    or access_context.permits(
                        snapshot.chunks[int(position)].workspace_id,
                        snapshot.chunks[int(position)].owner_id,
                        snapshot.chunks[int(position)].required_roles,
                    )
                )
            ]
            sparse_model, sparse_positions_map = (
                self._bm25_for(snapshot, access_context) if mode != "dense" else (None, [])
            )
            sparse_values = (
                sparse_model.get_scores(list(set(tokenize(query))))
                if sparse_model and tokenize(query)
                else np.zeros(len(sparse_positions_map), dtype=np.float32)
            )
            sparse_positions = np.argsort(sparse_values)[::-1]
            sparse_ranked = []
            for scoped_position in sparse_positions:
                position = sparse_positions_map[int(scoped_position)]
                chunk = snapshot.chunks[int(position)]
                score = float(sparse_values[scoped_position])
                has_exact_term = bool(set(tokenize(query)).intersection(tokenize(chunk.text)))
                if score > 0 or has_exact_term:
                    sparse_ranked.append((chunk.chunk_id, score))
                if len(sparse_ranked) >= self.settings.sparse_top_k:
                    break
            chunks_by_id = {
                chunk.chunk_id: chunk
                for chunk in snapshot.chunks
                if access_context is None
                or access_context.permits(
                    chunk.workspace_id, chunk.owner_id, chunk.required_roles
                )
            }

        query_tokens = set(tokenize(query))
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
