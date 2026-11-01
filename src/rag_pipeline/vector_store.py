"""Qdrant-backed dense vector storage with server-side document ACL filters."""

from collections.abc import Sequence
import math
from threading import Lock
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from rag_pipeline.models import Chunk, RetrievedChunk, WorkspaceAccessContext


class VectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str = "hybrid_rag_chunks",
        vector_size: int = 1536,
        client: QdrantClient | None = None,
    ):
        if not url.strip():
            raise ValueError("QDRANT_URL must not be empty.")
        if not collection_name.strip():
            raise ValueError("Qdrant collection name must not be empty.")
        if vector_size < 1:
            raise ValueError("QDRANT_VECTOR_SIZE must be positive.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = client or QdrantClient(url=url, timeout=5)
        self._collection_lock = Lock()
        self._initialized = False

    def ensure_collection(self) -> None:
        if self._initialized:
            return
        with self._collection_lock:
            if self._initialized:
                return
            try:
                if self._client.collection_exists(self.collection_name):
                    collection = self._client.get_collection(self.collection_name)
                    vectors = collection.config.params.vectors
                    if isinstance(vectors, dict):
                        raise RuntimeError(
                            "Qdrant collection must use one unnamed dense vector."
                        )
                    if vectors.size != self.vector_size:
                        raise RuntimeError(
                            "Qdrant collection vector size does not match configuration."
                        )
                    payload_schema = collection.payload_schema
                else:
                    self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.vector_size,
                            distance=models.Distance.COSINE,
                        ),
                        on_disk_payload=True,
                    )
                    payload_schema = {}
                for field_name in ("workspace_id", "owner_id", "required_roles"):
                    existing = payload_schema.get(field_name)
                    if existing is None:
                        self._client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field_name,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )
                    elif existing.data_type != models.PayloadSchemaType.KEYWORD:
                        raise RuntimeError(
                            f"Qdrant ACL index {field_name!r} must use keyword values."
                        )
                self._initialized = True
            except ApiException as error:
                raise RuntimeError("Unable to initialize the Qdrant collection.") from error

    @staticmethod
    def _point_id(chunk_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"hybrid-rag:{chunk_id}")

    @staticmethod
    def _query_filter(scope: WorkspaceAccessContext) -> models.Filter | None:
        if scope.platform_admin:
            return None
        workspace_condition = models.FieldCondition(
            key="workspace_id",
            match=models.MatchValue(value=scope.workspace_id),
        )
        if "admin" in scope.roles:
            return models.Filter(must=[workspace_condition])
        allowed_conditions: list[models.FieldCondition] = [
            models.FieldCondition(
                key="owner_id",
                match=models.MatchValue(value=scope.user_id),
            )
        ]
        if scope.roles:
            allowed_conditions.append(
                models.FieldCondition(
                    key="required_roles",
                    match=models.MatchAny(any=sorted(scope.roles)),
                )
            )
        return models.Filter(
            must=[workspace_condition],
            min_should=models.MinShould(
                conditions=allowed_conditions,
                min_count=1,
            ),
        )

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(chunks), self.vector_size):
            raise ValueError(
                "Each Qdrant chunk must have one vector matching QDRANT_VECTOR_SIZE."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("Qdrant vectors must contain only finite values.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Qdrant chunk IDs must be unique within an upsert.")
        for chunk in chunks:
            if not chunk.workspace_id.strip() or not chunk.owner_id.strip():
                raise ValueError("Qdrant chunks require workspace and owner ACL metadata.")
        if not chunks:
            return
        self.ensure_collection()
        points = [
            models.PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=vector.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, matrix, strict=True)
        ]
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except ApiException as error:
            raise RuntimeError("Unable to write vectors to Qdrant.") from error

    def search(
        self,
        vector: np.ndarray,
        scope: WorkspaceAccessContext,
        limit: int,
    ) -> list[RetrievedChunk]:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.vector_size,) or not np.isfinite(query).all():
            raise ValueError("Query vector does not match QDRANT_VECTOR_SIZE.")
        if limit < 1:
            raise ValueError("Qdrant search limit must be positive.")
        self.ensure_collection()
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query.tolist(),
                query_filter=self._query_filter(scope),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except ApiException as error:
            raise RuntimeError("Qdrant vector search failed.") from error
        results: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("workspace_id"), str)
                or not payload["workspace_id"].strip()
                or not isinstance(payload.get("owner_id"), str)
                or not payload["owner_id"].strip()
                or not isinstance(payload.get("required_roles"), list)
                or any(not isinstance(role, str) or not role for role in payload["required_roles"])
            ):
                raise RuntimeError("Qdrant returned malformed document ACL metadata.")
            if not scope.permits(
                payload["workspace_id"],
                payload["owner_id"],
                payload["required_roles"],
            ):
                raise RuntimeError(
                    "Qdrant returned a document outside the authorized workspace scope."
                )
            try:
                chunk = Chunk.model_validate(payload)
                score = float(point.score)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Qdrant returned an invalid document payload.") from error
            if not math.isfinite(score):
                raise RuntimeError("Qdrant returned a non-finite similarity score.")
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=score,
                    rank=rank,
                )
            )
        return results

    def delete_document(self, workspace_id: str, source_id: str) -> None:
        if not workspace_id.strip() or not source_id.strip():
            raise ValueError("Workspace and source IDs must not be empty.")
        self.ensure_collection()
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="workspace_id",
                                match=models.MatchValue(value=workspace_id),
                            ),
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source_id),
                            ),
                        ]
                    )
                ),
                wait=True,
            )
        except ApiException as error:
            raise RuntimeError("Unable to delete document vectors from Qdrant.") from error

    def check(self) -> None:
        try:
            if not self._client.collection_exists(self.collection_name):
                raise RuntimeError("Qdrant collection is unavailable.")
            collection = self._client.get_collection(self.collection_name)
        except ApiException as error:
            raise RuntimeError("Qdrant collection is unavailable.") from error
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict) or vectors.size != self.vector_size:
            raise RuntimeError("Qdrant collection vector configuration is invalid.")
        for field_name in ("workspace_id", "owner_id", "required_roles"):
            index = collection.payload_schema.get(field_name)
            if index is None or index.data_type != models.PayloadSchemaType.KEYWORD:
                raise RuntimeError(
                    f"Qdrant ACL index {field_name!r} is unavailable or invalid."
                )

    def close(self) -> None:
        self._client.close()
