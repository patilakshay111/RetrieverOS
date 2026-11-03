"""Bounded Redis cache for authorization-scoped stateless answers."""

import hashlib
import json
from typing import Protocol

from pydantic import ValidationError

from rag_pipeline.models import AnswerResponse, WorkspaceAccessContext


class RedisCacheClient(Protocol):
    def get(self, name: str) -> bytes | str | None: ...

    def set(self, name: str, value: str, ex: int) -> object: ...

    def delete(self, *names: str) -> object: ...


def cache_key(
    question: str,
    retrieval_mode: str,
    scope: WorkspaceAccessContext,
    index_version: str,
) -> str:
    """Hash request and authorization inputs so Redis keys contain no user text or IDs."""
    authorization = {
        "user_id": scope.user_id,
        "workspace_id": scope.workspace_id,
        "roles": sorted(scope.roles),
        "platform_admin": scope.platform_admin,
    }
    authorization_version = hashlib.sha256(
        json.dumps(authorization, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload = json.dumps(
        {
            "question": question,
            "retrieval_mode": retrieval_mode,
            "authorization": authorization,
            "authorization_version": authorization_version,
            "index_version": index_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"rag:answer:v1:{hashlib.sha256(payload).hexdigest()}"


class RedisAnswerCache:
    def __init__(
        self,
        client: RedisCacheClient,
        ttl_seconds: int = 60,
        max_value_bytes: int = 262_144,
    ):
        if ttl_seconds < 1:
            raise ValueError("Answer cache TTL must be positive.")
        if max_value_bytes < 1:
            raise ValueError("Answer cache value limit must be positive.")
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._max_value_bytes = max_value_bytes

    def get(self, key: str) -> AnswerResponse | None:
        value = self._client.get(key)
        if value is None:
            return None
        try:
            return AnswerResponse.model_validate_json(value)
        except ValidationError:
            self._client.delete(key)
            return None

    def set(self, key: str, answer: AnswerResponse) -> bool:
        value = answer.model_dump_json()
        if len(value.encode("utf-8")) > self._max_value_bytes:
            return False
        return bool(self._client.set(key, value, ex=self._ttl_seconds))
