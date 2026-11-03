"""Atomic Redis-backed request limits keyed by authenticated scope."""

import hashlib
import json
from typing import Protocol

from rag_pipeline.models import WorkspaceAccessContext

_INCREMENT_WITH_EXPIRY = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


class RedisRateLimitClient(Protocol):
    def eval(
        self, script: str, numkeys: int, *keys_and_args: str | int
    ) -> int: ...


def rate_limit_key(scope: WorkspaceAccessContext, operation: str) -> str:
    payload = json.dumps(
        {
            "user_id": scope.user_id,
            "workspace_id": scope.workspace_id,
            "operation": operation,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"rag:rate:v1:{hashlib.sha256(payload).hexdigest()}"


class RedisRateLimiter:
    def __init__(
        self,
        client: RedisRateLimitClient,
        requests: int = 120,
        window_seconds: int = 60,
    ):
        if requests < 1 or window_seconds < 1:
            raise ValueError("Rate-limit count and window must be positive.")
        self._client = client
        self._requests = requests
        self._window_seconds = window_seconds

    def allow(self, scope: WorkspaceAccessContext, operation: str) -> bool:
        count = self._client.eval(
            _INCREMENT_WITH_EXPIRY,
            1,
            rate_limit_key(scope, operation),
            self._window_seconds,
        )
        return count <= self._requests
