# api.py: exposes authenticated FastAPI endpoints for ingestion, retrieval, and grounded answers

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID, uuid4

import jwt
from psycopg import Error as PsycopgError
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_pipeline.auth import OIDCVerifier, build_oidc_verifier
from rag_pipeline.audit import AdminAuditStore
from rag_pipeline.cache import RedisAnswerCache, cache_key
from rag_pipeline.checkpointing import create_postgres_checkpointer
from rag_pipeline.conversations import (
    ConversationRecord,
    ConversationStore,
    conversation_thread_id,
)
from rag_pipeline.credentials import UserCredentialStore
from rag_pipeline.config import Settings
from rag_pipeline.embeddings import OpenAIEmbeddingProvider
from rag_pipeline.identity_store import WorkspaceStore
from rag_pipeline.models import (
    AnswerResponse,
    DocumentInfo,
    IngestionResult,
    ReindexDryRunReport,
    VerifiedIdentity,
    WorkspaceAccessContext,
)
from rag_pipeline.reindex import verify_archived_sources
from rag_pipeline.service import RAGService
from rag_pipeline.rate_limits import RedisRateLimiter
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.workspaces import create_workspace_router


# Expose one optional API-key header so development can stay frictionless.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


# Signal that a request exceeded the bounded ASGI body size.
class RequestBodyTooLarge(Exception):
    pass


class RequestIDMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_request_id)


# Reject oversized requests before multipart parsing can spool an unbounded upload.
class RequestBodyLimitMiddleware:
    # Store separate upload and ordinary-request limits.
    def __init__(self, app: ASGIApp, max_upload_bytes: int):
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    # Count streamed body bytes and return 413 as soon as the route limit is exceeded.
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_upload = scope.get("path") == "/v1/ingest" and scope.get("method") == "POST"
        limit = self.max_upload_bytes + 1024 * 1024 if is_upload else 1024 * 1024
        content_length = next(
            (
                value.decode("ascii", errors="ignore")
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                response = JSONResponse(
                    {"detail": "Content-Length must be a non-negative integer."},
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_length < 0:
                response = JSONResponse(
                    {"detail": "Content-Length must be a non-negative integer."},
                    status_code=400,
                )
                await response(scope, receive, send)
                return
            if declared_length > limit:
                response = JSONResponse(
                    {"detail": "The request body exceeds the configured size limit."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
        received = 0
        response_started = False

        # Count each ASGI request message without buffering the complete upload.
        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestBodyTooLarge
            return message

        # Track whether the downstream app has already started its response.
        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            response = JSONResponse(
                {"detail": "The request body exceeds the configured size limit."},
                status_code=413,
            )
            await response(scope, receive, send)


# Validate user-facing question and stable conversation identifier fields.
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    thread_id: str = Field(default="chat_session_1", min_length=1, max_length=128)
    conversation_id: UUID | None = None

    # Reject whitespace-only prompts and unsafe thread identifier characters.
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question

    # Keep MemorySaver checkpoint identifiers simple and bounded.
    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        if not value.strip() or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
            raise ValueError("thread_id may contain only letters, numbers, '.', '_' and '-'.")
        return value


class OpenAIKeyRequest(BaseModel):
    api_key: str = Field(min_length=20, max_length=512)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        key = value.strip()
        if not key:
            raise ValueError("The OpenAI API key cannot be blank.")
        return key


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=160)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("Conversation title cannot be blank.")
        return title


# Build the API and enforce OIDC or local API-key authentication outside health.
def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    configured.validate_production()
    postgres_database_configured = bool(
        configured.database_url
        and make_url(configured.database_url).drivername.startswith("postgresql")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        vector_store = (
            VectorStore(
                configured.qdrant_url,
                vector_size=configured.qdrant_vector_size,
            )
            if configured.qdrant_url
            else None
        )
        app.state.vector_store = vector_store
        if vector_store is not None:
            vector_store.ensure_collection()
        app.state.checkpointer = None
        redis_client = (
            Redis.from_url(
                configured.redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                health_check_interval=30,
            )
            if configured.redis_url
            else None
        )
        app.state.redis_client = redis_client
        app.state.answer_cache = (
            RedisAnswerCache(
                redis_client,
                ttl_seconds=configured.redis_cache_ttl_seconds,
                max_value_bytes=configured.redis_cache_max_value_bytes,
            )
            if redis_client is not None
            else None
        )
        app.state.rate_limiter = (
            RedisRateLimiter(
                redis_client,
                requests=configured.redis_rate_limit_requests,
                window_seconds=configured.redis_rate_limit_window_seconds,
            )
            if redis_client is not None
            else None
        )
        managed_checkpointer = None
        try:
            if postgres_database_configured and configured.database_url is not None:
                managed_checkpointer = create_postgres_checkpointer(
                    configured.database_url
                )
                app.state.checkpointer = managed_checkpointer.saver
            yield
        finally:
            if managed_checkpointer is not None:
                managed_checkpointer.close()
            if redis_client is not None:
                redis_client.close()
            if isinstance(runtime, RAGService):
                runtime.close()
            if conversation_store is not None:
                conversation_store.close()
            if vector_store is not None:
                vector_store.close()

    app = FastAPI(
        title="RetrieverOS",
        version="1.0.0",
        description="Hybrid retrieval with grounded answers, citation verification, and confidence scoring.",
        lifespan=lifespan,
    )
    app.state.vector_store = None
    app.state.checkpointer = None
    app.state.redis_client = None
    app.state.answer_cache = None
    app.state.rate_limiter = None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Workspace-ID",
        ],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_upload_bytes=configured.max_upload_bytes,
    )
    app.add_middleware(RequestIDMiddleware)
    runtime = service
    oidc_verifier = build_oidc_verifier(configured)
    if bool(configured.database_url) != bool(configured.user_key_encryption_key):
        raise ValueError(
            "DATABASE_URL and USER_KEY_ENCRYPTION_KEY must be configured together."
        )
    credential_store = (
        UserCredentialStore(
            configured.database_url, configured.user_key_encryption_key
        )
        if configured.database_url and configured.user_key_encryption_key
        else None
    )
    workspace_store = (
        WorkspaceStore(configured.database_url) if configured.database_url else None
    )
    audit_store = (
        AdminAuditStore(configured.database_url) if configured.database_url else None
    )
    conversation_store = (
        ConversationStore(configured.database_url) if configured.database_url else None
    )

    # Authenticate a signed Keycloak identity, retaining API-key mode only for local use.
    def authenticate(
        api_key: Annotated[str | None, Depends(api_key_header)],
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
        ],
    ) -> VerifiedIdentity:
        if oidc_verifier is not None:
            if credentials is None or credentials.scheme.casefold() != "bearer":
                raise HTTPException(status_code=401, detail="A valid bearer token is required.")
            try:
                return oidc_verifier.verify(credentials.credentials)
            except (jwt.PyJWTError, OSError, ValueError) as error:
                raise HTTPException(status_code=401, detail="The bearer token is invalid.") from error
        if configured.app_env == "production":
            raise HTTPException(status_code=503, detail="OIDC authentication is not configured.")
        if configured.api_key and (
            api_key is None or not secrets.compare_digest(api_key, configured.api_key)
        ):
            raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")
        return VerifiedIdentity(user_id="local", platform_admin=True)

    def authorize(
        request: Request,
        identity: Annotated[VerifiedIdentity, Depends(authenticate)],
        workspace_header: Annotated[
            str | None, Header(alias="X-Workspace-ID", max_length=512)
        ] = None,
    ) -> WorkspaceAccessContext:
        if oidc_verifier is not None:
            if workspace_store is None:
                raise HTTPException(
                    status_code=503,
                    detail="Workspace membership storage is not configured.",
                )
            try:
                if workspace_header is not None and not workspace_header.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="X-Workspace-ID must not be empty.",
                    )
                if workspace_header is None:
                    raise HTTPException(
                        status_code=400,
                        detail="X-Workspace-ID is required for this request.",
                    )
                selected_workspace_id = workspace_header
                workspace = workspace_store.get_workspace(selected_workspace_id)
                if workspace is None:
                    raise HTTPException(
                        status_code=403, detail="The selected workspace is not available."
                    )
                membership = workspace_store.get_membership(
                    identity.user_id, selected_workspace_id
                )
                personal_membership = (
                    workspace_store.get_membership_for_personal_workspace(
                        identity.user_id
                    )
                    if identity.platform_admin
                    else None
                )
                if not identity.platform_admin and (
                    membership is None or membership.status != "active"
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="You are not an active member of the selected workspace.",
                    )
                roles = (
                    membership.roles
                    if membership is not None and membership.status == "active"
                    else frozenset()
                )
                try:
                    permissions = workspace_store.get_role_permissions(
                        selected_workspace_id, roles
                    )
                except ValueError as error:
                    raise HTTPException(
                        status_code=403,
                        detail="A workspace membership role is unavailable.",
                    ) from error
                access_context = WorkspaceAccessContext(
                    user_id=identity.user_id,
                    workspace_id=selected_workspace_id,
                    roles=roles,
                    permissions=permissions,
                    platform_admin=identity.platform_admin,
                )
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503,
                    detail="Unable to resolve workspace membership.",
                ) from error
            route = request.scope.get("route")
            route_path = getattr(route, "path", request.url.path)
            global_scope = route_path in {
                "/v1/workspaces",
                "/v1/roles",
                "/v1/admin/operations/reindex-dry-run",
            }
            target_workspace_id = request.path_params.get(
                "workspace_id", selected_workspace_id
            )
            cross_workspace_access = (
                personal_membership is None
                or selected_workspace_id != personal_membership.workspace_id
                or target_workspace_id != personal_membership.workspace_id
                or global_scope
            )
            if identity.platform_admin and cross_workspace_access:
                if audit_store is None:
                    raise HTTPException(
                        status_code=503,
                        detail="Privileged access audit storage is not configured.",
                    )
                try:
                    audit_store.record_cross_workspace_access(
                        actor_user_id=identity.user_id,
                        action=f"{request.method} {route_path}",
                        workspace_id=None if global_scope else target_workspace_id,
                        request_id=request.state.request_id,
                    )
                except (SQLAlchemyError, ValueError) as error:
                    raise HTTPException(
                        status_code=503,
                        detail="Unable to record privileged access audit.",
                    ) from error
            return access_context
        return WorkspaceAccessContext(
            user_id=identity.user_id,
            workspace_id="local",
            roles=frozenset({"admin"}),
            platform_admin=identity.platform_admin,
        )

    def audit_platform_workspace_access(
        identity: VerifiedIdentity,
        request: Request,
        action: str,
        workspace_id: str | None,
    ) -> None:
        if not identity.platform_admin:
            return
        if audit_store is None:
            raise HTTPException(
                status_code=503,
                detail="Privileged access audit storage is not configured.",
            )
        try:
            audit_store.record_cross_workspace_access(
                actor_user_id=identity.user_id,
                action=action,
                workspace_id=workspace_id,
                request_id=request.state.request_id,
            )
        except (SQLAlchemyError, ValueError) as error:
            raise HTTPException(
                status_code=503,
                detail="Unable to record privileged access audit.",
            ) from error

    def resolve_conversation(
        access_context: WorkspaceAccessContext,
        requested_id: UUID | None,
    ) -> ConversationRecord | None:
        if conversation_store is None:
            return None
        try:
            if requested_id is None:
                return conversation_store.create(access_context)
            conversation = conversation_store.get(access_context, requested_id)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to access conversation storage."
            ) from error
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conversation

    app.include_router(
        create_workspace_router(
            workspace_store,
            configured,
            authorize,
            authenticate,
            audit_platform_workspace_access,
        )
    )

    @app.post(
        "/v1/conversations",
        response_model=ConversationRecord,
        status_code=201,
    )
    def create_conversation(
        request: ConversationCreateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> ConversationRecord:
        require_permission(access_context, "documents:query")
        if conversation_store is None:
            raise HTTPException(
                status_code=503, detail="Conversation storage is not configured."
            )
        try:
            return conversation_store.create(access_context, request.title)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to create the conversation."
            ) from error

    @app.get("/v1/conversations", response_model=list[ConversationRecord])
    def list_conversations(
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> list[ConversationRecord]:
        require_permission(access_context, "documents:query")
        if conversation_store is None:
            raise HTTPException(
                status_code=503, detail="Conversation storage is not configured."
            )
        try:
            return conversation_store.list(access_context, limit)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to list conversations."
            ) from error

    @app.get(
        "/v1/conversations/{conversation_id}",
        response_model=ConversationRecord,
    )
    def get_conversation(
        conversation_id: UUID,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> ConversationRecord:
        require_permission(access_context, "documents:query")
        if conversation_store is None:
            raise HTTPException(
                status_code=503, detail="Conversation storage is not configured."
            )
        try:
            conversation = conversation_store.get(access_context, conversation_id)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to retrieve the conversation."
            ) from error
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        return conversation

    @app.delete("/v1/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: UUID,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> None:
        require_permission(access_context, "documents:query")
        if conversation_store is None:
            raise HTTPException(
                status_code=503, detail="Conversation storage is not configured."
            )
        try:
            conversation = conversation_store.get(access_context, conversation_id)
            if conversation is None:
                raise HTTPException(
                    status_code=404, detail="Conversation not found."
                )
            if app.state.checkpointer is not None:
                for variant in (None, "hybrid-compare", "dense-compare"):
                    app.state.checkpointer.delete_thread(
                        conversation_thread_id(
                            access_context, conversation_id, variant
                        )
                    )
            elif isinstance(runtime, RAGService):
                runtime.delete_conversation_checkpoint(
                    access_context, conversation_id
                )
            deleted = conversation_store.delete(access_context, conversation_id)
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to delete the conversation."
            ) from error
        except PsycopgError as error:
            raise HTTPException(
                status_code=503, detail="Unable to delete conversation history."
            ) from error
        except RuntimeError as error:
            raise HTTPException(
                status_code=503, detail="Unable to delete conversation history."
            ) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    # Resolve the shared service lazily and return setup errors as explicit service failures.
    def get_runtime() -> RAGService:
        nonlocal runtime
        if runtime is None:
            if postgres_database_configured and app.state.checkpointer is None:
                raise HTTPException(
                    status_code=503,
                    detail="PostgreSQL conversation checkpointing is unavailable.",
                )
            try:
                runtime = RAGService(
                    configured,
                    vector_store=app.state.vector_store,
                    checkpointer=app.state.checkpointer,
                )
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503,
                    detail="Unable to initialize shared retrieval storage.",
                ) from error
            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        return runtime

    def enforce_rate_limit(
        access_context: WorkspaceAccessContext, operation: str
    ) -> None:
        limiter = app.state.rate_limiter
        if limiter is None:
            if configured.app_env == "production":
                raise HTTPException(
                    status_code=503, detail="Request limiting is unavailable."
                )
            return
        try:
            allowed = limiter.allow(access_context, operation)
        except RedisError as error:
            logger.warning("Redis rate limiting failed for operation %s.", operation)
            if configured.app_env == "production":
                raise HTTPException(
                    status_code=503, detail="Request limiting is unavailable."
                ) from error
            return
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please retry after the current limit window.",
            )

    def require_permission(
        access_context: WorkspaceAccessContext, permission: str
    ) -> None:
        if (
            not access_context.platform_admin
            and permission not in access_context.permissions
        ):
            raise HTTPException(
                status_code=403,
                detail=f"The {permission} permission is required.",
            )

    # Resolve a user's private OpenAI key without exposing it through the API.
    def get_user_openai_key(access_context: WorkspaceAccessContext) -> str:
        if oidc_verifier is not None:
            if credential_store is None:
                raise HTTPException(
                    status_code=503,
                    detail="Per-user OpenAI key storage is not configured.",
                )
            try:
                api_key = credential_store.get_key(
                    access_context.user_id, access_context.workspace_id
                )
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503, detail="Unable to access credential storage."
                ) from error
            if api_key is None:
                raise HTTPException(
                    status_code=428,
                    detail="Add your OpenAI API key in account settings before using the RAG API.",
                )
            return api_key
        if configured.openai_api_key:
            return configured.openai_api_key
        if credential_store is None:
            raise HTTPException(
                status_code=428,
                detail="Configure an OpenAI API key before using the RAG API.",
            )
        try:
            api_key = credential_store.get_key(
                access_context.user_id, access_context.workspace_id
            )
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to access credential storage."
            ) from error
        if api_key is None:
            raise HTTPException(
                status_code=428,
                detail="Configure an OpenAI API key before using the RAG API.",
            )
        return api_key

    # Store only an authenticated user's own encrypted model credential.
    @app.get("/v1/account/openai-key")
    def openai_key_status(
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> dict[str, bool]:
        if credential_store is None:
            raise HTTPException(
                status_code=503, detail="Per-user OpenAI key storage is not configured."
            )
        try:
            return {
                "configured": credential_store.has_key(
                    access_context.user_id, access_context.workspace_id
                )
            }
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to access credential storage."
            ) from error

    @app.put("/v1/account/openai-key", status_code=204)
    def save_openai_key(
        request: OpenAIKeyRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> None:
        if credential_store is None:
            raise HTTPException(
                status_code=503, detail="Per-user OpenAI key storage is not configured."
            )
        try:
            credential_store.set_key(
                access_context.user_id,
                access_context.workspace_id,
                request.api_key,
            )
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to save the encrypted credential."
            ) from error

    @app.delete("/v1/account/openai-key", status_code=204)
    def delete_openai_key(
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> None:
        if credential_store is None:
            raise HTTPException(
                status_code=503, detail="Per-user OpenAI key storage is not configured."
            )
        try:
            credential_store.delete_key(
                access_context.user_id, access_context.workspace_id
            )
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to delete the encrypted credential."
            ) from error

    # Report process readiness without exposing private configuration.
    @app.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"])
    def readiness() -> dict[str, str | dict[str, str]]:
        dependencies = {
            "postgres": "not_configured",
            "qdrant": "not_configured",
            "redis": "not_configured",
        }
        if postgres_database_configured:
            try:
                if credential_store is None or app.state.checkpointer is None:
                    raise RuntimeError("PostgreSQL-backed application storage is incomplete.")
                credential_store.check()
                dependencies["postgres"] = "ready"
            except (SQLAlchemyError, PsycopgError, RuntimeError, OSError):
                dependencies["postgres"] = "unavailable"
        if app.state.vector_store is not None:
            try:
                app.state.vector_store.check()
                dependencies["qdrant"] = "ready"
            except (RuntimeError, OSError):
                dependencies["qdrant"] = "unavailable"
        if app.state.redis_client is not None:
            try:
                app.state.redis_client.ping()
                dependencies["redis"] = "ready"
            except RedisError:
                dependencies["redis"] = "unavailable"

        required_dependencies = (
            set(dependencies)
            if configured.app_env == "production"
            else {
                name
                for name, state in dependencies.items()
                if state != "not_configured"
            }
        )
        ready = all(dependencies[name] == "ready" for name in required_dependencies)
        if not ready:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "dependencies": dependencies,
                },
            )
        return {"status": "ready", "dependencies": dependencies}

    @app.get(
        "/v1/admin/operations/reindex-dry-run",
        response_model=ReindexDryRunReport,
        tags=["operations"],
    )
    def reindex_dry_run(
        request: Request,
        identity: Annotated[VerifiedIdentity, Depends(authenticate)],
    ) -> ReindexDryRunReport:
        if not identity.platform_admin:
            raise HTTPException(
                status_code=403,
                detail="Platform administrator access is required.",
            )
        audit_platform_workspace_access(
            identity,
            request,
            "GET /v1/admin/operations/reindex-dry-run",
            workspace_id=None,
        )
        try:
            verified_sources = verify_archived_sources(
                get_runtime().knowledge_base
            )
        except RuntimeError as error:
            raise HTTPException(
                status_code=409,
                detail="The archived source manifest could not be verified.",
            ) from error
        except OSError as error:
            raise HTTPException(
                status_code=503,
                detail="Unable to read archived sources for the dry-run report.",
            ) from error
        return ReindexDryRunReport(
            documents=[
                {
                    "source": document.source,
                    "source_sha256": document.source_sha256,
                    "bytes": byte_count,
                    "chunking_strategy": document.chunking_strategy,
                }
                for document, byte_count in verified_sources
            ]
        )

    # Return the current indexed-document inventory.
    @app.get("/v1/documents", response_model=list[DocumentInfo])
    def documents(
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> list[DocumentInfo]:
        require_permission(access_context, "documents:read")
        try:
            return get_runtime().knowledge_base.list_documents(access_context)
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    # Ingest one bounded Markdown, text, HTML, or PDF upload.
    @app.post("/v1/ingest", response_model=IngestionResult)
    async def ingest(
        file: Annotated[UploadFile, File()],
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
        strategy: Annotated[Literal["fixed", "recursive", "semantic"], Form()] = "recursive",
        roles: Annotated[str, Form()] = "employee",
    ) -> IngestionResult:
        if not file.filename:
            raise HTTPException(status_code=400, detail="An upload filename is required.")
        raw_roles = [role.strip() for role in roles.split(",")]
        assigned_roles = sorted({role for role in raw_roles if role})
        if not assigned_roles:
            raise HTTPException(status_code=422, detail="At least one document role is required.")
        if workspace_store is not None:
            try:
                workspace_roles = workspace_store.list_workspace_roles(
                    access_context.workspace_id
                )
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503, detail="Unable to validate document roles."
                ) from error
            known_roles = {role.name for role in workspace_roles}
            unknown_roles = set(assigned_roles) - known_roles
            if unknown_roles:
                raise HTTPException(
                    status_code=422,
                    detail="Document roles must belong to the selected workspace.",
                )
        if (
            not access_context.platform_admin
            and "admin" not in access_context.roles
            and not set(assigned_roles).issubset(access_context.roles)
        ):
            raise HTTPException(
                status_code=403,
                detail="You may assign only roles granted to your account.",
            )
        try:
            existing_document = get_runtime().knowledge_base.get_document(
                access_context.workspace_id, file.filename
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(
                status_code=503, detail="Unable to inspect the existing document."
            ) from error
        if existing_document is None:
            require_permission(access_context, "documents:upload")
        else:
            require_permission(access_context, "documents:update")
            if not access_context.platform_admin and not access_context.permits(
                existing_document.workspace_id,
                existing_document.owner_id,
                existing_document.required_roles,
            ):
                raise HTTPException(
                    status_code=403,
                    detail="You do not have access to the document being replaced.",
                )
        enforce_rate_limit(access_context, "ingest")
        content = await file.read(configured.max_upload_bytes + 1)
        if len(content) > configured.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The uploaded document exceeds the size limit.")
        try:
            user_api_key = get_user_openai_key(access_context)
            embeddings = OpenAIEmbeddingProvider(configured, user_api_key)
            return await run_in_threadpool(
                get_runtime().knowledge_base.ingest,
                file.filename,
                content,
                strategy,
                access_context,
                assigned_roles,
                embeddings,
            )
        except HTTPException:
            raise
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Shared retrieval storage is unavailable."
            ) from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(
                status_code=503, detail="Ingestion dependencies are unavailable."
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    # Retrieve and answer one question with source-level evidence and quality scores.
    @app.post(
        "/v1/ask",
        response_model=AnswerResponse,
    )
    async def ask(
        request: AskRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> AnswerResponse:
        require_permission(access_context, "documents:query")
        enforce_rate_limit(access_context, "ask")
        try:
            user_api_key = get_user_openai_key(access_context)
            conversation = resolve_conversation(
                access_context, request.conversation_id
            )
            conversation_id = (
                conversation.conversation_id if conversation is not None else None
            )
            thread_id = (
                conversation_thread_id(access_context, conversation_id)
                if conversation_id is not None
                else request.thread_id
            )
            runtime_service = get_runtime()
            cache = app.state.answer_cache
            cache_allowed = (
                cache is not None
                and conversation is None
                and isinstance(runtime_service, RAGService)
                and runtime_service.answer_cache_enabled
            )
            key = (
                cache_key(
                    request.question,
                    request.retrieval_mode,
                    access_context,
                    runtime_service.answer_cache_index_version,
                )
                if cache_allowed
                else None
            )
            if cache is not None and key is not None:
                try:
                    cached_answer = cache.get(key)
                except RedisError:
                    logger.warning("Redis answer cache read failed; bypassing cache.")
                else:
                    if cached_answer is not None:
                        return cached_answer
            answer = await run_in_threadpool(
                runtime_service.ask,
                request.question,
                request.retrieval_mode,
                thread_id,
                access_context,
                user_api_key,
                conversation_id=conversation_id,
            )
            if conversation_id is not None:
                answer = answer.model_copy(
                    update={"conversation_id": conversation_id}
                )
            if cache is not None and key is not None:
                try:
                    cache.set(key, answer)
                except RedisError:
                    logger.warning("Redis answer cache write failed; returning generated answer.")
            return answer
        except HTTPException:
            raise
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Shared retrieval storage is unavailable."
            ) from error
        except (OSError, PsycopgError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    # Run dense-only and hybrid retrieval separately for dashboard comparison.
    @app.post(
        "/v1/compare",
        response_model=dict[str, AnswerResponse],
    )
    async def compare(
        request: AskRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> dict[str, AnswerResponse]:
        require_permission(access_context, "documents:query")
        enforce_rate_limit(access_context, "compare")
        runtime_service = get_runtime()
        try:
            user_api_key = get_user_openai_key(access_context)
            conversation = resolve_conversation(
                access_context, request.conversation_id
            )
            conversation_id = (
                conversation.conversation_id if conversation is not None else None
            )
            hybrid_thread_id = (
                conversation_thread_id(
                    access_context, conversation_id, "hybrid-compare"
                )
                if conversation_id is not None
                else f"{request.thread_id}-hybrid"
            )
            dense_thread_id = (
                conversation_thread_id(
                    access_context, conversation_id, "dense-compare"
                )
                if conversation_id is not None
                else f"{request.thread_id}-dense"
            )
            hybrid, dense = await run_in_threadpool(
                lambda: (
                    runtime_service.ask(
                        request.question,
                        "hybrid",
                        hybrid_thread_id,
                        access_context,
                        user_api_key,
                        conversation_id=conversation_id,
                        checkpoint_variant="hybrid-compare",
                    ),
                    runtime_service.ask(
                        request.question,
                        "dense",
                        dense_thread_id,
                        access_context,
                        user_api_key,
                        conversation_id=conversation_id,
                        checkpoint_variant="dense-compare",
                    ),
                )
            )
            if conversation_id is not None:
                hybrid = hybrid.model_copy(
                    update={"conversation_id": conversation_id}
                )
                dense = dense.model_copy(update={"conversation_id": conversation_id})
            return {"hybrid": hybrid, "dense": dense}
        except HTTPException:
            raise
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Shared retrieval storage is unavailable."
            ) from error
        except (OSError, PsycopgError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


# Export the ASGI application used by Uvicorn and container deployments.
app = create_app()
