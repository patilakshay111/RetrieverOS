# api.py: exposes authenticated FastAPI endpoints for ingestion, retrieval, and grounded answers

import secrets
from typing import Annotated, Literal

import jwt
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_pipeline.config import Settings
from rag_pipeline.auth import OIDCVerifier, build_oidc_verifier
from rag_pipeline.credentials import UserCredentialStore
from rag_pipeline.embeddings import OpenAIEmbeddingProvider
from rag_pipeline.identity_store import WorkspaceStore
from rag_pipeline.models import (
    AnswerResponse,
    DocumentInfo,
    IngestionResult,
    VerifiedIdentity,
    WorkspaceAccessContext,
)
from rag_pipeline.service import RAGService
from rag_pipeline.workspaces import create_workspace_router


# Expose one optional API-key header so development can stay frictionless.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# Signal that a request exceeded the bounded ASGI body size.
class RequestBodyTooLarge(Exception):
    pass


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


# Build the API and enforce OIDC or local API-key authentication outside health.
def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    configured = settings or Settings.from_env()
    configured.validate_production()
    app = FastAPI(
        title="Internal Docs Hybrid RAG",
        version="1.0.0",
        description="Hybrid retrieval with grounded answers, citation verification, and confidence scoring.",
    )
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
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_upload_bytes=configured.max_upload_bytes,
    )
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
                personal_membership = workspace_store.create_personal_workspace(
                    identity.user_id
                )
                if workspace_header is not None and not workspace_header.strip():
                    raise HTTPException(
                        status_code=422,
                        detail="X-Workspace-ID must not be empty.",
                    )
                selected_workspace_id = (
                    workspace_header
                    if workspace_header is not None
                    else personal_membership.workspace_id
                )
                workspace = workspace_store.get_workspace(selected_workspace_id)
                if workspace is None:
                    raise HTTPException(
                        status_code=403, detail="The selected workspace is not available."
                    )
                membership = workspace_store.get_membership(
                    identity.user_id, selected_workspace_id
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
                return WorkspaceAccessContext(
                    user_id=identity.user_id,
                    workspace_id=selected_workspace_id,
                    roles=roles,
                    platform_admin=identity.platform_admin,
                )
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503,
                    detail="Unable to resolve workspace membership.",
                ) from error
        return WorkspaceAccessContext(
            user_id=identity.user_id,
            workspace_id="local",
            roles=frozenset({"admin"}),
            platform_admin=identity.platform_admin,
        )

    app.include_router(
        create_workspace_router(workspace_store, configured, authorize, authenticate)
    )

    # Resolve the shared service lazily and return setup errors as explicit service failures.
    def get_runtime() -> RAGService:
        nonlocal runtime
        if runtime is None:
            try:
                runtime = RAGService(configured)
            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        return runtime

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
    def readiness() -> dict[str, str]:
        if configured.app_env == "production" and credential_store is None:
            raise HTTPException(
                status_code=503, detail="Credential storage is not configured."
            )
        if credential_store is not None:
            try:
                credential_store.check()
            except SQLAlchemyError as error:
                raise HTTPException(
                    status_code=503, detail="Credential storage is unavailable."
                ) from error
        return {"status": "ready"}

    # Return the current indexed-document inventory.
    @app.get("/v1/documents", response_model=list[DocumentInfo])
    def documents(
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> list[DocumentInfo]:
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
        assigned_roles = sorted({role.strip() for role in roles.split(",") if role.strip()})
        if not assigned_roles:
            raise HTTPException(status_code=422, detail="At least one document role is required.")
        if not access_context.platform_admin and not set(assigned_roles).issubset(
            access_context.roles
        ):
            raise HTTPException(
                status_code=403,
                detail="You may assign only roles granted to your account.",
            )
        if not file.filename:
            raise HTTPException(status_code=400, detail="An upload filename is required.")
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
        except (OSError, RuntimeError, ValueError) as error:
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
        try:
            user_api_key = get_user_openai_key(access_context)
            return await run_in_threadpool(
                get_runtime().ask,
                request.question,
                request.retrieval_mode,
                request.thread_id,
                access_context,
                user_api_key,
            )
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
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
        runtime_service = get_runtime()
        try:
            user_api_key = get_user_openai_key(access_context)
            hybrid, dense = await run_in_threadpool(
                lambda: (
                    runtime_service.ask(
                        request.question,
                        "hybrid",
                        f"{request.thread_id}-hybrid",
                        access_context,
                        user_api_key,
                    ),
                    runtime_service.ask(
                        request.question,
                        "dense",
                        f"{request.thread_id}-dense",
                        access_context,
                        user_api_key,
                    ),
                )
            )
            return {"hybrid": hybrid, "dense": dense}
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


# Export the ASGI application used by Uvicorn and container deployments.
app = create_app()
