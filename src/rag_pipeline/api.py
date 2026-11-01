# api.py: exposes authenticated FastAPI endpoints for ingestion, retrieval, and grounded answers

import secrets
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_pipeline.config import Settings
from rag_pipeline.models import AnswerResponse, DocumentInfo, IngestionResult
from rag_pipeline.service import RAGService


# Expose one optional API-key header so development can stay frictionless.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


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


# Build the API and enforce configured API keys outside the health endpoint.
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
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_upload_bytes=configured.max_upload_bytes,
    )
    runtime = service

    # Require a constant-time API key match when an API key is configured.
    def authorize(api_key: Annotated[str | None, Depends(api_key_header)]) -> None:
        if configured.app_env == "production" and not configured.api_key:
            raise HTTPException(status_code=503, detail="API authentication is not configured.")
        if configured.api_key and (
            api_key is None or not secrets.compare_digest(api_key, configured.api_key)
        ):
            raise HTTPException(status_code=401, detail="A valid X-API-Key header is required.")

    # Resolve the shared service lazily and return setup errors as explicit service failures.
    def get_runtime() -> RAGService:
        nonlocal runtime
        if runtime is None:
            try:
                runtime = RAGService(configured)
            except (RuntimeError, ValueError) as error:
                raise HTTPException(status_code=503, detail=str(error)) from error
        return runtime

    # Report process readiness without exposing private configuration.
    @app.get("/healthz", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Return the current indexed-document inventory.
    @app.get("/v1/documents", response_model=list[DocumentInfo], dependencies=[Depends(authorize)])
    def documents() -> list[DocumentInfo]:
        try:
            return get_runtime().knowledge_base.list_documents()
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    # Ingest one bounded Markdown, text, HTML, or PDF upload.
    @app.post(
        "/v1/ingest",
        response_model=IngestionResult,
        dependencies=[Depends(authorize)],
    )
    async def ingest(
        file: Annotated[UploadFile, File()],
        strategy: Annotated[Literal["fixed", "recursive", "semantic"], Form()] = "recursive",
    ) -> IngestionResult:
        if not file.filename:
            raise HTTPException(status_code=400, detail="An upload filename is required.")
        content = await file.read(configured.max_upload_bytes + 1)
        if len(content) > configured.max_upload_bytes:
            raise HTTPException(status_code=413, detail="The uploaded document exceeds the size limit.")
        try:
            return await run_in_threadpool(
                get_runtime().knowledge_base.ingest, file.filename, content, strategy
            )
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    # Retrieve and answer one question with source-level evidence and quality scores.
    @app.post(
        "/v1/ask",
        response_model=AnswerResponse,
        dependencies=[Depends(authorize)],
    )
    async def ask(request: AskRequest) -> AnswerResponse:
        try:
            return await run_in_threadpool(
                get_runtime().ask,
                request.question,
                request.retrieval_mode,
                request.thread_id,
            )
        except HTTPException:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    # Run dense-only and hybrid retrieval separately for dashboard comparison.
    @app.post(
        "/v1/compare",
        response_model=dict[str, AnswerResponse],
        dependencies=[Depends(authorize)],
    )
    async def compare(request: AskRequest) -> dict[str, AnswerResponse]:
        runtime_service = get_runtime()
        try:
            hybrid, dense = await run_in_threadpool(
                lambda: (
                    runtime_service.ask(request.question, "hybrid", f"{request.thread_id}-hybrid"),
                    runtime_service.ask(request.question, "dense", f"{request.thread_id}-dense"),
                )
            )
            return {"hybrid": hybrid, "dense": dense}
        except (OSError, RuntimeError, ValueError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


# Export the ASGI application used by Uvicorn and container deployments.
app = create_app()
