# Durable RAG State and Shared Services Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RAG retrieval, conversations, caching, and request limits safe for concurrent multi-user requests and API restarts.

**Architecture:** Keep FastAPI and LangGraph, but replace process-local conversation checkpoints and request-time FAISS as the production retrieval path with PostgreSQL checkpoints, PostgreSQL full-text lexical search, and Qdrant-filtered vectors. Store chunk text/ACL metadata in PostgreSQL for shared lexical search; Qdrant stores dense vectors and matching ACL payloads. Add Redis caching/rate limits with complete authorization-scope keys; preserve FAISS for local migration/reindex support only where useful. Keep the existing document archive on the durable application volume for the initial single-API Render test; do not horizontally scale the API until raw files use shared object storage.

**Tech Stack:** Python 3.11+, LangGraph, PostgreSQL, Qdrant, Redis, SQLAlchemy/Alembic, pytest, Docker Compose.

**Spec:** [2026-11-01-multiuser-web-dashboard-design.md](../specs/2026-11-01-multiuser-web-dashboard-design.md)

**Order:** Implement after [workspace-authorization](2026-11-01-workspace-authorization.md)
and before [react-dashboard](2026-11-01-react-dashboard.md); Qdrant/cache
interfaces and conversation routes depend on its authenticated workspace scope.

## Global Constraints

- Production retrieval must apply workspace, owner, and role filters server-side before evidence reaches the answer model.
- Checkpoint identifiers include immutable user, active workspace, and conversation ID.
- Cache keys include effective authorization scope and the relevant corpus/index version.
- Do not cache prompts/answers across authorization scopes or log sensitive model input/output.
- Keep current local development and deterministic offline tests working.
- PostgreSQL schema changes use Alembic revisions; no production `metadata.create_all`.
- No Kubernetes.
- Do not horizontally scale the API while raw uploads remain on service-local storage.

## Review Focus

- Empty, stale, or malformed Qdrant payload ACL fields must fail closed; test missing scope metadata.
- A user with overlapping roles in two workspaces must retrieve only that workspace's points; test both dense and lexical paths.
- Redis outage must return explicit behavior, not serve a success-shaped cross-scope stale answer; test cache failure handling.
- Concurrent updates/reindex must not expose mixed index generations; test publish/version behavior.
- Database/Qdrant/Redis readiness failure must fail readiness and give a bounded service error; test each dependency failure.

---

### Task 1: Add Qdrant configuration and client lifecycle

**Files:**
- Modify: `src/rag_pipeline/config.py`
- Create: `src/rag_pipeline/vector_store.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `docker-compose.yml`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `VectorStore.upsert(chunks: Sequence[Chunk], vectors: np.ndarray) -> None`, `VectorStore.search(vector: np.ndarray, scope: WorkspaceAccessContext, limit: int) -> list[RetrievedChunk]`, `VectorStore.delete_document(workspace_id: str, source_id: str) -> None`, and `VectorStore.check() -> None`.
- `scope` is the server-derived `WorkspaceAccessContext` from the workspace authorization plan.

- [ ] **Step 1: Write failing configuration/client tests** for required production `QDRANT_URL`, collection readiness, exact ACL filters, and Qdrant failure status.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k qdrant`; confirm failures.
- [ ] **Step 3: Implement** one managed Qdrant client and collection schema with keyword payload indexes for `workspace_id`, `owner_id`, and `required_roles`; use exact points rather than in-process vector copies for production search.
- [ ] **Step 4: Run** focused tests with a local Qdrant fixture or deterministic fake and verify Compose config.
- [ ] **Step 5: Commit** client, configuration, Compose, and tests.

### Task 2: Route ingestion and hybrid retrieval through Qdrant

**Files:**
- Create: `src/rag_pipeline/lexical_store.py`
- Create: `migrations/versions/0004_document_chunk_catalog.py`
- Modify: `src/rag_pipeline/storage.py`
- Modify: `src/rag_pipeline/retrieval.py`
- Modify: `src/rag_pipeline/service.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/reindex.py`
- Test: `tests/test_core.py`

**Interfaces:**
- `KnowledgeBase.ingest(..., access_context: WorkspaceAccessContext, ...)` assigns workspace/owner/role payloads.
- `HybridRetriever.search(query: str, mode: str, access_context: WorkspaceAccessContext) -> tuple[list[RetrievedChunk], float]` obtains dense candidates from Qdrant and lexical candidates from PostgreSQL full-text search, with workspace, owner, and role predicates applied in each store before ranking.

- [ ] **Step 1: Write failing tests** for workspace/owner/role filters at both query boundaries, same-name documents in separate workspaces, lexical ranking, document replacement/deletion, and reindex payload preservation.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k retrieval_acl`; confirm unauthorized candidate tests fail.
- [ ] **Step 3: Implement** the shared chunk catalog and indexed PostgreSQL `tsvector` search with ACL predicates, plus Qdrant writes and server-filtered dense retrieval. Fuse authorized dense and lexical ranks; do not rely on post-generation filtering or a process-local corpus.
- [ ] **Step 4: Run** existing retrieval/ingestion tests plus Qdrant ACL tests; verify same-filename documents in different workspaces remain distinct.
- [ ] **Step 5: Commit** the Qdrant retrieval path and regression tests.

### Task 3: Make conversation history durable

**Files:**
- Create: `migrations/versions/0005_conversations.py`
- Create: `src/rag_pipeline/conversations.py`
- Modify: `src/rag_pipeline/service.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/graph.py`
- Modify: `pyproject.toml`
- Modify: `migrations/versions/`
- Test: `tests/test_core.py`

**Interfaces:**
- Produce `GET/POST /v1/conversations`, `GET /v1/conversations/{conversation_id}`, and `DELETE /v1/conversations/{conversation_id}`; list, read, and delete only within `(user_id, workspace_id)`.
- `conversation_id` is a server-generated UUID. The user-provided thread identifier is not trusted as a database authorization key.
- Configure a supported PostgreSQL LangGraph checkpointer; thread IDs are server-derived from the authenticated user, workspace, and conversation ID.

- [ ] **Step 1: Write failing tests** for cross-user/workspace thread isolation, thread listing, API process/service recreation, and deletion authorization.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k conversation`; confirm persistence assertions fail.
- [ ] **Step 3: Implement** durable checkpoint storage and conversation metadata migrations; keep an explicit in-memory checkpointer only for tests/local opt-in.
- [ ] **Step 4: Run** tests using a temporary PostgreSQL test service or supported integration fixture; verify checkpoint recovery after creating a new service instance.
- [ ] **Step 5: Commit** dependency, migrations, service integration, and tests.

### Task 4: Add Redis rate limiting and authorization-scoped answer cache

**Files:**
- Create: `src/rag_pipeline/cache.py`
- Create: `src/rag_pipeline/rate_limits.py`
- Modify: `src/rag_pipeline/config.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `docker-compose.yml`
- Test: `tests/test_core.py`

**Interfaces:**
- `cache_key(question, retrieval_mode, scope, index_version) -> str` includes user/workspace/effective roles and authorization version.
- Redis cache values are bounded/expiring and contain only the authorized API response for that scope.
- Rate limit keys are derived server-side from authenticated subject/workspace and do not contain raw tokens.

- [ ] **Step 1: Write failing tests** for key separation by user, workspace, roles, corpus version, TTL expiry, cache miss, and rate-limit denial.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k redis`; confirm key and rate-limit tests fail.
- [ ] **Step 3: Implement** Redis integration with explicit timeout/error behavior; fail closed for rate limiting when required by production configuration, and bypass cache rather than return unsafe data on cache errors.
- [ ] **Step 4: Run** unit tests and Redis integration tests; ensure no bearer token or prompt appears in Redis keys or logs.
- [ ] **Step 5: Commit** Redis services, settings, and tests.

### Task 5: Integrate readiness and migration/reindex operations

**Files:**
- Modify: `src/rag_pipeline/api.py`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `docs/PROJECT_RECREATION_GUIDE.md`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing readiness tests** for PostgreSQL, Qdrant, and Redis readiness dependencies and explicit failure responses.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k readiness`; confirm dependency failures are reported.
- [ ] **Step 3: Implement** production readiness checks and a documented one-time FAISS-to-Qdrant reindex procedure; do not make health checks call OpenAI.
- [ ] **Step 4: Run** full tests, migration tests, Compose config, and reindex dry-run; verify no existing index is deleted before successful publication.
- [ ] **Step 5: Commit** operational integration and docs.
