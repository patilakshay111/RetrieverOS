# Recreate the Internal Docs Hybrid RAG Project

This guide describes how to recreate and run the implemented project from a
checkout. Python 3.11 is the tested CI version; do not treat the incomplete
production roadmap below as already implemented.

## 1. Create the project and environment

Install the project into the globally installed Python 3.11 selected for this
project:

```powershell
python --version
python -m pip install -e ".[dev]"
```

The dependency manifest is `pyproject.toml`; it includes the API, dashboard,
retrieval, evaluation, migration, and test dependencies. The source package is
under `src/rag_pipeline/`; tests, docs, sample documents, and the 50-case
evaluation set are kept in their corresponding top-level folders.

FAISS and PyTorch platform wheels vary by operating system. If Windows cannot resolve a compatible FAISS wheel, use the supplied Docker image or a supported Linux/WSL environment rather than silently switching index formats.

## 2. Establish configuration and safe defaults

Generate local service passwords and the Fernet encryption key with
`python scripts/init_local_env.py`, then add `OPENAI_API_KEY` to `.env`. The
script refuses to overwrite an existing `.env`. Never put credentials in
source code or commit `.env`.

The native-Python development setup uses SQLite for account-key, workspace,
invitation, and audit storage; set
`$env:DATABASE_URL = "sqlite:///./data/runtime/credentials.db"` in the API
terminal before startup. Docker Compose overrides this with PostgreSQL and
runs Alembic migrations before starting the API. Streamlit OIDC settings live
in `.streamlit/secrets.toml` (use the example template and replace every
placeholder). Set `SMTP_HOST`, `SMTP_PORT`, and `SMTP_FROM` to route invitation
mail; Compose defaults to Mailpit at <http://localhost:8025>.

Production mode requires HTTPS OIDC, PostgreSQL, an encryption key, and
restricted CORS. API-key authentication is retained for local development; it
is not the production identity mechanism. Uploads are size limited, and the
health/readiness endpoints do not expose secrets.

## 3. Build document loading and chunking

Implement a loader for Markdown, plain text, HTML, and PDF. Convert content to text, strip HTML script/style elements, preserve headings and PDF page numbers, and reject unknown or invalid inputs.

Implement three named strategies:

1. `fixed`: recursive character-size splitting with overlap.
2. `recursive`: heading-aware section splitting with paragraph/sentence boundaries.
3. `semantic`: sentence grouping based on adjacent embedding similarity, bounded by the maximum chunk size.

Record source, source digest, chunk index, section, page, strategy, and character count for every chunk.

## 4. Persist source files and synchronized indexes

Keep raw uploaded bytes in a content-addressed archive so they can be
reprocessed. Embed chunks with the configured embedding model; normalize
vectors and use a FAISS inner-product index as cosine retrieval.

Before committing an incoming chunk, compare its vector against the retained
corpus and the other accepted chunks. Skip it when cosine similarity exceeds
the configured deduplication threshold. Save chunk metadata, vectors, FAISS
index, and document inventory in a fresh generation directory; only point
`CURRENT` at that generation after every file is written successfully. BM25 is
built lazily from the active chunk snapshot and cached per authorization scope.

Add a re-index command that verifies each current document's raw-file digest, reloads the original source, and reruns the same chunking/embedding path without requiring an upload. Build/cache BM25 from the active chunk snapshot on the first sparse or hybrid query for that generation.

## 5. Add hybrid retrieval and reranking

For each query, retrieve dense and BM25 top-k results. Fuse their ranked chunk IDs with configurable weighted Reciprocal Rank Fusion, send the top candidates through a cross-encoder, and keep the configured final top-k. Preserve per-stage scores for inspection. Keep dense-only and sparse-only modes available for comparison.

## 6. Add grounded answers and citation checks

Create a deterministic OpenAI chat helper. Tell the model to treat retrieved text as evidence rather than instructions, answer only from evidence, cite every factual claim with a numbered chunk reference, and explicitly abstain when evidence is insufficient.

Parse each claim and cited chunk separately. Ask an LLM judge whether the complete claim is supported, flag invalid or unsupported citations, and calculate claim-level citation coverage. Ask a separate structured judge whether the answer addresses all parts of the question.

## 7. Wire the LangGraph workflow

Define the shared graph state as a Pydantic `BaseModel`. Use small sequential nodes for retrieval, generation/abstention, citation verification, and confidence scoring. Compile with `MemorySaver` and pass a stable `thread_id` in every invocation; retain a bounded recent-turn history so follow-up answers can use earlier context. Use the same state and service from API and evaluation code.

Confidence is a weighted composite of retrieval confidence (40%), verified
claim coverage (35%), and answer completeness (25%). Keep component scores in
the response. Below the retrieval threshold, do not call the generator;
return a clear no-answer response and the closest documents to check manually.
The current LangGraph checkpointer is in-memory; its thread ID is namespaced by
user and workspace but history is not durable across process restarts.

## 8. Expose the service and dashboard

Create FastAPI endpoints:

- `GET /healthz`
- `GET /readyz`
- `GET /v1/documents`
- `POST /v1/ingest`
- `POST /v1/ask`
- `POST /v1/compare`
- `GET`, `PUT`, `DELETE /v1/account/openai-key`
- `GET /v1/workspaces`
- `GET`, `POST /v1/workspaces/{workspace_id}/members`
- `PUT`, `DELETE /v1/workspaces/{workspace_id}/members/{user_id}`
- `POST /v1/workspaces/{workspace_id}/invitations`
- `POST /v1/invitations/{token}/accept`
- `GET`, `POST /v1/roles`

Validate inputs, restrict uploads to supported formats and a bounded size,
authenticate local requests with `X-API-Key` when configured, and document
request and response schemas through OpenAPI. OIDC-authenticated users can
save, rotate, and delete their own encrypted OpenAI key. Build a Streamlit
dashboard that asks questions, displays ranked chunks/citation checks and
confidence, and compares hybrid with dense-only retrieval.

The API provisions a personal workspace for each verified OIDC subject and
uses PostgreSQL—not token-supplied workspace or role claims—as the source of
membership and workspace roles. Clients select a workspace with
`X-Workspace-ID`; without the header, the personal workspace is selected.
Workspace admins are confined to their active workspace, existing global role
definitions are assigned through membership, and only platform admins may
create global roles or administer across workspaces. Invitations contain
single-use, expiring random tokens, persist only the SHA-256 token hash, and
can be accepted only by an account with the matching verified OIDC email.
Platform-admin cross-workspace activity is persisted with the actor, action,
target workspace, timestamp, and request ID; request content and secrets are
excluded.

## 9. Seed, test, and evaluate

The repository includes a non-sensitive sample corpus and a hand-written
50-question golden dataset. It includes simple lookups, multi-document
reasoning, ambiguous prompts, and questions whose answers are absent. Tests
cover chunk provenance, RRF math, deduplication, persistence, index alignment,
citation failures, access checks, key isolation, and database migrations.

Run the local checks:

```powershell
python -m pytest
rag-pipeline-seed
rag-pipeline-reindex
python -m rag_pipeline.evaluation.run
```

The evaluation runner compares fixed, recursive, and semantic chunkers using
correctness, faithfulness, retrieval relevance, citation accuracy, and
coverage. It requires valid OpenAI credentials and reports model-judge outputs;
CI runs the offline deterministic tests rather than making paid model calls.

## 10. Containerize and demonstrate

Docker Compose starts the API/dashboard, PostgreSQL, Redis, Qdrant, Keycloak,
and Mailpit. It runs the Alembic migration before the API, and mounts a
persistent volume for raw documents and the versioned FAISS index. Run:

```powershell
docker compose up --build
docker compose exec api rag-pipeline-seed
```

GitHub Actions runs tests on Python 3.11, validates Compose, and builds the
container image after the tests pass. It does not publish or deploy the image.
Although Redis and Qdrant are included in Compose, application retrieval still
uses local FAISS and Redis is not wired into the request path. Durable
checkpoints, the React/Vite dashboard, OpenTelemetry, Kubernetes, and
deployment automation remain future work. Workspace authorization,
invitation endpoints, and privileged-access auditing are implemented; the
Streamlit dashboard does not yet expose their full management workflows.

For a short demo, ingest a new runbook, ask a direct lookup, ask a multi-hop
question, ask something absent from the corpus, then compare hybrid and
dense-only retrieval. Show citation verification and confidence breakdown.
Capture the full evaluation report before making any claim about measured
quality.

## Rebuild checklist

- [ ] `.env` was created by the secret generator and `OPENAI_API_KEY` is configured.
- [ ] The data directory is writable and persistent.
- [ ] Native Python uses SQLite, or Compose successfully migrates PostgreSQL.
- [ ] Sample ingestion reports a nonzero chunk count.
- [ ] `GET /healthz`, `GET /readyz`, and `/docs` load.
- [ ] A supported question returns numbered citations and a confidence breakdown.
- [ ] An unanswerable question produces a structured abstention.
- [ ] Dense/hybrid comparison returns both result sets.
- [ ] Unit tests pass and all three evaluation strategies complete.
- [ ] All offline tests pass; Compose config and the container build are checked.
- [ ] Production use additionally has HTTPS OIDC, PostgreSQL, durable storage,
      secret management, observability, recovery, and a tested deployment target.
