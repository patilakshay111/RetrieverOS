# Recreate the RetrieverOS Project

This guide describes how to recreate and run the implemented project from a
checkout. Python 3.11 is the tested CI version; do not treat the incomplete
production roadmap below as already implemented.

## 1. Create the project and environment

Install the project into the globally installed Python 3.11-or-newer
interpreter selected for this project. CI currently tests Python 3.11:

```powershell
python --version
python -m pip install -e ".[dev]"
```

The Python dependency manifest is `pyproject.toml`; it includes the API,
retrieval, evaluation, migration, and test dependencies. The independent React
dashboard and its pinned dependencies are in `web/package.json` and
`web/package-lock.json`. The source package is under `src/rag_pipeline/`;
tests, docs, sample documents, and the 50-case evaluation set are kept in
their corresponding top-level folders.

FAISS and PyTorch platform wheels vary by operating system. If Windows cannot resolve a compatible FAISS wheel, use the supplied Docker image or a supported Linux/WSL environment rather than silently switching index formats.

## 2. Establish configuration and safe defaults

Generate local service passwords and the Fernet encryption key with
`python scripts/init_local_env.py`, then add `OPENAI_API_KEY` to `.env`. The
script refuses to overwrite an existing `.env`. Never put credentials in
source code or commit `.env`.

The native-Python development setup uses SQLite for account-key, workspace,
invitation, and audit storage; set
`$env:DATABASE_URL = "sqlite:///./data/runtime/credentials.db"` in the API
terminal before startup, then upgrade the database schema:

```powershell
python -m alembic upgrade head
```

Docker Compose overrides this with PostgreSQL and runs Alembic migrations
before starting the API. The Keycloak service is the stock 26.3.2 image plus
the RetrieverOS Google-verified-email provider, built from
`deploy/keycloak-provider/` via `deploy/keycloak/Dockerfile`. On a new
Keycloak data volume, Compose imports `deploy/keycloak/retrieveros-realm.json`,
including the public React client configured for PKCE S256, the API audience
mapper, and the custom `retrieveros first broker login` flow (Google identities
must carry `email_verified=true`; existing same-email accounts are auto-linked
only on a matching verified Google email). Keycloak does not overwrite an
existing realm from an import; update an already-persisted `retrieveros`
realm in the Keycloak Admin Console — an older persisted realm will also lack
the custom flow, so import it there as well or re-create the volume. Set
`SMTP_HOST`, `SMTP_PORT`, and `SMTP_FROM` to route invitation mail; Compose
defaults to Mailpit at <http://localhost:8025>.

Production mode requires HTTPS OIDC, PostgreSQL, an encryption key, and
restricted CORS. The SPA uses Keycloak Authorization Code + PKCE and keeps
tokens in memory; the API validates tokens and checks workspace memberships
on every protected request. API-key authentication is retained for local
development; it is not the production identity mechanism. Uploads are size
limited, and the health/readiness endpoints do not expose secrets.

## 3. Build document loading and chunking

Implement a loader for Markdown, plain text, HTML, and PDF. Convert content to text, strip HTML script/style elements, preserve headings and PDF page numbers, and reject unknown or invalid inputs.

Implement three named strategies:

1. `fixed`: recursive character-size splitting with overlap.
2. `recursive`: heading-aware section splitting with paragraph/sentence boundaries.
3. `semantic`: sentence grouping based on adjacent embedding similarity, bounded by the maximum chunk size.

Record source, source digest, chunk index, section, page, strategy, and character count for every chunk.

## 4. Persist source files and synchronized indexes

Keep raw uploaded bytes in a content-addressed archive so they can be
reprocessed. Embed chunks with the configured embedding model and normalize
vectors. When running natively without both `QDRANT_URL` and `DATABASE_URL`,
use a FAISS inner-product index for cosine retrieval.

Before committing an incoming chunk, compare its vector against the retained
corpus and the other accepted chunks. Skip it when cosine similarity exceeds
the configured deduplication threshold. In local mode, save chunk metadata,
vectors, the FAISS index, and document inventory in a fresh generation
directory; only point `CURRENT` at that generation after every file is written
successfully. BM25 is built lazily from the active chunk snapshot and cached
per authorization scope.

When both `QDRANT_URL` and `DATABASE_URL` are configured, write accepted dense
vectors to Qdrant with workspace, owner, and required-role payloads, then
replace that workspace/source's current chunk rows in the PostgreSQL catalog.
The catalog indexes full-text data for lexical search and is the source of
current chunk text and ACL metadata. Publish the local snapshot and then remove
superseded Qdrant point IDs. These writes span separate systems and are not one
transaction; retrieval checks Qdrant candidates against the current catalog
before using their evidence. Run the Alembic migrations through revision
`0005_document_chunk_catalog` before enabling this mode.

The re-index command verifies each current document's raw-file digest, reloads
the original source, and reruns the same chunking/embedding path without
requiring an upload. In local mode, build/cache BM25 from the active snapshot
on the first sparse or hybrid query for that generation; shared mode uses
PostgreSQL full-text search instead.

## 5. Add hybrid retrieval and reranking

For local mode, retrieve dense and BM25 top-k results from the FAISS snapshot.
For shared mode, retrieve dense candidates from Qdrant and lexical candidates
from PostgreSQL full-text search, applying workspace, owner, and role ACL
predicates inside each store before ranking. Confirm dense chunk IDs against
the current catalog, fuse the authorized ranked IDs with configurable weighted
Reciprocal Rank Fusion, send top candidates through a cross-encoder, and keep
the configured final top-k. Preserve per-stage scores for inspection. Keep
dense-only and sparse-only modes available for comparison.

## 6. Add grounded answers and citation checks

Create a deterministic OpenAI chat helper. Tell the model to treat retrieved text as evidence rather than instructions, answer only from evidence, cite every factual claim with a numbered chunk reference, and explicitly abstain when evidence is insufficient.

Parse each claim and cited chunk separately. Ask an LLM judge whether the complete claim is supported, flag invalid or unsupported citations, and calculate claim-level citation coverage. Ask a separate structured judge whether the answer addresses all parts of the question.

## 7. Wire the LangGraph workflow

Define the shared graph state as a Pydantic `BaseModel`. Use small sequential nodes for retrieval, generation/abstention, citation verification, and confidence scoring. Retain a bounded recent-turn history so follow-up answers can use earlier context. Use the same state and service from API and evaluation code. In PostgreSQL-backed API operation, compile with the managed LangGraph PostgreSQL checkpointer; use `MemorySaver` only for local/test operation without PostgreSQL.

Confidence is a weighted composite of retrieval confidence (40%), verified
claim coverage (35%), and answer completeness (25%). Keep component scores in
the response. Below the retrieval threshold, do not call the generator;
return a clear no-answer response and the closest documents to check manually.
Create conversation metadata in PostgreSQL and expose scoped list/create/read/delete
endpoints. Generate conversation UUIDs on the server, validate ownership against
the authenticated user and active workspace before every ask, and derive LangGraph
thread IDs from that scope plus the conversation UUID. Do not treat a client-supplied
thread identifier as an authorization key. Deleting a conversation also deletes
its regular and comparison checkpoint variants. The PostgreSQL-backed path survives
API process restarts; local SQLite/test mode can use an in-memory checkpointer.

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
- `POST /v1/workspaces` (create a named workspace and initial admin membership)
- `PATCH /v1/workspaces/{workspace_id}` (workspace-admin edit)
- `GET`, `POST /v1/workspaces/{workspace_id}/roles` (workspace-scoped role definitions)
- `GET`, `POST /v1/workspaces/{workspace_id}/members`
- `PUT`, `DELETE /v1/workspaces/{workspace_id}/members/{user_id}`
- `POST /v1/workspaces/{workspace_id}/invitations`
- `POST /v1/invitations/accept` (the invitation token is supplied in the JSON
  request body)

Validate inputs, restrict uploads to supported formats and a bounded size,
authenticate local requests with `X-API-Key` when configured, and document
request and response schemas through OpenAPI. OIDC-authenticated users can
save, rotate, and delete their own encrypted OpenAI key. The React dashboard
in `web/` provides sign-in, workspace switching, chat and conversation
history, citation/evidence display, hybrid-versus-dense comparison, document
upload with chunking/access-role choices, private key management, and
role-gated workspace administration and operations. Keep UI permission checks
as usability only; the API remains authoritative for access control.

The API does not provision a default workspace. A first-time signed-in user
creates a named workspace with a description and becomes its initial admin.
Users can create additional isolated workspaces later. A workspace invitee
can accept an invitation before creating a workspace; after acceptance the
dashboard selects the invited workspace. Invitees must register and verify
the same email address used for the invitation.

Workspace memberships and roles come from PostgreSQL—not token-supplied
workspace or role claims. Workspace-scoped routes require an active
`X-Workspace-ID` and verify membership. Each workspace has its own built-in
roles and role definitions; its admins can create roles, choose document
permissions, edit workspace details, manage members, and invite users with
one or more roles. Capabilities are `documents:read`, `documents:query`,
`documents:upload`, and `documents:update`. A member receives the union of
their assigned-role permissions. Document uploads can be ACL-limited to
multiple roles in that workspace; the API validates those roles and enforces
the required permission on inventory, query, upload, and replacement routes.
The Keycloak `platform-admin` realm role is separate from workspace roles and
its cross-workspace actions are audited with the actor, action, target
workspace, timestamp, and request ID; request content and secrets are
excluded.

### Onboard users and test workspace access

1. Register and verify a user through Keycloak. Local verification messages
   appear in Mailpit at <http://localhost:8025>.
2. On first sign-in with no workspace memberships, enter a workspace name and
   description. The creator becomes that workspace's administrator. To create
   another isolated workspace later, use **Create another workspace**.
3. In **Workspace**, edit the name or description if needed. Create a role,
   select its permissions, and save it before inviting a user. Assign one or
   more existing roles in the invitation form; role permissions are not
   Keycloak realm roles.
4. The invitee registers with the invited email, verifies it through Mailpit,
   and accepts the invitation. The app selects the invited workspace
   automatically, even if this is the invitee's first workspace.
5. In **Documents**, choose a chunking strategy and one or more workspace
   roles for the document ACL. Non-admin members can assign only roles they
   hold. `documents:read` gates inventory, `documents:upload` gates new
   sources, and `documents:update` gates replacements. `documents:query`
   gates chat and conversation history. The API remains authoritative if a
   browser control is bypassed.
6. To verify isolation, invite a second user with a different role and
   confirm each user can list/query only documents permitted by their
   workspace membership and document ACL.

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
npm --prefix web ci
npm --prefix web test
npm --prefix web run lint
npm --prefix web run build
npm --prefix web audit
```

The evaluation runner compares fixed, recursive, and semantic chunkers using
correctness, faithfulness, retrieval relevance, citation accuracy, and
coverage. It requires valid OpenAI credentials and reports model-judge outputs;
CI runs the offline deterministic tests rather than making paid model calls.

## 10. Containerize and demonstrate

Docker Compose starts the API, React dashboard, PostgreSQL, Redis, Qdrant,
Keycloak, and Mailpit. It runs the Alembic migration before the API, and
mounts persistent volumes for application data and service state. Run:

```powershell
docker compose up --build
docker compose exec api rag-pipeline-seed
```

GitHub Actions runs Python and frontend tests, lint, and build checks; it then
validates Compose and builds the API image. It does not publish or deploy the
image.
With Compose's Qdrant and PostgreSQL settings present, shared retrieval uses
Qdrant and the PostgreSQL chunk catalog; native Python without both settings
uses local FAISS/BM25. PostgreSQL-backed conversation checkpoints are used when
`DATABASE_URL` is configured; local operation uses in-memory checkpoints.
Redis backs API rate limits and optional stateless answer caching. Production
requires it; Compose configures the service automatically. Native development
can leave `REDIS_URL` blank unless a Redis server is running—the VS Code
extension is only a client. `/readyz` checks configured PostgreSQL, Qdrant,
and Redis services, and reports sanitized dependency status. Before rebuilding
the index, run `python -m rag_pipeline.reindex --dry-run` to verify every
archived source digest without writing to the index. Workspace authorization,
invitation endpoints, privileged-access auditing, and their React dashboard
workflows are implemented, but must be exercised against the full local stack.
OpenTelemetry and deployment automation remain future work; Kubernetes is out
of scope.
For native local development, leave `QDRANT_URL` blank; Compose configures its
internal Qdrant endpoint. The API manages the configured Qdrant client with its
application lifespan. `QDRANT_VECTOR_SIZE` defaults to 1536 and must match
the embedding model. Docker runtime validation is required before claiming
the Compose service integration has been exercised end to end.

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
- [ ] Frontend tests, lint, and production build pass.
- [ ] Local Compose sign-in, invitation, upload/chat, and two-workspace
      isolation scenarios pass before describing the integration as verified.
- [ ] Production use additionally has HTTPS OIDC, PostgreSQL, durable storage,
      secret management, observability, recovery, and a tested deployment target.
