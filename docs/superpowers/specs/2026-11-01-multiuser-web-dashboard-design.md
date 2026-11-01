# Multi-user Web Dashboard and Production Readiness Design

## Intent

Replace the Streamlit user interface with a browser-based React/TypeScript
dashboard so multiple users can use the product through independent browser
sessions. Keep the existing Python FastAPI service and RAG implementation,
which already own authentication verification, document APIs, access control,
and per-user encrypted model-key endpoints.

Continue the existing local-first production-readiness effort. Docker Compose
remains the local integration environment. Do not add Kubernetes. Once the
implementation is complete and validated locally, test the deployment on
Render as requested; do not claim Render compatibility before that test.

## Goals

- Replace Streamlit with a responsive, accessible, multi-user web dashboard.
- Preserve the current FastAPI/RAG service and authenticated API contracts
  unless secure multi-user behavior requires extending them.
- Support Keycloak sign-in (email/password and Google through configured
  Keycloak identity provider) and enforce the signed user's workspace/role
  permissions throughout the UI and API.
- Provide user workflows for document ingestion and access roles, grounded
  question answering, hybrid-versus-dense comparison, evidence/citation review,
  conversation threads, and private OpenAI-key management.
- Add the missing persistence and service-side authorization needed for
  authenticated multi-user operation and safe restarts: PostgreSQL-backed
  domain/checkpoint state, Qdrant-filtered retrieval, Redis-scoped caching and
  rate limiting, auditable administrative access, redacted tracing, and
  security/evaluation gates.
- Keep local development reproducible with Docker Compose and CI; defer the
  Render deployment test until implementation and local validation are done.
- Keep the planned test cohort at 100 registered accounts. Define concurrency,
  request rate, and latency objectives before making capacity claims.
- Allow an account to belong to multiple workspaces and switch active workspace.
  Workspace admins manage memberships and assign existing roles only inside
  their own workspace; only platform admins define global role names.
  Platform-admin access to other workspaces must be audited.

## Non-goals

- Rewriting RAG, identity verification, or document APIs in Node.js.
- Kubernetes or Helm configuration.
- Claiming production readiness, Render operation, or capacity for 100
  concurrent users / requests per second without measured evidence.
- Storing OIDC access or refresh tokens in browser local storage.
- Sending document text, prompts, answers, tokens, or provider credentials to
  logs or trace attributes.

## Current state and constraints

- The API is FastAPI/Python and already verifies Keycloak RS256 bearer tokens,
  applies workspace/role document ACLs, scopes LangGraph thread IDs by user and
  workspace, and saves each user's OpenAI key encrypted in PostgreSQL.
- Registration is open and each new account starts with a private personal
  workspace. A user can also join multiple shared workspaces.
- Streamlit carries UI state in its session model and performs direct OIDC
  login. It is the only dashboard implementation; there is no existing
  JavaScript app or package manifest.
- API persistence is incomplete: PostgreSQL currently stores credentials only,
  LangGraph uses an in-memory checkpointer, and retrieval is a local FAISS
  snapshot. Redis and Qdrant start in Compose but are not used in request
  handling. The raw source archive is on the application data volume, so an
  initial Render test must keep a durable disk attached to one API service;
  horizontal API scaling requires a shared object-storage backend first.
- Workspace memberships, invitations, and role administration are managed
  manually in Keycloak. There is no application audit trail, OpenTelemetry
  instrumentation, or production-ready invitation/admin API.
- Production API configuration requires HTTPS OIDC, PostgreSQL, and a
  credential-encryption key. Current CORS configuration is a fixed list and
  must support exact configured frontend origins, not wildcards.
- Docker Engine has not been available in the prior work environment, so
  Compose has not been smoke-tested there.
- Render account credentials, Google OAuth credentials, and production mail
  provider settings are not available in the repository. They must be
  configured by the owner for the eventual Render test.

## Approaches considered

1. **React + TypeScript + Vite SPA with the existing FastAPI backend
   (selected).** Build a static browser app and keep RAG, API authorization,
   and OIDC token verification in FastAPI. Use Keycloak's browser OIDC client
   with Authorization Code + PKCE. This replaces the Streamlit UI while
   limiting the migration surface and avoiding a Node server for a static
   dashboard.
2. **Next.js frontend plus FastAPI backend.** Adds SSR/server-runtime options
   and a Node deployment service. Those capabilities are not required by the
   current authenticated internal dashboard; this adds hosting and session
   complexity without removing the Python API.
3. **Rewrite the backend and dashboard in Node.js.** Rejected because it
   duplicates or ports the existing RAG, retrieval, ingestion, authorization,
   and evaluation implementation. It increases migration risk and does not
   address the actual UI/session limitation better than a separate SPA.

## Selected design

### Browser application

- Add a standalone React + TypeScript + Vite app, with its own package
  manifest, lockfile, lint/test/build commands, and Docker build target.
- Use the `ui-ux-pro-max` and relevant design skills for a consistent dashboard
  design system, then implement the dashboard with accessible, responsive
  components appropriate to the chosen React stack. Avoid assuming that the
  existing Python project requires Tailwind or shadcn; select and document the
  smallest compatible component/styling setup during implementation.
- Use Keycloak Authorization Code flow with PKCE S256. Keep access/refresh
  tokens in the OIDC client’s in-memory session; do not persist tokens to
  localStorage or sessionStorage. Attach the access token as a bearer token to
  API requests. Handle expiry, refresh, sign-out, and unauthorized API
  responses explicitly. After a reload, restore the session through the
  Keycloak SSO session; if it is no longer valid, require sign-in again.
- Provide an explicit workspace switcher for accounts with multiple
  memberships. Persist only non-sensitive UI preferences client-side; obtain
  the membership list and effective roles from the API.
- Include authenticated views for chat, conversation history, document
  library/ingestion, citations/evidence, retrieval comparison, account API-key
  controls, and workspace administration based on the verified role.
- Keep each browser user's current thread and UI state isolated. The backend,
  not client-side visibility, remains authoritative for every workspace,
  document, and role permission.

### API and persistent state

- Retain FastAPI as the only application API. Add typed endpoints needed by
  the dashboard for conversation listing/creation, workspace membership,
  invitations/roles, and administrative actions. Authenticate and authorize
  every request; do not trust workspace IDs or roles supplied only by the
  browser.
- Store workspace membership and workspace-scoped roles in PostgreSQL. Treat
  the Keycloak subject as the immutable identity; derive effective roles from
  the selected workspace membership on the server. A client-selected workspace
  identifier is only a request to switch scope and must be checked against the
  caller's membership on every request.
- Open registration provisions an isolated personal workspace. Workspace
  administrators can invite/remove members and assign existing HR/CEO/employee
  roles in their own workspace. Platform administrators alone manage global
  role definitions and may access all workspaces; cross-workspace access is
  audited.
- Add PostgreSQL migrations and domain models for workspaces, memberships,
  invitations (or a documented Keycloak-backed alternative), conversations,
  and audit events. Continue using the separate Fernet key for per-user
  provider credentials.
- Replace process-local LangGraph checkpoint storage with a supported
  PostgreSQL checkpointer. Scope checkpoint identifiers by immutable user,
  workspace, and conversation ID.
- Replace local FAISS request retrieval with Qdrant collections/payloads
  containing workspace, owner, and required-role metadata. Apply Qdrant-side
  filters before returning candidates; preserve equivalent ACL filtering for
  BM25 or any hybrid lexical candidate source. Add migration/reindex tooling
  and test cross-user/workspace/role isolation.
- Integrate Redis for rate limiting and authorization-scoped answer caching.
  Cache keys must include the complete effective authorization scope and
  relevant corpus/index version; never allow results to cross ACL boundaries.
- Add an auditable event for platform-admin cross-workspace access and other
  privileged workspace administration.

### Security, evaluation, and operations

- Add redacted OpenTelemetry traces/metrics for request and dependency
  boundaries. Do not record user prompts, document contents, answers, auth
  tokens, or API keys. Make exporter, sampling, retention, and deployment
  configuration explicit. Until retention/export policy is selected, keep
  collection opt-in and avoid enabling an external exporter by default.
- Add deterministic tests for API and UI auth flows, private-key isolation,
  workspace and document ACLs, cache isolation, checkpoint isolation, and
  prompt-injection boundaries. Run offline tests and build checks on every CI
  change. Keep model-judge evaluations opt-in or separately gated until stable,
  non-flaky thresholds and model-cost handling are reviewed.
- Update Docker Compose to run the SPA, API, PostgreSQL migrations, Keycloak,
  Redis, Qdrant, and Mailpit locally. Remove Streamlit from the runtime path
  after feature parity and auth behavior are validated.
- Prepare an eventual Render test using externally configured HTTPS OIDC,
  database/vector/cache services, secrets, and real public callback/origin
  URLs. Do not put credentials in the repository. Do not deploy or run that
  test until local validation is complete and the owner provides/sets the
  required Render and identity-provider configuration.
- Do not create Kubernetes manifests or a cluster deployment path.
- Do not claim horizontal scaling while uploaded raw documents depend on a
  service-local disk; use a single API service for the initial Render test.

## Acceptance criteria

- A browser user can sign in, sign out, use the dashboard, refresh/expire a
  session safely, and manage their own key without seeing another user's data.
- Multiple independently authenticated users can use separate conversations
  and workspace/document scopes without cross-user state or evidence leakage.
- The UI provides the existing ask, compare, ingest, evidence/citation, and
  key-management behavior, plus approved workspace/admin operations.
- Conversation state survives an API process restart and is separated by
  user/workspace/conversation.
- Vector search enforces access filters server-side. Hybrid retrieval cannot
  expose unauthorized lexical or dense evidence.
- All API routes validate authorization on the server; administrative
  cross-workspace access is audited.
- Cache/rate-limit behavior is isolated by authenticated identity and effective
  access scope.
- CI validates Python and frontend tests, security-sensitive isolation
  scenarios, migrations, Compose configuration, and container builds.
- Local Compose runs and is smoke-tested end-to-end before a Render test.
- Render validation is documented with measured results and actual limitations;
  no Kubernetes path is included.

## Implementation decisions left to the builder

- Choose a maintained Keycloak browser client and compatible React component
  and styling libraries; do not add dependencies until the app scaffold and
  support status are checked.
- Choose appropriate PostgreSQL LangGraph checkpointer and Qdrant client APIs
  compatible with current dependency versions.
- Choose a secure Keycloak SSO session restoration configuration consistent
  with in-memory browser tokens and the deployed HTTPS origins.
- Design backward-compatible Alembic revisions and reindex/migration behavior
  for existing FAISS corpora; preserve local dev/test support where practical.
- Prefer smallest complete UI design; expose capabilities only when supported
  by server-side APIs.
- Resolve exact Render topology and external service choices during the
  post-implementation deployment test, with user-provided account settings.

## Explicitly unresolved before capacity testing

- Peak concurrent active users and request rate for a cohort of 100 registered
  users.
- Latency/error-rate targets and the expected upload/query mix.
- Production SMTP provider and retention/export settings for observability.
