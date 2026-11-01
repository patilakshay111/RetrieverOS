# Workspace Authorization and Admin API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist workspaces, memberships, and invitations; authorize every request against the selected workspace; and audit privileged administration.

**Architecture:** Keep Keycloak as identity and PostgreSQL as the source of truth for workspace membership and workspace-scoped roles. Derive role claims on the server for the selected `X-Workspace-ID`; never trust a workspace or role sent only by the browser.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, Alembic, PostgreSQL, PyJWT, pytest.

**Spec:** [2026-11-01-multiuser-web-dashboard-design.md](../specs/2026-11-01-multiuser-web-dashboard-design.md)

**Order:** Implement before [durable-rag-state](2026-11-01-durable-rag-state.md)
and [react-dashboard](2026-11-01-react-dashboard.md); both consume its
server-derived workspace scope and membership API.

## Global Constraints

- Production identity is Keycloak OIDC over HTTPS; API-key authentication remains local-development-only.
- Open registration provisions an isolated personal workspace.
- Accounts may belong to multiple workspaces; workspace admins manage only their workspace and assign existing roles.
- Only platform admins define global roles; cross-workspace access is audited.
- Never trust workspace IDs or roles supplied only by the browser.
- Do not log prompts, document contents, answers, tokens, or provider credentials.
- Do not add Kubernetes configuration.

## Review Focus

- A valid token with a forged workspace header must be denied; test unknown and non-member workspace IDs.
- A workspace admin must not manage another workspace; test all member/invite/role endpoints cross-scope.
- A platform admin may cross scopes only with an audit event; test success and durable audit details without content/secrets.
- Open registration must create exactly one private workspace and initial membership; test idempotency/retry behavior.
- Membership removal or role changes must take effect on the next request; test authorization against current persisted membership, not stale claims.

---

### Task 1: Persist workspace membership and roles

**Files:**
- Modify: `src/rag_pipeline/credentials.py` (shared SQLAlchemy metadata only if still the established model location; otherwise create focused `src/rag_pipeline/identity_store.py`)
- Create: `migrations/versions/0002_workspace_membership.py`
- Modify: `migrations/env.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: verified immutable Keycloak `AccessContext.user_id`.
- Produces: `WorkspaceMembership(user_id, workspace_id, roles, status)` persistence and repository methods `list_memberships(user_id)`, `get_membership(user_id, workspace_id)`, `create_personal_workspace(user_id)`.
- Personal-workspace provisioning is idempotent and runs when a verified new
  subject first reaches an authenticated API route.

- [ ] **Step 1: Write failing persistence tests** for multiple memberships, unique `(user_id, workspace_id)`, workspace creation, role assignment, and membership status.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k workspace_membership`; confirm the new cases fail.
- [ ] **Step 3: Implement** typed SQLAlchemy models/repository and an Alembic revision; keep credential storage behavior unchanged.
- [ ] **Step 4: Run** focused tests and `alembic upgrade head` against a temporary SQLite database; confirm the tables and uniqueness constraints.
- [ ] **Step 5: Commit** the migration, repository, and tests together.

### Task 2: Resolve active workspace from persisted membership

**Files:**
- Modify: `src/rag_pipeline/models.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/auth.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: signed `AccessContext`, optional `X-Workspace-ID`, and Task 1 membership repository.
- Produces: `WorkspaceAccessContext(user_id, workspace_id, roles, platform_admin)` returned from an API dependency; no endpoint accepts client-supplied roles as authorization.

- [ ] **Step 1: Write failing API tests** for personal-workspace default, valid member workspace selection, unknown workspace, non-member workspace, and role changes taking effect without token refresh.
- [ ] **Step 2: Run** `python -m pytest tests/test_core.py -k workspace_scope`; confirm denied scopes currently fail the expected assertions.
- [ ] **Step 3: Implement** server-side workspace resolution. Use the user's personal-workspace default when no header is supplied; when a header is supplied, require a current membership unless the caller is a platform admin. Platform-admin access to a workspace other than the user's own must be audited. Resolve roles from PostgreSQL, not the workspace claim or client input.
- [ ] **Step 4: Run** focused tests plus existing OIDC/ACL tests; confirm legacy local-development auth remains functional.
- [ ] **Step 5: Commit** scope resolution and regression tests.

### Task 3: Add workspace membership and invitation endpoints

**Files:**
- Create: `src/rag_pipeline/workspaces.py`
- Create: `src/rag_pipeline/email.py`
- Create: `migrations/versions/0003_workspace_invitations.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/models.py`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: `GET /v1/workspaces`, `GET/POST /v1/workspaces/{workspace_id}/members`, `PUT/DELETE /v1/workspaces/{workspace_id}/members/{user_id}`, `POST /v1/workspaces/{workspace_id}/invitations`, authenticated `GET /v1/roles`, and platform-admin-only `POST /v1/roles`.
- Invitation acceptance must bind the authenticated Keycloak subject and
  verified email to a single-use, expiring invitation. Email uses configured
  SMTP; local Compose routes to Mailpit. Role values must be existing
  administrator-approved global roles.
- `POST /v1/invitations/{token}/accept` is the invitation acceptance endpoint.

- [x] **Step 1: Write failing tests** for listing only caller memberships, platform-admin workspace listing, workspace-admin invite/remove/role assignment, employee denial, platform-admin-only role creation, role validation, SMTP invitation delivery, authenticated invitation acceptance, expired/reused/email-mismatch invitation rejection, and cross-workspace denial.
- [x] **Step 2: Run** focused workspace/OIDC tests; confirm endpoint authorization tests fail before route registration.
- [x] **Step 3: Implement** typed request/response models, routes, and an SMTP email adapter. Workspace admins may administer only their own workspace; only platform admins may define global role names. Store only a hash of each invitation token.
- [x] **Step 4: Run** focused tests and API OpenAPI generation; confirm no route trusts an unverified workspace ID.
- [ ] **Step 5: Commit** route, service, and endpoint tests.

### Task 4: Audit privileged workspace access

**Files:**
- Create: `src/rag_pipeline/audit.py`
- Create: `migrations/versions/0004_admin_audit_events.py`
- Modify: `src/rag_pipeline/workspaces.py`
- Modify: `src/rag_pipeline/api.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces: append-only `AuditEvent` records with actor subject, action, workspace target, timestamp, and request correlation ID; exclude prompt/document/answer/key/token values.

- [x] **Step 1: Write failing tests** asserting every platform-admin cross-workspace API access (document list, ingest, ask/compare, member/role operations) writes an audit event; same-workspace access does not create a false cross-workspace event; event records contain no sensitive content.
- [x] **Step 2: Run** focused audit tests; confirm they fail before request IDs and audit event persistence are implemented.
- [x] **Step 3: Implement** the audit repository and write events in every cross-workspace request path, including reads and RAG operations, not only admin mutations.
- [x] **Step 4: Run** focused audit and authorization tests; verify persisted audit rows through a separate database connection.
- [ ] **Step 5: Commit** audit migration, service, and tests.

### Task 5: Verify the authorization API contract

**Files:**
- Modify: `tests/test_core.py`
- Modify: `README.md`
- Modify: `docs/PROJECT_RECREATION_GUIDE.md`

- [ ] **Step 1: Add regression tests** for same user across two workspaces, two users in one workspace, role removal, platform-admin audit, and request without OIDC in production.
- [ ] **Step 2: Run** `python -m pytest`; confirm all authorization tests pass.
- [ ] **Step 3: Document** workspace selection header, registration, invitation acceptance, and admin boundaries.
- [ ] **Step 4: Run** `git diff --check` and the full test suite; confirm docs match implemented routes.
- [ ] **Step 5: Commit** contract docs and tests.
