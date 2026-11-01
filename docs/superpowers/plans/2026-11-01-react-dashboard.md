# React Multi-user Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Streamlit with an accessible React/TypeScript dashboard that supports independent Keycloak users and multiple workspace contexts.

**Architecture:** Build a static React SPA with Vite and retain FastAPI as the only application API. Authenticate using Keycloak Authorization Code + PKCE, keep tokens in memory, and attach bearer tokens to server-authorized API calls. The UI depends on the workspace API and durable conversation contracts in the preceding plans.

**Tech Stack:** React, TypeScript, Vite, Keycloak OIDC browser client, a tested component/styling library, Vitest/Testing Library, Docker Compose.

**Spec:** [2026-11-01-multiuser-web-dashboard-design.md](../specs/2026-11-01-multiuser-web-dashboard-design.md)

## Global Constraints

- Keep FastAPI/Python as the application API; do not rewrite RAG or auth in Node.
- Use Keycloak Authorization Code + PKCE S256.
- Never persist access or refresh tokens in localStorage or sessionStorage.
- Treat server authorization as authoritative; UI hiding is not access control.
- Support workspace switching from API-provided memberships and effective roles.
- Preserve ask, compare, ingest, evidence/citation, and per-user key-management workflows.
- Use responsive/accessibility rules and the installed UI/UX Pro Max skill; do not add dependencies before compatibility review.
- No Kubernetes. Render deployment validation comes after local end-to-end validation.

## Review Focus

- Expired, rejected, or revoked token while requests are in flight must return to a safe sign-in state; test refresh failure and 401 handling.
- User reload or browser restart must restore only through Keycloak SSO, not recover a token from browser storage; test storage is never written.
- Workspace switch must clear or reload workspace-specific thread/document data; test stale results cannot display after switching.
- API/network failure during upload or ask must be visible and allow retry without accidental duplicate submissions; test loading/error states.
- Keyboard-only and narrow/mobile viewport users must access core flows and dialogs; test focus, labels, contrast-relevant states, and responsive layout.

---

### Task 1: Scaffold the React application and local developer checks

**Files:**
- Create: `web/package.json`
- Create: `web/package-lock.json`
- Create: `web/index.html`
- Create: `web/vite.config.ts`
- Create: `web/tsconfig.json`
- Create: `web/src/main.tsx`
- Create: `web/src/App.tsx`
- Test: `web/src/App.test.tsx`
- Modify: `.github/workflows/tests.yml`

**Interfaces:**
- Produce: standalone `web` commands `npm run dev`, `npm run test`, `npm run lint`, and `npm run build`.

- [ ] **Step 1: Create the smallest app scaffold and failing smoke test** asserting the unauthenticated sign-in view renders.
- [ ] **Step 2: Run** `npm --prefix web test`; confirm the new test fails before the app is implemented.
- [ ] **Step 3: Implement** React/Vite/TypeScript setup and test tooling with pinned lockfile; do not alter Python API dependencies.
- [ ] **Step 4: Run** web tests, lint, and production build; verify no Streamlit runtime is required to build the SPA.
- [ ] **Step 5: Add** Node setup/cache and frontend test/build steps to CI; keep the Python 3.11 test job and make image build depend on both jobs.
- [ ] **Step 6: Commit** the scaffold, lockfile, and CI changes.

### Task 2: Implement Keycloak OIDC and authenticated API client

**Files:**
- Create: `web/src/auth/AuthProvider.tsx`
- Create: `web/src/auth/authConfig.ts`
- Create: `web/src/api/client.ts`
- Create: `web/src/api/client.test.ts`
- Modify: `deploy/keycloak/hybrid-rag-realm.json`
- Modify: `.streamlit/secrets.toml.example`
- Modify: `web/src/App.tsx`

**Interfaces:**
- `AuthProvider` exposes `user`, `accessToken`, `status`, `login()`, and `logout()`.
- `apiRequest<T>(path, init, accessToken, workspaceId?) -> Promise<T>` uses bearer auth and optional `X-Workspace-ID`; it never logs credentials.

- [ ] **Step 1: Write failing tests** for PKCE config, memory-only tokens, bearer headers, workspace header, 401/403 handling, and safe network errors.
- [ ] **Step 2: Run** `npm --prefix web test -- AuthProvider client`; confirm tests fail.
- [ ] **Step 3: Implement** a maintained Keycloak browser client with Authorization Code + PKCE S256. Configure local Vite origin/callback and production origin settings; restore reload sessions via Keycloak SSO; never use browser storage for tokens. Remove the obsolete Streamlit OIDC example after migration.
- [ ] **Step 4: Run** auth/API client tests; verify URL and client ID are environment-configured and no token appears in logs/storage.
- [ ] **Step 5: Commit** auth and request client.

### Task 3: Build workspace-aware application shell

**Files:**
- Create: `web/src/workspaces/WorkspaceProvider.tsx`
- Create: `web/src/workspaces/WorkspaceSwitcher.tsx`
- Create: `web/src/layout/AppShell.tsx`
- Create: `web/src/routes.tsx`
- Test: `web/src/workspaces/WorkspaceProvider.test.tsx`

**Interfaces:**
- Workspace context contains only API-returned memberships and selected workspace ID; selection changes add the ID to each request and invalidate workspace-scoped UI data.

- [ ] **Step 1: Write failing tests** for membership loading, invalid workspace selection, switching context, and clearing previous workspace data.
- [ ] **Step 2: Run** `npm --prefix web test -- WorkspaceProvider`; confirm workspace switching tests fail.
- [ ] **Step 3: Implement** authenticated shell, route guards, workspace switcher, navigation, and role-aware menus.
- [ ] **Step 4: Run** tests and check keyboard navigation/focus management for navigation and workspace controls.
- [ ] **Step 5: Commit** the shell and workspace state.

### Task 4: Implement dashboard workflows and responsive UI

**Files:**
- Create: `web/src/features/chat/`
- Create: `web/src/features/conversations/`
- Create: `web/src/features/documents/`
- Create: `web/src/features/account/`
- Create: `web/src/features/workspace-admin/`
- Create: `web/src/styles/`
- Test: `web/src/features/**/*.test.tsx`

**Interfaces:**
- Consume API contracts for conversation CRUD, `/v1/ask`, `/v1/compare`, `/v1/ingest`, `/v1/documents`, `/v1/account/openai-key`, and workspace membership/admin endpoints.

- [ ] **Step 1: Write failing component tests** for chat/thread isolation, ask/compare evidence rendering, document upload/role assignment, private-key save/delete, and role-gated workspace admin screens.
- [ ] **Step 2: Run** `npm --prefix web test`; confirm behavior tests fail.
- [ ] **Step 3: Implement** the core dashboard and a consistent design system using the installed UI/UX design skills; include loading, empty, error, and success states.
- [ ] **Step 4: Run** frontend tests plus accessibility and responsive checks at 375, 768, 1024, and 1440 px; ensure focus visibility and no horizontal overflow.
- [ ] **Step 5: Commit** dashboard workflows and UI tests.

### Task 5: Replace Streamlit runtime and containerize the SPA

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `.env.example`
- Modify: `pyproject.toml`
- Delete after parity: `src/rag_pipeline/ui.py`
- Modify: `README.md`
- Modify: `docs/PROJECT_RECREATION_GUIDE.md`

- [ ] **Step 1: Add integration configuration tests** for frontend/API origin, OIDC issuer/client ID, exact CORS origins, and a production SPA build.
- [ ] **Step 2: Run** web tests and `docker compose --env-file .env.example config --quiet`; verify frontend configuration does not contain secrets.
- [ ] **Step 3: Implement** multi-stage frontend build and a static file server/container; replace Streamlit service while retaining local Compose API/identity dependencies.
- [ ] **Step 4: Run** full Python/frontend suites and a Compose smoke test: register/sign in two users, switch workspaces, ask/ingest, and verify isolation. Remove Streamlit dependency/configuration only after parity passes.
- [ ] **Step 5: Commit** container/runtime changes and setup documentation.
