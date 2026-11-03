# Workspace Onboarding and Permissions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace implicit personal-workspace provisioning with named workspace onboarding and add workspace-scoped, server-enforced roles and document permissions.

**Architecture:** Keep FastAPI, SQLAlchemy/Alembic, and the React SPA. Add workspace-local role definitions with an allowlisted permission set, resolve permissions from current PostgreSQL membership for each request, and keep document ACLs scoped to their workspace. Let identity-only endpoints list/create workspaces and accept invitations without fabricating a workspace; workspace-scoped endpoints continue to require authorized membership.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, SQLAlchemy, Alembic, PostgreSQL/SQLite, React, TypeScript, Vitest, pytest.

**Spec:** [2026-11-02-workspace-onboarding-and-permissions-design.md](../specs/2026-11-02-workspace-onboarding-and-permissions-design.md)

## Global Constraints

- A new authenticated user with no workspace membership sees a create-workspace flow that asks for a name and description. Creating it makes that user its initial administrator.
- A user who accepts an invitation lands in that invited workspace first, including when it is their first sign-in. They may create another workspace later.
- Workspace administrators can edit workspace name and description.
- Workspace administrators can create roles scoped to their workspace and select supported permissions. Role creation, invitation, membership administration, and workspace editing remain admin-only.
- Invitation acceptance requires the invitee's verified OIDC email to match the invitation.
- A member with multiple roles has the union of their capabilities. A document is visible/queryable only when its ACL intersects the member's workspace roles and the member has the corresponding capability.
- On upload, allow selecting multiple roles. The API validates that every ACL role belongs to the current workspace and that a non-admin uploader may assign only roles they hold. Admins may assign any role in that workspace.
- Keep existing automatically created personal workspaces as existing workspaces so their documents and conversations are not orphaned. Stop creating new ones.
- Map the built-in `admin` role to full capabilities. Map existing non-admin roles to document read/query/upload capabilities to preserve existing behavior; do not grant update capability automatically.
- Never trust role names, permissions, workspace IDs, or ACL selections sent by the browser; validate them against authenticated membership and persisted workspace role definitions.
- Keep workspace IDs in every document query and mutation. Role names are not globally meaningful.
- Preserve platform-admin authorization as a separate mechanism, including required audit logging for cross-workspace access.
- Reject unknown permission keys and attempts by custom roles to grant administrative capabilities.
- Keep invitation tokens out of query strings and logs.

## Review Focus

- A first-time invitee must accept an invitation without an existing workspace or `X-Workspace-ID`, then land in that exact workspace; test in backend API and frontend integration.
- A user with no memberships must not gain a workspace merely by calling a protected route; test authorization and workspace listing independently.
- Identically named roles in two workspaces must not share permission definitions or grant cross-workspace document access; test store, API, and retrieval boundaries.
- A forged ACL role or permission in an upload/update request must fail before storage mutation or embedding; test API and ingestion.
- Legacy database migration must preserve existing memberships, invitation roles, indexed ACLs, and workspace data while safely mapping role permissions; test populated and empty migration paths.

---

### Task 1: Persist Named Workspaces and Workspace-Scoped Roles

**Files:**
- Modify: `src/rag_pipeline/identity_store.py`
- Modify: `migrations/versions/0007_workspace_local_permissions.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces `WorkspaceRecord(id, name, description, personal_owner_id, created_at)`.
- Produces `WorkspaceRoleRecord(workspace_id, name, description, permissions, created_by, created_at)`.
- Preserve `WorkspaceStore.create_workspace(name: str, workspace_id: str | None = None) -> WorkspaceRecord` for existing callers.
- Add `WorkspaceStore.create_workspace_for_user(user_id: str, name: str, description: str = "") -> tuple[WorkspaceRecord, WorkspaceMembershipRecord]` to create a named workspace and initial admin membership atomically.
- `WorkspaceStore.update_workspace(workspace_id: str, name: str, description: str) -> WorkspaceRecord | None`.
- `WorkspaceStore.list_workspace_roles(workspace_id: str) -> list[WorkspaceRoleRecord]`.
- `WorkspaceStore.create_workspace_role(workspace_id: str, name: str, description: str, permissions: Iterable[str], created_by: str) -> WorkspaceRoleRecord`.
- `WorkspaceStore.get_role_permissions(workspace_id: str, roles: Iterable[str]) -> frozenset[str]`.
- Consumes: Existing `Workspace`, `WorkspaceMembership`, `WorkspaceInvitation`, and `RoleDefinition` data and ACL role-name strings.
- Existing native SQLite databases must be upgraded with `alembic upgrade head`; `Base.metadata.create_all()` does not alter existing tables.

- [ ] **Step 1: Add failing store/migration tests**

Add tests named `test_workspace_creation_creates_named_workspace_and_admin_membership`,
`test_workspace_update_changes_name_and_description`,
`test_roles_are_scoped_by_workspace_and_permissions`,
`test_workspace_local_role_creation_rejects_unknown_permissions`, and
`test_workspace_permission_migration_preserves_existing_data`.
Assert:
- Creating a workspace stores its name/description and creates only its creator's active `admin` membership.
- A workspace cannot read or assign a role definition from another workspace, even when the role names match.
- Unknown capability names are rejected.
- Existing workspace names, memberships, invitations, and ACL role names remain intact after migration.
- Legacy `admin` gets full capabilities; legacy non-admin roles get `documents:read`, `documents:query`, and `documents:upload`, but not `documents:update`.

- [ ] **Step 2: Run focused tests and verify the new tests fail**

Run: `python -m pytest tests/test_core.py -k "workspace_creation_creates_named or workspace_update_changes or roles_are_scoped or workspace_local_role_creation or workspace_permission_migration" -q`
Expected: FAIL because the store and workspace-local permission schema do not exist yet.

- [ ] **Step 3: Implement workspace and role persistence**

Add workspace descriptions and a `workspace_roles` table keyed by `(workspace_id, name)` with description and JSON permission keys. Keep the existing global `role_definitions` table as migration source/legacy data; seed each existing workspace from its legacy definitions, mapping permissions exactly as specified above. New workspaces receive workspace-local `admin`, `ceo`, `employee`, and `hr` definitions; `admin` is reserved and full-access, while the other built-ins receive read/query/upload only. Create the workspace and creator membership in one transaction. Validate role permissions against the fixed allowlist.

- [ ] **Step 4: Run focused store and migration tests**

Run: `python -m pytest tests/test_core.py -k "workspace_creation_creates_named or workspace_update_changes or roles_are_scoped or workspace_local_role_creation or workspace_permission_migration" -q`
Expected: PASS for empty SQLite stores and populated legacy migration fixtures.

### Task 2: Implement No-Default Workspace Onboarding and Invitation-First Selection

**Files:**
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/workspaces.py`
- Modify: `src/rag_pipeline/identity_store.py`
- Modify: `web/src/App.tsx`
- Modify: `web/src/workspaces/WorkspaceProvider.tsx`
- Test: `tests/test_core.py`
- Test: `web/src/App.test.tsx`
- Test: `web/src/workspaces/WorkspaceProvider.test.tsx`

**Interfaces:**
- Consumes Task 1 `WorkspaceStore.create_workspace` and `WorkspaceStore.update_workspace`.
- `POST /v1/workspaces` accepts `{name, description}` and returns the created workspace plus the caller's admin membership.
- `PATCH /v1/workspaces/{workspace_id}` accepts `{name, description}` and returns updated workspace metadata; only workspace admins may call it.
- `GET /v1/workspaces` works for authenticated identities with zero memberships and never creates one.
- `POST /v1/invitations/accept` remains identity-authenticated without requiring `X-Workspace-ID` and returns `workspace_id`.
- Workspace-specific routes derive the target from `workspace_id` and verify current membership; retrieval/document routes require an authorized active workspace.
- Invitation acceptance tells `WorkspaceProvider` which workspace to select after refreshing membership.

- [ ] **Step 1: Add failing API and UI tests**

Add tests named `test_workspace_list_does_not_provision_workspace`,
`test_workspace_creation_requires_identity_and_creates_admin`,
`test_workspace_update_requires_admin`,
`test_invitation_acceptance_works_without_workspace_header`, and
`test_invitation_acceptance_adds_roles_and_reactivates_membership`.
Add frontend assertions that a zero-membership account sees workspace creation,
submitting it creates and loads the named workspace, and accepting an invitation
selects the response's `workspace_id` even when another membership already exists.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_core.py -k "workspace_list_does_not_provision or workspace_creation_requires_identity or workspace_update_requires_admin or invitation_acceptance_works_without or invitation_acceptance_adds_roles" -q`
Run: `npm test -- App.test.tsx WorkspaceProvider.test.tsx`
Expected: New assertions fail because authorization currently auto-creates a personal workspace and invitation acceptance does not select its workspace.

- [ ] **Step 3: Implement identity-only workspace operations and no implicit workspace creation**

Remove `create_personal_workspace` from request authorization. Use authenticated-identity dependencies for workspace listing, creation, and invitation acceptance; require a selected, active membership for workspace-scoped document/chat operations. Add workspace create/update request/response models and admin checks. Preserve the existing invitation behavior: merge invited roles into an existing membership and reactivate it. Return the accepted `workspace_id`. Do not require `X-Workspace-ID` to accept an invitation.

- [ ] **Step 4: Implement the empty-membership and invitation-selected frontend flow**

Add a first-run form for workspace name and description. After creation, refresh memberships and select the created workspace. On invitation acceptance, use the response workspace ID as the selected workspace after refresh. Do not select or synthesize a default workspace when membership is empty. Keep the existing invitation fragment-token handling.

- [ ] **Step 5: Run focused backend and frontend tests**

Run: `python -m pytest tests/test_core.py -k "workspace_list_does_not_provision or workspace_creation_requires_identity or workspace_update_requires_admin or invitation_acceptance_works_without or invitation_acceptance_adds_roles" -q`
Run: `npm test -- App.test.tsx WorkspaceProvider.test.tsx`
Expected: PASS; a user without membership remains workspace-less until create/accept, and an invitee lands in the invited workspace.

### Task 3: Enforce Workspace Permissions and Multi-Role Document ACLs

**Files:**
- Modify: `src/rag_pipeline/models.py`
- Modify: `src/rag_pipeline/api.py`
- Modify: `src/rag_pipeline/workspaces.py`
- Modify: `src/rag_pipeline/identity_store.py`
- Modify: `src/rag_pipeline/storage.py`
- Modify: `src/rag_pipeline/lexical_store.py`
- Modify: `src/rag_pipeline/vector_store.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes Task 1 `get_role_permissions` and Task 2 authorized workspace context.
- Workspace access context carries the union of persisted role names and permissions for its active membership.
- Workspace role endpoints are `GET/POST /v1/workspaces/{workspace_id}/roles`; create input is `{name, description, permissions}`.
- Any active workspace member may list that workspace's role definitions for document ACL selection; only an admin may create them.
- `POST /v1/ingest` continues to use `roles` as a comma-separated form field and validates each role against the selected workspace.
- `POST /v1/ask` and `POST /v1/compare` require `documents:query`; `GET /v1/documents` requires `documents:read`; new-file ingestion requires `documents:upload`; replacing an existing source requires `documents:update`.
- Document query/list results require both the capability and intersection of the member's role names with the document ACL. Workspace isolation remains mandatory.
- `documents:update` authorizes re-upload/re-index of an existing source and replacement of its ACL; it does not grant access to unrelated workspaces or other roles' documents.

- [ ] **Step 1: Add failing permission and ACL tests**

Add tests named `test_workspace_admin_creates_workspace_role_permissions`,
`test_non_admin_cannot_create_roles_or_invite`,
`test_documents_read_and_query_require_capabilities`,
`test_ingest_requires_upload_and_replacement_requires_update`,
`test_ingest_accepts_multiple_workspace_roles`,
`test_ingest_rejects_role_from_another_workspace_before_embedding`, and
`test_role_permission_union_and_acl_intersection_are_enforced`.
Assert denied operations return 403 without embedding or storage writes, custom roles cannot grant admin permissions, users with multiple roles see only ACL-intersecting documents, and same-named roles remain isolated by workspace.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_core.py -k "workspace_admin_creates_workspace_role or non_admin_cannot_create_roles or documents_read_and_query_require or ingest_requires_upload or ingest_accepts_multiple_workspace or ingest_rejects_role_from_another or role_permission_union" -q`
Expected: FAIL where endpoints currently authorize by literal role names or ignore capability permissions.

- [ ] **Step 3: Add role-scoped API contracts and permission resolution**

Scope role listing/creation to a workspace; require the built-in admin role for role creation, invitations, workspace edits, and membership administration. Validate role names and permission keys against the current workspace. Populate request access context with the union of permission keys for active assigned roles; keep platform-admin elevation separate and audited.

- [ ] **Step 4: Enforce read/query/upload/update permissions**

Require `documents:read` for the inventory, `documents:query` for ask/compare and conversation history that may contain prior answers, `documents:upload` for new sources, and `documents:update` for replacement of an existing source. Apply role ACL intersection on list and retrieval paths. Require role ACL values to exist in the current workspace and, for non-admin uploads, be among the caller's assigned roles. Keep checks before embedding and persistence. Ensure all lexical/vector/document-catalog queries remain workspace-filtered.

- [ ] **Step 5: Run backend authorization, retrieval, and migration regression tests**

Run: `python -m pytest tests/test_core.py -k "workspace_admin_creates_workspace_role or non_admin_cannot_create_roles or documents_read_and_query_require or ingest_requires_upload or ingest_accepts_multiple_workspace or ingest_rejects_role_from_another or role_permission_union or document_acl or retrieval_excludes_documents or vector_store_delete_document" -q`
Expected: PASS with all existing ACL and cross-workspace tests still passing.

### Task 4: Build Permission-Aware Workspace and Document Controls

**Files:**
- Modify: `web/src/api/types.ts`
- Modify: `web/src/App.tsx`
- Modify: `web/src/workspaces/WorkspaceProvider.tsx`
- Modify: `web/src/layout/AppShell.tsx`
- Modify: `web/src/features/workspace-admin/WorkspaceAdminPage.tsx`
- Modify: `web/src/features/documents/DocumentsPage.tsx`
- Modify: `web/src/features/chat/ChatPage.tsx`
- Test: `web/src/App.test.tsx`
- Test: `web/src/workspaces/WorkspaceProvider.test.tsx`
- Test: `web/src/features/workspace-admin/WorkspaceAdminPage.test.tsx`
- Test: `web/src/features/documents/DocumentsPage.test.tsx`
- Test: `web/src/features/chat/ChatPage.test.tsx`

**Interfaces:**
- Consumes Task 2 workspace create/update and invitation contracts and Task 3 permission-aware workspace/role APIs.
- `Workspace` exposes `description`, `roles`, and the active member's effective `permissions`.
- `Role` exposes `name`, `description`, and `permissions`.
- `WorkspaceAdminPage` accepts the active workspace and uses workspace-scoped role endpoints.
- `DocumentsPage` accepts `workspaceId: string` and `permissions: string[]`, fetches role definitions from the active workspace, and limits non-admin ACL choices to roles the user holds. Its ACL control is a multi-select and upload is enabled only with `documents:upload`.
- `ChatPage` accepts `canQuery: boolean`; unauthorized users cannot submit ask/compare requests from the UI.

- [ ] **Step 1: Add failing React tests for workspace and permission workflows**

Test `App.test.tsx` for zero-workspace create flow and workspace description submission. Test `WorkspaceProvider.test.tsx` for invitation-selected workspace. Test `WorkspaceAdminPage.test.tsx` for admin-only role creation with permission choices, workspace rename/description editing, and role selection on invitation. Test `DocumentsPage.test.tsx` for multi-select ACL roles, hidden/disabled upload without permission, and replacement control only with update permission. Test `ChatPage.test.tsx` for query-disabled controls when the user lacks `documents:query`.

- [ ] **Step 2: Run focused frontend tests and verify failure**

Run: `npm test -- App.test.tsx WorkspaceProvider.test.tsx WorkspaceAdminPage.test.tsx DocumentsPage.test.tsx ChatPage.test.tsx`
Expected: FAIL because the UI has no workspace-creation flow, role permission editor, workspace editor, multi-role document selection, or capability gating.

- [ ] **Step 3: Implement workspace onboarding and admin UI**

Render workspace creation when membership is empty; collect name and description. Add admin-only workspace name/description editing and workspace-local role creation with the supported permission checkboxes. Display role permission summaries and assign the selected role(s) in invitations and membership updates.

- [ ] **Step 4: Implement permission-aware document and chat UI**

Use the effective permission list to gate document list, upload, replacement, chat, and compare actions. Replace the single document ACL select with an accessible multi-select populated by current workspace roles. Render a clear read-only or access-denied state when capabilities are absent; retain server authorization as authoritative.

- [ ] **Step 5: Run frontend tests, lint, and production build**

Run: `npm test -- App.test.tsx WorkspaceProvider.test.tsx WorkspaceAdminPage.test.tsx DocumentsPage.test.tsx ChatPage.test.tsx`
Run: `npm run lint`
Run: `npm run build`
Expected: all selected tests, TypeScript checks, lint, and build pass.

### Task 5: Document and Verify End-to-End Behavior

**Files:**
- Modify: `README.md`
- Modify: `docs/PROJECT_RECREATION_GUIDE.md`
- Test: `tests/test_core.py`
- Test: `web/src/App.test.tsx`

**Interfaces:**
- Consumes the shipped API and UI contracts from Tasks 1–4.
- Documentation describes first workspace creation, workspace switching, admin role creation and permissions, invitation verification/acceptance through local Mailpit, document ACL multi-select, and the difference between workspace roles and Keycloak platform-admin.

- [ ] **Step 1: Add end-to-end regression tests for the invite-to-workspace path**

Add a backend flow test that creates a workspace, creates a custom `hr` role with `documents:read` and `documents:query`, sends an invitation, accepts it with the matching verified identity without a workspace header, and confirms the member's permissions and workspace ID. Add frontend coverage asserting the returned workspace is selected after acceptance.

- [ ] **Step 2: Run the integrated backend and frontend tests**

Run: `python -m pytest tests/test_core.py`
Run: `npm test`
Expected: all backend and frontend tests pass.

- [ ] **Step 3: Update setup and demo instructions**

Document fresh-user workspace creation and invited-user first login in both README and the recreation guide. Explain that local verification and invitation emails appear at `http://localhost:8025`, that the invitee must use the invited email and accept the invitation, and that workspace role permissions—not Keycloak role claims—control application access.
For native SQLite development, document `alembic upgrade head` after setting `DATABASE_URL`; Compose already applies migrations before API startup.

- [ ] **Step 4: Run final validation and inspect the complete diff**

Run: `python -m pytest`
Run: `npm test`
Run: `npm run lint`
Run: `npm run build`
Review the diff for migration safety, workspace scoping on every document operation, permission enforcement on every relevant API route, and accidental changes to unrelated user work.
Expected: all validation succeeds; existing workspace data and ACL behavior remain compatible.
