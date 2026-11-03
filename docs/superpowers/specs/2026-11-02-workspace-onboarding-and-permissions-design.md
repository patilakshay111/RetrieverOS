# Workspace Onboarding and Permissions Design

## Intent

Make workspaces explicit, named collaboration spaces instead of silently
creating a personal workspace for every new identity. Let workspace
administrators define workspace-local roles with clear document permissions,
then grant those roles through invitations and document access controls.

## Goals

- A new authenticated user with no workspace membership sees a create-workspace
  flow that asks for a name and description. Creating it makes that user its
  initial administrator.
- A user who accepts an invitation lands in that invited workspace first,
  including when it is their first sign-in. They may create another workspace
  later.
- Workspace administrators can edit workspace name and description.
- Workspace administrators can create roles scoped to their workspace and
  select supported permissions. Role creation, invitation, membership
  administration, and workspace editing remain admin-only.
- Invitations assign one or more roles. Each role's effective permissions come
  from the workspace's role definition and are checked by the API.
- Document ingestion accepts multiple workspace roles as its access policy.
  Listing and querying documents require the matching document roles and the
  appropriate read/query permissions.
- Document permissions distinguish reading/listing, querying, uploading, and
  updating/replacing document content or access policy.
- Preserve existing workspaces, memberships, invitations, document ACLs, and
  conversation data during migration.

## Non-goals

- Cross-workspace roles or permissions.
- Making a member of one workspace an administrator of another.
- Platform-wide custom roles; the existing platform-admin role remains
  separate from workspace roles.
- Public or anonymous workspaces.
- Implementing a general-purpose policy language or arbitrary permission
  expressions.
- Document deletion in this change unless required by the builder's assessment
  of the existing document lifecycle.

## Current behavior and constraints

- OIDC authorization currently provisions a personal workspace automatically.
- Workspace records have a name but no description.
- Role definitions are global names. Memberships and invitations store role
  names, and role creation currently requires a platform administrator.
- Workspace administration checks for the literal `admin` role.
- Document ACLs store required role names. Ingestion currently accepts one role
  in the dashboard, while the API accepts a comma-separated list.
- Retrieval filters documents by workspace and role ACL. Document listing and
  chat do not yet have separate explicit capabilities.
- Document upload exists; a document replacement/edit endpoint does not.
- PostgreSQL deployments use Alembic migrations; SQLite development creates
  tables from SQLAlchemy metadata.
- The React SPA authenticates with Keycloak Authorization Code + PKCE and keeps
  tokens in memory. Invitation tokens are carried through the callback in the
  URL fragment and submitted in an authenticated request body.

## Approaches considered

1. **Workspace-scoped roles with a fixed set of permission keys (selected).**
   A role has a name, description, and allowlisted capabilities such as
   `documents:read`, `documents:query`, `documents:upload`, and
   `documents:update`. API handlers enforce those capabilities. This keeps
   behavior understandable, testable, and safe to evolve.
2. **Configurable permission rules.** Administrators compose arbitrary rules
   or resource/action expressions. This is more flexible, but introduces a
   policy language, evaluation semantics, and a larger security review surface
   than the current product needs.

The selected approach follows the user's choice of explicit supported
permissions and workspace-local roles. The built-in `admin` role remains
reserved and grants full workspace access; custom roles cannot grant
administrator-only operations. Other existing roles remain workspace-local
role names with explicit permission definitions.

## Proposed behavior

### Workspace lifecycle

- Remove automatic personal-workspace creation from request authorization.
- Allow authenticated users to list their memberships and create a workspace
  without requiring an active workspace header.
- A user-created workspace has a required name, an optional description, and
  an initial active `admin` membership for its creator.
- Add an admin-only update operation for workspace name and description.
- When a user has no memberships, the dashboard presents workspace creation
  rather than an empty workspace or a silently generated personal workspace.
- When an invitation is accepted, return its workspace ID and select that
  workspace in the dashboard after refreshing memberships. This selection
  takes precedence for that acceptance, even if the user later joins other
  workspaces.
- If an invited user has no membership yet, workspace listing and invitation
  acceptance must work without fabricating a default workspace or requiring an
  `X-Workspace-ID`.

### Roles and permissions

- Store role definitions per workspace, including a display name/slug,
  description, supported permission keys, and creation metadata.
- Provide workspace-scoped role listing and creation. Only a workspace admin
  may create roles; names are unique within a workspace.
- Keep the reserved `admin` role non-editable and non-removable. It grants all
  workspace capabilities and the workspace-management operations.
- Custom roles can grant only allowlisted document capabilities:
  - `documents:read`: list and inspect documents the member is allowed to see.
  - `documents:query`: ask questions using only accessible documents.
  - `documents:upload`: add documents to the workspace.
  - `documents:update`: replace/re-index a document and change its access
    roles.
- Do not allow custom roles to grant workspace administration, role creation,
  invitations, membership changes, or platform administration.
- Permission checks apply on every protected API request. Hiding a button in
  the SPA is supplementary, not an authorization boundary.
- A member with multiple roles has the union of their capabilities. A document
  is visible/queryable only when its ACL intersects the member's workspace
  roles and the member has the corresponding capability.
- On upload, allow selecting multiple roles. The API validates that every ACL
  role belongs to the current workspace and that a non-admin uploader may
  assign only roles they hold. Admins may assign any role in that workspace.
- The dashboard should show only actions the current permissions allow. A
  member lacking `documents:read` cannot browse document inventory; lacking
  `documents:query` cannot submit chat/retrieval requests; lacking
  `documents:upload` cannot upload; lacking `documents:update` cannot replace
  documents or alter their ACL.

### Invitations and membership

- Invitation creation accepts one or more role names that exist in the current
  workspace. Only admins may invite.
- Acceptance requires the invitee's verified OIDC email to match the
  invitation, as today.
- The accepted membership receives the invitation's selected roles and the
  permissions those roles define in that workspace.
- If the invitee already has a membership in the workspace, acceptance adds
  the invited roles to the existing roles and reactivates a suspended
  membership, preserving the existing behavior.
- Membership role assignment and status changes remain admin-only and validate
  roles against the target workspace.

### Existing data and migration

- Add workspace descriptions with a safe empty default for existing rows.
- Migrate global role definitions to workspace-scoped definitions for each
  existing workspace, preserving membership and invitation role names.
- Preserve document ACL role names and existing indexed data; ACLs remain
  interpreted only inside their workspace.
- Keep existing automatically created personal workspaces as existing
  workspaces so their documents and conversations are not orphaned. Stop
  creating new ones. Existing users may continue using them and create
  additional named workspaces.
- Map the built-in `admin` role to full capabilities. Map existing non-admin
  roles to document read/query/upload capabilities to preserve existing
  behavior; do not grant update capability automatically.
- Do not silently authorize existing custom roles beyond the documented
  compatibility mapping. The migration must explicitly handle pre-existing
  custom global roles and report or fail on ambiguous assignments rather than
  lose access data.

## API and UI surfaces

- Workspace list/create/update contracts and workspace description fields.
- Workspace-local role list/create contracts with descriptions and validated
  permission lists.
- Invitation and membership contracts referencing only current-workspace roles.
- Document upload form with multi-select ACL roles and permission-aware
  controls.
- Document replacement/edit surface for users with `documents:update`.
- Chat/document APIs that enforce `documents:query` / `documents:read` and
  filter results by the intersection of role ACL and workspace membership.
- Empty-membership onboarding, invitation-first workspace selection, and
  workspace selector behavior in the React dashboard.
- README and project recreation guide updates for onboarding, roles,
  permissions, and demo invitations.

## Security and compatibility

- Never trust role names, permissions, workspace IDs, or ACL selections sent by
  the browser; validate them against authenticated membership and persisted
  workspace role definitions.
- Keep workspace IDs in every document query and mutation. Role names are not
  globally meaningful.
- Preserve platform-admin authorization as a separate mechanism, including
  required audit logging for cross-workspace access.
- Reject unknown permission keys and attempts by custom roles to grant
  administrative capabilities.
- Keep invitation tokens out of query strings and logs.
- Apply migrations transactionally where supported and test them against both
  the SQLite development path and the PostgreSQL deployment path.

## Testing and documentation

- Backend tests for workspace creation/rename, no implicit workspace creation,
  invitation acceptance without prior membership, initial invited-workspace
  selection, role scoping, permission validation, and unauthorized API access.
- Retrieval tests proving workspace isolation and role-ACL filtering still
   apply when users hold multiple roles.
- Ingestion tests for multiple ACL roles and rejection of roles the caller
  cannot assign.
- Frontend tests for create-workspace onboarding, invited-workspace selection,
  workspace editing, role creation and permission selection, multi-role
  document access, and permission-gated actions.
- Migration tests proving existing memberships, invitation roles, document
  ACLs, and stored documents remain usable.
- Update README and recreation guide with the tested first-run and invitation
  demo flows.

## Left to the builder

- Exact display labels and concise descriptions for supported permissions.
- Database representation and API naming that best fit existing conventions.
- The safest additive migration strategy for existing installations, including
  any legacy global custom roles.
- How to represent the no-selected-workspace authorization context without
  weakening protections on workspace-scoped endpoints.
- The precise UX for changing an active workspace when its role set removes
  capabilities from a currently visible page.
- If a document replacement API cannot be safely added within the existing
  storage/re-index design, report the gap rather than pretending an
  upload-only flow is an update capability.
