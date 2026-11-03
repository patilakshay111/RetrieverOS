import { useEffect, useState, type FormEvent } from "react";
import type { Role, Workspace, WorkspaceMember, WorkspaceRequest } from "../../api/types";

const permissionOptions = [
  { key: "documents:read", label: "View documents" },
  { key: "documents:query", label: "Query documents and history" },
  { key: "documents:upload", label: "Upload documents" },
  { key: "documents:update", label: "Replace accessible documents" },
];

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Retry when ready.";
}

export default function WorkspaceAdminPage({
  request,
  workspace,
  platformAdmin = false,
  onWorkspaceUpdated,
}: {
  request: WorkspaceRequest;
  workspace: Workspace;
  platformAdmin?: boolean;
  onWorkspaceUpdated?(): Promise<void> | void;
}) {
  const canAdminister = platformAdmin || workspace.roles.includes("admin");
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [roles, setRoles] = useState<Role[]>([]);
  const [email, setEmail] = useState("");
  const [inviteRoles, setInviteRoles] = useState<string[]>([]);
  const [memberRoles, setMemberRoles] = useState<Record<string, string[]>>({});
  const [workspaceName, setWorkspaceName] = useState(workspace.name);
  const [workspaceDescription, setWorkspaceDescription] = useState(workspace.description);
  const [roleName, setRoleName] = useState("");
  const [roleDescription, setRoleDescription] = useState("");
  const [rolePermissions, setRolePermissions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!canAdminister) {
      setLoading(false);
      return;
    }
    let active = true;
    Promise.all([
      request<WorkspaceMember[]>(`/v1/workspaces/${encodeURIComponent(workspace.id)}/members`),
      request<Role[]>(`/v1/workspaces/${encodeURIComponent(workspace.id)}/roles`),
    ])
      .then(([workspaceMembers, roleCatalog]) => {
        if (!active) return;
        setMembers(workspaceMembers);
        setRoles(roleCatalog);
        setMemberRoles(
          Object.fromEntries(workspaceMembers.map((member) => [member.user_id, member.roles])),
        );
        const defaultRole = roleCatalog.find((role) => role.name === "employee")
          ?? roleCatalog.find((role) => role.name !== "admin");
        setInviteRoles(defaultRole ? [defaultRole.name] : []);
      })
      .catch((loadError: unknown) => {
        if (active) setError(messageFor(loadError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [canAdminister, request, workspace.id]);

  useEffect(() => {
    setWorkspaceName(workspace.name);
    setWorkspaceDescription(workspace.description);
  }, [workspace.description, workspace.name]);

  async function sendInvitation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inviteRoles.length === 0) {
      setError("Choose at least one role for the invitation.");
      return;
    }
    setError(null);
    setSuccess(null);
    try {
      await request(`/v1/workspaces/${encodeURIComponent(workspace.id)}/invitations`, {
        method: "POST",
        body: JSON.stringify({ email, roles: inviteRoles }),
      });
      setEmail("");
      setSuccess("Invitation sent. The recipient must verify their email before joining.");
    } catch (inviteError) {
      setError(messageFor(inviteError));
    }
  }

  async function saveWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccess(null);
    try {
      const updated = await request<Workspace>(
        `/v1/workspaces/${encodeURIComponent(workspace.id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            name: workspaceName,
            description: workspaceDescription,
          }),
        },
      );
      setWorkspaceName(updated.name);
      setWorkspaceDescription(updated.description);
      await onWorkspaceUpdated?.();
      setSuccess("Workspace details updated.");
    } catch (updateError) {
      setError(messageFor(updateError));
    }
  }

  async function createRole(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSuccess(null);
    try {
      const created = await request<Role>(
        `/v1/workspaces/${encodeURIComponent(workspace.id)}/roles`,
        {
          method: "POST",
          body: JSON.stringify({
            name: roleName,
            description: roleDescription,
            permissions: rolePermissions,
          }),
        },
      );
      setRoles((current) =>
        [...current.filter((role) => role.name !== created.name), created]
          .sort((left, right) => left.name.localeCompare(right.name)),
      );
      setInviteRoles([created.name]);
      setRoleName("");
      setRoleDescription("");
      setRolePermissions([]);
      setSuccess(`Created the ${created.name} role.`);
    } catch (createError) {
      setError(messageFor(createError));
    }
  }

  async function updateMember(member: WorkspaceMember) {
    setError(null);
    setSuccess(null);
    try {
      const updated = await request<WorkspaceMember>(
        `/v1/workspaces/${encodeURIComponent(workspace.id)}/members/${encodeURIComponent(member.user_id)}`,
        {
          method: "PUT",
          body: JSON.stringify({
            roles: memberRoles[member.user_id] ?? member.roles,
            status: member.status,
          }),
        },
      );
      setMembers((items) =>
        items.map((item) => (item.user_id === updated.user_id ? updated : item)),
      );
      setSuccess(`Updated access for ${member.user_id}.`);
    } catch (updateError) {
      setError(messageFor(updateError));
    }
  }

  if (!canAdminister) {
    return (
      <section className="page-section" aria-labelledby="workspace-title">
        <h1 id="workspace-title">Workspace members</h1>
        <p className="notice notice-warning">Workspace administrator access is required.</p>
      </section>
    );
  }

  return (
    <section className="page-section" aria-labelledby="workspace-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Workspace access</p>
          <h1 id="workspace-title">Members and roles</h1>
          <p className="muted-copy">
            Manage access to <strong>{workspace.name}</strong>. Server-side membership rules remain
            authoritative.
          </p>
        </div>
      </div>

      {error && <p className="notice notice-error" role="alert">{error}</p>}
      {success && <output className="notice notice-success">{success}</output>}

      <form className="form-card" onSubmit={(event) => void saveWorkspace(event)}>
        <h2>Workspace details</h2>
        <div className="form-grid workspace-details-grid">
          <label>
            Workspace name
            <input
              value={workspaceName}
              onChange={(event) => setWorkspaceName(event.target.value)}
              maxLength={255}
              required
            />
          </label>
          <label>
            Workspace description
            <textarea
              value={workspaceDescription}
              onChange={(event) => setWorkspaceDescription(event.target.value)}
              maxLength={2000}
              rows={2}
            />
          </label>
          <button className="button button-secondary" type="submit">
            Save workspace details
          </button>
        </div>
      </form>

      <form className="form-card" onSubmit={(event) => void sendInvitation(event)}>
        <h2>Invite a member</h2>
        <div className="form-grid invite-grid">
          <label>
            Member email
            <input
              type="email"
              required
              maxLength={254}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
            />
          </label>
          <label>
            Invite roles
            <select
              multiple
              value={inviteRoles}
              onChange={(event) =>
                setInviteRoles(
                  Array.from(event.target.selectedOptions, (option) => option.value),
                )
              }
              required
            >
              {roles.map((role) => (
                <option key={role.name} value={role.name}>
                  {role.name} — {role.description}
                </option>
              ))}
            </select>
          </label>
          <button className="button button-primary" type="submit" disabled={inviteRoles.length === 0}>
            Send invitation
          </button>
        </div>
      </form>

      <form className="form-card" onSubmit={(event) => void createRole(event)}>
        <h2>Create a workspace role</h2>
        <div className="form-grid role-form-grid">
          <label>
            Role name
            <input
              value={roleName}
              onChange={(event) => setRoleName(event.target.value)}
              minLength={1}
              maxLength={50}
              pattern="[a-z][a-z0-9_-]*"
              required
            />
          </label>
          <label>
            Role description
            <input
              value={roleDescription}
              onChange={(event) => setRoleDescription(event.target.value)}
              maxLength={500}
            />
          </label>
        </div>
        <fieldset className="permission-options">
          <legend>Role permissions</legend>
          {permissionOptions.map((permission) => (
            <label key={permission.key}>
              <input
                type="checkbox"
                checked={rolePermissions.includes(permission.key)}
                onChange={(event) =>
                  setRolePermissions((current) =>
                    event.target.checked
                      ? [...current, permission.key]
                      : current.filter((item) => item !== permission.key),
                  )
                }
              />
              {permission.label}
            </label>
          ))}
        </fieldset>
        <button className="button button-primary" type="submit" disabled={!roleName.trim()}>
          Create role
        </button>
      </form>

      <section className="data-card" aria-labelledby="roles-title">
        <div className="section-heading">
          <div>
            <h2 id="roles-title">Workspace roles</h2>
            <p className="muted-copy">Each role is defined independently inside this workspace.</p>
          </div>
        </div>
        {loading ? (
          <output>Loading workspace roles…</output>
        ) : (
          <ul className="role-list">
            {roles.map((role) => (
              <li key={role.name}>
                <strong>{role.name}</strong>
                <span>{role.description || "No description"}</span>
                <span>
                  {role.permissions.length
                    ? role.permissions.map((permission) =>
                        permissionOptions.find((option) => option.key === permission)?.label
                        ?? permission,
                      ).join(", ")
                    : "No permissions"}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="data-card" aria-labelledby="members-title">
        <div className="section-heading">
          <div>
            <h2 id="members-title">Workspace members</h2>
            <p className="muted-copy">Assign existing roles or suspend a membership.</p>
          </div>
        </div>
        {loading ? (
          <output>Loading workspace members…</output>
        ) : members.length === 0 ? (
          <p className="empty-state">There are no members in this workspace.</p>
        ) : (
          <div className="member-list">
            {members.map((member) => (
              <div className="member-row" key={member.user_id}>
                <div>
                  <strong>{member.user_id}</strong>
                  <span className="muted-copy">Current roles: {member.roles.join(", ")}</span>
                </div>
                <label>
                  Roles for {member.user_id}
                  <select
                    multiple
                    value={memberRoles[member.user_id] ?? member.roles}
                    onChange={(event) =>
                      setMemberRoles((current) => ({
                        ...current,
                        [member.user_id]: Array.from(
                          event.target.selectedOptions,
                          (option) => option.value,
                        ),
                      }))
                    }
                  >
                    {roles.map((role) => (
                      <option key={role.name} value={role.name}>
                        {role.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Status for {member.user_id}
                  <select
                    value={member.status}
                    onChange={(event) =>
                      setMembers((current) =>
                        current.map((item) =>
                          item.user_id === member.user_id
                            ? { ...item, status: event.target.value }
                            : item,
                        ),
                      )
                    }
                  >
                    <option value="active">Active</option>
                    <option value="suspended">Suspended</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="button button-secondary"
                  onClick={() => void updateMember(member)}
                  disabled={(memberRoles[member.user_id] ?? member.roles).length === 0}
                >
                  Save access
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </section>
  );
}
