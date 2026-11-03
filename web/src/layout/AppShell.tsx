import { useState, type FormEvent } from "react";
import type { AuthUser } from "../auth/AuthProvider";
import type { Workspace } from "../api/types";
import AccountPage from "../features/account/AccountPage";
import ChatPage from "../features/chat/ChatPage";
import DocumentsPage from "../features/documents/DocumentsPage";
import OperationsPage from "../features/operations/OperationsPage";
import WorkspaceAdminPage from "../features/workspace-admin/WorkspaceAdminPage";
import { useWorkspace } from "../workspaces/WorkspaceProvider";

type Page = "chat" | "documents" | "workspace" | "account" | "operations";

const navigation: { id: Page; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "documents", label: "Documents" },
  { id: "workspace", label: "Workspace" },
  { id: "account", label: "Account" },
  { id: "operations", label: "Operations" },
];

export default function AppShell({
  user,
  onSignOut,
}: {
  user: AuthUser;
  onSignOut(): void;
}) {
  const { workspaces, workspace, loading, error, selectWorkspace, refresh, request } = useWorkspace();
  const [page, setPage] = useState<Page>("chat");
  const [showWorkspaceForm, setShowWorkspaceForm] = useState(false);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceDescription, setWorkspaceDescription] = useState("");
  const [workspaceCreateError, setWorkspaceCreateError] = useState<string | null>(null);
  const canAdminister =
    user.platformAdmin || Boolean(workspace?.roles.includes("admin"));
  const activePage =
    (!canAdminister && page === "workspace") || (!user.platformAdmin && page === "operations")
      ? "chat"
      : page;

  async function createWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setWorkspaceCreateError(null);
    setCreatingWorkspace(true);
    let workspaceCreated = false;
    try {
      const created = await request<Workspace>("/v1/workspaces", {
        method: "POST",
        body: JSON.stringify({
          name: workspaceName,
          description: workspaceDescription,
        }),
      });
      workspaceCreated = true;
      await refresh(created.id);
      setWorkspaceName("");
      setWorkspaceDescription("");
      setShowWorkspaceForm(false);
    } catch (createError) {
      setWorkspaceCreateError(
        workspaceCreated
          ? "Workspace created, but access could not be refreshed. Reload the page to continue."
          : createError instanceof Error
            ? createError.message
            : "Unable to create the workspace.",
      );
    } finally {
      setCreatingWorkspace(false);
    }
  }

  if (loading) {
    return (
      <main className="loading-view" aria-busy="true">
        <p>Loading your workspace…</p>
      </main>
    );
  }

  if (!workspace) {
    return (
      <main className="loading-view">
        <h1>Workspace unavailable</h1>
        <p role="alert">{error ?? "No active workspace is available to this account."}</p>
      </main>
    );
  }

  return (
    <div className="dashboard-layout">
      <aside className="sidebar">
        <a className="brand brand-sidebar" href="/" aria-label="RetrieverOS home">
          <img className="brand-logo brand-logo-sidebar" src="/logo-dark.png" alt="RetrieverOS" />
        </a>
        <p className="sidebar-label">Workspace</p>
        <label className="workspace-label">
          <span className="visually-hidden">Active workspace</span>
          <select
            aria-label="Active workspace"
            value={workspace.id}
            onChange={(event) => selectWorkspace(event.target.value)}
          >
            {workspaces.map((item) => (
              <option value={item.id} key={item.id}>{item.name}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="workspace-create-button"
          onClick={() => setShowWorkspaceForm(true)}
        >
          Create another workspace
        </button>
        <nav className="primary-nav" aria-label="Primary navigation">
          {navigation
            .filter((item) => {
              if (item.id === "workspace") return canAdminister;
              if (item.id === "operations") return user.platformAdmin;
              return true;
            })
            .map((item) => (
              <button
                type="button"
                key={item.id}
                className={`nav-link${activePage === item.id ? " nav-link-active" : ""}`}
                aria-current={activePage === item.id ? "page" : undefined}
                onClick={() => setPage(item.id)}
              >
                {item.label}
              </button>
            ))}
        </nav>
        <div className="sidebar-footer">
          <div className="user-summary">
            <span className="user-avatar" aria-hidden="true">
              {user.name.slice(0, 1).toUpperCase()}
            </span>
            <span className="user-name" title={user.name}>{user.name}</span>
          </div>
          <button type="button" className="sign-out-button" onClick={onSignOut}>Sign out</button>
        </div>
      </aside>

      <div className="dashboard-main">
        <header className="topbar">
          <div>
            <span className="topbar-workspace">{workspace.name}</span>
            <span className="topbar-separator" aria-hidden="true">/</span>
            <span>{navigation.find((item) => item.id === activePage)?.label}</span>
          </div>
          <div className="topbar-actions">
            <span className="role-summary">
              {user.platformAdmin ? "Platform administrator" : workspace.roles.join(", ")}
            </span>
            <button type="button" className="mobile-sign-out" onClick={onSignOut}>Sign out</button>
          </div>
        </header>
        <main className="content-area">
          {error && <p className="notice notice-error" role="alert">{error}</p>}
          {showWorkspaceForm ? (
            <section className="page-section" aria-labelledby="create-workspace-title">
              <div className="page-heading">
                <div>
                  <p className="eyebrow">Workspace setup</p>
                  <h1 id="create-workspace-title">Create another workspace</h1>
                  <p className="muted-copy">
                    New workspaces start isolated with you as their administrator.
                  </p>
                </div>
              </div>
              <form className="form-card workspace-creation-card" onSubmit={(event) => void createWorkspace(event)}>
                <label>
                  New workspace name
                  <input
                    value={workspaceName}
                    onChange={(event) => setWorkspaceName(event.target.value)}
                    maxLength={255}
                    required
                  />
                </label>
                <label>
                  New workspace description
                  <textarea
                    value={workspaceDescription}
                    onChange={(event) => setWorkspaceDescription(event.target.value)}
                    maxLength={2000}
                    rows={3}
                  />
                </label>
                {workspaceCreateError && (
                  <p className="notice notice-error" role="alert">{workspaceCreateError}</p>
                )}
                <div className="form-actions">
                  <button
                    type="submit"
                    className="button button-primary"
                    disabled={creatingWorkspace}
                  >
                    {creatingWorkspace ? "Creating workspace…" : "Create workspace"}
                  </button>
                  <button
                    type="button"
                    className="button button-secondary"
                    onClick={() => setShowWorkspaceForm(false)}
                  >
                    Cancel
                  </button>
                </div>
              </form>
            </section>
          ) : (
          <div key={workspace.id} className="workspace-content">
            {activePage === "chat" && (
              <ChatPage
                request={request}
                canQuery={user.platformAdmin || workspace.permissions.includes("documents:query")}
              />
            )}
            {activePage === "documents" && (
              <DocumentsPage
                request={request}
                workspaceId={workspace.id}
                roles={workspace.roles}
                permissions={user.platformAdmin
                  ? ["documents:read", "documents:query", "documents:upload", "documents:update"]
                  : workspace.permissions}
              />
            )}
            {activePage === "workspace" && (
              <WorkspaceAdminPage
                request={request}
                workspace={workspace}
                platformAdmin={user.platformAdmin}
                onWorkspaceUpdated={() => refresh(workspace.id)}
              />
            )}
            {activePage === "account" && <AccountPage request={request} />}
            {activePage === "operations" && <OperationsPage request={request} />}
          </div>
          )}
        </main>
      </div>
    </div>
  );
}
