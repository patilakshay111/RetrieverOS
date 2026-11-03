import { useEffect, useLayoutEffect, useState, type FormEvent } from "react";
import type { OidcClient } from "./auth/AuthProvider";
import { AuthProvider, useAuth } from "./auth/AuthProvider";
import AppShell from "./layout/AppShell";
import {
  WorkspaceProvider,
  useWorkspace,
  type Workspace,
} from "./workspaces/WorkspaceProvider";

const INVITATION_STORAGE_KEY = "hybrid-rag.invitation-token";

function readInvitationToken(): string | null {
  const isInvitationPath = window.location.pathname === "/invite";
  const queryToken = isInvitationPath
    ? new URLSearchParams(window.location.search).get("token")
    : null;
  const fragmentToken = isInvitationPath
    ? new URLSearchParams(window.location.hash.slice(1)).get("token")
    : null;
  return queryToken ?? fragmentToken ?? sessionStorage.getItem(INVITATION_STORAGE_KEY);
}

function removeInvitationTokenFromUrl() {
  const url = new URL(window.location.href);
  url.searchParams.delete("token");
  const fragment = new URLSearchParams(url.hash.slice(1));
  fragment.delete("token");
  url.hash = fragment.toString();
  window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function LandingPage() {
  const { login, register, loginWithGoogle, forgotPassword, error } = useAuth();

  return (
    <main className="landing-layout">
      <section className="landing-intro">
        <a className="brand" href="/" aria-label="Hybrid RAG Studio">
          <span className="brand-mark" aria-hidden="true">H</span>
          <span>Hybrid RAG Studio</span>
        </a>
        <div className="landing-copy">
          <p className="eyebrow">Private knowledge, grounded answers</p>
          <h1>Your documents.<br />Your workspace.</h1>
          <p className="landing-lede">
            Search trusted workspace documents with answers linked to their source evidence.
          </p>
          <section className="feature-list" aria-label="Workspace features">
            <div className="feature-item">
              <span className="feature-number">01</span>
              <div><strong>Isolated by workspace</strong><span>Membership and document access are checked by the API.</span></div>
            </div>
            <div className="feature-item">
              <span className="feature-number">02</span>
              <div><strong>Evidence with every answer</strong><span>Inspect retrieved passages and citation checks.</span></div>
            </div>
            <div className="feature-item">
              <span className="feature-number">03</span>
              <div><strong>Your model key</strong><span>Store a private key encrypted by the application API.</span></div>
            </div>
          </section>
        </div>
        <p className="landing-footnote">Secure access powered by your organization identity provider.</p>
      </section>

      <section className="sign-in-panel" aria-labelledby="sign-in-title">
        <div className="sign-in-card">
          <p className="eyebrow">Welcome back</p>
          <h2 id="sign-in-title">Sign in to your workspace</h2>
          <p className="muted-copy">
            Sign in with your organization account, or create a new one.
            Sign-up and password reset are handled by your identity provider.
          </p>
          {error && <p className="notice notice-error" role="alert">{error}</p>}
          <div className="sign-in-actions">
            <button
              type="button"
              className="button button-primary sign-in-button"
              onClick={() => void login().catch(() => undefined)}
            >
              Sign in with email &amp; password
            </button>
            <button
              type="button"
              className="button button-outline sign-in-button"
              onClick={() => void loginWithGoogle().catch(() => undefined)}
            >
              <span aria-hidden="true">G</span> Continue with Google
            </button>
          </div>
          <p className="sign-in-divider"><span>or</span></p>
          <button
            type="button"
            className="button button-text create-account-button"
            onClick={() => void register().catch(() => undefined)}
          >
            Create an account
          </button>
          <p className="sign-in-note sign-in-help">
            <button
              type="button"
              className="link-button"
              onClick={() => void forgotPassword().catch(() => undefined)}
            >
              Forgot your password?
            </button>
            &nbsp;·&nbsp;
            Your session is managed by the identity provider. Access tokens stay
            in memory.
          </p>
        </div>
      </section>
    </main>
  );
}

function WorkspaceContent({
  onSignOut,
  invitationToken,
  clearInvitationToken,
}: {
  onSignOut(): void;
  invitationToken: string | null;
  clearInvitationToken(): void;
}) {
  const auth = useAuth();
  const { workspaces, loading, error, refresh } = useWorkspace();
  const [inviteState, setInviteState] = useState<"idle" | "pending" | "accepted" | "error">("idle");
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceDescription, setWorkspaceDescription] = useState("");
  const [workspaceCreateError, setWorkspaceCreateError] = useState<string | null>(null);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  useEffect(() => {
    if (!invitationToken || inviteState !== "idle") return;
    setInviteState("pending");
    auth
      .request<{ workspace_id: string }>("/v1/invitations/accept", {
        method: "POST",
        body: JSON.stringify({ token: invitationToken }),
      })
      .then(async (acceptedInvitation) => {
        clearInvitationToken();
        window.history.replaceState(null, "", "/");
        setInviteState("accepted");
        try {
          await refresh(acceptedInvitation.workspace_id);
        } catch {
          setInviteError("Invitation accepted. Reload the page to refresh workspace access.");
        }
      })
      .catch((acceptError: unknown) => {
        setInviteError(acceptError instanceof Error ? acceptError.message : "Unable to accept invitation.");
        setInviteState("error");
      });
  }, [auth, clearInvitationToken, invitationToken, inviteState, refresh]);

  async function createWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setWorkspaceCreateError(null);
    setCreatingWorkspace(true);
    let workspaceCreated = false;
    try {
      const created = await auth.request<Workspace>("/v1/workspaces", {
        method: "POST",
        body: JSON.stringify({
          name: workspaceName,
          description: workspaceDescription,
        }),
      });
      workspaceCreated = true;
      await refresh(created.id);
    } catch (creationError: unknown) {
      setWorkspaceCreateError(
        workspaceCreated
          ? "Workspace created, but access could not be refreshed. Reload the page to continue."
          : creationError instanceof Error
            ? creationError.message
            : "Unable to create the workspace.",
      );
    } finally {
      setCreatingWorkspace(false);
    }
  }

  if (invitationToken || inviteState !== "idle") {
    return (
      <main className="loading-view">
        {inviteState === "pending" && <output>Accepting your workspace invitation…</output>}
        {inviteState === "accepted" && <output>Invitation accepted. Your workspace access is ready.</output>}
        {inviteState === "error" && <p className="notice notice-error" role="alert">{inviteError}</p>}
        {inviteState === "accepted" && (
          <button type="button" className="button button-primary" onClick={() => setInviteState("idle")}>
            Continue to workspace
          </button>
        )}
      </main>
    );
  }

  if (loading) {
    return <main className="loading-view" aria-busy="true"><p>Loading your workspace…</p></main>;
  }

  if (error) {
    return (
      <main className="loading-view">
        <h1>Workspace could not be loaded</h1>
        <p className="notice notice-error" role="alert">{error}</p>
        <button type="button" className="button button-primary" onClick={() => void refresh().catch(() => undefined)}>
          Retry
        </button>
      </main>
    );
  }

  if (workspaces.length === 0) {
    return (
      <main className="loading-view">
        <section className="sign-in-card workspace-creation-card">
          <p className="eyebrow">First-time setup</p>
          <h1>Create a workspace</h1>
          <p className="muted-copy">
            Give your team a private place to organize documents and grounded answers.
          </p>
          {workspaceCreateError && (
            <p className="notice notice-error" role="alert">{workspaceCreateError}</p>
          )}
          <form onSubmit={(event) => void createWorkspace(event)}>
            <label>
              Workspace name
              <input
                autoComplete="organization"
                maxLength={255}
                minLength={1}
                name="workspaceName"
                onChange={(event) => setWorkspaceName(event.target.value)}
                required
                value={workspaceName}
              />
            </label>
            <label>
              Description
              <textarea
                maxLength={2000}
                name="workspaceDescription"
                onChange={(event) => setWorkspaceDescription(event.target.value)}
                rows={4}
                value={workspaceDescription}
              />
            </label>
            <button
              type="submit"
              className="button button-primary"
              disabled={creatingWorkspace}
            >
              {creatingWorkspace ? "Creating workspace…" : "Create workspace"}
            </button>
          </form>
        </section>
      </main>
    );
  }

  return auth.user ? <AppShell key={auth.user.id} user={auth.user} onSignOut={onSignOut} /> : null;
}

function Application({
  invitationToken,
  clearInvitationToken,
}: {
  invitationToken: string | null;
  clearInvitationToken(): void;
}) {
  const auth = useAuth();
  if (auth.status === "loading") {
    return <main className="loading-view" aria-busy="true"><p>Checking your sign-in…</p></main>;
  }
  if (auth.status === "error") {
    return (
      <main className="loading-view">
        <h1>Secure sign-in is unavailable</h1>
        <p className="notice notice-error" role="alert">{auth.error}</p>
        <button type="button" className="button button-primary" onClick={() => void auth.login().catch(() => undefined)}>
          Try sign-in again
        </button>
      </main>
    );
  }
  if (auth.status === "unauthenticated" || !auth.user) return <LandingPage />;

  return (
    <WorkspaceProvider key={auth.user.id} request={auth.request}>
      <WorkspaceContent
        onSignOut={() => void auth.logout().catch(() => undefined)}
        invitationToken={invitationToken}
        clearInvitationToken={clearInvitationToken}
      />
    </WorkspaceProvider>
  );
}

export default function App({ client }: { client?: OidcClient }) {
  const [invitationToken, setInvitationToken] = useState(readInvitationToken);

  useLayoutEffect(() => {
    if (!invitationToken) return;
    sessionStorage.setItem(INVITATION_STORAGE_KEY, invitationToken);
    removeInvitationTokenFromUrl();
  }, [invitationToken]);

  function clearInvitationToken() {
    sessionStorage.removeItem(INVITATION_STORAGE_KEY);
    setInvitationToken(null);
  }

  return (
    <AuthProvider client={client}>
      <Application
        invitationToken={invitationToken}
        clearInvitationToken={clearInvitationToken}
      />
    </AuthProvider>
  );
}
