import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OidcClient } from "./auth/AuthProvider";
import App from "./App";

function unauthenticatedClient(): OidcClient {
  return {
    authenticated: false,
    init: async () => false,
    login: async () => undefined,
    logout: async () => undefined,
    updateToken: async () => false,
    clearToken: () => undefined,
  };
}

function authenticatedClient(): OidcClient {
  return {
    authenticated: true,
    token: "access-token",
    tokenParsed: { sub: "member-1", name: "Workspace Member" },
    init: async () => true,
    login: async () => undefined,
    logout: async () => undefined,
    updateToken: async () => true,
    clearToken: () => undefined,
  };
}

describe("sign-in landing page", () => {
  it("offers organization sign-in without showing workspace data", async () => {
    render(<App client={unauthenticatedClient()} />);

    expect(await screen.findByRole("heading", { name: "Sign in to your workspace" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Continue with organization account" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("navigation")).not.toBeInTheDocument();
  });

  it("removes invitation tokens from the URL and sends them in the request body", async () => {
    const invitationToken = "one-time-invitation-secret";
    window.history.replaceState(null, "", `/invite#token=${invitationToken}`);
    sessionStorage.clear();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      const payload = url.endsWith("/v1/invitations/accept")
        ? { user_id: "member-1", workspace_id: "workspace-2", roles: ["employee"], status: "active" }
        : [
            { id: "workspace-1", name: "Personal", description: "", roles: ["admin"], permissions: ["documents:read", "documents:query", "documents:upload", "documents:update"], status: "active", is_personal: true },
            { id: "workspace-2", name: "Research", description: "Research documents", roles: ["employee"], permissions: ["documents:read", "documents:query", "documents:upload"], status: "active", is_personal: false },
          ];
      return {
        status: 200,
        ok: true,
        json: async () => payload,
      } as Response;
    });

    render(<App client={authenticatedClient()} />);

    expect(await screen.findByText("Invitation accepted. Your workspace access is ready.")).toBeInTheDocument();
    const acceptRequest = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith("/v1/invitations/accept"),
    );
    expect(acceptRequest).toBeDefined();
    expect(String(acceptRequest?.[0])).not.toContain(invitationToken);
    expect(JSON.parse(String(acceptRequest?.[1]?.body))).toEqual({ token: invitationToken });
    expect(window.location.search).toBe("");
    expect(window.location.hash).toBe("");
    expect(sessionStorage.getItem("hybrid-rag.invitation-token")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Continue to workspace" }));
    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "Active workspace" })).toHaveValue("workspace-2");
    });
  });

  it("creates and loads a named workspace for an account with no memberships", async () => {
    let workspace: Record<string, unknown> | null = null;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/v1/workspaces") && init?.method === "POST") {
        workspace = {
          id: "workspace-created",
          name: "Product team",
          description: "Private product evidence",
          roles: ["admin"],
          permissions: ["documents:read", "documents:query", "documents:upload", "documents:update"],
          status: "active",
          is_personal: false,
        };
        return {
          status: 201,
          ok: true,
          json: async () => workspace,
        } as Response;
      }

      const payload = url.endsWith("/v1/workspaces") && workspace ? [workspace] : [];
      return {
        status: 200,
        ok: true,
        json: async () => payload,
      } as Response;
    });

    render(<App client={authenticatedClient()} />);

    expect(await screen.findByRole("heading", { name: "Create a workspace" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("textbox", { name: "Workspace name" }), {
      target: { value: "Product team" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Description" }), {
      target: { value: "Private product evidence" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create workspace" }));

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "Active workspace" })).toHaveValue(
        "workspace-created",
      );
    });
    const createRequest = fetchMock.mock.calls.find(
      ([input, init]) => String(input).endsWith("/v1/workspaces") && init?.method === "POST",
    );
    expect(JSON.parse(String(createRequest?.[1]?.body))).toEqual({
      name: "Product team",
      description: "Private product evidence",
    });
  });

  it("lets a signed-in member create and switch to another workspace", async () => {
    const firstWorkspace = {
      id: "workspace-1",
      name: "Research",
      description: "",
      roles: ["admin"],
      permissions: ["documents:read", "documents:query", "documents:upload", "documents:update"],
      status: "active",
      is_personal: false,
    };
    const secondWorkspace = {
      ...firstWorkspace,
      id: "workspace-2",
      name: "Operations",
    };
    let workspaces = [firstWorkspace];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/v1/workspaces") && init?.method === "POST") {
        workspaces = [firstWorkspace, secondWorkspace];
        return {
          status: 201,
          ok: true,
          json: async () => secondWorkspace,
        } as Response;
      }
      const payload = url.endsWith("/v1/workspaces")
        ? workspaces
        : url.endsWith("/v1/conversations")
          ? []
          : [];
      return {
        status: 200,
        ok: true,
        json: async () => payload,
      } as Response;
    });

    render(<App client={authenticatedClient()} />);

    await screen.findByRole("combobox", { name: "Active workspace" });
    fireEvent.click(screen.getByRole("button", { name: "Create another workspace" }));
    fireEvent.change(screen.getByRole("textbox", { name: "New workspace name" }), {
      target: { value: "Operations" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "New workspace description" }), {
      target: { value: "Operations evidence" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create workspace" }));

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "Active workspace" })).toHaveValue("workspace-2");
    });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/v1/workspaces"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          name: "Operations",
          description: "Operations evidence",
        }),
      }),
    );
  });

  afterEach(() => {
    window.history.replaceState(null, "", "/");
    sessionStorage.clear();
  });
});
