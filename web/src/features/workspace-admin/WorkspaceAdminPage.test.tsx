import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import WorkspaceAdminPage from "./WorkspaceAdminPage";
import type { Workspace, WorkspaceRequest } from "../../workspaces/WorkspaceProvider";

const workspace: Workspace = {
  id: "workspace-1",
  name: "Operations",
  description: "Operations workspace",
  roles: ["admin"],
  permissions: ["documents:read", "documents:query", "documents:upload", "documents:update"],
  status: "active",
  is_personal: true,
};

function createRequest(): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(async <T,>(path: string, init?: RequestInit): Promise<T> => {
    if (path === "/v1/workspaces/workspace-1/members") {
      return [
        {
          user_id: "user-1",
          workspace_id: "workspace-1",
          roles: ["admin"],
          status: "active",
        },
      ] as T;
    }
    if (path === "/v1/workspaces/workspace-1/roles" && init?.method === "POST") {
      return {
        ...JSON.parse(String(init.body)),
        workspace_id: "workspace-1",
        created_by: "admin-1",
        created_at: "2026-11-02T00:00:00Z",
      } as T;
    }
    if (path === "/v1/workspaces/workspace-1/roles") {
      return [
        { name: "employee", description: "Team member", permissions: ["documents:read"] },
        { name: "hr", description: "People team", permissions: ["documents:read", "documents:query"] },
      ] as T;
    }
    if (path === "/v1/workspaces/workspace-1" && init?.method === "PATCH") {
      return {
        ...workspace,
        ...JSON.parse(String(init.body)),
      } as T;
    }
    if (path.endsWith("/invitations")) return { id: "invite-1" } as T;
    throw new Error(`Unexpected request: ${path}`);
  }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

describe("WorkspaceAdminPage", () => {
  it("invites a workspace member with multiple selected workspace roles", async () => {
    const request = createRequest();
    render(<WorkspaceAdminPage request={request} workspace={workspace} />);

    await screen.findByText("user-1");
    fireEvent.change(screen.getByLabelText("Member email"), {
      target: { value: "new.member@example.test" },
    });
    await userEvent.selectOptions(screen.getByLabelText("Invite roles"), ["employee", "hr"]);
    fireEvent.click(screen.getByRole("button", { name: "Send invitation" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Invitation sent");
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        "/v1/workspaces/workspace-1/invitations",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    const invitationCall = request.mock.calls.find(([path]) => path.endsWith("/invitations"));
    const body = invitationCall?.[1]?.body;
    expect(typeof body).toBe("string");
    if (typeof body !== "string") throw new Error("The invitation body must be JSON.");
    expect(JSON.parse(body)).toEqual({
      email: "new.member@example.test",
      roles: ["employee", "hr"],
    });
  });

  it("creates a workspace role with selected capabilities", async () => {
    const request = createRequest();
    render(<WorkspaceAdminPage request={request} workspace={workspace} />);

    await screen.findByText("user-1");
    fireEvent.change(screen.getByLabelText("Role name"), {
      target: { value: "auditor" },
    });
    fireEvent.change(screen.getByLabelText("Role description"), {
      target: { value: "Read-only audit access" },
    });
    fireEvent.click(screen.getByLabelText("View documents"));
    fireEvent.click(screen.getByLabelText("Query documents and history"));
    fireEvent.click(screen.getByRole("button", { name: "Create role" }));

    await waitFor(() => {
      expect(request).toHaveBeenCalledWith(
        "/v1/workspaces/workspace-1/roles",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const roleCall = request.mock.calls.find(
      ([path, init]) => path.endsWith("/roles") && init?.method === "POST",
    );
    expect(JSON.parse(String(roleCall?.[1]?.body))).toEqual({
      name: "auditor",
      description: "Read-only audit access",
      permissions: ["documents:read", "documents:query"],
    });
  });

  it("updates the workspace name and description", async () => {
    const request = createRequest();
    render(<WorkspaceAdminPage request={request} workspace={workspace} />);

    fireEvent.change(await screen.findByLabelText("Workspace name"), {
      target: { value: "Customer operations" },
    });
    fireEvent.change(screen.getByLabelText("Workspace description"), {
      target: { value: "Customer-facing documents" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save workspace details" }));

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        "/v1/workspaces/workspace-1",
        expect.objectContaining({ method: "PATCH" }),
      ),
    );
    const updateCall = request.mock.calls.find(
      ([path, init]) => path.endsWith("/workspace-1") && init?.method === "PATCH",
    );
    expect(JSON.parse(String(updateCall?.[1]?.body))).toEqual({
      name: "Customer operations",
      description: "Customer-facing documents",
    });
  });

  it("does not render administration forms for non-admin workspace roles", () => {
    render(
      <WorkspaceAdminPage
        request={createRequest()}
        workspace={{ ...workspace, roles: ["employee"] }}
      />,
    );

    expect(screen.queryByLabelText("Member email")).not.toBeInTheDocument();
    expect(screen.getByText("Workspace administrator access is required.")).toBeInTheDocument();
  });
});
