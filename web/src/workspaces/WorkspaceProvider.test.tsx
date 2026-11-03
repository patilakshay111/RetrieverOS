import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  WorkspaceProvider,
  useWorkspace,
  type WorkspaceRequest,
} from "./WorkspaceProvider";

const workspaces = [
  {
    id: "workspace-a",
    name: "Personal",
    description: "",
    roles: ["admin"],
    permissions: ["documents:read", "documents:query", "documents:upload", "documents:update"],
    status: "active",
    is_personal: true,
  },
  {
    id: "workspace-b",
    name: "Research",
    description: "",
    roles: ["employee"],
    permissions: ["documents:read", "documents:query", "documents:upload"],
    status: "active",
    is_personal: false,
  },
];

function createRequest(): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(async <T,>(
    path: string,
    _init?: RequestInit,
    workspaceId?: string,
  ): Promise<T> => {
    if (path === "/v1/workspaces") return workspaces as T;
    return { selectedWorkspace: workspaceId } as T;
  }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

function WorkspaceProbe() {
  const { workspace, loading, selectWorkspace, refresh, request, error } = useWorkspace();
  if (loading) return <p>Loading workspaces</p>;

  return (
    <div>
      <p>{workspace?.name ?? "No workspace"}</p>
      {error && <p role="alert">{error}</p>}
      <button type="button" onClick={() => selectWorkspace("workspace-b")}>Switch workspace</button>
      <button type="button" onClick={() => selectWorkspace("workspace-missing")}>Select invalid workspace</button>
      <button type="button" onClick={() => void refresh("workspace-b")}>Prefer workspace B</button>
      <button type="button" onClick={() => void request("/v1/documents")}>Load documents</button>
    </div>
  );
}

describe("WorkspaceProvider", () => {
  it("loads memberships and selects the personal workspace by default", async () => {
    const request = createRequest();

    render(
      <WorkspaceProvider request={request}>
        <WorkspaceProbe />
      </WorkspaceProvider>,
    );

    expect(await screen.findByText("Personal")).toBeInTheDocument();
    expect(request).toHaveBeenCalledWith("/v1/workspaces");
  });

  it("adds the selected workspace to authenticated API requests", async () => {
    const request = createRequest();

    render(
      <WorkspaceProvider request={request}>
        <WorkspaceProbe />
      </WorkspaceProvider>,
    );

    await screen.findByText("Personal");
    fireEvent.click(screen.getByRole("button", { name: "Switch workspace" }));
    fireEvent.click(screen.getByRole("button", { name: "Load documents" }));

    await waitFor(() =>
      expect(request).toHaveBeenLastCalledWith("/v1/documents", undefined, "workspace-b"),
    );
    expect(screen.getByText("Research")).toBeInTheDocument();
  });

  it("selects the preferred workspace returned by a membership refresh", async () => {
    const request = createRequest();

    render(
      <WorkspaceProvider request={request}>
        <WorkspaceProbe />
      </WorkspaceProvider>,
    );

    await screen.findByText("Personal");
    fireEvent.click(screen.getByRole("button", { name: "Prefer workspace B" }));

    await waitFor(() => expect(screen.getByText("Research")).toBeInTheDocument());
  });

  it("rejects workspace IDs not present in the authenticated membership list", async () => {
    const request = createRequest();

    render(
      <WorkspaceProvider request={request}>
        <WorkspaceProbe />
      </WorkspaceProvider>,
    );

    await screen.findByText("Personal");
    fireEvent.click(screen.getByRole("button", { name: "Select invalid workspace" }));

    expect(screen.getByRole("alert")).toHaveTextContent(
      "That workspace is not available to your account.",
    );
    expect(screen.getByText("Personal")).toBeInTheDocument();
  });
});
