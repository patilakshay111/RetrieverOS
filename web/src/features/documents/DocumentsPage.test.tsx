import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import DocumentsPage from "./DocumentsPage";
import type { WorkspaceRequest } from "../../workspaces/WorkspaceProvider";

const documentInfo = {
  source: "handbook.md",
  source_sha256: "digest",
  bytes: 42,
  chunk_count: 2,
  chunking_strategy: "recursive",
  ingested_at: "2026-11-02T00:00:00Z",
  workspace_id: "workspace-1",
  owner_id: "user-1",
  required_roles: ["employee"],
};

function createRequest(): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(async <T,>(path: string): Promise<T> => {
    if (path === "/v1/documents") return [documentInfo] as T;
    if (path === "/v1/workspaces/workspace-1/roles") {
      return [
        { name: "employee", description: "Team member", permissions: ["documents:read"] },
        { name: "hr", description: "People team", permissions: ["documents:read"] },
      ] as T;
    }
    if (path === "/v1/ingest") {
      return {
        document: documentInfo,
        inserted_chunks: 2,
        duplicate_chunks_skipped: 0,
      } as T;
    }
    throw new Error(`Unexpected request: ${path}`);
  }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

describe("DocumentsPage", () => {
  it("uploads a selected file using the chosen chunking strategy and multiple roles", async () => {
    const request = createRequest();
    render(
      <DocumentsPage
        request={request}
        workspaceId="workspace-1"
        roles={["employee", "hr"]}
        permissions={["documents:read", "documents:upload"]}
      />,
    );

    await screen.findByText("handbook.md");
    const file = new File(["policy body"], "new-policy.md", { type: "text/markdown" });
    fireEvent.change(screen.getByLabelText("Document file"), {
      target: { files: [file] },
    });
    fireEvent.change(screen.getByLabelText("Chunking strategy"), {
      target: { value: "semantic" },
    });
    fireEvent.change(screen.getByLabelText("Document access roles"), {
      target: { value: "employee" },
    });
    const roleSelect = screen.getByLabelText("Document access roles") as HTMLSelectElement;
    Array.from(roleSelect.options).forEach((option) => {
      option.selected = ["employee", "hr"].includes(option.value);
    });
    fireEvent.change(roleSelect);
    fireEvent.click(screen.getByRole("button", { name: "Upload document" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Document indexed");
    await waitFor(() => expect(request).toHaveBeenCalledWith("/v1/ingest", expect.any(Object)));
    const uploadCall = request.mock.calls.find(([path]) => path === "/v1/ingest");
    const body = uploadCall?.[1]?.body;
    expect(body).toBeInstanceOf(FormData);
    if (!(body instanceof FormData)) throw new Error("The upload body must be FormData.");
    expect(body.get("file")).toBe(file);
    expect(body.get("strategy")).toBe("semantic");
    expect(body.get("roles")).toBe("employee,hr");
  });

  it("reports inventory refresh failure separately after a successful upload", async () => {
    let documentListRequests = 0;
    const request = vi.fn(async <T,>(path: string): Promise<T> => {
      if (path === "/v1/documents") {
        documentListRequests += 1;
        if (documentListRequests === 2) throw new Error("Inventory unavailable");
        return [documentInfo] as T;
      }
      if (path === "/v1/workspaces/workspace-1/roles") {
        return [
          { name: "employee", description: "Team member", permissions: ["documents:read"] },
        ] as T;
      }
      if (path === "/v1/ingest") {
        return {
          document: documentInfo,
          inserted_chunks: 2,
          duplicate_chunks_skipped: 0,
        } as T;
      }
      throw new Error(`Unexpected request: ${path}`);
    }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
    render(
      <DocumentsPage
        request={request}
        workspaceId="workspace-1"
        roles={["employee"]}
        permissions={["documents:read", "documents:upload"]}
      />,
    );

    await screen.findByText("handbook.md");
    fireEvent.change(screen.getByLabelText("Document file"), {
      target: { files: [new File(["policy body"], "new-policy.md", { type: "text/markdown" })] },
    });
    fireEvent.click(screen.getByRole("button", { name: "Upload document" }));

    expect(await screen.findByRole("status")).toHaveTextContent("Document indexed");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Document indexed, but the document list could not be refreshed",
    );
    expect(request.mock.calls.filter(([path]) => path === "/v1/ingest")).toHaveLength(1);
  });

  it("does not submit a replacement without update permission", async () => {
    const request = createRequest();
    render(
      <DocumentsPage
        request={request}
        workspaceId="workspace-1"
        roles={["employee"]}
        permissions={["documents:read", "documents:upload"]}
      />,
    );

    await screen.findByText("handbook.md");
    fireEvent.change(screen.getByLabelText("Document file"), {
      target: { files: [new File(["replacement"], "handbook.md", { type: "text/markdown" })] },
    });

    expect(screen.getByRole("button", { name: "Replace document" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Replace document" }));
    expect(request.mock.calls.filter(([path]) => path === "/v1/ingest")).toHaveLength(0);
  });

  it("hides document controls when the user has no read or upload permissions", () => {
    const request = createRequest();
    render(
      <DocumentsPage
        request={request}
        workspaceId="workspace-1"
        roles={["employee"]}
        permissions={[]}
      />,
    );

    expect(screen.getByRole("heading", { name: "Document access unavailable" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Upload document" })).not.toBeInTheDocument();
    expect(request).not.toHaveBeenCalledWith("/v1/documents");
  });
});
