import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import OperationsPage from "./OperationsPage";
import type { WorkspaceRequest } from "../../api/types";

function createRequest(
  implementation: <T>(path: string, init?: RequestInit) => Promise<T>,
): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(implementation) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

describe("OperationsPage", () => {
  it("shows dependency readiness and the read-only reindex report", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            status: "ready",
            dependencies: { postgres: "ready", qdrant: "ready", redis: "ready" },
          }),
          { status: 200 },
        ),
      ),
    );
    const request = createRequest(async <T,>() =>
      ({
        dry_run: true,
        read_only: true,
        documents: [
          {
            source: "policy.md",
            source_sha256: "a".repeat(64),
            bytes: 24,
            chunking_strategy: "recursive",
          },
        ],
      }) as T,
    );

    render(<OperationsPage request={request} />);

    expect(await screen.findByText("policy.md")).toBeInTheDocument();
    expect(screen.getByText("Postgres")).toBeInTheDocument();
    expect(screen.getAllByText("ready")).toHaveLength(4);
    expect(screen.getByText("Read-only dry run")).toBeInTheDocument();
  });

  it("reports unavailable dependencies and safely handles a denied dry run", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: {
              status: "not_ready",
              dependencies: { postgres: "ready", qdrant: "unavailable", redis: "ready" },
            },
          }),
          { status: 503 },
        ),
      ),
    );
    const request = createRequest(async () => {
      throw new Error("You do not have permission to perform this action.");
    });

    render(<OperationsPage request={request} />);

    expect(await screen.findByText("unavailable")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "You do not have permission to perform this action.",
    );
    expect(screen.queryByText("private authorization detail")).not.toBeInTheDocument();
  });
});
