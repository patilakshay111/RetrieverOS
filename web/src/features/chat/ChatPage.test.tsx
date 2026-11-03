import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ChatPage from "./ChatPage";
import type { WorkspaceRequest } from "../../workspaces/WorkspaceProvider";

const conversation = {
  conversation_id: "conversation-1",
  title: "Workplace policy",
  created_at: "2026-11-02T00:00:00Z",
  updated_at: "2026-11-02T00:00:00Z",
};

const answer = {
  question: "How many vacation days?",
  answer: "Employees receive 20 vacation days.",
  retrieval_mode: "hybrid" as const,
  citations: [
    {
      claim: "Employees receive 20 vacation days.",
      citation_number: 1,
      source: "policy.md",
      verified: true,
      rationale: "The policy states 20 days.",
    },
  ],
  retrieved_chunks: [
    {
      chunk: {
        chunk_id: "chunk-1",
        text: "Full-time employees receive 20 vacation days each year.",
        source: "policy.md",
        source_sha256: "abc",
        chunk_index: 0,
        chunking_strategy: "recursive",
        character_count: 55,
      },
      dense_score: 0.9,
      bm25_score: 2,
      rrf_score: 0.8,
      rerank_score: 0.95,
      rank: 1,
    },
  ],
  confidence: 0.91,
  confidence_breakdown: { retrieval: 0.9, citation: 1 },
  insufficient_context: false,
  documents_to_check: [],
  thread_id: "thread-1",
  conversation_id: "conversation-1",
};

function createRequest(): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(async <T,>(path: string): Promise<T> => {
    if (path === "/v1/conversations") return [conversation] as T;
    if (path === "/v1/ask") return answer as T;
    throw new Error(`Unexpected request: ${path}`);
  }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

describe("ChatPage", () => {
  it("sends a question in the selected conversation and renders answer evidence", async () => {
    const request = createRequest();
    render(<ChatPage request={request} canQuery />);

    await screen.findByText("Workplace policy");
    fireEvent.change(screen.getByRole("textbox", { name: "Ask your documents" }), {
      target: { value: "How many vacation days?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findAllByText("Employees receive 20 vacation days.")).toHaveLength(2);
    expect(screen.getByText("policy.md")).toBeInTheDocument();
    expect(
      screen.getByText("Full-time employees receive 20 vacation days each year."),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        "/v1/ask",
        expect.objectContaining({ method: "POST" }),
      ),
    );
  });

  it("clears the previous answer when a new ask fails", async () => {
    const request = createRequest();
    render(<ChatPage request={request} canQuery />);

    await screen.findByText("Workplace policy");
    const questionField = screen.getByRole("textbox", { name: "Ask your documents" });
    fireEvent.change(questionField, { target: { value: "How many vacation days?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByRole("article", { name: "Answer" });

    request.mockRejectedValueOnce(new Error("Service unavailable"));
    fireEvent.change(questionField, { target: { value: "What is the office address?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Service unavailable");
    expect(screen.queryByRole("article", { name: "Answer" })).not.toBeInTheDocument();
  });

  it("blocks conversation and query actions without query permission", () => {
    const request = createRequest();
    render(<ChatPage request={request} canQuery={false} />);

    expect(screen.getByRole("heading", { name: "Chat access unavailable" })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Ask your documents" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "New conversation" })).not.toBeInTheDocument();
    expect(request).not.toHaveBeenCalled();
  });
});
