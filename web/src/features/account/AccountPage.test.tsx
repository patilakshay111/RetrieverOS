import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AccountPage from "./AccountPage";
import type { WorkspaceRequest } from "../../workspaces/WorkspaceProvider";

function createRequest(): ReturnType<typeof vi.fn> & WorkspaceRequest {
  return vi.fn(async <T,>(path: string): Promise<T> => {
    if (path === "/v1/account/openai-key") return { configured: true } as T;
    return undefined as T;
  }) as ReturnType<typeof vi.fn> & WorkspaceRequest;
}

describe("AccountPage", () => {
  it("saves a private API key without displaying it after submission", async () => {
    const request = createRequest();
    render(<AccountPage request={request} />);

    await screen.findByText("API key configured");
    fireEvent.change(screen.getByLabelText("OpenAI API key", { selector: "input" }), {
      target: { value: "sk-test-user-private-key" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save API key" }));

    expect(await screen.findByRole("status")).toHaveTextContent("API key saved");
    expect(screen.queryByDisplayValue("sk-test-user-private-key")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith(
        "/v1/account/openai-key",
        expect.objectContaining({ method: "PUT" }),
      ),
    );
  });
});
