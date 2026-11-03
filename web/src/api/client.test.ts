import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, apiRequest } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("apiRequest", () => {
  it("sends the bearer token and selected workspace with JSON requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify([{ id: "workspace-a" }]), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiRequest(
      "/v1/workspaces/workspace-a/members",
      { method: "POST", body: JSON.stringify({ roles: ["employee"] }) },
      "access-token",
      "workspace-a",
    );

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const headers = new Headers(init.headers);
    expect(url).toBe("/v1/workspaces/workspace-a/members");
    expect(headers.get("Authorization")).toBe("Bearer access-token");
    expect(headers.get("X-Workspace-ID")).toBe("workspace-a");
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(init.body).toBe(JSON.stringify({ roles: ["employee"] }));
  });

  it("returns undefined for successful empty responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    await expect(
      apiRequest<void>("/v1/account/openai-key", { method: "DELETE" }, "access-token"),
    ).resolves.toBeUndefined();
  });

  it("reports unauthorized responses without exposing the bearer token", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response('{"detail":"expired"}', { status: 401 })),
    );

    const result = apiRequest("/v1/documents", {}, "sensitive-access-token");

    await expect(result).rejects.toMatchObject({ status: 401 });
    await expect(result).rejects.not.toThrow("sensitive-access-token");
  });

  it("turns network failures into a safe retryable error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));

    await expect(apiRequest("/v1/documents", {}, "access-token")).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        message: "Unable to reach the application API. Check your connection and retry.",
      }),
    );
  });

  it("blocks requests outside the versioned API", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiRequest("https://example.test", {}, "access-token")).rejects.toBeInstanceOf(
      ApiError,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
