import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AuthProvider, type OidcClient, useAuth } from "./AuthProvider";

const accessToken = "in-memory-access-token-that-must-not-be-persisted";

function createOidcClient(overrides: Partial<OidcClient> = {}): OidcClient {
  return {
    authenticated: true,
    token: accessToken,
    tokenParsed: { sub: "user-1", preferred_username: "akshay@example.test" },
    init: vi.fn().mockResolvedValue(true),
    login: vi.fn().mockResolvedValue(undefined),
    logout: vi.fn().mockResolvedValue(undefined),
    updateToken: vi.fn().mockResolvedValue(false),
    clearToken: vi.fn(),
    ...overrides,
  };
}

function AuthStatus() {
  const { status, user } = useAuth();
  return <p>{status === "authenticated" ? user?.name : status}</p>;
}

describe("AuthProvider", () => {
  it("uses PKCE S256 and keeps access tokens out of browser storage", async () => {
    localStorage.clear();
    sessionStorage.clear();
    const client = createOidcClient();

    render(
      <AuthProvider client={client}>
        <AuthStatus />
      </AuthProvider>,
    );

    expect(await screen.findByText("akshay@example.test")).toBeInTheDocument();
    expect(client.init).toHaveBeenCalledWith({
      onLoad: "check-sso",
      flow: "standard",
      pkceMethod: "S256",
      checkLoginIframe: false,
    });
    expect(localStorage.getItem("access_token")).toBeNull();
    expect(sessionStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("refresh_token")).toBeNull();
    expect(sessionStorage.getItem("refresh_token")).toBeNull();
  });

  it("returns to the unauthenticated state when token refresh fails", async () => {
    const client = createOidcClient({
      updateToken: vi.fn().mockRejectedValue(new Error("refresh failed")),
    });

    function RequestProbe() {
      const { request } = useAuth();
      return (
        <button
          type="button"
          onClick={() => {
            void request("/v1/documents").catch(() => undefined);
          }}
        >
          Load documents
        </button>
      );
    }

    render(
      <AuthProvider client={client}>
        <AuthStatus />
        <RequestProbe />
      </AuthProvider>,
    );

    await screen.findByText("akshay@example.test");
    await act(async () => screen.getByRole("button", { name: "Load documents" }).click());

    expect(await screen.findByText("unauthenticated")).toBeInTheDocument();
    expect(client.clearToken).toHaveBeenCalledOnce();
  });
});
