import Keycloak from "keycloak-js";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ApiError, apiRequest } from "../api/client";
import type { ApiRequest } from "../api/types";
import { authConfig } from "./authConfig";

export type AuthStatus = "loading" | "authenticated" | "unauthenticated" | "error";

export interface AuthUser {
  id: string;
  name: string;
  email?: string;
  platformAdmin: boolean;
}

export interface OidcClient {
  authenticated?: boolean;
  token?: string;
  tokenParsed?: {
    sub?: string;
    name?: string;
    preferred_username?: string;
    email?: string;
    realm_access?: { roles?: string[] };
  };
  init(options: {
    onLoad: "check-sso";
    flow: "standard";
    pkceMethod: "S256";
    checkLoginIframe: false;
  }): Promise<boolean>;
  login(): Promise<void>;
  logout(): Promise<void>;
  register(): Promise<void>;
  loginWithGoogle(): Promise<void>;
  forgotPassword(): Promise<void>;
  updateToken(minValidity: number): Promise<boolean>;
  clearToken(): void;
}

interface AuthContextValue {
  status: AuthStatus;
  user: AuthUser | null;
  error: string | null;
  login(): Promise<void>;
  logout(): Promise<void>;
  register(): Promise<void>;
  loginWithGoogle(): Promise<void>;
  forgotPassword(): Promise<void>;
  request: ApiRequest;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function createOidcClient(): OidcClient {
  const client = new Keycloak(authConfig);
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  return {
    get authenticated() {
      return client.authenticated;
    },
    get token() {
      return client.token;
    },
    get tokenParsed() {
      return client.tokenParsed;
    },
    init: (options) => client.init(options),
    login: () => client.login(),
    logout: () => client.logout(),
    register: () => client.register(),
    loginWithGoogle: () =>
      (client as any).login({ idpHint: "google" }),
    forgotPassword: () => {
      const resetUrl = new URL(
        `${authConfig.url}/realms/${authConfig.realm}/login-actions/reset-credentials`
      );
      resetUrl.searchParams.set("client_id", authConfig.clientId);
      resetUrl.searchParams.set("redirect_uri", `${origin}/`);
      resetUrl.searchParams.set("scope", "openid");
      window.location.assign(resetUrl.toString());
      return Promise.resolve();
    },
    updateToken: (minValidity) => client.updateToken(minValidity),
    clearToken: () => client.clearToken(),
  };
}

const defaultClient = createOidcClient();

function getUser(client: OidcClient): AuthUser | null {
  const id = client.tokenParsed?.sub;
  if (!id) return null;
  return {
    id,
    name:
      client.tokenParsed?.name ??
      client.tokenParsed?.preferred_username ??
      client.tokenParsed?.email ??
      "Workspace member",
    email: client.tokenParsed?.email,
    platformAdmin: client.tokenParsed?.realm_access?.roles?.includes("platform-admin") ?? false,
  };
}

export function AuthProvider({
  children,
  client = defaultClient,
}: {
  children: ReactNode;
  client?: OidcClient;
}) {
  const [status, setStatus] = useState<AuthStatus>("loading");
  const [user, setUser] = useState<AuthUser | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    client
      .init({
        onLoad: "check-sso",
        flow: "standard",
        pkceMethod: "S256",
        checkLoginIframe: false,
      })
      .then(() => {
        if (!active) return;
        const authenticatedUser = client.authenticated ? getUser(client) : null;
        if (client.authenticated && !authenticatedUser) {
          setStatus("error");
          setError("The identity provider did not return a valid user identity.");
          return;
        }
        setUser(authenticatedUser);
        setStatus(authenticatedUser ? "authenticated" : "unauthenticated");
      })
      .catch(() => {
        if (!active) return;
        setStatus("error");
        setError("Unable to initialize secure sign-in. Check the identity provider and retry.");
      });

    return () => {
      active = false;
    };
  }, [client]);

  const clearSession = useCallback(() => {
    client.clearToken();
    setUser(null);
    setStatus("unauthenticated");
    setError(null);
  }, [client]);

  const login = useCallback(async () => {
    try {
      await client.login();
    } catch {
      const message = "Unable to start secure sign-in. Check the identity provider and retry.";
      setError(message);
      setStatus("error");
      throw new Error(message);
    }
  }, [client]);

  const register = useCallback(async () => {
    try {
      await client.register();
    } catch {
      const message = "Unable to open the sign-up page. Check the identity provider and retry.";
      setError(message);
      setStatus("error");
      throw new Error(message);
    }
  }, [client]);

  const loginWithGoogle = useCallback(async () => {
    try {
      await (client as any).login({ idpHint: "google" });
    } catch {
      const message = "Unable to start sign-in with Google. Check the identity provider and retry.";
      setError(message);
      setStatus("error");
      throw new Error(message);
    }
  }, [client]);

  const forgotPassword = useCallback(async () => {
    try {
      await client.forgotPassword();
    } catch {
      const message = "Unable to open the password reset page. Check the identity provider and retry.";
      setError(message);
      setStatus("error");
      throw new Error(message);
    }
  }, [client]);

  const logout = useCallback(async () => {
    try {
      await client.logout();
      clearSession();
    } catch {
      const message = "Unable to complete sign-out. Please retry.";
      setError(message);
      setStatus("error");
      throw new Error(message);
    }
  }, [clearSession, client]);

  const getAccessToken = useCallback(async () => {
    if (!client.authenticated || !client.token) {
      clearSession();
      throw new ApiError("Your session has expired. Sign in again.", 401);
    }

    try {
      await client.updateToken(30);
    } catch {
      clearSession();
      throw new ApiError("Your session has expired. Sign in again.", 401);
    }

    if (!client.authenticated || !client.token) {
      clearSession();
      throw new ApiError("Your session has expired. Sign in again.", 401);
    }

    return client.token;
  }, [clearSession, client]);

  const request = useCallback<ApiRequest>(
    async <T,>(path: string, init: RequestInit = {}, workspaceId?: string) => {
      const token = await getAccessToken();
      try {
        return await apiRequest<T>(path, init, token, workspaceId);
      } catch (requestError) {
        if (requestError instanceof ApiError && requestError.status === 401) {
          clearSession();
        }
        throw requestError;
      }
    },
    [clearSession, getAccessToken],
  );

  const value = useMemo(
    () => ({
      status,
      user,
      error,
      login,
      logout,
      register,
      loginWithGoogle,
      forgotPassword,
      request,
    }),
    [status, user, error, login, logout, register, loginWithGoogle, forgotPassword, request]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used within AuthProvider.");
  return value;
}
