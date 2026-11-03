export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  dependencies: Record<string, "ready" | "unavailable" | "not_configured">;
}

function apiUrl(path: string): string {
  const configuredOrigin = import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, "");
  if (!configuredOrigin) return path;

  let url: URL;
  try {
    url = new URL(configuredOrigin);
  } catch {
    throw new ApiError("The configured application API origin is invalid.", 500);
  }
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new ApiError("The configured application API origin is invalid.", 500);
  }
  return `${url.origin}${path}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function errorMessage(status: number, body: unknown): string {
  if (status === 401) return "Your session has expired. Sign in again.";
  if (status === 403) return "You do not have permission to perform this action.";
  if (isRecord(body) && typeof body.detail === "string") return body.detail;
  if (status === 422) return "Please check the submitted information and try again.";
  return `The request failed (${status}). Try again.`;
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  accessToken: string,
  workspaceId?: string,
): Promise<T> {
  if (!path.startsWith("/v1/") || path.startsWith("//") || path.includes("\\")) {
    throw new ApiError("Requests must target the versioned application API.", 400);
  }

  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  headers.set("Authorization", `Bearer ${accessToken}`);
  if (workspaceId) headers.set("X-Workspace-ID", workspaceId);
  if (typeof init.body === "string" && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(apiUrl(path), { ...init, headers });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ApiError(
        "Unable to reach the application API. Check your connection and retry.",
        0,
      );
    }
    throw error;
  }

  if (response.status === 204) return undefined as T;

  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
    if (response.ok) throw new ApiError("The application API returned an invalid response.", 502);
  }

  if (!response.ok) {
    throw new ApiError(errorMessage(response.status, body), response.status);
  }

  return body as T;
}

export async function fetchReadiness(): Promise<ReadinessResponse> {
  let response: Response;
  try {
    response = await fetch(apiUrl("/readyz"), { headers: { Accept: "application/json" } });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ApiError(
        "Unable to reach the application API. Check your connection and retry.",
        0,
      );
    }
    throw error;
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new ApiError("The application API returned an invalid readiness response.", 502);
    }
    throw error;
  }

  const payload =
    response.status === 503 && isRecord(body) && isRecord(body.detail) ? body.detail : body;
  if (
    (response.status !== 200 && response.status !== 503) ||
    !isRecord(payload) ||
    (payload.status !== "ready" && payload.status !== "not_ready") ||
    !isRecord(payload.dependencies)
  ) {
    throw new ApiError(`Unable to read application readiness (${response.status}).`, response.status);
  }

  const dependencies: ReadinessResponse["dependencies"] = {};
  for (const [name, state] of Object.entries(payload.dependencies)) {
    if (
      state !== "ready" &&
      state !== "unavailable" &&
      state !== "not_configured"
    ) {
      throw new ApiError("The application API returned an invalid readiness response.", 502);
    }
    dependencies[name] = state;
  }

  return { status: payload.status, dependencies };
}
