import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiError } from "../api/client";
import type { Workspace, WorkspaceRequest } from "../api/types";

export type { Workspace, WorkspaceRequest } from "../api/types";

interface WorkspaceContextValue {
  workspaces: Workspace[];
  workspace: Workspace | null;
  loading: boolean;
  error: string | null;
  selectWorkspace(id: string): void;
  refresh(preferredWorkspaceId?: string): Promise<void>;
  request: WorkspaceRequest;
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Unable to load workspace data. Retry the request.";
}

export function WorkspaceProvider({
  children,
  request,
}: {
  children: ReactNode;
  request: <T>(path: string, init?: RequestInit, workspaceId?: string) => Promise<T>;
}) {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [workspaceId, setWorkspaceId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadGeneration = useRef(0);

  const refresh = useCallback(async (preferredWorkspaceId?: string) => {
    const generation = ++loadGeneration.current;
    setLoading(true);
    setError(null);
    try {
      const memberships = await request<Workspace[]>("/v1/workspaces");
      if (generation !== loadGeneration.current) return;
      setWorkspaces(memberships);
      if (
        preferredWorkspaceId &&
        !memberships.some((workspace) => workspace.id === preferredWorkspaceId)
      ) {
        setWorkspaceId(null);
        throw new Error("The invited workspace is not available to your account.");
      }
      setWorkspaceId((current) => {
        if (preferredWorkspaceId) return preferredWorkspaceId;
        if (current && memberships.some((workspace) => workspace.id === current)) {
          return current;
        }
        return (
          memberships.find((workspace) => workspace.is_personal)?.id ??
          memberships[0]?.id ??
          null
        );
      });
    } catch (loadError) {
      if (generation === loadGeneration.current) setError(getErrorMessage(loadError));
      throw loadError;
    } finally {
      if (generation === loadGeneration.current) setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    let active = true;
    const generation = ++loadGeneration.current;
    setLoading(true);
    request<Workspace[]>("/v1/workspaces")
      .then((memberships) => {
        if (!active || generation !== loadGeneration.current) return;
        setWorkspaces(memberships);
        setWorkspaceId(
          memberships.find((workspace) => workspace.is_personal)?.id ??
            memberships[0]?.id ??
            null,
        );
        setError(null);
      })
      .catch((loadError: unknown) => {
        if (active && generation === loadGeneration.current) {
          setError(getErrorMessage(loadError));
        }
      })
      .finally(() => {
        if (active && generation === loadGeneration.current) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [request]);

  const workspace = workspaces.find((item) => item.id === workspaceId) ?? null;

  const selectWorkspace = useCallback(
    (id: string) => {
      if (!workspaces.some((item) => item.id === id)) {
        setError("That workspace is not available to your account.");
        return;
      }
      setError(null);
      setWorkspaceId(id);
    },
    [workspaces],
  );

  const scopedRequest = useCallback<WorkspaceRequest>(
    (path, init) => {
      if (!workspace) {
        return Promise.reject(
          new ApiError("Select an available workspace before making this request.", 400),
        );
      }
      return request(path, init, workspace.id);
    },
    [request, workspace],
  );

  const value = useMemo(
    () => ({ workspaces, workspace, loading, error, selectWorkspace, refresh, request: scopedRequest }),
    [workspaces, workspace, loading, error, selectWorkspace, refresh, scopedRequest],
  );

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace(): WorkspaceContextValue {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error("useWorkspace must be used within WorkspaceProvider.");
  return value;
}
