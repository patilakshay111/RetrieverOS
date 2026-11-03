import { useEffect, useState, type FormEvent } from "react";
import type {
  DocumentInfo,
  IngestionResult,
  Role,
  WorkspaceRequest,
} from "../../api/types";

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Retry when ready.";
}

export default function DocumentsPage({
  request,
  workspaceId,
  roles,
  permissions,
}: {
  request: WorkspaceRequest;
  workspaceId: string;
  roles: string[];
  permissions: string[];
}) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [workspaceRoles, setWorkspaceRoles] = useState<Role[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState("recursive");
  const [aclRoles, setAclRoles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const canRead = permissions.includes("documents:read");
  const canUpload = permissions.includes("documents:upload");
  const canUpdate = permissions.includes("documents:update");
  const canManageDocuments = canUpload || canUpdate;
  const isWorkspaceAdmin = roles.includes("admin");
  const availableRoles = isWorkspaceAdmin
    ? workspaceRoles
    : workspaceRoles.filter((role) => roles.includes(role.name));
  const existingDocument = file
    ? documents.find((document) => document.source === file.name)
    : undefined;
  const replacingDocument = existingDocument !== undefined;
  const canSubmit = Boolean(
    file &&
      aclRoles.length > 0 &&
      (replacingDocument ? canUpdate : canUpload) &&
      !loading,
  );

  async function loadDocuments() {
    if (!canRead) return;
    const items = await request<DocumentInfo[]>("/v1/documents");
    setDocuments(items);
  }

  useEffect(() => {
    let active = true;
    const requests: Promise<void>[] = [];
    if (canRead) {
      requests.push(
        request<DocumentInfo[]>("/v1/documents").then((items) => {
          if (active) setDocuments(items);
        }),
      );
    }
    if (canManageDocuments) {
      requests.push(
        request<Role[]>(`/v1/workspaces/${encodeURIComponent(workspaceId)}/roles`).then(
          (items) => {
            if (!active) return;
            setWorkspaceRoles(items);
            const allowedRoles = isWorkspaceAdmin
              ? items
              : items.filter((role) => roles.includes(role.name));
            setAclRoles((current) => {
              const retained = current.filter((name) =>
                allowedRoles.some((role) => role.name === name),
              );
              if (retained.length > 0) return retained;
              const preferred =
                allowedRoles.find((role) => role.name === "employee") ?? allowedRoles[0];
              return preferred ? [preferred.name] : [];
            });
          },
        ),
      );
    }
    Promise.all(requests)
      .catch((loadError: unknown) => {
        if (active) setError(messageFor(loadError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [canManageDocuments, canRead, isWorkspaceAdmin, request, roles, workspaceId]);

  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || aclRoles.length === 0) {
      setError("Choose a document and at least one access role before uploading.");
      return;
    }
    if (replacingDocument && !canUpdate) {
      setError("Replacing an existing document requires the documents:update permission.");
      return;
    }
    if (!replacingDocument && !canUpload) {
      setError("Uploading a new document requires the documents:upload permission.");
      return;
    }
    setLoading(true);
    setError(null);
    setSuccess(null);
    const body = new FormData();
    body.set("file", file);
    body.set("strategy", strategy);
    body.set("roles", aclRoles.join(","));
    try {
      const result = await request<IngestionResult>("/v1/ingest", {
        method: "POST",
        body,
      });
      setFile(null);
      setSuccess(
        `Document indexed: ${result.inserted_chunks} chunks added; ${result.duplicate_chunks_skipped} duplicates skipped.`,
      );
      try {
        await loadDocuments();
      } catch (refreshError) {
        setError(
          `Document indexed, but the document list could not be refreshed. ${messageFor(refreshError)}`,
        );
      }
    } catch (uploadError) {
      setError(`Document upload failed. ${messageFor(uploadError)}`);
    } finally {
      setLoading(false);
    }
  }

  if (!canRead && !canManageDocuments) {
    return (
      <section className="page-section" aria-labelledby="documents-title">
        <div className="page-heading">
          <div>
            <p className="eyebrow">Workspace library</p>
            <h1 id="documents-title">Document access unavailable</h1>
          </div>
        </div>
        <p className="notice notice-warning">
          Your workspace role does not include permission to view or manage documents.
        </p>
      </section>
    );
  }

  return (
    <section className="page-section" aria-labelledby="documents-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Workspace library</p>
          <h1 id="documents-title">Documents</h1>
          <p className="muted-copy">
            Upload files and choose how their content is chunked and shared.
          </p>
        </div>
      </div>

      {canManageDocuments && (
        <form className="form-card" onSubmit={(event) => void upload(event)}>
          <h2>{replacingDocument ? "Replace a document" : "Add a document"}</h2>
          <div className="form-grid">
            <label>
              Document file
              <input
                type="file"
                accept=".txt,.md,.html,.pdf,text/plain,text/markdown,text/html,application/pdf"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              Chunking strategy
              <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
                <option value="recursive">Recursive (recommended)</option>
                <option value="fixed">Fixed size</option>
                <option value="semantic">Semantic</option>
              </select>
            </label>
            <label>
              Document access roles
              <select
                multiple
                value={aclRoles}
                onChange={(event) =>
                  setAclRoles(
                    Array.from(event.target.selectedOptions, (option) => option.value),
                  )
                }
                required
              >
                {availableRoles.map((role) => (
                  <option key={role.name} value={role.name}>
                    {role.name} — {role.description}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <p className="field-help">
            {replacingDocument && !canUpdate
              ? "Replacing this document requires documents:update."
              : !canUpload
                ? "You may replace accessible documents, but cannot upload new ones."
                : "Select one or more roles. Non-admin members may assign only their own roles."}
          </p>
          {error && <p className="notice notice-error" role="alert">{error}</p>}
          {success && <output className="notice notice-success">{success}</output>}
          <button
            className="button button-primary"
            type="submit"
            disabled={!canSubmit}
          >
            {loading
              ? "Uploading…"
              : replacingDocument
                ? "Replace document"
                : "Upload document"}
          </button>
        </form>
      )}

      {!canRead ? (
        <section className="data-card" aria-labelledby="document-list-title">
          <h2 id="document-list-title">Indexed documents</h2>
          <p className="notice notice-warning">
            Your role does not include permission to view the workspace document inventory.
          </p>
        </section>
      ) : (
        <section className="data-card" aria-labelledby="document-list-title">
          <div className="section-heading">
            <div>
              <h2 id="document-list-title">Indexed documents</h2>
              <p className="muted-copy">
                {documents.length} documents available in this workspace
              </p>
            </div>
            <button
              type="button"
              className="button button-secondary"
              onClick={() => void loadDocuments()}
            >
              Refresh
            </button>
          </div>
          {documents.length === 0 ? (
            <p className="empty-state">No documents have been indexed in this workspace yet.</p>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Document</th>
                    <th scope="col">Chunks</th>
                    <th scope="col">Chunking</th>
                    <th scope="col">Access roles</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((document) => (
                    <tr key={`${document.source}-${document.source_sha256}`}>
                      <td>{document.source}</td>
                      <td>{document.chunk_count}</td>
                      <td>{document.chunking_strategy}</td>
                      <td>{document.required_roles.join(", ") || "Workspace members"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </section>
  );
}
