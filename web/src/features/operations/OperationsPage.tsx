import { useEffect, useState } from "react";
import { ApiError, fetchReadiness, type ReadinessResponse } from "../../api/client";
import type { WorkspaceRequest } from "../../api/types";

interface ReindexDocument {
  source: string;
  source_sha256: string;
  bytes: number;
  chunking_strategy: "fixed" | "recursive" | "semantic";
}

interface ReindexReport {
  dry_run: true;
  read_only: true;
  documents: ReindexDocument[];
}

function messageFor(error: unknown): string {
  if (error instanceof ApiError && error.status === 403) {
    return "Platform administrator access is required to view the reindex report.";
  }
  if (error instanceof Error) return error.message;
  return "Unable to load operations data. Retry the request.";
}

export default function OperationsPage({ request }: { request: WorkspaceRequest }) {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [report, setReport] = useState<ReindexReport | null>(null);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    Promise.allSettled([
      fetchReadiness(),
      request<ReindexReport>("/v1/admin/operations/reindex-dry-run"),
    ]).then(([readinessResult, reportResult]) => {
      if (!active) return;
      if (readinessResult.status === "fulfilled") {
        setReadiness(readinessResult.value);
        setReadinessError(null);
      } else {
        setReadinessError(messageFor(readinessResult.reason));
      }
      if (reportResult.status === "fulfilled") {
        setReport(reportResult.value);
        setReportError(null);
      } else {
        setReportError(messageFor(reportResult.reason));
      }
      setLoading(false);
    });
    return () => {
      active = false;
    };
  }, [request]);

  return (
    <section className="page-section" aria-labelledby="operations-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Platform administration</p>
          <h1 id="operations-title">Operations</h1>
          <p className="muted-copy">Dependency readiness and a read-only index verification report.</p>
        </div>
      </div>

      <section className="data-card" aria-labelledby="readiness-title">
        <div className="section-heading">
          <div>
            <h2 id="readiness-title">Service readiness</h2>
            <p className="muted-copy">Live status from the API readiness endpoint.</p>
          </div>
          {readiness && <span className="status-pill">{readiness.status}</span>}
        </div>
        {readinessError && <p className="notice notice-error" role="alert">{readinessError}</p>}
        {loading && <output>Loading service status…</output>}
        {readiness && (
          <div className="readiness-grid">
            {Object.entries(readiness.dependencies).map(([name, state]) => (
              <div className="readiness-item" key={name}>
                <span>{name === "postgres" ? "Postgres" : name === "qdrant" ? "Qdrant" : name}</span>
                <strong className={`readiness-${state}`}>{state}</strong>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="data-card" aria-labelledby="reindex-title">
        <div className="section-heading">
          <div>
            <h2 id="reindex-title">Reindex verification</h2>
            <p className="muted-copy">Checks archived source metadata without writing to the index.</p>
          </div>
          {report?.read_only && <span className="status-pill">Read-only dry run</span>}
        </div>
        {reportError && <p className="notice notice-error" role="alert">{reportError}</p>}
        {report && (
          report.documents.length === 0 ? (
            <p className="empty-state">No archived sources are available to verify.</p>
          ) : (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Source</th>
                    <th scope="col">SHA-256</th>
                    <th scope="col">Bytes</th>
                    <th scope="col">Chunking</th>
                  </tr>
                </thead>
                <tbody>
                  {report.documents.map((document) => (
                    <tr key={`${document.source}-${document.source_sha256}`}>
                      <td>{document.source}</td>
                      <td><code>{document.source_sha256}</code></td>
                      <td>{document.bytes}</td>
                      <td>{document.chunking_strategy}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </section>
    </section>
  );
}
