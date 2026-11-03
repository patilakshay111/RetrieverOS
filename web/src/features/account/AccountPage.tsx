import { useEffect, useState, type FormEvent } from "react";
import type { WorkspaceRequest } from "../../api/types";

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Retry when ready.";
}

export default function AccountPage({ request }: { request: WorkspaceRequest }) {
  const [configured, setConfigured] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    request<{ configured: boolean }>("/v1/account/openai-key")
      .then((result) => {
        if (active) setConfigured(result.configured);
      })
      .catch((loadError: unknown) => {
        if (active) setError(messageFor(loadError));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [request]);

  async function saveKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (apiKey.trim().length < 20) {
      setError("Enter a valid API key with at least 20 characters.");
      return;
    }
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await request<void>("/v1/account/openai-key", {
        method: "PUT",
        body: JSON.stringify({ api_key: apiKey }),
      });
      setApiKey("");
      setConfigured(true);
      setSuccess("API key saved. The secret is not displayed or stored in this browser.");
    } catch (saveError) {
      setError(messageFor(saveError));
    } finally {
      setSaving(false);
    }
  }

  async function removeKey() {
    setSaving(true);
    setError(null);
    setSuccess(null);
    try {
      await request<void>("/v1/account/openai-key", { method: "DELETE" });
      setConfigured(false);
      setSuccess("API key removed.");
    } catch (removeError) {
      setError(messageFor(removeError));
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="page-section narrow-section" aria-labelledby="account-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Personal settings</p>
          <h1 id="account-title">Account</h1>
          <p className="muted-copy">
            Your provider key is encrypted by the API and scoped to your account and workspace.
          </p>
        </div>
      </div>
      <section className="form-card" aria-labelledby="key-title">
        <div className="section-heading">
          <div>
            <h2 id="key-title">OpenAI API key</h2>
            <p className="muted-copy">
              {loading
                ? "Checking key status…"
                : configured
                  ? "API key configured"
                  : "No API key configured"}
            </p>
          </div>
          {configured && <span className="status-pill">Configured</span>}
        </div>
        <form onSubmit={(event) => void saveKey(event)}>
          <label>
            OpenAI API key
            <input
              type="password"
              autoComplete="new-password"
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
              minLength={20}
              maxLength={512}
              aria-describedby="key-help"
            />
          </label>
          <p id="key-help" className="field-help">
            The full key is never returned by the API or saved in browser storage.
          </p>
          {error && <p className="notice notice-error" role="alert">{error}</p>}
          {success && <output className="notice notice-success">{success}</output>}
          <div className="form-actions">
            <button className="button button-primary" type="submit" disabled={saving || loading}>
              {saving ? "Saving…" : "Save API key"}
            </button>
            {configured && (
              <button
                className="button button-danger"
                type="button"
                onClick={() => void removeKey()}
                disabled={saving}
              >
                Remove key
              </button>
            )}
          </div>
        </form>
      </section>
    </section>
  );
}
