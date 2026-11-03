import { useEffect, useState, type FormEvent } from "react";
import type { AnswerResponse, Conversation } from "../../api/types";
import type { WorkspaceRequest } from "../../api/types";

function messageFor(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed. Retry when ready.";
}

export default function ChatPage({
  request,
  canQuery,
}: {
  request: WorkspaceRequest;
  canQuery: boolean;
}) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState("");
  const [question, setQuestion] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<"hybrid" | "dense" | "sparse">("hybrid");
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [comparison, setComparison] = useState<Record<string, AnswerResponse> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!canQuery) {
      setConversations([]);
      setConversationId("");
      return;
    }
    let active = true;
    request<Conversation[]>("/v1/conversations")
      .then((items) => {
        if (!active) return;
        setConversations(items);
        setConversationId(items[0]?.conversation_id ?? "");
      })
      .catch((loadError: unknown) => {
        if (active) setError(messageFor(loadError));
      });
    return () => {
      active = false;
    };
  }, [canQuery, request]);

  async function createConversation() {
    if (!canQuery) return;
    setError(null);
    try {
      const created = await request<Conversation>("/v1/conversations", {
        method: "POST",
        body: JSON.stringify({ title: "New conversation" }),
      });
      setConversations((items) => [created, ...items]);
      setConversationId(created.conversation_id);
      setAnswer(null);
      setComparison(null);
    } catch (createError) {
      setError(messageFor(createError));
    }
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canQuery) return;
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !conversationId) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    setComparison(null);
    try {
      const result = await request<AnswerResponse>("/v1/ask", {
        method: "POST",
        body: JSON.stringify({
          question: trimmedQuestion,
          retrieval_mode: retrievalMode,
          conversation_id: conversationId,
        }),
      });
      setAnswer(result);
    } catch (askError) {
      setError(messageFor(askError));
    } finally {
      setLoading(false);
    }
  }

  async function compareRetrieval() {
    if (!canQuery) return;
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !conversationId) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const result = await request<Record<string, AnswerResponse>>("/v1/compare", {
        method: "POST",
        body: JSON.stringify({
          question: trimmedQuestion,
          retrieval_mode: retrievalMode,
          conversation_id: conversationId,
        }),
      });
      setComparison(result);
    } catch (compareError) {
      setError(messageFor(compareError));
    } finally {
      setLoading(false);
    }
  }

  if (!canQuery) {
    return (
      <section className="page-section" aria-labelledby="chat-title">
        <div className="page-heading">
          <div>
            <p className="eyebrow">Knowledge assistant</p>
            <h1 id="chat-title">Chat access unavailable</h1>
          </div>
        </div>
        <p className="notice notice-warning">
          Your workspace role does not include permission to query documents or conversation history.
        </p>
      </section>
    );
  }

  return (
    <section className="page-section" aria-labelledby="chat-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Knowledge assistant</p>
          <h1 id="chat-title">Ask your documents</h1>
          <p className="muted-copy">
            Answers use only the documents your workspace allows you to access.
          </p>
        </div>
        <button type="button" className="button button-secondary" onClick={() => void createConversation()}>
          New conversation
        </button>
      </div>

      <div className="chat-panel">
        <div className="chat-toolbar">
          <label>
            Conversation
            <select
              aria-label="Conversation"
              value={conversationId}
              onChange={(event) => {
                setConversationId(event.target.value);
                setAnswer(null);
                setComparison(null);
              }}
            >
              {conversations.length === 0 && <option value="">No conversations yet</option>}
              {conversations.map((conversation) => (
                <option key={conversation.conversation_id} value={conversation.conversation_id}>
                  {conversation.title}
                </option>
              ))}
            </select>
          </label>
          <label>
            Retrieval mode
            <select
              value={retrievalMode}
              onChange={(event) =>
                setRetrievalMode(event.target.value as "hybrid" | "dense" | "sparse")
              }
            >
              <option value="hybrid">Hybrid search</option>
              <option value="dense">Dense search</option>
              <option value="sparse">Keyword search</option>
            </select>
          </label>
        </div>

        {error && <p className="notice notice-error" role="alert">{error}</p>}

        <form className="question-form" onSubmit={(event) => void submitQuestion(event)}>
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            aria-label="Ask your documents"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={4000}
            rows={3}
            placeholder="Ask a question about your workspace documents"
          />
          <div className="form-actions">
            <button
              className="button button-primary"
              type="submit"
              disabled={loading || !conversationId || !question.trim()}
            >
              {loading ? "Searching…" : "Send question"}
            </button>
            <button
              className="button button-secondary"
              type="button"
              disabled={loading || !conversationId || !question.trim()}
              onClick={() => void compareRetrieval()}
            >
              Compare retrieval
            </button>
          </div>
        </form>

        {loading && <output>Searching authorized workspace documents…</output>}
        {!answer && !comparison && conversations.length === 0 && (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Create a conversation to ask questions and keep its history in this workspace.</p>
          </div>
        )}
        {answer && <AnswerPanel answer={answer} />}
        {comparison && (
          <section className="comparison-grid" aria-label="Retrieval comparison">
            {(["hybrid", "dense"] as const).map((mode) => (
              <div className="answer-card" key={mode}>
                <h2>{mode === "hybrid" ? "Hybrid search" : "Dense search"}</h2>
                <p>{comparison[mode]?.answer ?? "No result returned."}</p>
                <p className="muted-copy">
                  Confidence: {Math.round((comparison[mode]?.confidence ?? 0) * 100)}%
                </p>
                {comparison[mode]?.retrieved_chunks.map((item) => (
                  <details key={item.chunk.chunk_id}>
                    <summary>
                      {item.chunk.source} · rank {item.rank}
                    </summary>
                    <p>{item.chunk.text}</p>
                  </details>
                ))}
              </div>
            ))}
          </section>
        )}
      </div>
    </section>
  );
}

function AnswerPanel({ answer }: { answer: AnswerResponse }) {
  return (
    <article className="answer-card" aria-label="Answer">
      <div className="answer-heading">
        <h2>Answer</h2>
        <span className="confidence-chip">{Math.round(answer.confidence * 100)}% confidence</span>
      </div>
      {answer.insufficient_context && (
        <p className="notice notice-warning">
          The available evidence may not fully answer this question.
        </p>
      )}
      <p className="answer-copy">{answer.answer}</p>
      {answer.citations.length > 0 && (
        <section aria-label="Citation checks">
          <h3>Citation checks</h3>
          <ul className="citation-list">
            {answer.citations.map((citation) => (
              <li
                key={`${citation.citation_number ?? "unreferenced"}-${citation.source ?? "unknown"}-${citation.claim}`}
              >
                <span>{citation.verified ? "Verified" : "Needs review"}</span>
                <span>{citation.claim}</span>
                {citation.source && <strong>{citation.source}</strong>}
              </li>
            ))}
          </ul>
        </section>
      )}
      <section aria-label="Retrieved evidence">
        <h3>Retrieved evidence</h3>
        {answer.retrieved_chunks.length === 0 ? (
          <p className="muted-copy">No supporting passages were returned.</p>
        ) : (
          answer.retrieved_chunks.map((item) => (
            <details key={item.chunk.chunk_id}>
              <summary>
                {item.chunk.source} · result {item.rank}
              </summary>
              <p>{item.chunk.text}</p>
              <p className="muted-copy">
                Hybrid score: {item.rrf_score.toFixed(3)}
                {item.chunk.page_number ? ` · page ${item.chunk.page_number}` : ""}
              </p>
            </details>
          ))
        )}
      </section>
    </article>
  );
}
