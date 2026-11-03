export type ApiRequest = <T>(
  path: string,
  init?: RequestInit,
  workspaceId?: string,
) => Promise<T>;

export type WorkspaceRequest = <T>(path: string, init?: RequestInit) => Promise<T>;

export interface Workspace {
  id: string;
  name: string;
  description: string;
  roles: string[];
  permissions: string[];
  status: string | null;
  is_personal: boolean;
}

export interface WorkspaceMember {
  user_id: string;
  workspace_id: string;
  roles: string[];
  status: string;
}

export interface Role {
  name: string;
  description: string;
  permissions: string[];
  created_by?: string | null;
  created_at?: string;
}

export interface Conversation {
  conversation_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentInfo {
  source: string;
  source_sha256: string;
  bytes: number;
  chunk_count: number;
  chunking_strategy: string;
  ingested_at: string;
  workspace_id: string;
  owner_id: string;
  required_roles: string[];
}

export interface Citation {
  claim: string;
  citation_number: number | null;
  source: string | null;
  verified: boolean;
  rationale: string;
}

export interface RetrievedChunk {
  chunk: {
    chunk_id: string;
    text: string;
    source: string;
    source_sha256: string;
    chunk_index: number;
    section_heading?: string | null;
    page_number?: number | null;
    chunking_strategy: string;
    character_count: number;
    workspace_id?: string;
    owner_id?: string;
    required_roles?: string[];
  };
  dense_score?: number | null;
  bm25_score?: number | null;
  rrf_score: number;
  rerank_score?: number | null;
  rank: number;
}

export interface AnswerResponse {
  question: string;
  answer: string;
  retrieval_mode: "hybrid" | "dense" | "sparse";
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  confidence: number;
  confidence_breakdown: Record<string, number>;
  insufficient_context: boolean;
  documents_to_check: string[];
  thread_id: string;
  conversation_id: string | null;
}

export interface IngestionResult {
  document: DocumentInfo;
  inserted_chunks: number;
  duplicate_chunks_skipped: number;
}
