# models.py: defines typed document, retrieval, citation, and API response objects

from typing import Annotated, Literal

from pydantic import BaseModel, Field


# Append new checkpoint history while retaining only the most recent 20 exchanges.
def merge_history(previous: list[str], update: list[str]) -> list[str]:
    return (previous + update)[-40:]


# Preserve source and chunking provenance for every indexed text chunk.
class Chunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    source_sha256: str
    chunk_index: int
    section_heading: str | None = None
    page_number: int | None = None
    chunking_strategy: Literal["fixed", "recursive", "semantic"]
    character_count: int


# Represent one ranked chunk and retain scores from each retrieval stage.
class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    rank: int


# Record whether an individual claim is supported by its cited chunk.
class CitationCheck(BaseModel):
    claim: str
    citation_number: int | None
    source: str | None = None
    verified: bool
    rationale: str


# Keep graph state Pydantic-typed for predictable LangGraph transitions.
class AnswerState(BaseModel):
    question: str
    thread_id: str = "chat_session_1"
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    history: Annotated[list[str], merge_history] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: str = ""
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    retrieval_confidence: float = 0.0
    citation_coverage: float = 0.0
    answer_completeness: float = 0.0
    confidence: float = 0.0
    insufficient_context: bool = False
    documents_to_check: list[str] = Field(default_factory=list)


# Return source citations, scores, and retrieval evidence to API clients.
class AnswerResponse(BaseModel):
    question: str
    answer: str
    retrieval_mode: Literal["hybrid", "dense", "sparse"]
    citations: list[CitationCheck]
    retrieved_chunks: list[RetrievedChunk]
    confidence: float
    confidence_breakdown: dict[str, float]
    insufficient_context: bool
    documents_to_check: list[str]
    thread_id: str


# Expose one document's indexed and source-file provenance.
class DocumentInfo(BaseModel):
    source: str
    source_sha256: str
    bytes: int
    chunk_count: int
    chunking_strategy: str
    ingested_at: str


# Describe what ingestion changed, including duplicates skipped by cosine similarity.
class IngestionResult(BaseModel):
    document: DocumentInfo
    inserted_chunks: int
    duplicate_chunks_skipped: int


# Share one independent judgment result for evaluation metrics.
class EvaluationScore(BaseModel):
    score: float = Field(ge=0, le=1)
    rationale: str
