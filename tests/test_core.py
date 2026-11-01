# test_core.py: verifies chunking, ranking, citations, ingestion, persistence, and index alignment

from dataclasses import replace
import hashlib

import numpy as np

from rag_pipeline.chunking import chunk_sections
from rag_pipeline.config import Settings
from rag_pipeline.documents import DocumentSection, load_document
from rag_pipeline.generation import AnswerGenerator, CitationVerdict
from rag_pipeline.graph import build_graph
from rag_pipeline.models import AnswerState, Chunk, CitationCheck, RetrievedChunk
from rag_pipeline.retrieval import HybridRetriever, reciprocal_rank_fusion
from rag_pipeline.storage import KnowledgeBase


# Provide deterministic local embeddings for tests without external API credentials.
class FakeEmbeddings:
    # Create reproducible normalized vectors from stable text hashes.
    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.casefold().encode()).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).normal(size=32).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    # Embed a test query using the same stable local hash strategy.
    def embed_query(self, text):
        return self.embed_documents([text])[0]


# Return a positive structured citation decision for isolated citation tests.
class SupportingJudge:
    # Return the expected judge result without making an external model call.
    def invoke(self, prompt):
        return CitationVerdict(supported=True, rationale="The supplied text directly supports the claim.")


# Return deterministic graph dependencies without OpenAI or reranker calls.
class FakeGraphGenerator:
    # Initialize captured conversation context for the graph integration test.
    def __init__(self):
        self.received_history = []

    # Produce a stable cited answer for the graph integration test.
    def generate(self, question, chunks, history=None):
        self.received_history = history or []
        return "The service keeps logs for 30 days [1]."

    # Mark the one local test citation as supported.
    def verify_citations(self, answer, chunks):
        return [CitationCheck(claim="The service keeps logs for 30 days.", citation_number=1, source="policy.md", verified=True, rationale="Supported.")]

    # Return full coverage for the deterministic test result.
    def citation_coverage(self, checks):
        return 1.0

    # Return a complete deterministic answer assessment.
    def assess_completeness(self, question, answer):
        return 1.0, "Complete."


# Return the indexed test chunk and a known retrieval confidence.
class FakeGraphRetriever:
    # Configure the confidence returned for each local test retrieval.
    def __init__(self, confidence=0.9):
        self.confidence = confidence

    # Return one local chunk for every query.
    def search(self, question, mode):
        chunk = Chunk(
            chunk_id="graph",
            text="The service retains logs for 30 days.",
            source="policy.md",
            source_sha256="b" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=36,
        )
        return [RetrievedChunk(chunk=chunk, rank=1)], self.confidence


# Verify RRF combines ranks with configurable dense and sparse weights.
def test_rrf_uses_weighted_reciprocal_ranks():
    scores = reciprocal_rank_fusion(["a", "b"], ["b", "c"], 0.7, 0.3, 60)
    assert scores["a"] == 0.7 / 61
    assert scores["b"] == 0.7 / 62 + 0.3 / 61
    assert scores["c"] == 0.3 / 62


# Verify Markdown and HTML loaders preserve headings and remove executable content.
def test_loaders_preserve_sections_and_clean_html():
    markdown = load_document("guide.md", b"# Setup\nInstall the service.\n\n## Verify\nRun tests.")
    assert [section.heading for section in markdown] == ["Setup", "Verify"]
    html = load_document("guide.html", b"<h1>Policy</h1><p>Keep data private.</p><script>secret()</script>")
    assert html[0].heading == "Policy"
    assert "secret" not in html[0].text


# Verify recursive chunking preserves the section heading and the configured overlap.
def test_recursive_chunks_retain_heading_and_overlap():
    sections = [DocumentSection("Operations", "alpha beta gamma delta epsilon zeta eta theta")]
    chunks = chunk_sections(
        sections, "recursive", 22, 5, 0.72, FakeEmbeddings().embed_documents
    )
    assert len(chunks) > 1
    assert all(chunk.heading == "Operations" for chunk in chunks)
    assert any(
        set(chunks[index - 1].text.split()) & set(chunks[index].text.split())
        for index in range(1, len(chunks))
    )


# Verify fixed and semantic chunkers both produce bounded, nonempty chunks.
def test_fixed_and_semantic_chunkers_produce_chunks():
    sections = [
        DocumentSection(
            "Topics",
            "alpha topic sentence. beta topic sentence. gamma topic sentence.",
        )
    ]
    for strategy in ("fixed", "semantic"):
        chunks = chunk_sections(
            sections, strategy, 30, 4, 0.72, FakeEmbeddings().embed_documents
        )
        assert chunks
        assert all(chunk.text.strip() for chunk in chunks)


# Verify every claim citation is checked and invalid reference numbers fail closed.
def test_citation_verification_rejects_missing_and_invalid_references():
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.citation_judge = SupportingJudge()
    chunk = Chunk(
        chunk_id="one",
        text="The service retains logs for 30 days.",
        source="policy.md",
        source_sha256="a" * 64,
        chunk_index=0,
        chunking_strategy="recursive",
        character_count=36,
    )
    retrieved = [RetrievedChunk(chunk=chunk, rank=1)]
    checks = generator.verify_citations(
        "Logs are retained for 30 days [1]. This second claim has no reference. "
        "This claim points nowhere [3].",
        retrieved,
    )
    assert [item.verified for item in checks] == [True, False, False]
    assert checks[-1].citation_number == 3
    assert generator.citation_coverage(checks) == 1 / 3


# Verify ingestion skips duplicates, persists the snapshot, and keeps FAISS and chunks aligned.
def test_ingestion_deduplicates_and_reloads_consistent_snapshot(tmp_path):
    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    embeddings = FakeEmbeddings()
    knowledge_base = KnowledgeBase(settings, embeddings)
    content = b"# Policy\nKeep the service available for all customers."
    first = knowledge_base.ingest("policy.md", content, "recursive")
    duplicate = knowledge_base.ingest("copy.md", content, "recursive")
    assert first.inserted_chunks == 1
    assert duplicate.duplicate_chunks_skipped == 1
    assert knowledge_base.raw_path("policy.md", first.document.source_sha256).exists()
    assert len(knowledge_base.snapshot().chunks) == 1
    assert knowledge_base.snapshot().dense_index.ntotal == 1
    reloaded = KnowledgeBase(settings, embeddings)
    assert len(reloaded.snapshot().chunks) == reloaded.snapshot().dense_index.ntotal == 1
    assert {item.source for item in reloaded.list_documents()} == {"policy.md", "copy.md"}


# Verify dense and sparse retrieval both return persisted chunks for supported query terms.
def test_dense_and_sparse_search_use_same_indexed_corpus(tmp_path):
    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    embeddings = FakeEmbeddings()
    knowledge_base = KnowledgeBase(settings, embeddings)
    knowledge_base.ingest(
        "operations.txt",
        b"SEARCH-429 means apply exponential backoff with jitter.",
        "fixed",
    )
    retriever = HybridRetriever(knowledge_base, embeddings, settings)
    dense, _ = retriever.search("SEARCH-429", "dense")
    sparse, _ = retriever.search("SEARCH-429", "sparse")
    assert dense and sparse
    assert dense[0].chunk.chunk_id == sparse[0].chunk.chunk_id


# Verify the Pydantic graph state, sequential nodes, and stable MemorySaver thread.
def test_langgraph_runs_typed_retrieval_and_scoring():
    settings = replace(Settings.from_env(), min_retrieval_confidence=0.25)
    generator = FakeGraphGenerator()
    graph = build_graph(FakeGraphRetriever(), generator, settings)
    config = {"configurable": {"thread_id": "unit-test-thread"}}
    result = AnswerState.model_validate(
        graph.invoke(
            AnswerState(question="How long are logs kept?", thread_id="unit-test-thread"),
            config=config,
        )
    )
    assert result.answer.endswith("[1].")
    assert result.citation_coverage == 1.0
    assert result.confidence == 0.96
    assert result.history == [
        "User: How long are logs kept?",
        "Assistant: The service keeps logs for 30 days [1].",
    ]
    next_result = AnswerState.model_validate(
        graph.invoke(
            AnswerState(question="And where is that policy?", thread_id="unit-test-thread"),
            config=config,
        )
    )
    assert generator.received_history == result.history
    assert len(next_result.history) == 4


# Verify weak retrieval produces an explicit no-answer and names the closest document.
def test_langgraph_abstains_when_retrieval_confidence_is_low():
    settings = replace(Settings.from_env(), min_retrieval_confidence=0.4)
    graph = build_graph(
        FakeGraphRetriever(confidence=0.1), FakeGraphGenerator(), settings
    )
    result = AnswerState.model_validate(
        graph.invoke(
            AnswerState(question="What is the policy?", thread_id="abstain-thread"),
            config={"configurable": {"thread_id": "abstain-thread"}},
        )
    )
    assert result.insufficient_context
    assert "don't have enough reliable evidence" in result.answer
    assert result.documents_to_check == ["policy.md"]
    assert result.citation_checks == []


# Verify production API configuration requires credentials and protects routes with an API key.
def test_production_api_authentication():
    from fastapi.testclient import TestClient
    from types import SimpleNamespace

    from rag_pipeline.api import create_app

    settings = replace(
        Settings.from_env(),
        app_env="production",
        api_key="test-key-that-is-at-least-thirty-two-characters",
        openai_api_key="test-openai-key",
    )
    fake_service = SimpleNamespace(
        knowledge_base=SimpleNamespace(list_documents=lambda: [])
    )
    client = TestClient(create_app(settings=settings, service=fake_service))
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/documents").status_code == 401
    assert client.get(
        "/v1/documents",
        headers={"X-API-Key": "test-key-that-is-at-least-thirty-two-characters"},
    ).json() == []
    oversized = client.post(
        "/v1/ask",
        content=b"x" * (1024 * 1024 + 1),
        headers={
            "X-API-Key": "test-key-that-is-at-least-thirty-two-characters",
            "Content-Type": "application/json",
        },
    )
    assert oversized.status_code == 413
