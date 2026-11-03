# test_core.py: verifies chunking, ranking, citations, ingestion, persistence, and index alignment

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from rag_pipeline.chunking import chunk_sections
from rag_pipeline.config import Settings
from rag_pipeline.documents import DocumentSection, load_document
from rag_pipeline.generation import AnswerGenerator, CitationVerdict
from rag_pipeline.graph import build_graph
from rag_pipeline.models import AnswerState, Chunk, CitationCheck, RetrievedChunk
from rag_pipeline.retrieval import HybridRetriever, reciprocal_rank_fusion
from rag_pipeline.storage import KnowledgeBase


def test_production_configuration_requires_qdrant_url(monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OIDC_ISSUER_URL", "https://identity.example.test/realms/rag")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db.example.test/rag")
    monkeypatch.setenv(
        "USER_KEY_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii")
    )
    monkeypatch.setenv("WEB_BASE_URL", "https://app.example.test")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.test")
    monkeypatch.setenv("QDRANT_URL", "")

    settings = Settings.from_env()

    with pytest.raises(ValueError, match="QDRANT_URL"):
        settings.validate_production()


def test_vector_store_creates_collection_and_acl_payload_indexes():
    from types import SimpleNamespace

    from qdrant_client import models

    from rag_pipeline.vector_store import VectorStore

    class FakeQdrantClient:
        def __init__(self):
            self.collection = None
            self.indexed_fields = {}

        def collection_exists(self, collection_name):
            return self.collection is not None

        def create_collection(
            self, collection_name, vectors_config, on_disk_payload
        ):
            self.collection = SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=vectors_config)
                ),
                payload_schema=self.indexed_fields,
            )

        def create_payload_index(
            self, collection_name, field_name, field_schema
        ):
            self.indexed_fields[field_name] = SimpleNamespace(data_type=field_schema)

        def get_collection(self, collection_name):
            return self.collection

        def close(self):
            pass

    client = FakeQdrantClient()
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="workspace-test",
        vector_size=3,
        client=client,
    )
    store.ensure_collection()
    collection = client.get_collection("workspace-test")

    assert collection.config.params.vectors.size == 3
    assert {
        "workspace_id",
        "owner_id",
        "required_roles",
    } <= set(collection.payload_schema)
    assert all(
        collection.payload_schema[field].data_type == models.PayloadSchemaType.KEYWORD
        for field in ("workspace_id", "owner_id", "required_roles")
    )
    store.check()


def test_api_creates_and_closes_vector_store_with_application_lifespan():
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from rag_pipeline.api import create_app

    settings = replace(
        Settings.from_env(),
        app_env="test",
        qdrant_url="http://qdrant.example.test:6333",
    )

    class FakeVectorStore:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    store = FakeVectorStore()
    with patch("rag_pipeline.api.VectorStore", return_value=store) as factory:
        app = create_app(settings=settings)
        factory.assert_not_called()
        with TestClient(app):
            factory.assert_called_once_with(
                settings.qdrant_url,
                vector_size=settings.qdrant_vector_size,
            )
            assert app.state.vector_store is store

    assert store.closed


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect.*:UserWarning")
def test_vector_store_search_applies_workspace_owner_and_role_filters():
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct
    from uuid import uuid4

    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.vector_store import VectorStore

    client = QdrantClient(":memory:")
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="workspace-acl-test",
        vector_size=3,
        client=client,
    )
    chunks = [
        Chunk(
            chunk_id="owner-chunk",
            text="owner-visible",
            source="same.md",
            source_sha256="a" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=13,
            workspace_id="workspace-a",
            owner_id="alice",
            required_roles=["hr"],
        ),
        Chunk(
            chunk_id="role-chunk",
            text="role-visible",
            source="same.md",
            source_sha256="b" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=12,
            workspace_id="workspace-a",
            owner_id="bob",
            required_roles=["employee"],
        ),
        Chunk(
            chunk_id="other-workspace-chunk",
            text="other-workspace",
            source="same.md",
            source_sha256="c" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=15,
            workspace_id="workspace-b",
            owner_id="alice",
            required_roles=["employee"],
        ),
    ]
    store.upsert(chunks, np.asarray([[1, 0, 0]] * len(chunks), dtype=np.float32))
    client.upsert(
        collection_name="workspace-acl-test",
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=[1, 0, 0],
                payload={"chunk_id": "missing-workspace", "owner_id": "alice"},
            )
        ],
    )

    results = store.search(
        np.asarray([1, 0, 0], dtype=np.float32),
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-a",
            roles=frozenset({"employee"}),
        ),
        limit=10,
    )

    assert {item.chunk.chunk_id for item in results} == {
        "owner-chunk",
        "role-chunk",
    }
    store.close()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect.*:UserWarning")
@pytest.mark.filterwarnings("ignore:Payload indexes have no effect.*:UserWarning")
def test_vector_store_rejects_malformed_acl_metadata_from_qdrant(monkeypatch):
    from types import SimpleNamespace

    from qdrant_client import QdrantClient

    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.vector_store import VectorStore

    client = QdrantClient(":memory:")
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="workspace-malformed-acl-test",
        vector_size=3,
        client=client,
    )
    store.ensure_collection()
    monkeypatch.setattr(
        client,
        "query_points",
        lambda **_kwargs: SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={"owner_id": "alice", "required_roles": []},
                    score=1.0,
                )
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="malformed document ACL metadata"):
        store.search(
            np.asarray([1, 0, 0], dtype=np.float32),
            WorkspaceAccessContext(
                user_id="alice",
                workspace_id="workspace-a",
                roles=frozenset(),
            ),
            limit=1,
        )

    store.close()


def test_vector_store_reports_qdrant_unavailable():
    from qdrant_client.http.exceptions import ResponseHandlingException

    from rag_pipeline.vector_store import VectorStore

    class UnavailableQdrantClient:
        def collection_exists(self, collection_name):
            raise ResponseHandlingException("Qdrant is offline.")

        def close(self):
            pass

    store = VectorStore(
        "http://qdrant.example.test:6333",
        vector_size=3,
        client=UnavailableQdrantClient(),
    )

    with pytest.raises(RuntimeError, match="Qdrant collection is unavailable"):
        store.check()


def test_vector_store_readiness_does_not_create_a_missing_collection():
    from qdrant_client import QdrantClient

    from rag_pipeline.vector_store import VectorStore

    client = QdrantClient(":memory:")
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="missing-collection-test",
        vector_size=3,
        client=client,
    )

    with pytest.raises(RuntimeError, match="Qdrant collection is unavailable"):
        store.check()

    assert not client.collection_exists("missing-collection-test")
    store.close()


def test_vector_store_delete_document_is_workspace_scoped():
    from qdrant_client import QdrantClient

    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.vector_store import VectorStore

    client = QdrantClient(":memory:")
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="workspace-delete-test",
        vector_size=3,
        client=client,
    )
    chunks = [
        Chunk(
            chunk_id=f"chunk-{workspace_id}",
            text="shared filename",
            source="same.md",
            source_sha256=source_digest,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=16,
            workspace_id=workspace_id,
            owner_id="alice",
        )
        for workspace_id, source_digest in (
            ("workspace-a", "a" * 64),
            ("workspace-b", "b" * 64),
        )
    ]
    store.upsert(chunks, np.asarray([[1, 0, 0]] * 2, dtype=np.float32))

    store.delete_document("workspace-a", "same.md")
    remaining = store.search(
        np.asarray([1, 0, 0], dtype=np.float32),
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-b",
            roles=frozenset(),
        ),
        limit=10,
    )

    assert [item.chunk.chunk_id for item in remaining] == ["chunk-workspace-b"]
    store.close()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect.*:UserWarning")
def test_vector_store_deletes_only_selected_chunk_ids():
    from qdrant_client import QdrantClient

    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.vector_store import VectorStore

    client = QdrantClient(":memory:")
    store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="selected-chunk-delete-test",
        vector_size=3,
        client=client,
    )
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            text=chunk_id,
            source="same.md",
            source_sha256="a" * 64,
            chunk_index=index,
            chunking_strategy="recursive",
            character_count=len(chunk_id),
            workspace_id="workspace-a",
            owner_id="alice",
        )
        for index, chunk_id in enumerate(("keep", "remove"))
    ]
    store.upsert(chunks, np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32))

    store.delete_chunks(["remove"])
    results = store.search(
        np.asarray([1, 0, 0], dtype=np.float32),
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-a",
            roles=frozenset(),
        ),
        limit=10,
    )

    assert [item.chunk.chunk_id for item in results] == ["keep"]
    store.close()


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
    def search(self, question, mode, access_context=None):
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


# Verify identical filenames remain isolated by workspace and document role.
def test_document_acl_filters_workspace_and_role_access(tmp_path):
    from rag_pipeline.models import AccessContext

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    knowledge_base = KnowledgeBase(settings, FakeEmbeddings())
    hr = AccessContext(user_id="hr-user", workspace_id="workspace-a", roles=frozenset({"hr"}))
    employee = AccessContext(
        user_id="employee-user", workspace_id="workspace-a", roles=frozenset({"employee"})
    )
    other_workspace = AccessContext(
        user_id="other-user", workspace_id="workspace-b", roles=frozenset({"hr"})
    )
    knowledge_base.ingest(
        "policy.md", b"# HR\nCompensation data is confidential.", "recursive", hr, ["hr"]
    )
    knowledge_base.ingest(
        "policy.md", b"# Employee\nOffice hours are 9 to 5.", "recursive", other_workspace, ["hr"]
    )

    assert [doc.workspace_id for doc in knowledge_base.list_documents(hr)] == ["workspace-a"]
    assert knowledge_base.list_documents(employee) == []
    assert [doc.workspace_id for doc in knowledge_base.list_documents(other_workspace)] == [
        "workspace-b"
    ]
    assert len(knowledge_base.snapshot().documents) == 2


# Verify a non-owner cannot replace another user's document in the same workspace.
def test_document_acl_blocks_unauthorized_replacement_before_embedding(tmp_path):
    from rag_pipeline.models import AccessContext

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    embeddings = FakeEmbeddings()
    knowledge_base = KnowledgeBase(settings, embeddings)
    owner = AccessContext(user_id="owner", workspace_id="workspace-a", roles=frozenset({"hr"}))
    intruder = AccessContext(
        user_id="intruder", workspace_id="workspace-a", roles=frozenset({"employee"})
    )
    knowledge_base.ingest("policy.md", b"# Policy\nOriginal document.", "recursive", owner, ["hr"])

    class NoEmbeddingCalls:
        def embed_documents(self, _texts):
            raise AssertionError("Unauthorized replacement must be rejected before embedding.")

    knowledge_base.embeddings = NoEmbeddingCalls()
    with pytest.raises(PermissionError):
        knowledge_base.ingest(
            "policy.md", b"# Policy\nReplacement document.", "recursive", intruder, ["employee"]
        )
    assert knowledge_base.list_documents(owner)[0].owner_id == "owner"


def test_denied_racing_upload_does_not_archive_raw_content(tmp_path, monkeypatch):
    from rag_pipeline.models import AccessContext

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    knowledge_base = KnowledgeBase(settings, FakeEmbeddings())
    uploader = AccessContext(
        user_id="uploader",
        workspace_id="workspace-a",
        roles=frozenset({"employee"}),
        permissions=frozenset({"documents:upload"}),
    )
    writer = AccessContext(
        user_id="writer",
        workspace_id="workspace-a",
        roles=frozenset({"admin"}),
        permissions=frozenset(
            {"documents:upload", "documents:update"}
        ),
    )
    unauthorized_content = b"An upload racing with document creation."
    original_load_document = load_document
    concurrent_write_started = False

    def create_document_during_parse(filename, content):
        nonlocal concurrent_write_started
        if not concurrent_write_started and content == unauthorized_content:
            concurrent_write_started = True
            knowledge_base.ingest(
                filename,
                b"The concurrent authorized document.",
                "recursive",
                writer,
                ["employee"],
            )
        return original_load_document(filename, content)

    monkeypatch.setattr("rag_pipeline.storage.load_document", create_document_during_parse)

    with pytest.raises(PermissionError, match="documents:update"):
        knowledge_base.ingest(
            "policy.md",
            unauthorized_content,
            "recursive",
            uploader,
            ["employee"],
        )

    unauthorized_digest = hashlib.sha256(unauthorized_content).hexdigest()
    assert not knowledge_base.raw_path("policy.md", unauthorized_digest).exists()


# Verify dense and sparse retrieval never return evidence outside the caller's document ACL.
def test_retrieval_excludes_documents_outside_workspace_and_roles(tmp_path):
    from rag_pipeline.models import AccessContext

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
        "hr.md",
        b"# HR\nSalary bands and compensation details.",
        "recursive",
        AccessContext(user_id="hr", workspace_id="acme", roles=frozenset({"hr"})),
        ["hr"],
    )
    knowledge_base.ingest(
        "staff.md",
        b"# Staff\nOffice holiday schedule for employees.",
        "recursive",
        AccessContext(user_id="staff", workspace_id="acme", roles=frozenset({"employee"})),
        ["employee"],
    )
    retriever = HybridRetriever(knowledge_base, embeddings, settings)
    caller = AccessContext(user_id="staff", workspace_id="acme", roles=frozenset({"employee"}))

    for mode in ("dense", "sparse", "hybrid"):
        results, _ = retriever.search("salary compensation", mode, caller)
        assert all(item.chunk.workspace_id == "acme" for item in results)
        assert all(
            caller.permits(
                item.chunk.workspace_id, item.chunk.owner_id, item.chunk.required_roles
            )
            for item in results
        )
        assert all(item.chunk.source != "hr.md" for item in results)


def test_retrieval_acl_lexical_store_filters_workspace_owner_and_roles():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from rag_pipeline.credentials import Base
    from rag_pipeline.lexical_store import LexicalStore
    from rag_pipeline.models import WorkspaceAccessContext

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    store = LexicalStore("sqlite://", engine=engine)
    chunks = [
        Chunk(
            chunk_id="owner",
            text="Salary compensation owner record.",
            source="same.md",
            source_sha256="a" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=34,
            workspace_id="workspace-a",
            owner_id="alice",
            required_roles=["hr"],
        ),
        Chunk(
            chunk_id="role",
            text="Salary compensation HR record.",
            source="same.md",
            source_sha256="b" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=31,
            workspace_id="workspace-a",
            owner_id="bob",
            required_roles=["hr"],
        ),
        Chunk(
            chunk_id="private",
            text="Salary compensation CEO record.",
            source="same.md",
            source_sha256="c" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=32,
            workspace_id="workspace-a",
            owner_id="carol",
            required_roles=["ceo"],
        ),
        Chunk(
            chunk_id="other-workspace",
            text="Salary compensation other workspace record.",
            source="same.md",
            source_sha256="d" * 64,
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=44,
            workspace_id="workspace-b",
            owner_id="alice",
            required_roles=["hr"],
        ),
    ]
    store.replace_document("workspace-a", "same.md", chunks[:3])
    store.replace_document("workspace-b", "same.md", chunks[3:])

    results = store.search(
        "salary compensation",
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-a",
            roles=frozenset({"hr"}),
        ),
        limit=10,
    )

    assert {item.chunk.chunk_id for item in results} == {"owner", "role"}
    assert [item.rank for item in results] == [1, 2]
    store.close()


def test_retrieval_acl_lexical_store_escapes_sqlite_like_wildcards():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from rag_pipeline.credentials import Base
    from rag_pipeline.lexical_store import LexicalStore
    from rag_pipeline.models import WorkspaceAccessContext

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    store = LexicalStore("sqlite://", engine=engine)
    chunks = [
        Chunk(
            chunk_id=chunk_id,
            text=text,
            source="same.md",
            source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            chunk_index=index,
            chunking_strategy="recursive",
            character_count=len(text),
            workspace_id="workspace-a",
            owner_id="alice",
            required_roles=[],
        )
        for index, (chunk_id, text) in enumerate(
            [
                ("literal", "annual_report policy"),
                ("wildcard", "annualXreport policy"),
            ]
        )
    ]
    store.replace_document("workspace-a", "same.md", chunks)

    results = store.search(
        "annual_report",
        WorkspaceAccessContext(
            user_id="alice", workspace_id="workspace-a", roles=frozenset()
        ),
        limit=10,
    )

    assert [item.chunk.chunk_id for item in results] == ["literal"]
    store.close()


def test_retrieval_acl_lexical_store_replaces_and_deletes_by_workspace_source():
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from rag_pipeline.credentials import Base
    from rag_pipeline.lexical_store import LexicalStore
    from rag_pipeline.models import WorkspaceAccessContext

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    store = LexicalStore("sqlite://", engine=engine)

    def make_chunk(chunk_id: str, workspace_id: str, text: str) -> Chunk:
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source="policy.md",
            source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            chunk_index=0,
            chunking_strategy="recursive",
            character_count=len(text),
            workspace_id=workspace_id,
            owner_id="alice",
            required_roles=["employee"],
        )

    old_a = make_chunk("old-a", "workspace-a", "Old workspace A policy.")
    current_a = make_chunk("new-a", "workspace-a", "Current workspace A policy.")
    current_b = make_chunk("current-b", "workspace-b", "Current workspace B policy.")
    store.replace_document("workspace-a", "policy.md", [old_a])
    store.replace_document("workspace-b", "policy.md", [current_b])
    store.replace_document("workspace-a", "policy.md", [current_a])

    results_a = store.search(
        "current workspace",
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-a",
            roles=frozenset({"employee"}),
        ),
        limit=10,
    )
    results_b = store.search(
        "current workspace",
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-b",
            roles=frozenset({"employee"}),
        ),
        limit=10,
    )

    assert [item.chunk.chunk_id for item in results_a] == ["new-a"]
    assert [item.chunk.chunk_id for item in results_b] == ["current-b"]
    store.delete_document("workspace-a", "policy.md")
    assert store.search(
        "current workspace",
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-a",
            roles=frozenset({"employee"}),
        ),
        limit=10,
    ) == []
    assert [item.chunk.chunk_id for item in store.search(
        "current workspace",
        WorkspaceAccessContext(
            user_id="alice",
            workspace_id="workspace-b",
            roles=frozenset({"employee"}),
        ),
        limit=10,
    )] == ["current-b"]
    store.close()


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect.*:UserWarning")
def test_retrieval_acl_ingestion_and_hybrid_search_use_shared_stores(tmp_path):
    from qdrant_client import QdrantClient
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    from rag_pipeline.credentials import Base
    from rag_pipeline.lexical_store import LexicalStore
    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.vector_store import VectorStore

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    embeddings = FakeEmbeddings()
    vector_client = QdrantClient(":memory:")
    vector_store = VectorStore(
        "http://qdrant.example.test:6333",
        collection_name="shared-retrieval-test",
        vector_size=32,
        client=vector_client,
    )
    lexical_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(lexical_engine)
    lexical_store = LexicalStore("sqlite://", engine=lexical_engine)
    knowledge_base = KnowledgeBase(
        settings,
        embeddings,
        vector_store=vector_store,
        lexical_store=lexical_store,
    )
    owner = WorkspaceAccessContext(
        user_id="hr-owner",
        workspace_id="workspace-a",
        roles=frozenset({"hr"}),
        permissions=frozenset({"documents:upload", "documents:update"}),
    )
    employee = WorkspaceAccessContext(
        user_id="employee",
        workspace_id="workspace-a",
        roles=frozenset({"employee"}),
    )
    other_workspace = WorkspaceAccessContext(
        user_id="hr-owner",
        workspace_id="workspace-b",
        roles=frozenset({"hr"}),
        permissions=frozenset({"documents:upload", "documents:update"}),
    )
    hr_content = b"Salary compensation bands are confidential to HR."
    other_content = b"Salary compensation bands in the other workspace."
    knowledge_base.ingest(
        "policy.md", hr_content, "recursive", owner, ["hr"]
    )
    knowledge_base.ingest(
        "policy.md", other_content, "recursive", other_workspace, ["hr"]
    )
    retriever = HybridRetriever(
        knowledge_base,
        embeddings,
        settings,
        vector_store=vector_store,
        lexical_store=lexical_store,
    )

    for mode in ("dense", "sparse", "hybrid"):
        employee_results, _ = retriever.search(
            "salary compensation", mode, employee
        )
        owner_results, _ = retriever.search("salary compensation", mode, owner)
        other_results, _ = retriever.search(
            "salary compensation", mode, other_workspace
        )
        assert employee_results == []
        assert {item.chunk.workspace_id for item in owner_results} == {"workspace-a"}
        assert {item.chunk.workspace_id for item in other_results} == {"workspace-b"}

    replacement_content = b"New annual salary bands are restricted to HR."
    knowledge_base.ingest(
        "policy.md", replacement_content, "recursive", owner, ["hr"]
    )
    replaced, _ = retriever.search("annual salary", "hybrid", owner)
    assert replaced
    assert all(item.chunk.source_sha256 == hashlib.sha256(replacement_content).hexdigest()
               for item in replaced)
    assert all(item.chunk.required_roles == ["hr"] for item in replaced)
    vector_store.close()
    lexical_store.close()


def test_reindex_preserves_document_acl_and_closes_service(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    import rag_pipeline.reindex as reindex

    settings = Settings.from_env()
    content = b"Reindexed HR-only policy."
    digest = hashlib.sha256(content).hexdigest()
    raw_path = tmp_path / "policy.md"
    raw_path.write_bytes(content)
    document = SimpleNamespace(
        source="policy.md",
        source_sha256=digest,
        chunking_strategy="recursive",
        owner_id="hr-owner",
        workspace_id="workspace-a",
        required_roles=["hr"],
    )
    ingest_calls = []
    closed = []

    class FakeKnowledgeBase:
        def list_documents(self):
            return [document]

        def raw_path(self, source, source_digest):
            assert source == document.source
            assert source_digest == document.source_sha256
            return raw_path

        def ingest(self, *args, **kwargs):
            ingest_calls.append((args, kwargs))
            return SimpleNamespace(model_dump=lambda mode: {"source": args[0]})

    class FakeService:
        def __init__(self, _settings):
            self.knowledge_base = FakeKnowledgeBase()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(reindex.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(reindex, "RAGService", FakeService)
    monkeypatch.setattr(sys, "argv", ["rag-pipeline-reindex"])

    reindex.main()
    capsys.readouterr()

    args, kwargs = ingest_calls[0]
    access_context = kwargs["access_context"]
    assert args[:3] == ("policy.md", content, "recursive")
    assert access_context.user_id == "hr-owner"
    assert access_context.workspace_id == "workspace-a"
    assert access_context.roles == frozenset({"hr"})
    assert access_context.platform_admin
    assert kwargs["required_roles"] == ["hr"]
    assert closed == [True]


def test_api_returns_service_unavailable_for_lexical_database_errors(monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy.exc import SQLAlchemyError
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    def unavailable(*_args, **_kwargs):
        raise SQLAlchemyError("catalog database is offline")

    fake_service = SimpleNamespace(
        knowledge_base=SimpleNamespace(
            ingest=unavailable,
            get_document=lambda *_args: None,
        ),
        ask=unavailable,
    )
    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        database_url=None,
        qdrant_url=None,
        user_key_encryption_key=None,
        oidc_issuer_url=None,
        oidc_jwks_url=None,
        openai_api_key="test-key",
    )
    with (
        patch("rag_pipeline.api.OpenAIEmbeddingProvider", return_value=object()),
        TestClient(create_app(settings=settings, service=fake_service)) as client,
    ):
        ask_response = client.post(
            "/v1/ask",
            json={"question": "What does the policy say?"},
        )
        ingest_response = client.post(
            "/v1/ingest",
            files={"file": ("policy.txt", b"Policy text")},
        )

    assert ask_response.status_code == 503
    assert ingest_response.status_code == 503


def test_api_returns_service_unavailable_for_qdrant_ingestion_errors():
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    def qdrant_unavailable(*_args, **_kwargs):
        raise RuntimeError("Unable to write vectors to Qdrant.")

    fake_service = SimpleNamespace(
        knowledge_base=SimpleNamespace(
            ingest=qdrant_unavailable,
            get_document=lambda *_args: None,
        ),
    )
    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        database_url=None,
        qdrant_url=None,
        user_key_encryption_key=None,
        oidc_issuer_url=None,
        oidc_jwks_url=None,
        openai_api_key="test-key",
    )
    with (
        patch("rag_pipeline.api.OpenAIEmbeddingProvider", return_value=object()),
        TestClient(create_app(settings=settings, service=fake_service)) as client,
    ):
        response = client.post(
            "/v1/ingest",
            files={"file": ("policy.txt", b"Policy text")},
        )

    assert response.status_code == 503


def test_conversation_store_isolates_user_workspace_and_persists_metadata(tmp_path):
    from uuid import UUID

    from rag_pipeline.conversations import ConversationStore
    from rag_pipeline.models import WorkspaceAccessContext

    database_url = f"sqlite:///{tmp_path / 'conversations.db'}"
    owner = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"employee"})
    )
    other_user = owner.model_copy(update={"user_id": "bob"})
    other_workspace = owner.model_copy(update={"workspace_id": "workspace-b"})

    store = ConversationStore(database_url)
    conversation = store.create(owner, "Incident response")
    assert isinstance(conversation.conversation_id, UUID)
    assert conversation.title == "Incident response"
    assert store.list(owner) == [conversation]
    assert store.get(owner, conversation.conversation_id) == conversation
    assert store.get(other_user, conversation.conversation_id) is None
    assert store.get(other_workspace, conversation.conversation_id) is None
    assert not store.delete(other_user, conversation.conversation_id)
    store.close()

    reopened_store = ConversationStore(database_url)
    assert reopened_store.list(owner) == [conversation]
    assert reopened_store.delete(owner, conversation.conversation_id)
    assert reopened_store.get(owner, conversation.conversation_id) is None
    reopened_store.close()


def test_conversation_thread_id_is_derived_from_server_scope():
    from uuid import UUID, uuid4

    from rag_pipeline.conversations import conversation_thread_id
    from rag_pipeline.models import WorkspaceAccessContext

    conversation_id = uuid4()
    owner = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"employee"})
    )

    owner_thread = conversation_thread_id(owner, conversation_id)

    assert UUID(owner_thread)
    assert conversation_thread_id(owner, conversation_id) == owner_thread
    assert conversation_thread_id(
        owner.model_copy(update={"user_id": "bob"}), conversation_id
    ) != owner_thread
    assert conversation_thread_id(
        owner.model_copy(update={"workspace_id": "workspace-b"}), conversation_id
    ) != owner_thread


def test_postgres_checkpointer_uses_managed_pool_and_runs_setup(monkeypatch):
    from rag_pipeline import checkpointing

    calls = {}

    class FakePool:
        def __init__(self, conninfo, **options):
            calls["conninfo"] = conninfo
            calls["options"] = options
            self.closed = False

        def open(self, **options):
            calls["open"] = options

        def close(self):
            self.closed = True

    class FakeSaver:
        def __init__(self, pool):
            self.pool = pool

        def setup(self):
            calls["setup"] = True

    monkeypatch.setattr(checkpointing, "ConnectionPool", FakePool)
    monkeypatch.setattr(checkpointing, "PostgresSaver", FakeSaver)

    managed = checkpointing.create_postgres_checkpointer(
        "postgresql+psycopg://rag:secret@db.example.test:5432/rag"
    )

    assert calls["conninfo"].startswith("postgresql://")
    assert calls["options"]["kwargs"]["autocommit"] is True
    assert calls["open"]["wait"] is True
    assert calls["setup"] is True
    assert managed.saver.pool is managed.pool
    managed.close()
    assert managed.pool.closed


def test_redis_answer_cache_scopes_entries_and_enforces_ttl_and_size():
    from rag_pipeline.cache import RedisAnswerCache, cache_key
    from rag_pipeline.models import AnswerResponse, WorkspaceAccessContext

    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.expiries = {}
            self.now = 0

        def get(self, key):
            if self.expiries.get(key, 0) <= self.now:
                self.values.pop(key, None)
                self.expiries.pop(key, None)
                return None
            return self.values.get(key)

        def set(self, key, value, ex):
            self.values[key] = value
            self.expiries[key] = self.now + ex
            return True

    scope = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"employee"})
    )
    key = cache_key("secret prompt", "hybrid", scope, "generation-1")
    assert key != cache_key("secret prompt", "hybrid", scope, "generation-2")
    assert key != cache_key(
        "secret prompt",
        "hybrid",
        scope.model_copy(update={"user_id": "bob"}),
        "generation-1",
    )
    assert key != cache_key(
        "secret prompt",
        "hybrid",
        scope.model_copy(update={"workspace_id": "workspace-b"}),
        "generation-1",
    )
    assert key != cache_key(
        "secret prompt",
        "hybrid",
        scope.model_copy(update={"roles": frozenset({"hr"})}),
        "generation-1",
    )
    assert "secret prompt" not in key
    assert "alice" not in key

    redis_client = FakeRedis()
    cache = RedisAnswerCache(redis_client, ttl_seconds=120, max_value_bytes=4096)
    response = AnswerResponse(
        question="secret prompt",
        answer="The safe answer.",
        retrieval_mode="hybrid",
        citations=[],
        retrieved_chunks=[],
        confidence=0.5,
        confidence_breakdown={},
        insufficient_context=False,
        documents_to_check=[],
        thread_id="server-thread",
    )

    assert cache.get(key) is None
    assert cache.set(key, response)
    assert cache.get(key) == response
    assert redis_client.expiries[key] == 120
    redis_client.now = 120
    assert cache.get(key) is None

    tiny_cache = RedisAnswerCache(FakeRedis(), max_value_bytes=10)
    assert not tiny_cache.set(key, response)
    assert tiny_cache.get(key) is None


def test_redis_rate_limiter_uses_hashed_scopes_and_denies_over_limit():
    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.rate_limits import RedisRateLimiter, rate_limit_key

    class FakeRedis:
        def __init__(self):
            self.counts = {}

        def eval(self, script, key_count, key, window_seconds):
            assert key_count == 1
            assert "INCR" in script and "EXPIRE" in script
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

    scope = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"employee"})
    )
    limiter = RedisRateLimiter(FakeRedis(), requests=2, window_seconds=60)

    assert limiter.allow(scope, "ask")
    assert limiter.allow(scope, "ask")
    assert not limiter.allow(scope, "ask")
    assert rate_limit_key(scope, "ask") != rate_limit_key(
        scope.model_copy(update={"workspace_id": "workspace-b"}), "ask"
    )
    assert rate_limit_key(scope, "ask") != rate_limit_key(scope, "ingest")
    assert "alice" not in rate_limit_key(scope, "ask")
    assert "workspace-a" not in rate_limit_key(scope, "ask")


def test_ask_endpoint_enforces_shared_redis_rate_limit(monkeypatch):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.models import AnswerResponse

    class FakeRedis:
        def __init__(self):
            self.count = 0

        def eval(self, *_args):
            self.count += 1
            return self.count

        def close(self):
            pass

    calls = []

    def ask(question, retrieval_mode, thread_id, access_context, api_key, **_kwargs):
        calls.append(question)
        return AnswerResponse(
            question=question,
            answer="Grounded answer.",
            retrieval_mode=retrieval_mode,
            citations=[],
            retrieved_chunks=[],
            confidence=0.0,
            confidence_breakdown={},
            insufficient_context=False,
            documents_to_check=[],
            thread_id=thread_id,
        )

    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        database_url=None,
        user_key_encryption_key=None,
        qdrant_url=None,
        redis_url="redis://redis.example.test:6379/0",
        redis_rate_limit_requests=1,
        openai_api_key="test-key",
    )
    fake_redis = FakeRedis()
    with (
        patch("rag_pipeline.api.Redis.from_url", return_value=fake_redis),
        TestClient(
            create_app(
                settings=settings,
                service=SimpleNamespace(ask=ask),
            )
        ) as client,
    ):
        first = client.post("/v1/ask", json={"question": "first"})
        second = client.post("/v1/ask", json={"question": "second"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert calls == ["first"]


def test_ask_uses_answer_cache_only_for_stateless_service(tmp_path):
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.models import AnswerResponse
    from rag_pipeline.service import RAGService

    class FakeRedis:
        def __init__(self):
            self.values = {}
            self.counts = {}

        def eval(self, _script, _key_count, key, _window):
            self.counts[key] = self.counts.get(key, 0) + 1
            return self.counts[key]

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value, ex):
            self.values[key] = value
            assert ex == 60
            return True

        def delete(self, key):
            self.values.pop(key, None)

        def close(self):
            pass

    settings = replace(
        Settings.from_env(),
        app_env="test",
        database_url=None,
        user_key_encryption_key=None,
        qdrant_url=None,
        redis_url="redis://redis.example.test:6379/0",
        openai_api_key="test-key",
        data_dir=tmp_path,
        reranker_enabled=False,
    )
    service = RAGService(settings, use_memory_saver=False)
    calls = []

    def ask(question, retrieval_mode, thread_id, _scope, _api_key, **_kwargs):
        calls.append(question)
        return AnswerResponse(
            question=question,
            answer="Grounded answer.",
            retrieval_mode=retrieval_mode,
            citations=[],
            retrieved_chunks=[],
            confidence=0.0,
            confidence_breakdown={},
            insufficient_context=False,
            documents_to_check=[],
            thread_id=thread_id,
        )

    service.ask = ask
    fake_redis = FakeRedis()
    with (
        patch("rag_pipeline.api.Redis.from_url", return_value=fake_redis),
        TestClient(create_app(settings=settings, service=service)) as client,
    ):
        first = client.post("/v1/ask", json={"question": "same question"})
        second = client.post("/v1/ask", json={"question": "same question"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert calls == ["same question"]
    assert all("same question" not in key for key in fake_redis.values)


def test_readiness_checks_postgres_qdrant_and_redis(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    class FakeRedis:
        def ping(self):
            return True

        def close(self):
            pass

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            pass

        def check(self):
            return None

        def close(self):
            pass

    settings = replace(
        Settings.from_env(),
        app_env="test",
        database_url="postgresql+psycopg://user:password@postgres.example.test/rag",
        user_key_encryption_key="MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
        qdrant_url="http://qdrant.example.test:6333",
        redis_url="redis://redis.example.test:6379/0",
    )
    managed_checkpointer = SimpleNamespace(saver=object(), close=lambda: None)
    fake_store = SimpleNamespace(check=lambda: None, close=lambda: None)
    with (
        patch("rag_pipeline.api.VectorStore", FakeVectorStore),
        patch("rag_pipeline.api.Redis.from_url", return_value=FakeRedis()),
        patch(
            "rag_pipeline.api.create_postgres_checkpointer",
            return_value=managed_checkpointer,
        ),
        patch("rag_pipeline.api.UserCredentialStore", return_value=fake_store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=fake_store),
        patch("rag_pipeline.api.AdminAuditStore", return_value=fake_store),
        patch("rag_pipeline.api.ConversationStore", return_value=fake_store),
        TestClient(create_app(settings=settings)) as client,
    ):
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {
            "postgres": "ready",
            "qdrant": "ready",
            "redis": "ready",
        },
    }


def test_readiness_reports_configured_dependency_failure(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from sqlalchemy.exc import SQLAlchemyError
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    class FakeRedis:
        def ping(self):
            from redis.exceptions import ConnectionError

            raise ConnectionError("redis is offline")

        def close(self):
            pass

    settings = replace(
        Settings.from_env(),
        app_env="test",
        database_url="postgresql+psycopg://user:password@postgres.example.test/rag",
        user_key_encryption_key="MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
        qdrant_url=None,
        redis_url="redis://redis.example.test:6379/0",
    )
    managed_checkpointer = SimpleNamespace(saver=object(), close=lambda: None)
    fake_store = SimpleNamespace(check=None, close=lambda: None)
    def database_unavailable():
        raise SQLAlchemyError("database unavailable")

    credential_store = SimpleNamespace(check=database_unavailable, close=lambda: None)
    with (
        patch("rag_pipeline.api.Redis.from_url", return_value=FakeRedis()),
        patch(
            "rag_pipeline.api.create_postgres_checkpointer",
            return_value=managed_checkpointer,
        ),
        patch("rag_pipeline.api.UserCredentialStore", return_value=credential_store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=fake_store),
        patch("rag_pipeline.api.AdminAuditStore", return_value=fake_store),
        patch("rag_pipeline.api.ConversationStore", return_value=fake_store),
        TestClient(create_app(settings=settings)) as client,
    ):
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "status": "not_ready",
        "dependencies": {
            "postgres": "unavailable",
            "qdrant": "not_configured",
            "redis": "unavailable",
        },
    }


def test_production_readiness_requires_redis():
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    settings = replace(
        Settings.from_env(),
        app_env="production",
        web_base_url="https://app.example.test",
        api_key=None,
        qdrant_url="http://qdrant.example.test:6333",
        oidc_issuer_url="https://identity.example.test/realms/rag",
        database_url="postgresql://localhost/rag",
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
        redis_url="redis://redis.example.test:6379/0",
    )
    class OfflineRedis:
        def ping(self):
            from redis.exceptions import ConnectionError

            raise ConnectionError("redis is offline")

        def close(self):
            pass

    fake_store = SimpleNamespace(check=lambda: None, close=lambda: None)
    fake_vector_store = SimpleNamespace(check=lambda: None, close=lambda: None)
    managed_checkpointer = SimpleNamespace(saver=object(), close=lambda: None)
    with (
        patch("rag_pipeline.api.UserCredentialStore", return_value=fake_store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=fake_store),
        patch("rag_pipeline.api.AdminAuditStore", return_value=fake_store),
        patch("rag_pipeline.api.ConversationStore", return_value=fake_store),
        patch("rag_pipeline.api.VectorStore", return_value=fake_vector_store),
        patch(
            "rag_pipeline.api.create_postgres_checkpointer",
            return_value=managed_checkpointer,
        ),
        patch("rag_pipeline.api.Redis.from_url", return_value=OfflineRedis()),
        TestClient(create_app(settings=settings)) as client,
    ):
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"]["dependencies"]["redis"] == "unavailable"


def test_reindex_dry_run_verifies_sources_without_ingesting(
    monkeypatch, capsys, tmp_path
):
    import hashlib
    import json
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.reindex import main

    content = b"# Verified source"
    digest = hashlib.sha256(content).hexdigest()
    document = SimpleNamespace(
        source="policy.md",
        source_sha256=digest,
        chunking_strategy="recursive",
        workspace_id="workspace-a",
        owner_id="alice",
        required_roles=["employee"],
    )
    source_path = tmp_path / "policy.md"
    source_path.write_bytes(content)
    ingest_calls = []

    class FakeKnowledgeBase:
        def list_documents(self):
            return [document]

        def raw_path(self, source, source_digest):
            assert source == document.source
            assert source_digest == document.source_sha256
            return source_path

        def ingest(self, *args, **kwargs):
            ingest_calls.append((args, kwargs))

    fake_service = SimpleNamespace(
        knowledge_base=FakeKnowledgeBase(), close=lambda: None
    )
    monkeypatch.setattr(sys, "argv", ["rag-pipeline-reindex", "--dry-run"])
    with patch("rag_pipeline.reindex.RAGService", return_value=fake_service):
        main()

    result = json.loads(capsys.readouterr().out)
    assert result == {
        "dry_run": True,
        "documents": [
            {
                "source": "policy.md",
                "source_sha256": digest,
                "bytes": len(content),
                "workspace_id": "workspace-a",
                "owner_id": "alice",
                "required_roles": ["employee"],
                "chunking_strategy": "recursive",
            }
        ],
    }
    assert ingest_calls == []


def test_verify_archived_sources_streams_without_returning_source_bytes(
    monkeypatch, tmp_path
):
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    from rag_pipeline.reindex import verify_archived_sources

    content = b"A large archived source"
    digest = hashlib.sha256(content).hexdigest()
    document = SimpleNamespace(
        source="policy.md",
        source_sha256=digest,
        chunking_strategy="recursive",
    )
    source_path = tmp_path / "policy.md"
    source_path.write_bytes(content)
    knowledge_base = SimpleNamespace(
        list_documents=lambda: [document],
        raw_path=lambda _source, _digest: source_path,
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("archive verification must stream source bytes"),
    )

    assert verify_archived_sources(knowledge_base) == [(document, len(content))]


def test_operations_dry_run_requires_platform_admin(workspace_api):
    endpoint = "/v1/admin/operations/reindex-dry-run"

    assert workspace_api.client.get(endpoint).status_code == 401
    response = workspace_api.client.get(
        endpoint, headers={"Authorization": "Bearer employee-token"}
    )

    assert response.status_code == 403


def test_operations_dry_run_returns_verified_metadata_without_ingesting(
    workspace_api, tmp_path
):
    import hashlib
    from types import SimpleNamespace

    content = b"secret document contents"
    digest = hashlib.sha256(content).hexdigest()
    document = SimpleNamespace(
        source="policy.md",
        source_sha256=digest,
        chunking_strategy="recursive",
        workspace_id="private-workspace",
        owner_id="private-owner",
        required_roles=["private-role"],
    )
    source_path = tmp_path / "policy.md"
    source_path.write_bytes(content)
    ingest_calls = []
    knowledge_base = workspace_api.service.knowledge_base
    knowledge_base.list_documents = lambda: [document]
    knowledge_base.raw_path = lambda source, source_digest: source_path
    knowledge_base.ingest = lambda *args, **kwargs: ingest_calls.append(args)

    response = workspace_api.client.get(
        "/v1/admin/operations/reindex-dry-run",
        headers={"Authorization": "Bearer platform-token"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "dry_run": True,
        "read_only": True,
        "documents": [
            {
                "source": "policy.md",
                "source_sha256": digest,
                "bytes": len(content),
                "chunking_strategy": "recursive",
            }
        ],
    }
    assert "secret document contents" not in response.text
    assert "private-owner" not in response.text
    assert "private-role" not in response.text
    assert ingest_calls == []


def test_operations_dry_run_rejects_digest_mismatch(workspace_api, tmp_path):
    from types import SimpleNamespace

    document = SimpleNamespace(
        source="policy.md",
        source_sha256="a" * 64,
        chunking_strategy="recursive",
    )
    source_path = tmp_path / "policy.md"
    source_path.write_bytes(b"modified document contents")
    knowledge_base = workspace_api.service.knowledge_base
    knowledge_base.list_documents = lambda: [document]
    knowledge_base.raw_path = lambda source, source_digest: source_path

    response = workspace_api.client.get(
        "/v1/admin/operations/reindex-dry-run",
        headers={"Authorization": "Bearer platform-token"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "The archived source manifest could not be verified."
    assert "modified document contents" not in response.text


def test_operations_dry_run_reports_missing_manifest(workspace_api):
    workspace_api.service.knowledge_base.list_documents = lambda: []

    response = workspace_api.client.get(
        "/v1/admin/operations/reindex-dry-run",
        headers={"Authorization": "Bearer platform-token"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "The archived source manifest could not be verified."


def test_ask_uses_owned_conversation_and_ignores_client_thread_id(tmp_path):
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch
    from uuid import UUID

    from rag_pipeline.api import create_app
    from rag_pipeline.conversations import conversation_thread_id
    from rag_pipeline.identity_store import WorkspaceStore
    from rag_pipeline.models import AnswerResponse, VerifiedIdentity

    identities = {
        "alice-token": VerifiedIdentity(user_id="alice"),
        "bob-token": VerifiedIdentity(user_id="bob"),
    }

    class FakeVerifier:
        def verify(self, token):
            return identities[token]

    calls = []

    def ask(
        question,
        retrieval_mode,
        thread_id,
        access_context,
        api_key,
        conversation_id=None,
        checkpoint_variant=None,
    ):
        calls.append(
            (thread_id, access_context, api_key, conversation_id, checkpoint_variant)
        )
        return AnswerResponse(
            question=question,
            answer="Grounded answer.",
            retrieval_mode=retrieval_mode,
            citations=[],
            retrieved_chunks=[],
            confidence=0.0,
            confidence_breakdown={},
            insufficient_context=False,
            documents_to_check=[],
            thread_id=thread_id,
        )

    class FakeCredentialStore:
        def __init__(self, *_args):
            pass

        def get_key(self, *_args):
            return "test-user-key"

        def close(self):
            pass

    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        database_url=f"sqlite:///{tmp_path / 'ask-conversation.db'}",
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
        oidc_issuer_url="https://identity.example.test/realms/rag",
        oidc_jwks_url="https://identity.example.test/jwks",
        qdrant_url=None,
        openai_api_key=None,
    )
    workspace_store = WorkspaceStore(settings.database_url)
    alice_workspace = workspace_store.create_workspace("Alice workspace")
    bob_workspace = workspace_store.create_workspace("Bob workspace")
    workspace_store.add_membership("alice", alice_workspace.id, roles={"employee"})
    workspace_store.add_membership("bob", bob_workspace.id, roles={"employee"})
    workspace_store.close()
    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=FakeVerifier()),
        patch("rag_pipeline.api.UserCredentialStore", FakeCredentialStore),
        TestClient(
            create_app(
                settings=settings,
                service=SimpleNamespace(ask=ask),
            )
        ) as client,
    ):
        alice_headers = {
            "Authorization": "Bearer alice-token",
            "X-Workspace-ID": alice_workspace.id,
        }
        bob_headers = {
            "Authorization": "Bearer bob-token",
            "X-Workspace-ID": bob_workspace.id,
        }
        created = client.post(
            "/v1/conversations",
            headers=alice_headers,
            json={"title": "Owned thread"},
        )
        conversation_id = created.json()["conversation_id"]
        response = client.post(
            "/v1/ask",
            headers=alice_headers,
            json={
                "question": "What happened?",
                "conversation_id": conversation_id,
                "thread_id": "attacker-controlled",
            },
        )
        unauthorized = client.post(
            "/v1/ask",
            headers=bob_headers,
            json={
                "question": "What happened?",
                "conversation_id": conversation_id,
            },
        )

    assert response.status_code == 200
    assert response.json()["conversation_id"] == conversation_id
    expected_thread_id = conversation_thread_id(
        calls[0][1],
        UUID(conversation_id),
    )
    assert calls == [
        (expected_thread_id, calls[0][1], "test-user-key", UUID(conversation_id), None)
    ]
    assert calls[0][1].user_id == "alice"
    assert unauthorized.status_code == 404


def test_service_uses_conversation_id_for_scope_isolated_checkpoints(tmp_path):
    from langgraph.checkpoint.memory import MemorySaver
    from uuid import uuid4

    from rag_pipeline.models import WorkspaceAccessContext
    from rag_pipeline.service import RAGService

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        openai_api_key=None,
        qdrant_url=None,
        database_url=None,
        min_retrieval_confidence=0.0,
    )
    checkpointer = MemorySaver()
    generator = FakeGraphGenerator()
    service = RAGService(
        settings,
        embeddings=FakeEmbeddings(),
        generator=generator,
        checkpointer=checkpointer,
    )

    assert service._checkpointer is checkpointer
    conversation_id = uuid4()
    alice = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"member"})
    )
    bob = alice.model_copy(update={"user_id": "bob"})
    service.ask(
        "First question",
        thread_id="client-thread-a",
        access_context=alice,
        conversation_id=conversation_id,
    )
    service.close()

    next_generator = FakeGraphGenerator()
    restarted_service = RAGService(
        settings,
        embeddings=FakeEmbeddings(),
        generator=next_generator,
        checkpointer=checkpointer,
    )
    restarted_service.ask(
        "Second question",
        thread_id="client-thread-b",
        access_context=alice,
        conversation_id=conversation_id,
    )
    assert next_generator.received_history == [
        "User: First question",
        "Assistant: The service keeps logs for 30 days [1].",
    ]
    restarted_service.ask(
        "Bob question",
        thread_id="client-thread-a",
        access_context=bob,
        conversation_id=conversation_id,
    )
    assert next_generator.received_history == []
    restarted_service.close()


def test_service_does_not_use_memory_saver_for_postgres_by_default(tmp_path):
    from rag_pipeline.service import RAGService

    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        openai_api_key=None,
        qdrant_url=None,
        database_url="postgresql://localhost/rag",
    )
    service = RAGService(
        settings,
        embeddings=FakeEmbeddings(),
        generator=FakeGraphGenerator(),
    )

    assert service._checkpointer is None
    service.close()


def test_conversation_api_scopes_listing_and_deletion_by_user_and_workspace(tmp_path):
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.identity_store import WorkspaceStore
    from rag_pipeline.models import VerifiedIdentity

    database_url = f"sqlite:///{tmp_path / 'conversation-api.db'}"
    workspace_store = WorkspaceStore(database_url)
    shared_workspace = workspace_store.create_workspace("Shared workspace")
    other_workspace = workspace_store.create_workspace("Other workspace")
    bob_workspace = workspace_store.create_workspace("Bob workspace")
    workspace_store.add_membership(
        "alice", shared_workspace.id, roles={"employee"}
    )
    workspace_store.add_membership(
        "alice", other_workspace.id, roles={"employee"}
    )
    workspace_store.add_membership(
        "bob", bob_workspace.id, roles={"employee"}
    )
    workspace_store.close()

    class FakeVerifier:
        def verify(self, token):
            return VerifiedIdentity(user_id=token)

    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        database_url=database_url,
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
        oidc_issuer_url="https://identity.example.test/realms/rag",
        oidc_jwks_url="https://identity.example.test/jwks",
        qdrant_url=None,
    )
    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=FakeVerifier()),
        TestClient(create_app(settings=settings)) as client,
    ):
        alice_headers = {"Authorization": "Bearer alice"}
        bob_headers = {"Authorization": "Bearer bob"}
        alice_headers["X-Workspace-ID"] = shared_workspace.id
        bob_headers["X-Workspace-ID"] = bob_workspace.id
        created = client.post(
            "/v1/conversations",
            headers=alice_headers,
            json={"title": "Incident response"},
        )
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        alice_list = client.get("/v1/conversations", headers=alice_headers)
        bob_get = client.get(
            f"/v1/conversations/{conversation_id}", headers=bob_headers
        )
        bob_delete = client.delete(
            f"/v1/conversations/{conversation_id}", headers=bob_headers
        )
        other_workspace_get = client.get(
            f"/v1/conversations/{conversation_id}",
            headers={**alice_headers, "X-Workspace-ID": other_workspace.id},
        )
        alice_delete = client.delete(
            f"/v1/conversations/{conversation_id}", headers=alice_headers
        )

    assert alice_list.status_code == 200
    assert [item["conversation_id"] for item in alice_list.json()] == [
        conversation_id
    ]
    assert bob_get.status_code == 404
    assert bob_delete.status_code == 404
    assert other_workspace_get.status_code == 404
    assert alice_delete.status_code == 204


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


def test_langgraph_reads_authorization_scope_from_invocation_config():
    from rag_pipeline.models import WorkspaceAccessContext

    class CapturingRetriever(FakeGraphRetriever):
        def __init__(self):
            super().__init__()
            self.scopes = []

        def search(self, query, mode, access_context=None):
            self.scopes.append(access_context)
            return super().search(query, mode, access_context)

    scope = WorkspaceAccessContext(
        user_id="alice", workspace_id="workspace-a", roles=frozenset({"member"})
    )
    retriever = CapturingRetriever()
    graph = build_graph(
        retriever,
        FakeGraphGenerator(),
        replace(Settings.from_env(), min_retrieval_confidence=0.0),
    )
    graph.invoke(
        AnswerState(question="What does the policy say?"),
        config={
            "configurable": {
                "thread_id": "scope-config-thread",
                "access_context": scope.model_dump(mode="json"),
            }
        },
    )

    assert [item.model_dump() for item in retriever.scopes] == [scope.model_dump()]


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


# Verify production mode requires OIDC and rejects the legacy shared API-key login.
def test_production_api_authentication():
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from cryptography.fernet import Fernet
    from unittest.mock import patch

    from rag_pipeline.api import create_app

    settings = replace(
        Settings.from_env(),
        app_env="production",
        web_base_url="https://app.example.test",
        api_key=None,
        qdrant_url="http://qdrant.example.test:6333",
        redis_url="redis://redis.example.test:6379/0",
        oidc_issuer_url="https://identity.example.test/realms/retrieveros",
        openai_api_key="test-openai-key",
        database_url="postgresql+psycopg://user:password@localhost/test",
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
    )
    fake_service = SimpleNamespace(
        knowledge_base=SimpleNamespace(list_documents=lambda _access_context=None: [])
    )
    fake_store = SimpleNamespace(check=lambda: None)
    with (
        patch("rag_pipeline.api.UserCredentialStore", return_value=fake_store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=fake_store),
    ):
        client = TestClient(create_app(settings=settings, service=fake_service))
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/documents").status_code == 401
    assert client.get(
        "/v1/documents",
        headers={"X-API-Key": "test-key-that-is-at-least-thirty-two-characters"},
    ).status_code == 401
    oversized = client.post(
        "/v1/ask",
        content=b"x" * (1024 * 1024 + 1),
        headers={
            "X-API-Key": "test-key-that-is-at-least-thirty-two-characters",
            "Content-Type": "application/json",
        },
    )
    assert oversized.status_code == 413


# Verify Keycloak tokens yield identity and platform-admin status, not workspace roles.
def test_oidc_verifier_validates_signature_issuer_audience_and_access_claims():
    from datetime import datetime, timedelta, timezone

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    from rag_pipeline.auth import OIDCVerifier
    from rag_pipeline.models import VerifiedIdentity

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    issuer = "https://identity.example.test/realms/retrieveros"
    verifier = OIDCVerifier(issuer=issuer, audience="retrieveros-api")

    class StaticSigningKey:
        key = private_key.public_key()

    class StaticJwks:
        @staticmethod
        def get_signing_key_from_jwt(_token):
            return StaticSigningKey()

    verifier.__dict__["jwks"] = StaticJwks()
    now = datetime.now(timezone.utc)
    claims = {
        "iss": issuer,
        "aud": "retrieveros-api",
        "sub": "user-123",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "workspace_id": "workspace-abc",
        "roles": ["hr"],
        "email": "user@example.test",
        "email_verified": True,
        "realm_access": {"roles": ["platform-admin"]},
    }
    token = jwt.encode(claims, private_key, algorithm="RS256")

    context = verifier.verify(token)

    assert context == VerifiedIdentity(
        user_id="user-123",
        platform_admin=True,
        email="user@example.test",
        email_verified=True,
    )
    assert not hasattr(context, "workspace_id")
    assert not hasattr(context, "roles")
    assert context.email == "user@example.test"
    assert context.email_verified

    claims["aud"] = "different-api"
    invalid_token = jwt.encode(claims, private_key, algorithm="RS256")
    with pytest.raises(jwt.InvalidAudienceError):
        verifier.verify(invalid_token)


# Verify API keys are encrypted at rest and isolated to their user/workspace.
def test_user_credential_store_encrypts_and_scopes_keys():
    from cryptography.fernet import Fernet
    from sqlalchemy.orm import Session

    from rag_pipeline.credentials import UserCredentialStore, UserOpenAIKey

    plaintext = "sk-test-user-specific-key-never-store-plaintext"
    store = UserCredentialStore(
        "sqlite:///:memory:", Fernet.generate_key().decode("ascii")
    )
    store.set_key("user-a", "workspace-a", plaintext)

    assert store.has_key("user-a", "workspace-a")
    assert store.get_key("user-a", "workspace-a") == plaintext
    assert store.get_key("user-b", "workspace-a") is None
    assert store.get_key("user-a", "workspace-b") is None
    with Session(store._engine) as session:
        encrypted = session.get(UserOpenAIKey, ("user-a", "workspace-a")).encrypted_key
    assert plaintext.encode("utf-8") not in encrypted
    store.delete_key("user-a", "workspace-a")
    assert not store.has_key("user-a", "workspace-a")


# Verify the checked-in migration creates the schema used by credential storage.
def test_credential_schema_migration(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    database_path = tmp_path / "migration.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(Config("alembic.ini"), "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    tables = inspect(engine).get_table_names()
    engine.dispose()

    assert "user_openai_keys" in tables
    assert "conversations" in tables
    assert "alembic_version" in tables


# Verify workspace memberships are durable, unique, and scoped by user/workspace.
def test_workspace_membership_repository_supports_multiple_memberships_and_roles():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    personal = store.create_personal_workspace("user-a")
    shared = store.create_workspace("Shared workspace")
    store.add_membership("user-a", shared.id, roles={"hr", "employee"})

    memberships = store.list_memberships("user-a")

    assert {membership.workspace_id for membership in memberships} == {
        personal.workspace_id,
        shared.id,
    }
    assert store.get_membership("user-a", shared.id).roles == frozenset({"hr", "employee"})
    assert store.get_membership("user-b", shared.id) is None


def test_workspace_creation_creates_named_workspace_and_admin_membership():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")

    workspace, membership = store.create_workspace_for_user(
        "owner-1", "Research", "Internal research library"
    )

    assert workspace.name == "Research"
    assert workspace.description == "Internal research library"
    assert membership.user_id == "owner-1"
    assert membership.workspace_id == workspace.id
    assert membership.roles == frozenset({"admin"})
    assert membership.status == "active"
    assert store.list_workspace_roles(workspace.id)[0].permissions == frozenset(
        {"documents:read", "documents:query", "documents:upload", "documents:update"}
    )


def test_workspace_update_changes_name_and_description():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    workspace = store.create_workspace("Old name")

    updated = store.update_workspace(workspace.id, "New name", "Updated purpose")

    assert updated is not None
    assert updated.name == "New name"
    assert updated.description == "Updated purpose"


def test_roles_are_scoped_by_workspace_and_permissions():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    first = store.create_workspace("First")
    second = store.create_workspace("Second")
    store.create_workspace_role(
        first.id, "analyst", "Read-only analyst", {"documents:read"}, "owner-a"
    )
    store.create_workspace_role(
        second.id, "analyst", "Uploader analyst", {"documents:upload"}, "owner-b"
    )

    first_role = next(
        role for role in store.list_workspace_roles(first.id) if role.name == "analyst"
    )
    second_role = next(
        role for role in store.list_workspace_roles(second.id) if role.name == "analyst"
    )

    assert first_role.description == "Read-only analyst"
    assert first_role.permissions == frozenset({"documents:read"})
    assert second_role.permissions == frozenset({"documents:upload"})
    assert store.get_role_permissions(first.id, {"analyst"}) == frozenset(
        {"documents:read"}
    )


def test_workspace_local_role_creation_rejects_unknown_permissions():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    workspace = store.create_workspace("Research")

    with pytest.raises(ValueError, match="(?i)unknown permission"):
        store.create_workspace_role(
            workspace.id,
            "analyst",
            "Invalid permissions",
            {"documents:read", "users:impersonate"},
            "owner-1",
        )


def test_personal_workspace_provisioning_is_idempotent():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")

    first = store.create_personal_workspace("user-a")
    second = store.create_personal_workspace("user-a")

    assert first.workspace_id == second.workspace_id
    assert store.list_memberships("user-a") == [first]
    assert first.roles == frozenset({"admin"})


def test_workspace_membership_rejects_unknown_roles():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    workspace = store.create_workspace("Shared workspace")

    with pytest.raises(ValueError, match="(?i)unknown role"):
        store.add_membership("user-a", workspace.id, roles={"superuser"})


def test_workspace_membership_status_can_be_updated():
    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    workspace = store.create_workspace("Shared workspace")
    store.add_membership("user-a", workspace.id, roles={"employee"})

    suspended = store.set_membership_status("user-a", workspace.id, "suspended")

    assert suspended is not None
    assert suspended.status == "suspended"
    assert store.get_membership("user-a", workspace.id).status == "suspended"


def test_workspace_membership_unique_user_workspace_pair():
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Session

    from rag_pipeline.identity_store import WorkspaceMembership, WorkspaceStore

    store = WorkspaceStore("sqlite:///:memory:")
    workspace = store.create_workspace("Shared workspace")
    store.add_membership("user-a", workspace.id, roles={"employee"})

    with pytest.raises(IntegrityError):
        with Session(store._engine) as session, session.begin():
            session.add(
                WorkspaceMembership(
                    user_id="user-a",
                    workspace_id=workspace.id,
                    roles=["employee"],
                    status="active",
                )
            )


def test_workspace_membership_schema_migration(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    database_path = tmp_path / "workspace-migration.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    command.upgrade(Config("alembic.ini"), "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    workspace_id = next(
        column["type"].length
        for column in inspector.get_columns("workspaces")
        if column["name"] == "id"
    )
    membership_workspace_id = next(
        column["type"].length
        for column in inspector.get_columns("workspace_memberships")
        if column["name"] == "workspace_id"
    )
    engine.dispose()

    assert {
        "workspaces",
        "workspace_memberships",
        "role_definitions",
        "workspace_invitations",
        "admin_audit_events",
    } <= tables
    assert workspace_id == 512
    assert membership_workspace_id == 512


def test_workspace_permission_migration_preserves_existing_data(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timedelta, timezone

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect, text

    database_path = tmp_path / "workspace-permissions-migration.sqlite"
    database_url = f"sqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    config = Config("alembic.ini")
    command.upgrade(config, "0006_conversations")

    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO workspaces (id, name, personal_owner_id) "
                "VALUES ('legacy-workspace', 'Legacy workspace', 'legacy-owner')"
            )
        )
        connection.execute(
            text("INSERT INTO role_definitions (name) VALUES ('analyst')")
        )
        connection.execute(
            text(
                "INSERT INTO workspace_memberships "
                "(user_id, workspace_id, roles, status) "
                "VALUES ('member-1', 'legacy-workspace', :roles, 'active')"
            ),
            {"roles": json.dumps(["hr", "analyst"])},
        )
        connection.execute(
            text(
                "INSERT INTO workspace_invitations "
                "(id, workspace_id, email, token_hash, roles, expires_at, created_by) "
                "VALUES ('invite-1', 'legacy-workspace', 'invitee@example.test', "
                "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', "
                ":roles, :expires_at, 'legacy-owner')"
            ),
            {
                "roles": json.dumps(["analyst"]),
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            },
        )
        connection.execute(
            text(
                "INSERT INTO document_chunk_catalog "
                "(chunk_id, chunk_text, source, source_sha256, chunk_index, "
                "chunking_strategy, character_count, workspace_id, owner_id, required_roles) "
                "VALUES ('chunk-1', 'legacy data', 'guide.txt', :digest, 0, 'recursive', "
                "11, 'legacy-workspace', 'legacy-owner', :roles)"
            ),
            {
                "digest": "b" * 64,
                "roles": json.dumps(["hr"]),
            },
        )
    engine.dispose()

    command.upgrade(config, "head")

    from rag_pipeline.identity_store import WorkspaceStore

    store = WorkspaceStore(database_url)
    migrated_workspace = store.get_workspace("legacy-workspace")
    roles = {
        role.name: role
        for role in store.list_workspace_roles("legacy-workspace")
    }
    engine = create_engine(database_url)
    with engine.connect() as connection:
        membership_roles = json.loads(
            connection.scalar(
                text(
                    "SELECT roles FROM workspace_memberships "
                    "WHERE user_id = 'member-1' AND workspace_id = 'legacy-workspace'"
                )
            )
        )
        invitation_roles = json.loads(
            connection.scalar(
                text("SELECT roles FROM workspace_invitations WHERE id = 'invite-1'")
            )
        )
        document_roles = json.loads(
            connection.scalar(
                text("SELECT required_roles FROM document_chunk_catalog WHERE chunk_id = 'chunk-1'")
            )
        )
        tables = set(inspect(connection).get_table_names())
    engine.dispose()

    assert migrated_workspace is not None
    assert migrated_workspace.name == "Legacy workspace"
    assert migrated_workspace.description == ""
    assert membership_roles == ["hr", "analyst"]
    assert invitation_roles == ["analyst"]
    assert document_roles == ["hr"]
    assert {"admin", "ceo", "employee", "hr", "analyst"} <= set(roles)
    assert roles["admin"].permissions == frozenset(
        {"documents:read", "documents:query", "documents:upload", "documents:update"}
    )
    assert roles["hr"].permissions == frozenset(
        {"documents:read", "documents:query", "documents:upload"}
    )
    assert roles["analyst"].permissions == frozenset(
        {"documents:read", "documents:query", "documents:upload"}
    )
    assert "workspace_roles" in tables


def test_workspace_scope_uses_current_membership_not_token_claims(tmp_path):
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.identity_store import WorkspaceStore
    from rag_pipeline.models import VerifiedIdentity

    database_url = f"sqlite:///{(tmp_path / 'workspace-scope.sqlite').as_posix()}"
    seed_store = WorkspaceStore(database_url)
    personal = seed_store.create_personal_workspace("alice")
    shared = seed_store.create_workspace("Shared workspace")
    other_personal = seed_store.create_personal_workspace("bob")
    seed_store.add_membership("alice", shared.id, roles={"hr"})

    class StaticVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return VerifiedIdentity(
                user_id=token.removesuffix("-token"),
                platform_admin=token == "platform-admin",
            )

    seen_contexts = []

    def list_documents(access_context):
        seen_contexts.append(access_context)
        return []

    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        oidc_issuer_url="https://identity.example.test/realms/retrieveros",
        database_url=database_url,
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
    )
    fake_service = SimpleNamespace(
        knowledge_base=SimpleNamespace(list_documents=list_documents)
    )
    with patch("rag_pipeline.api.build_oidc_verifier", return_value=StaticVerifier()):
        client = TestClient(create_app(settings=settings, service=fake_service))

    preflight = client.options(
        "/v1/account/openai-key",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "authorization,content-type,x-workspace-id",
        },
    )
    assert preflight.status_code == 200
    assert "x-workspace-id" in preflight.headers["access-control-allow-headers"].lower()

    assert personal.workspace_id == "personal:alice"
    alice = {"Authorization": "Bearer alice-token"}
    alice["X-Workspace-ID"] = personal.workspace_id
    assert client.get("/v1/documents", headers=alice).status_code == 200
    assert seen_contexts[-1].workspace_id == personal.workspace_id
    assert seen_contexts[-1].roles == frozenset({"admin"})

    shared_headers = {**alice, "X-Workspace-ID": shared.id}
    assert client.get("/v1/documents", headers=shared_headers).status_code == 200
    assert seen_contexts[-1].roles == frozenset({"hr"})

    seed_store.add_membership("alice", shared.id, roles={"employee"})
    assert client.get("/v1/documents", headers=shared_headers).status_code == 200
    assert seen_contexts[-1].roles == frozenset({"employee"})

    assert client.get(
        "/v1/documents", headers={**alice, "X-Workspace-ID": "unknown-workspace"}
    ).status_code == 403
    assert client.get(
        "/v1/documents", headers={**alice, "X-Workspace-ID": ""}
    ).status_code == 422
    assert client.get(
        "/v1/documents",
        headers={**alice, "X-Workspace-ID": other_personal.workspace_id},
    ).status_code == 403

    platform_headers = {
        "Authorization": "Bearer platform-admin",
        "X-Workspace-ID": shared.id,
    }
    assert client.get("/v1/documents", headers=platform_headers).status_code == 200
    assert seen_contexts[-1].platform_admin


# Verify the account API can save, report, and delete a user's encrypted key.
def test_openai_key_account_endpoints():
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient

    from rag_pipeline.api import create_app

    settings = replace(
        Settings.from_env(),
        database_url="sqlite:///:memory:",
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
    )
    client = TestClient(create_app(settings=settings))
    api_key = "sk-test-user-specific-key-never-return-it"

    saved = client.put("/v1/account/openai-key", json={"api_key": api_key})
    assert saved.status_code == 204
    assert saved.text == ""
    assert client.get("/v1/account/openai-key").json() == {"configured": True}
    deleted = client.delete("/v1/account/openai-key")
    assert deleted.status_code == 204
    assert client.get("/v1/account/openai-key").json() == {"configured": False}


# Verify the authenticated identity selects its own key before an ask reaches the service.
def test_ask_uses_only_the_authenticated_users_openai_key():
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.conversations import ConversationStore
    from rag_pipeline.credentials import UserCredentialStore
    from rag_pipeline.identity_store import WorkspaceStore
    from rag_pipeline.models import AnswerResponse, VerifiedIdentity

    encryption_key = Fernet.generate_key().decode("ascii")
    store = UserCredentialStore("sqlite:///:memory:", encryption_key)
    workspace_store = WorkspaceStore("sqlite:///:memory:")
    workspace_store.create_personal_workspace("alice")
    workspace_store.create_personal_workspace("bob")
    identities = {
        "alice-token": VerifiedIdentity(user_id="alice"),
        "bob-token": VerifiedIdentity(user_id="bob"),
    }

    class StaticVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return identities[token]

    used_keys: list[str] = []

    def ask(
        question: str,
        retrieval_mode: str,
        thread_id: str,
        _access_context,
        api_key: str,
        conversation_id=None,
        checkpoint_variant=None,
    ) -> AnswerResponse:
        used_keys.append(api_key)
        return AnswerResponse(
            question=question,
            answer="Grounded answer.",
            retrieval_mode=retrieval_mode,
            citations=[],
            retrieved_chunks=[],
            confidence=0.0,
            confidence_breakdown={},
            insufficient_context=False,
            documents_to_check=[],
            thread_id=thread_id,
        )

    settings = replace(
        Settings.from_env(),
        app_env="production",
        web_base_url="https://app.example.test",
        api_key=None,
        qdrant_url="http://qdrant.example.test:6333",
        redis_url="redis://redis.example.test:6379/0",
        oidc_issuer_url="https://identity.example.test/realms/retrieveros",
        openai_api_key=None,
        database_url="postgresql+psycopg://user:password@localhost/test",
        user_key_encryption_key=encryption_key,
    )
    fake_service = SimpleNamespace(ask=ask)

    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=StaticVerifier()),
        patch("rag_pipeline.api.UserCredentialStore", return_value=store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=workspace_store),
        patch(
            "rag_pipeline.api.ConversationStore",
            return_value=ConversationStore("sqlite:///:memory:"),
        ),
    ):
        app = create_app(settings=settings, service=fake_service)
        app.state.rate_limiter = SimpleNamespace(allow=lambda *_args: True)
        client = TestClient(app)

    alice_headers = {
        "Authorization": "Bearer alice-token",
        "X-Workspace-ID": "personal:alice",
    }
    bob_headers = {
        "Authorization": "Bearer bob-token",
        "X-Workspace-ID": "personal:bob",
    }
    alice_key = "sk-alice-specific-test-key"
    alice_headers["X-Workspace-ID"] = "personal:alice"
    bob_headers["X-Workspace-ID"] = "personal:bob"
    saved = client.put(
        "/v1/account/openai-key",
        headers=alice_headers,
        json={"api_key": alice_key},
    )
    assert saved.status_code == 204

    request = {"question": "What does the document say?"}
    assert client.post("/v1/ask", headers=bob_headers, json=request).status_code == 428
    response = client.post("/v1/ask", headers=alice_headers, json=request)

    assert response.status_code == 200
    assert used_keys == [alice_key]
    assert store.get_key("bob", "personal:bob") is None


# Verify a per-user key is supplied to both embedding and answer-generation clients.
def test_graph_clients_are_built_with_the_callers_key(monkeypatch):
    from collections import OrderedDict
    from threading import Lock
    from types import SimpleNamespace

    import rag_pipeline.service as service_module
    from rag_pipeline.service import RAGService

    received_embedding_keys: list[str] = []
    received_generation_keys: list[str] = []

    class FakeEmbeddingProvider:
        def __init__(self, _settings, api_key: str):
            received_embedding_keys.append(api_key)
            self.api_key = api_key

    class FakeGenerator:
        def __init__(self, _settings, api_key: str):
            received_generation_keys.append(api_key)
            self.api_key = api_key

    class FakeRetriever:
        def __init__(
            self,
            _knowledge_base,
            embeddings,
            _settings,
            _reranker,
            **_store_options,
        ):
            self.embeddings = embeddings

    def fake_build_graph(retriever, generator, *_args, **_kwargs):
        return SimpleNamespace(
            embedding_key=retriever.embeddings.api_key,
            generation_key=generator.api_key,
        )

    monkeypatch.setattr(service_module, "OpenAIEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(service_module, "AnswerGenerator", FakeGenerator)
    monkeypatch.setattr(service_module, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(service_module, "build_graph", fake_build_graph)

    service = object.__new__(RAGService)
    service.settings = Settings.from_env()
    service.knowledge_base = object()
    service._reranker = None
    service._checkpointer = None
    service.graph = None
    service._graph_cache = OrderedDict()
    service._graph_lock = Lock()

    alice_graph = service._graph_for_key("alice-openai-key")
    bob_graph = service._graph_for_key("bob-openai-key")

    assert alice_graph.embedding_key == "alice-openai-key"
    assert alice_graph.generation_key == "alice-openai-key"
    assert bob_graph.embedding_key == "bob-openai-key"
    assert bob_graph.generation_key == "bob-openai-key"
    assert alice_graph is not bob_graph
    assert service._graph_for_key("alice-openai-key") is alice_graph
    assert received_embedding_keys == ["alice-openai-key", "bob-openai-key"]
    assert received_generation_keys == ["alice-openai-key", "bob-openai-key"]


def test_platform_admin_cross_workspace_requests_are_audited(workspace_api):
    from sqlalchemy import create_engine, inspect, text

    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": "personal:platform"})
    headers = {
        "Authorization": "Bearer platform-token",
        "X-Workspace-ID": workspace_api.shared_workspace_id,
    }
    responses = [
        client.get("/v1/documents", headers=headers),
        client.post(
            "/v1/ingest",
            headers=headers,
            files={"file": ("confidential.txt", b"private document text")},
        ),
        client.post(
            "/v1/ask",
            headers=headers,
            json={"question": "A private question"},
        ),
        client.post(
            "/v1/compare",
            headers=headers,
            json={"question": "A private comparison question"},
        ),
        client.get(
            f"/v1/workspaces/{workspace_api.shared_workspace_id}/members",
            headers=headers,
        ),
        client.get(
            f"/v1/workspaces/{workspace_api.shared_workspace_id}/members",
            headers={
                "Authorization": "Bearer platform-token",
                "X-Workspace-ID": "personal:platform",
            },
        ),
        client.get(
            f"/v1/workspaces/{workspace_api.shared_workspace_id}/roles",
            headers=headers,
        ),
        client.post(
            f"/v1/workspaces/{workspace_api.shared_workspace_id}/roles",
            headers=headers,
            json={"name": "auditor"},
        ),
        client.get(
            "/v1/workspaces",
            headers={"Authorization": "Bearer platform-token"},
        ),
        client.get(
            f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles",
            headers={"Authorization": "Bearer platform-token"},
        ),
    ]

    assert [response.status_code for response in responses] == [
        200,
        428,
        428,
        428,
        200,
        200,
        200,
        201,
        200,
        200,
    ]
    request_ids = [response.headers.get("x-request-id") for response in responses]
    assert all(request_ids)
    assert len(set(request_ids)) == len(responses)

    engine = create_engine(workspace_api.database_url)
    assert "admin_audit_events" in inspect(engine).get_table_names()
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT actor_user_id, action, workspace_id, request_id "
                "FROM admin_audit_events ORDER BY created_at, id"
            )
        ).all()
    engine.dispose()

    assert len(rows) == len(responses)
    assert {row.actor_user_id for row in rows} == {"platform"}
    assert {row.workspace_id for row in rows[:6]} == {
        workspace_api.shared_workspace_id
    }
    assert {row.workspace_id for row in rows[6:8]} == {
        workspace_api.shared_workspace_id
    }
    assert rows[8].workspace_id is None
    assert rows[9].workspace_id == workspace_api.owner_workspace_id
    assert {row.request_id for row in rows} == set(request_ids)
    assert {row.action for row in rows} == {
        "GET /v1/documents",
        "POST /v1/ingest",
        "POST /v1/ask",
        "POST /v1/compare",
        "GET /v1/workspaces/{workspace_id}/members",
        "GET /v1/workspaces/{workspace_id}/roles",
        "POST /v1/workspaces/{workspace_id}/roles",
        "GET /v1/workspaces",
    }
    assert all(
        "private" not in row.action and "confidential" not in row.action
        for row in rows
    )

    same_workspace = client.get(
        "/v1/documents",
        headers={
            "Authorization": "Bearer platform-token",
            "X-Workspace-ID": "personal:platform",
        },
    )
    assert same_workspace.status_code == 200
    with engine.connect() as connection:
        audit_count = connection.execute(
            text("SELECT count(*) FROM admin_audit_events")
        ).scalar_one()
    engine.dispose()
    assert audit_count == len(responses)


@pytest.fixture
def workspace_api(tmp_path):
    from cryptography.fernet import Fernet
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from unittest.mock import patch

    from rag_pipeline.api import create_app
    from rag_pipeline.config import Settings
    from rag_pipeline.identity_store import WorkspaceStore
    from rag_pipeline.models import VerifiedIdentity

    database_url = f"sqlite:///{(tmp_path / 'workspace-api.sqlite').as_posix()}"
    store = WorkspaceStore(database_url)
    owner_personal = store.create_personal_workspace("owner")
    shared = store.create_workspace("Operations")
    store.add_membership("owner", shared.id, roles={"admin"})
    employee_personal = store.create_personal_workspace("employee")
    store.add_membership("employee", owner_personal.workspace_id, roles={"employee"})
    other_personal = store.create_personal_workspace("other")
    store.create_personal_workspace("platform")
    identities = {
        "owner-token": VerifiedIdentity(
            user_id="owner", email="owner@example.test", email_verified=True
        ),
        "employee-token": VerifiedIdentity(
            user_id="employee", email="employee@example.test", email_verified=True
        ),
        "other-token": VerifiedIdentity(
            user_id="other", email="other@example.test", email_verified=True
        ),
        "invitee-token": VerifiedIdentity(
            user_id="invitee", email="invitee@example.test", email_verified=True
        ),
        "wrong-email-token": VerifiedIdentity(
            user_id="wrong-email", email="attacker@example.test", email_verified=True
        ),
        "unverified-email-token": VerifiedIdentity(
            user_id="unverified-email",
            email="invitee@example.test",
            email_verified=False,
        ),
        "platform-token": VerifiedIdentity(
            user_id="platform", platform_admin=True, email="platform@example.test",
            email_verified=True,
        ),
    }

    class StaticVerifier:
        def verify(self, token: str) -> VerifiedIdentity:
            return identities[token]

    sent_messages = []

    class FakeMailer:
        def __init__(self, _settings):
            pass

        def send_invitation(self, email: str, invite_url: str) -> None:
            if email == "smtp-failure@example.test":
                raise RuntimeError("SMTP unavailable")
            sent_messages.append((email, invite_url))

    settings = replace(
        Settings.from_env(),
        app_env="test",
        api_key=None,
        oidc_issuer_url="https://identity.example.test/realms/retrieveros",
        database_url=database_url,
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
    )
    service = SimpleNamespace(
        knowledge_base=SimpleNamespace(
            list_documents=lambda *_args: [],
            get_document=lambda *_args: None,
        )
    )
    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=StaticVerifier()),
        patch("rag_pipeline.api.WorkspaceStore", return_value=store),
        patch("rag_pipeline.workspaces.SMTPInvitationMailer", FakeMailer),
        TestClient(
            create_app(settings=settings, service=service)
        ) as client,
    ):
        yield SimpleNamespace(
            client=client,
            service=service,
            store=store,
            database_url=database_url,
            settings=settings,
            sent_messages=sent_messages,
            owner_workspace_id=owner_personal.workspace_id,
            shared_workspace_id=shared.id,
            employee_workspace_id=employee_personal.workspace_id,
            other_workspace_id=other_personal.workspace_id,
        )


def _create_test_workspace_role(workspace_api, workspace_id, name, permissions):
    workspace_api.store.create_role(name, "owner")
    return workspace_api.store.create_workspace_role(
        workspace_id,
        name,
        f"{name} role",
        permissions,
        "owner",
    )


def test_workspace_admin_creates_workspace_role_permissions(workspace_api):
    client = workspace_api.client
    workspace_id = workspace_api.shared_workspace_id
    headers = {
        "Authorization": "Bearer owner-token",
        "X-Workspace-ID": workspace_id,
    }

    created = client.post(
        f"/v1/workspaces/{workspace_id}/roles",
        headers=headers,
        json={
            "name": "analyst",
            "description": "Read-only research access",
            "permissions": ["documents:read", "documents:query"],
        },
    )
    listed = client.get(f"/v1/workspaces/{workspace_id}/roles", headers=headers)

    assert created.status_code == 201
    assert created.json()["name"] == "analyst"
    assert created.json()["description"] == "Read-only research access"
    assert created.json()["permissions"] == ["documents:query", "documents:read"]
    assert {role["name"] for role in listed.json()} >= {"admin", "analyst", "employee"}
    assert workspace_api.store.get_role_permissions(
        workspace_id, {"analyst"}
    ) == frozenset({"documents:read", "documents:query"})


def test_non_admin_cannot_create_roles_or_invite(workspace_api):
    client = workspace_api.client
    workspace_id = workspace_api.shared_workspace_id
    workspace_api.store.add_membership("employee", workspace_id, roles={"hr"})
    headers = {
        "Authorization": "Bearer employee-token",
        "X-Workspace-ID": workspace_id,
    }

    role_response = client.post(
        f"/v1/workspaces/{workspace_id}/roles",
        headers=headers,
        json={
            "name": "restricted",
            "description": "",
            "permissions": ["documents:read"],
        },
    )
    invitation_response = client.post(
        f"/v1/workspaces/{workspace_id}/invitations",
        headers=headers,
        json={"email": "new-member@example.test", "roles": ["hr"]},
    )

    assert role_response.status_code == 403
    assert invitation_response.status_code == 403
    assert workspace_api.sent_messages == []


def test_documents_read_and_query_require_capabilities(workspace_api):
    client = workspace_api.client
    workspace_id = workspace_api.shared_workspace_id
    _create_test_workspace_role(
        workspace_api, workspace_id, "reader", {"documents:read"}
    )
    _create_test_workspace_role(
        workspace_api, workspace_id, "queryer", {"documents:query"}
    )
    workspace_api.store.add_membership("employee", workspace_id, roles={"reader"})
    workspace_api.store.add_membership("invitee", workspace_id, roles={"queryer"})

    reader_headers = {
        "Authorization": "Bearer employee-token",
        "X-Workspace-ID": workspace_id,
    }
    queryer_headers = {
        "Authorization": "Bearer invitee-token",
        "X-Workspace-ID": workspace_id,
    }

    assert client.get("/v1/documents", headers=reader_headers).status_code == 200
    assert client.post(
        "/v1/ask",
        headers=reader_headers,
        json={"question": "What is in the workspace?"},
    ).status_code == 403
    assert client.post(
        "/v1/compare",
        headers=reader_headers,
        json={"question": "Compare the workspace sources."},
    ).status_code == 403
    assert client.get("/v1/documents", headers=queryer_headers).status_code == 403


def test_ingest_requires_upload_and_replacement_requires_update(workspace_api, monkeypatch):
    from types import SimpleNamespace

    client = workspace_api.client
    workspace_id = workspace_api.shared_workspace_id
    _create_test_workspace_role(
        workspace_api, workspace_id, "reader", {"documents:read"}
    )
    _create_test_workspace_role(
        workspace_api, workspace_id, "uploader", {"documents:upload"}
    )
    workspace_api.store.add_membership("employee", workspace_id, roles={"reader"})
    workspace_api.store.add_membership("invitee", workspace_id, roles={"uploader"})
    workspace_api.service.knowledge_base.get_document = lambda *_args: SimpleNamespace(
        source="existing.txt",
        source_sha256="a" * 64,
        bytes=10,
        chunk_count=1,
        chunking_strategy="recursive",
        ingested_at="2026-01-01T00:00:00+00:00",
        workspace_id=workspace_id,
        owner_id="owner",
        required_roles=["uploader"],
    )
    workspace_api.service.knowledge_base.ingest = lambda *_args: pytest.fail(
        "Denied uploads must not write indexed data."
    )
    embedding_calls = []
    monkeypatch.setattr(
        "rag_pipeline.api.OpenAIEmbeddingProvider",
        lambda *_args, **_kwargs: embedding_calls.append(True),
    )

    read_only_response = client.post(
        "/v1/ingest",
        headers={
            "Authorization": "Bearer employee-token",
            "X-Workspace-ID": workspace_id,
        },
        files={"file": ("new.txt", b"New policy text")},
        data={"roles": "reader"},
    )
    upload_only_response = client.post(
        "/v1/ingest",
        headers={
            "Authorization": "Bearer invitee-token",
            "X-Workspace-ID": workspace_id,
        },
        files={"file": ("existing.txt", b"Replacement policy text")},
        data={"roles": "uploader"},
    )

    assert read_only_response.status_code == 403
    assert upload_only_response.status_code == 403
    assert embedding_calls == []


def test_ingest_accepts_multiple_workspace_roles(workspace_api, monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from rag_pipeline.credentials import UserCredentialStore
    from rag_pipeline.models import DocumentInfo, IngestionResult

    workspace_id = workspace_api.shared_workspace_id
    workspace_api.store.add_membership(
        "employee", workspace_id, roles={"employee", "hr"}
    )
    credentials = UserCredentialStore(
        workspace_api.database_url,
        workspace_api.settings.user_key_encryption_key,
    )
    credentials.set_key("employee", workspace_id, "test-openai-key")
    ingest_calls = []

    def ingest(filename, content, strategy, context, required_roles, _embeddings):
        ingest_calls.append((filename, content, strategy, context, required_roles))
        return IngestionResult(
            document=DocumentInfo(
                source=filename,
                source_sha256="a" * 64,
                bytes=len(content),
                chunk_count=1,
                chunking_strategy=strategy,
                ingested_at=datetime.now(timezone.utc).isoformat(),
                workspace_id=context.workspace_id,
                owner_id=context.user_id,
                required_roles=required_roles,
            ),
            inserted_chunks=1,
            duplicate_chunks_skipped=0,
        )

    workspace_api.service.knowledge_base.get_document = lambda *_args: None
    workspace_api.service.knowledge_base.ingest = ingest
    monkeypatch.setattr(
        "rag_pipeline.api.OpenAIEmbeddingProvider",
        lambda *_args, **_kwargs: object(),
    )
    response = workspace_api.client.post(
        "/v1/ingest",
        headers={
            "Authorization": "Bearer employee-token",
            "X-Workspace-ID": workspace_id,
        },
        files={"file": ("policy.txt", b"Policy text")},
        data={"roles": "hr,employee"},
    )

    assert response.status_code == 200
    assert response.json()["document"]["required_roles"] == ["employee", "hr"]
    assert ingest_calls[0][4] == ["employee", "hr"]
    assert ingest_calls[0][3].permissions == frozenset(
        {"documents:read", "documents:query", "documents:upload"}
    )


def test_ingest_rejects_role_from_another_workspace_before_embedding(
    workspace_api, monkeypatch
):
    from sqlalchemy.orm import Session

    from rag_pipeline.identity_store import WorkspaceMembership

    workspace_id = workspace_api.shared_workspace_id
    other_workspace = workspace_api.store.create_workspace("Other department")
    workspace_api.store.create_workspace_role(
        other_workspace.id,
        "foreign",
        "Role from another workspace",
        {"documents:upload"},
        "owner",
    )
    workspace_api.store.add_membership(
        "employee", workspace_id, roles={"employee"}
    )
    with Session(workspace_api.store._engine) as session, session.begin():
        membership = session.get(WorkspaceMembership, ("employee", workspace_id))
        membership.roles = ["foreign"]
    embedding_calls = []
    monkeypatch.setattr(
        "rag_pipeline.api.OpenAIEmbeddingProvider",
        lambda *_args, **_kwargs: embedding_calls.append(True),
    )

    response = workspace_api.client.post(
        "/v1/ingest",
        headers={
            "Authorization": "Bearer employee-token",
            "X-Workspace-ID": workspace_id,
        },
        files={"file": ("policy.txt", b"Policy text")},
        data={"roles": "foreign"},
    )

    assert response.status_code == 403
    assert embedding_calls == []


def test_role_permission_union_and_acl_intersection_are_enforced(
    workspace_api, tmp_path
):
    from types import SimpleNamespace

    from rag_pipeline.models import AccessContext
    from rag_pipeline.storage import KnowledgeBase

    workspace_id = workspace_api.shared_workspace_id
    _create_test_workspace_role(
        workspace_api, workspace_id, "reader", {"documents:read"}
    )
    _create_test_workspace_role(
        workspace_api, workspace_id, "queryer", {"documents:query"}
    )
    workspace_api.store.add_membership(
        "employee", workspace_id, roles={"reader", "queryer"}
    )
    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path,
        reranker_enabled=False,
        chunk_size=1000,
        chunk_overlap=0,
    )
    knowledge_base = KnowledgeBase(settings, FakeEmbeddings())
    document_owner = AccessContext(
        user_id="document-owner",
        workspace_id=workspace_id,
        roles=frozenset({"admin"}),
    )
    knowledge_base.ingest(
        "reader.txt",
        b"Reader-only evidence.",
        "recursive",
        document_owner,
        ["reader"],
    )
    knowledge_base.ingest(
        "hr.txt",
        b"Human-resources-only evidence.",
        "recursive",
        document_owner,
        ["hr"],
    )
    observed_contexts = []
    workspace_api.service.knowledge_base = SimpleNamespace(
        list_documents=lambda context: (
            observed_contexts.append(context) or knowledge_base.list_documents(context)
        )
    )

    response = workspace_api.client.get(
        "/v1/documents",
        headers={
            "Authorization": "Bearer employee-token",
            "X-Workspace-ID": workspace_id,
        },
    )

    assert response.status_code == 200
    assert [document["source"] for document in response.json()] == ["reader.txt"]
    assert observed_contexts[0].roles == frozenset({"reader", "queryer"})
    assert observed_contexts[0].permissions == frozenset(
        {"documents:read", "documents:query"}
    )


def test_workspace_member_management_is_scoped_and_uses_existing_roles(workspace_api):
    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    owner_headers = {"Authorization": "Bearer owner-token"}
    employee_headers = {"Authorization": "Bearer employee-token"}
    workspace_id = workspace_api.owner_workspace_id

    listed = client.get("/v1/workspaces", headers=owner_headers)
    assert listed.status_code == 200
    assert {workspace["id"] for workspace in listed.json()} == {
        workspace_id,
        workspace_api.shared_workspace_id,
    }
    employee_workspaces = client.get(
        "/v1/workspaces", headers=employee_headers
    )
    assert employee_workspaces.status_code == 200
    assert {workspace["id"] for workspace in employee_workspaces.json()} == {
        workspace_api.employee_workspace_id,
        workspace_id,
    }

    members = client.get(
        f"/v1/workspaces/{workspace_id}/members", headers=owner_headers
    )
    assert members.status_code == 200
    assert {member["user_id"] for member in members.json()} == {"employee", "owner"}

    assert client.get(
        f"/v1/workspaces/{workspace_id}/members", headers=employee_headers
    ).status_code == 403
    assert client.get(
        f"/v1/workspaces/{workspace_api.other_workspace_id}/members",
        headers=owner_headers,
    ).status_code == 403
    invitation_count_before = len(workspace_api.sent_messages)
    assert client.post(
        f"/v1/workspaces/{workspace_api.other_workspace_id}/invitations",
        headers=owner_headers,
        json={"email": "cross-scope@example.test", "roles": ["employee"]},
    ).status_code == 403
    assert len(workspace_api.sent_messages) == invitation_count_before

    updated = client.put(
        f"/v1/workspaces/{workspace_id}/members/employee",
        headers=owner_headers,
        json={"roles": ["hr"], "status": "active"},
    )
    assert updated.status_code == 200
    assert updated.json()["roles"] == ["hr"]

    created = client.post(
        f"/v1/workspaces/{workspace_id}/members",
        headers=owner_headers,
        json={"user_id": "new-member", "roles": ["employee"]},
    )
    assert created.status_code == 201
    assert created.json()["roles"] == ["employee"]

    assert client.put(
        f"/v1/workspaces/{workspace_id}/members/employee",
        headers=employee_headers,
        json={"roles": ["admin"], "status": "active"},
    ).status_code == 403
    assert client.delete(
        f"/v1/workspaces/{workspace_id}/members/employee",
        headers=owner_headers,
    ).status_code == 204
    assert client.get(
        f"/v1/workspaces/{workspace_id}/members",
        headers=employee_headers,
    ).status_code == 403


def test_workspace_list_does_not_provision_workspace(workspace_api):
    client = workspace_api.client
    headers = {"Authorization": "Bearer invitee-token"}

    listed = client.get("/v1/workspaces", headers=headers)
    documents = client.get("/v1/documents", headers=headers)

    assert listed.status_code == 200
    assert listed.json() == []
    assert workspace_api.store.list_memberships("invitee") == []
    assert documents.status_code == 400
    assert workspace_api.store.list_memberships("invitee") == []


def test_workspace_creation_requires_identity_and_creates_admin(workspace_api):
    client = workspace_api.client
    body = {"name": "Product research", "description": "Shared product evidence"}

    assert client.post("/v1/workspaces", json=body).status_code == 401
    created = client.post(
        "/v1/workspaces",
        headers={"Authorization": "Bearer invitee-token"},
        json=body,
    )

    assert created.status_code == 201
    assert created.json()["name"] == "Product research"
    assert created.json()["description"] == "Shared product evidence"
    assert created.json()["roles"] == ["admin"]
    assert created.json()["status"] == "active"
    membership = workspace_api.store.get_membership(
        "invitee", created.json()["id"]
    )
    assert membership is not None
    assert membership.roles == frozenset({"admin"})


def test_workspace_update_requires_admin(workspace_api):
    client = workspace_api.client
    workspace_id = workspace_api.shared_workspace_id
    client.headers.update({"X-Workspace-ID": workspace_id})
    body = {"name": "Operations renamed", "description": "Updated description"}

    denied = client.patch(
        f"/v1/workspaces/{workspace_id}",
        headers={"Authorization": "Bearer employee-token"},
        json=body,
    )
    updated = client.patch(
        f"/v1/workspaces/{workspace_id}",
        headers={"Authorization": "Bearer owner-token"},
        json=body,
    )

    assert denied.status_code == 403
    assert updated.status_code == 200
    assert updated.json()["name"] == "Operations renamed"
    assert updated.json()["description"] == "Updated description"


def test_invitation_acceptance_works_without_workspace_header(workspace_api):
    from datetime import datetime, timedelta, timezone
    from hashlib import sha256

    token = "invitation-token-without-workspace-header"
    invitation = workspace_api.store.create_invitation(
        workspace_id=workspace_api.shared_workspace_id,
        email="invitee@example.test",
        roles={"hr"},
        token_hash=sha256(token.encode("utf-8")).hexdigest(),
        created_by="owner",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    response = workspace_api.client.post(
        "/v1/invitations/accept",
        headers={"Authorization": "Bearer invitee-token"},
        json={"token": token},
    )

    assert response.status_code == 200
    assert response.json()["workspace_id"] == invitation.workspace_id
    assert response.json()["roles"] == ["hr"]
    assert workspace_api.store.get_membership(
        "invitee", invitation.workspace_id
    ).roles == frozenset({"hr"})


def test_invitation_acceptance_adds_roles_and_reactivates_membership(workspace_api):
    from datetime import datetime, timedelta, timezone
    from hashlib import sha256

    workspace_id = workspace_api.shared_workspace_id
    workspace_api.store.add_membership(
        "invitee", workspace_id, roles={"employee"}, status="suspended"
    )
    token = "invitation-token-reactivates-member"
    workspace_api.store.create_invitation(
        workspace_id=workspace_id,
        email="invitee@example.test",
        roles={"hr"},
        token_hash=sha256(token.encode("utf-8")).hexdigest(),
        created_by="owner",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )

    response = workspace_api.client.post(
        "/v1/invitations/accept",
        headers={"Authorization": "Bearer invitee-token"},
        json={"token": token},
    )

    assert response.status_code == 200
    assert response.json()["roles"] == ["employee", "hr"]
    assert response.json()["status"] == "active"


def test_first_time_invitee_gets_custom_workspace_permissions(workspace_api):
    from urllib.parse import parse_qs, urlparse

    client = workspace_api.client
    owner_headers = {"Authorization": "Bearer owner-token"}
    invitee_headers = {"Authorization": "Bearer invitee-token"}

    assert client.get("/v1/workspaces", headers=invitee_headers).json() == []
    created = client.post(
        "/v1/workspaces",
        headers=owner_headers,
        json={
            "name": "People Operations",
            "description": "Private people policies",
        },
    )
    assert created.status_code == 201
    workspace_id = created.json()["id"]

    role = client.post(
        f"/v1/workspaces/{workspace_id}/roles",
        headers={**owner_headers, "X-Workspace-ID": workspace_id},
        json={
            "name": "hr-analyst",
            "description": "People team document access",
            "permissions": ["documents:read", "documents:query"],
        },
    )
    assert role.status_code == 201

    invitation = client.post(
        f"/v1/workspaces/{workspace_id}/invitations",
        headers={**owner_headers, "X-Workspace-ID": workspace_id},
        json={"email": "invitee@example.test", "roles": ["hr-analyst"]},
    )
    assert invitation.status_code == 201
    token = parse_qs(urlparse(workspace_api.sent_messages[-1][1]).fragment)["token"][0]

    accepted = client.post(
        "/v1/invitations/accept",
        headers=invitee_headers,
        json={"token": token},
    )

    assert accepted.status_code == 200
    assert accepted.json()["workspace_id"] == workspace_id
    assert accepted.json()["roles"] == ["hr-analyst"]
    invitee_workspaces = client.get(
        "/v1/workspaces",
        headers=invitee_headers,
    )
    assert invitee_workspaces.status_code == 200
    assert invitee_workspaces.json() == [
        {
            "id": workspace_id,
            "name": "People Operations",
            "description": "Private people policies",
            "roles": ["hr-analyst"],
            "permissions": ["documents:query", "documents:read"],
            "status": "active",
            "is_personal": False,
        }
    ]


def test_workspace_roles_are_scoped_and_workspace_admin_can_define(
    workspace_api,
):
    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})

    roles = client.get(f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles", headers={"Authorization": "Bearer owner-token"})
    assert roles.status_code == 200
    assert {"admin", "ceo", "employee", "hr"} <= {role["name"] for role in roles.json()}

    body = {"name": "analyst"}
    assert client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles", headers={"Authorization": "Bearer owner-token"}, json=body
    ).status_code == 201
    assert client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles", headers={"Authorization": "Bearer platform-token"}, json={"name": "platform-analyst"}
    ).status_code == 201
    assert client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles",
        headers={"Authorization": "Bearer platform-token"},
        json={"name": "invalid role"},
    ).status_code == 422
    assert client.get(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/roles", headers={"Authorization": "Bearer employee-token"}
    ).status_code == 200

    all_workspaces = client.get(
        "/v1/workspaces", headers={"Authorization": "Bearer platform-token"}
    )
    assert all_workspaces.status_code == 200
    assert len(all_workspaces.json()) == 5


def test_workspace_invitation_is_hashed_single_use_and_email_bound(workspace_api):
    from hashlib import sha256
    from urllib.parse import parse_qs, urlparse

    from sqlalchemy import create_engine, text

    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    headers = {"Authorization": "Bearer owner-token"}
    response = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers=headers,
        json={"email": "invitee@example.test", "roles": ["employee"]},
    )
    assert response.status_code == 201
    assert "token" not in response.json()
    email, invite_url = workspace_api.sent_messages[-1]
    assert email == "invitee@example.test"
    parsed_invite_url = urlparse(invite_url)
    assert not parsed_invite_url.query
    token = parse_qs(parsed_invite_url.fragment)["token"][0]

    engine = create_engine(workspace_api.database_url)
    with engine.connect() as connection:
        token_hash = connection.execute(
            text("SELECT token_hash FROM workspace_invitations WHERE id = :id"),
            {"id": response.json()["id"]},
        ).scalar_one()
    engine.dispose()
    assert token_hash == sha256(token.encode("utf-8")).hexdigest()
    assert token not in token_hash

    mismatched_email = client.post(
        "/v1/invitations/accept",
        json={"token": token},
        headers={"Authorization": "Bearer wrong-email-token"},
    )
    assert mismatched_email.status_code == 403
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    accepted = client.post(
        "/v1/invitations/accept",
        json={"token": token},
        headers={"Authorization": "Bearer invitee-token"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["roles"] == ["employee"]
    assert client.post(
        "/v1/invitations/accept",
        json={"token": token},
        headers={"Authorization": "Bearer invitee-token"},
    ).status_code == 409


def test_workspace_invitation_rejects_expired_or_unverified_email(workspace_api):
    from datetime import datetime, timedelta, timezone
    from urllib.parse import parse_qs, urlparse

    from sqlalchemy import create_engine, text

    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    headers = {"Authorization": "Bearer owner-token"}
    invitation = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers=headers,
        json={"email": "invitee@example.test", "roles": ["employee"]},
    )
    assert invitation.status_code == 201
    token = parse_qs(urlparse(workspace_api.sent_messages[-1][1]).fragment)["token"][0]
    assert invitation.status_code == 201

    assert client.post(
        "/v1/invitations/accept",
        json={"token": token},
        headers={"Authorization": "Bearer wrong-email-token"},
    ).status_code == 403
    assert client.post(
        "/v1/invitations/accept",
        json={"token": token},
        headers={"Authorization": "Bearer unverified-email-token"},
    ).status_code == 403

    expired = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers=headers,
        json={"email": "invitee@example.test", "roles": ["employee"]},
    )
    assert expired.status_code == 201
    expired_token = parse_qs(urlparse(workspace_api.sent_messages[-1][1]).fragment)["token"][0]
    engine = create_engine(workspace_api.database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE workspace_invitations SET expires_at = :expires_at WHERE id = :id"
            ),
            {
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(minutes=1)
                ).isoformat(),
                "id": expired.json()["id"],
            },
        )
    engine.dispose()
    assert client.post(
        "/v1/invitations/accept",
        json={"token": expired_token},
        headers={"Authorization": "Bearer invitee-token"},
    ).status_code == 410


def test_workspace_invitation_reports_smtp_failure_without_leaking_token(workspace_api):
    from sqlalchemy import create_engine, text

    client = workspace_api.client
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    response = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers={"Authorization": "Bearer owner-token"},
        json={"email": "smtp-failure@example.test", "roles": ["employee"]},
    )

    assert response.status_code == 503
    assert "SMTP unavailable" not in response.text
    assert "token" not in response.text.lower()
    engine = create_engine(workspace_api.database_url)
    with engine.connect() as connection:
        pending_count = connection.execute(
            text("SELECT count(*) FROM workspace_invitations")
        ).scalar_one()
    engine.dispose()
    assert pending_count == 0
