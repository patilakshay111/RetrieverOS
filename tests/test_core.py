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
        oidc_issuer_url="https://identity.example.test/realms/hybrid-rag",
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
    issuer = "https://identity.example.test/realms/hybrid-rag"
    verifier = OIDCVerifier(issuer=issuer, audience="hybrid-rag-api")

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
        "aud": "hybrid-rag-api",
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
    } <= tables
    assert workspace_id == 512
    assert membership_workspace_id == 512


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
        oidc_issuer_url="https://identity.example.test/realms/hybrid-rag",
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
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "authorization,content-type,x-workspace-id",
        },
    )
    assert preflight.status_code == 200
    assert "x-workspace-id" in preflight.headers["access-control-allow-headers"].lower()

    assert personal.workspace_id == "personal:alice"
    alice = {"Authorization": "Bearer alice-token"}
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
        oidc_issuer_url="https://identity.example.test/realms/hybrid-rag",
        openai_api_key=None,
        database_url="postgresql+psycopg://user:password@localhost/test",
        user_key_encryption_key=encryption_key,
    )
    fake_service = SimpleNamespace(ask=ask)
    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=StaticVerifier()),
        patch("rag_pipeline.api.UserCredentialStore", return_value=store),
        patch("rag_pipeline.api.WorkspaceStore", return_value=workspace_store),
    ):
        client = TestClient(create_app(settings=settings, service=fake_service))

    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}
    alice_key = "sk-alice-specific-test-key"
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
        def __init__(self, _knowledge_base, embeddings, _settings, _reranker):
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
        oidc_issuer_url="https://identity.example.test/realms/hybrid-rag",
        database_url=database_url,
        user_key_encryption_key=Fernet.generate_key().decode("ascii"),
    )
    with (
        patch("rag_pipeline.api.build_oidc_verifier", return_value=StaticVerifier()),
        patch("rag_pipeline.api.WorkspaceStore", return_value=store),
        patch("rag_pipeline.workspaces.SMTPInvitationMailer", FakeMailer),
        TestClient(create_app(settings=settings, service=SimpleNamespace())) as client,
    ):
        yield SimpleNamespace(
            client=client,
            store=store,
            database_url=database_url,
            sent_messages=sent_messages,
            owner_workspace_id=owner_personal.workspace_id,
            shared_workspace_id=shared.id,
            employee_workspace_id=employee_personal.workspace_id,
            other_workspace_id=other_personal.workspace_id,
        )


def test_workspace_member_management_is_scoped_and_uses_existing_roles(workspace_api):
    client = workspace_api.client
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


def test_workspace_roles_are_visible_to_members_but_only_platform_admin_can_define(
    workspace_api,
):
    client = workspace_api.client

    roles = client.get("/v1/roles", headers={"Authorization": "Bearer owner-token"})
    assert roles.status_code == 200
    assert {"admin", "ceo", "employee", "hr"} <= {role["name"] for role in roles.json()}

    body = {"name": "analyst"}
    assert client.post(
        "/v1/roles", headers={"Authorization": "Bearer owner-token"}, json=body
    ).status_code == 403
    assert client.post(
        "/v1/roles", headers={"Authorization": "Bearer platform-token"}, json=body
    ).status_code == 201
    assert client.post(
        "/v1/roles",
        headers={"Authorization": "Bearer platform-token"},
        json={"name": "invalid role"},
    ).status_code == 422
    assert client.get(
        "/v1/roles", headers={"Authorization": "Bearer employee-token"}
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
    token = parse_qs(urlparse(invite_url).query)["token"][0]

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
        f"/v1/invitations/{token}/accept",
        headers={"Authorization": "Bearer wrong-email-token"},
    )
    assert mismatched_email.status_code == 403
    client.headers.update({"X-Workspace-ID": workspace_api.owner_workspace_id})
    accepted = client.post(
        f"/v1/invitations/{token}/accept",
        headers={"Authorization": "Bearer invitee-token"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["roles"] == ["employee"]
    assert client.post(
        f"/v1/invitations/{token}/accept",
        headers={"Authorization": "Bearer invitee-token"},
    ).status_code == 409


def test_workspace_invitation_rejects_expired_or_unverified_email(workspace_api):
    from datetime import datetime, timedelta, timezone
    from urllib.parse import parse_qs, urlparse

    from sqlalchemy import create_engine, text

    client = workspace_api.client
    headers = {"Authorization": "Bearer owner-token"}
    invitation = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers=headers,
        json={"email": "invitee@example.test", "roles": ["employee"]},
    )
    assert invitation.status_code == 201
    token = parse_qs(urlparse(workspace_api.sent_messages[-1][1]).query)["token"][0]
    assert invitation.status_code == 201

    assert client.post(
        f"/v1/invitations/{token}/accept",
        headers={"Authorization": "Bearer wrong-email-token"},
    ).status_code == 403
    assert client.post(
        f"/v1/invitations/{token}/accept",
        headers={"Authorization": "Bearer unverified-email-token"},
    ).status_code == 403

    expired = client.post(
        f"/v1/workspaces/{workspace_api.owner_workspace_id}/invitations",
        headers=headers,
        json={"email": "invitee@example.test", "roles": ["employee"]},
    )
    assert expired.status_code == 201
    expired_token = parse_qs(urlparse(workspace_api.sent_messages[-1][1]).query)["token"][0]
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
        f"/v1/invitations/{expired_token}/accept",
        headers={"Authorization": "Bearer invitee-token"},
    ).status_code == 410


def test_workspace_invitation_reports_smtp_failure_without_leaking_token(workspace_api):
    from sqlalchemy import create_engine, text

    response = workspace_api.client.post(
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
