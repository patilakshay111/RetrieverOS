"""PostgreSQL full-text chunk catalog with workspace-scoped ACL enforcement."""

import re
import json
from collections.abc import Sequence

from sqlalchemy import (
    JSON,
    Index,
    String,
    Text,
    bindparam,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from rag_pipeline.credentials import Base
from rag_pipeline.models import Chunk, RetrievedChunk, WorkspaceAccessContext


class DocumentChunk(Base):
    __tablename__ = "document_chunk_catalog"
    __table_args__ = (
        Index(
            "ix_document_chunk_catalog_workspace_owner",
            "workspace_id",
            "owner_id",
        ),
        Index(
            "ix_document_chunk_catalog_workspace_source",
            "workspace_id",
            "source",
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    section_heading: Mapped[str | None] = mapped_column(String(1024))
    page_number: Mapped[int | None] = mapped_column()
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    character_count: Mapped[int] = mapped_column(nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(512), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False)
    required_roles: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        default=list,
    )


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_./:-]+", value.casefold())


class LexicalStore:
    """Persist current chunks and search only the caller's authorized scope."""

    def __init__(self, database_url: str, engine: Engine | None = None):
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty.")
        if engine is not None:
            self._engine = engine
        else:
            engine_options: dict[str, object] = {"pool_pre_ping": True}
            if database_url.startswith("sqlite"):
                engine_options.update(
                    poolclass=StaticPool,
                    connect_args={"check_same_thread": False},
                )
            else:
                engine_options.update(pool_size=5, max_overflow=10)
            self._engine = create_engine(database_url, **engine_options)

    def _acl_sql(
        self, scope: WorkspaceAccessContext, params: dict[str, object]
    ) -> str:
        if scope.platform_admin:
            return "1 = 1"
        params["workspace_id"] = scope.workspace_id
        if "admin" in scope.roles:
            return "workspace_id = :workspace_id"
        params["user_id"] = scope.user_id
        if not scope.roles:
            return (
                "workspace_id = :workspace_id "
                "AND owner_id = :user_id"
            )
        params["roles"] = sorted(scope.roles)
        if self._engine.dialect.name == "sqlite":
            role_params = []
            for index, role in enumerate(params["roles"]):
                key = f"role_{index}"
                params[key] = role
                role_params.append(f":{key}")
            role_match = ", ".join(role_params)
            return (
                "workspace_id = :workspace_id "
                "AND (owner_id = :user_id OR EXISTS ("
                "SELECT 1 FROM json_each(required_roles) AS document_role "
                f"WHERE document_role.value IN ({role_match})"
                "))"
            )
        return (
            "workspace_id = :workspace_id "
            "AND (owner_id = :user_id "
            "OR required_roles ?| CAST(:roles AS text[]))"
        )

    @staticmethod
    def _row_chunk(row: dict[str, object]) -> Chunk:
        roles = row["required_roles"]
        if isinstance(roles, str):
            roles = json.loads(roles)
        return Chunk.model_validate(
            {
                "chunk_id": row["chunk_id"],
                "text": row["chunk_text"],
                "source": row["source"],
                "source_sha256": row["source_sha256"],
                "chunk_index": row["chunk_index"],
                "section_heading": row["section_heading"],
                "page_number": row["page_number"],
                "chunking_strategy": row["chunking_strategy"],
                "character_count": row["character_count"],
                "workspace_id": row["workspace_id"],
                "owner_id": row["owner_id"],
                "required_roles": roles,
            }
        )

    def replace_document(
        self,
        workspace_id: str,
        source: str,
        chunks: Sequence[Chunk],
    ) -> list[str]:
        if not workspace_id.strip() or not source.strip():
            raise ValueError("Workspace and source IDs must not be empty.")
        if any(
            chunk.workspace_id != workspace_id or chunk.source != source
            for chunk in chunks
        ):
            raise ValueError("Catalog chunks must match the replacement workspace and source.")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Catalog chunk IDs must be unique within a document.")
        with Session(self._engine) as session, session.begin():
            old_ids = list(
                session.scalars(
                    text(
                        "SELECT chunk_id FROM document_chunk_catalog "
                        "WHERE workspace_id = :workspace_id AND source = :source"
                    ),
                    {"workspace_id": workspace_id, "source": source},
                )
            )
            session.execute(
                text(
                    "DELETE FROM document_chunk_catalog "
                    "WHERE workspace_id = :workspace_id AND source = :source"
                ),
                {"workspace_id": workspace_id, "source": source},
            )
            if chunks:
                session.add_all(
                    DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        chunk_text=chunk.text,
                        source=chunk.source,
                        source_sha256=chunk.source_sha256,
                        chunk_index=chunk.chunk_index,
                        section_heading=chunk.section_heading,
                        page_number=chunk.page_number,
                        chunking_strategy=chunk.chunking_strategy,
                        character_count=chunk.character_count,
                        workspace_id=chunk.workspace_id,
                        owner_id=chunk.owner_id,
                        required_roles=chunk.required_roles,
                    )
                    for chunk in chunks
                )
        return old_ids

    def delete_document(self, workspace_id: str, source: str) -> list[str]:
        return self.replace_document(workspace_id, source, [])

    def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
        scope: WorkspaceAccessContext,
    ) -> list[Chunk]:
        if not chunk_ids:
            return []
        params: dict[str, object] = {"chunk_ids": list(set(chunk_ids))}
        acl_sql = self._acl_sql(scope, params)
        statement = text(
            "SELECT chunk_id, chunk_text, source, source_sha256, chunk_index, "
            "section_heading, page_number, chunking_strategy, character_count, "
            "workspace_id, owner_id, required_roles "
            "FROM document_chunk_catalog "
            f"WHERE {acl_sql} AND chunk_id IN :chunk_ids"
        ).bindparams(bindparam("chunk_ids", expanding=True))
        with Session(self._engine) as session:
            rows = session.execute(statement, params).mappings()
            return [self._row_chunk(dict(row)) for row in rows]

    def search(
        self,
        query: str,
        scope: WorkspaceAccessContext,
        limit: int,
    ) -> list[RetrievedChunk]:
        if limit < 1:
            raise ValueError("Lexical search limit must be positive.")
        query_tokens = list(dict.fromkeys(_tokens(query)))
        if not query_tokens:
            return []
        params: dict[str, object] = {}
        acl_sql = self._acl_sql(scope, params)
        if self._engine.dialect.name == "postgresql":
            params.update(query=query, limit=limit)
            statement = text(
                "SELECT chunk_id, chunk_text, source, source_sha256, chunk_index, "
                "section_heading, page_number, chunking_strategy, character_count, "
                "workspace_id, owner_id, required_roles, "
                "ts_rank_cd(to_tsvector('simple', chunk_text), "
                "plainto_tsquery('simple', :query)) AS score "
                "FROM document_chunk_catalog "
                f"WHERE {acl_sql} AND to_tsvector('simple', chunk_text) "
                "@@ plainto_tsquery('simple', :query) "
                "ORDER BY score DESC, chunk_id LIMIT :limit"
            )
            with Session(self._engine) as session:
                rows = session.execute(statement, params).mappings()
                return [
                    RetrievedChunk(
                        chunk=self._row_chunk(dict(row)),
                        bm25_score=float(row["score"]),
                        rank=rank,
                    )
                    for rank, row in enumerate(rows, start=1)
                ]

        params["tokens"] = query_tokens
        if self._engine.dialect.name == "sqlite":
            token_params: list[str] = []
            for index, token in enumerate(query_tokens):
                key = f"token_{index}"
                params[key] = f"%{token.replace('%', '\\%').replace('_', '\\_')}%"
                token_params.append(f"lower(chunk_text) LIKE :{key} ESCAPE '\\'")
            match_sql = " OR ".join(token_params)
            statement = text(
                "SELECT chunk_id, chunk_text, source, source_sha256, chunk_index, "
                "section_heading, page_number, chunking_strategy, character_count, "
                "workspace_id, owner_id, required_roles "
                "FROM document_chunk_catalog "
                f"WHERE {acl_sql} AND ({match_sql})"
            )
            with Session(self._engine) as session:
                rows = session.execute(statement, params).mappings()
                ranked: list[tuple[int, Chunk]] = []
                query_set = set(query_tokens)
                for row in rows:
                    chunk = self._row_chunk(dict(row))
                    overlap = len(query_set.intersection(_tokens(chunk.text)))
                    if overlap:
                        ranked.append((overlap, chunk))
                ranked.sort(key=lambda result: (-result[0], result[1].chunk_id))
                return [
                    RetrievedChunk(
                        chunk=chunk,
                        bm25_score=float(score),
                        rank=rank,
                    )
                    for rank, (score, chunk) in enumerate(ranked[:limit], start=1)
                ]
        raise RuntimeError("Lexical search requires PostgreSQL or SQLite.")

    def check(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(
                text("SELECT chunk_id FROM document_chunk_catalog LIMIT 0")
            )

    def close(self) -> None:
        self._engine.dispose()
