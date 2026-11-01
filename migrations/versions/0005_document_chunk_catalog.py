"""Add a shared PostgreSQL document-chunk catalog for scoped lexical search."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0005_document_chunk_catalog"
down_revision: str | None = "0004_admin_audit_events"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    dialect_name = op.get_bind().dialect.name
    roles_type = (
        postgresql.JSONB(astext_type=sa.Text())
        if dialect_name == "postgresql"
        else sa.JSON()
    )
    roles_default = (
        sa.text("'[]'::jsonb")
        if dialect_name == "postgresql"
        else sa.text("'[]'")
    )
    op.create_table(
        "document_chunk_catalog",
        sa.Column("chunk_id", sa.String(length=128), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("section_heading", sa.String(length=1024), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("chunking_strategy", sa.String(length=32), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column(
            "required_roles",
            roles_type,
            nullable=False,
            server_default=roles_default,
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
    )
    op.create_index(
        "ix_document_chunk_catalog_workspace_owner",
        "document_chunk_catalog",
        ["workspace_id", "owner_id"],
    )
    op.create_index(
        "ix_document_chunk_catalog_workspace_source",
        "document_chunk_catalog",
        ["workspace_id", "source"],
    )
    if dialect_name == "postgresql":
        op.create_index(
            "ix_document_chunk_catalog_required_roles",
            "document_chunk_catalog",
            ["required_roles"],
            postgresql_using="gin",
        )
        op.create_index(
            "ix_document_chunk_catalog_search",
            "document_chunk_catalog",
            [sa.text("to_tsvector('simple', chunk_text)")],
            postgresql_using="gin",
        )


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.drop_index(
            "ix_document_chunk_catalog_search",
            table_name="document_chunk_catalog",
        )
        op.drop_index(
            "ix_document_chunk_catalog_required_roles",
            table_name="document_chunk_catalog",
        )
    op.drop_index(
        "ix_document_chunk_catalog_workspace_source",
        table_name="document_chunk_catalog",
    )
    op.drop_index(
        "ix_document_chunk_catalog_workspace_owner",
        table_name="document_chunk_catalog",
    )
    op.drop_table("document_chunk_catalog")
