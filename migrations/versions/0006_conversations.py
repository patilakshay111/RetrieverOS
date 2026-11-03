"""Add user/workspace-scoped conversation metadata."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0006_conversations"
down_revision: str | None = "0005_document_chunk_catalog"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversations_user_workspace_updated",
        "conversations",
        ["user_id", "workspace_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversations_user_workspace_updated",
        table_name="conversations",
    )
    op.drop_table("conversations")
