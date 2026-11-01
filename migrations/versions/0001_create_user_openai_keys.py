"""Create encrypted per-user OpenAI credential storage."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0001_user_openai_keys"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "user_openai_keys",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=False),
        sa.Column("encrypted_key", sa.LargeBinary(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id", "workspace_id"),
    )


def downgrade() -> None:
    op.drop_table("user_openai_keys")
