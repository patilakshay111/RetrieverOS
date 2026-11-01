"""Create one-time workspace invitation storage."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_workspace_invitations"
down_revision: str | None = "0002_workspace_membership"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_invitations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("roles", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(
        "ix_workspace_invitations_workspace_id",
        "workspace_invitations",
        ["workspace_id"],
    )
    op.create_index(
        "ix_workspace_invitations_expires_at",
        "workspace_invitations",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workspace_invitations_expires_at",
        table_name="workspace_invitations",
    )
    op.drop_index(
        "ix_workspace_invitations_workspace_id",
        table_name="workspace_invitations",
    )
    op.drop_table("workspace_invitations")
