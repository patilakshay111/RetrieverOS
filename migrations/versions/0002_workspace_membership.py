"""Create workspace, membership, and role-definition persistence."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_workspace_membership"
down_revision: str | None = "0001_user_openai_keys"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspaces",
        sa.Column("id", sa.String(length=512), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("personal_owner_id", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("personal_owner_id"),
    )
    op.create_table(
        "role_definitions",
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("name"),
    )
    op.bulk_insert(
        sa.table("role_definitions", sa.column("name", sa.String(length=100))),
        [{"name": role} for role in ("admin", "ceo", "employee", "hr")],
    )
    op.create_table(
        "workspace_memberships",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("roles", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'suspended')",
            name="ck_workspace_membership_status",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "workspace_id"),
    )
    op.create_index(
        "ix_workspace_memberships_workspace_id",
        "workspace_memberships",
        ["workspace_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_workspace_memberships_workspace_id",
        table_name="workspace_memberships",
    )
    op.drop_table("workspace_memberships")
    op.drop_table("role_definitions")
    op.drop_table("workspaces")
