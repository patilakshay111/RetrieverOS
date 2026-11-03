"""Add named workspace descriptions and workspace-scoped role permissions."""

from typing import Sequence
from datetime import datetime

from alembic import op
import sqlalchemy as sa


revision: str = "0007_workspace_local_permissions"
down_revision: str | None = "0006_conversations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


ADMIN_PERMISSIONS = [
    "documents:read",
    "documents:query",
    "documents:upload",
    "documents:update",
]
MEMBER_PERMISSIONS = [
    "documents:read",
    "documents:query",
    "documents:upload",
]


def upgrade() -> None:
    op.add_column(
        "workspaces",
        sa.Column(
            "description",
            sa.String(length=2000),
            nullable=False,
            server_default="",
        ),
    )
    op.create_table(
        "workspace_roles",
        sa.Column("workspace_id", sa.String(length=512), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False, server_default=""),
        sa.Column("permissions", sa.JSON(), nullable=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("workspace_id", "name"),
    )

    connection = op.get_bind()
    legacy_roles = connection.execute(
        sa.text(
            "SELECT name, created_by, created_at FROM role_definitions ORDER BY name"
        )
    ).mappings()
    workspaces = connection.execute(
        sa.text("SELECT id FROM workspaces ORDER BY id")
    ).scalars()
    workspace_role_table = sa.table(
        "workspace_roles",
        sa.column("workspace_id", sa.String(length=512)),
        sa.column("name", sa.String(length=50)),
        sa.column("description", sa.String(length=500)),
        sa.column("permissions", sa.JSON()),
        sa.column("created_by", sa.String(length=255)),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    roles = list(legacy_roles)
    for workspace_id in workspaces:
        for role in roles:
            permissions = (
                ADMIN_PERMISSIONS if role["name"] == "admin" else MEMBER_PERMISSIONS
            )
            created_at = role["created_at"]
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            connection.execute(
                sa.insert(workspace_role_table).values(
                    workspace_id=workspace_id,
                    name=role["name"],
                    description="",
                    permissions=permissions,
                    created_by=role["created_by"],
                    created_at=created_at,
                )
            )


def downgrade() -> None:
    op.drop_table("workspace_roles")
    op.drop_column("workspaces", "description")
