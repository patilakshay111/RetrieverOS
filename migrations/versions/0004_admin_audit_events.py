"""Create append-only administrator access audit events."""

from typing import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0004_admin_audit_events"
down_revision: str | None = "0003_workspace_invitations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "admin_audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("actor_user_id", sa.String(length=255), nullable=False),
        sa.Column("action", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=512), nullable=True),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_admin_audit_events_workspace_created",
        "admin_audit_events",
        ["workspace_id", "created_at"],
    )
    op.create_index(
        "ix_admin_audit_events_actor_created",
        "admin_audit_events",
        ["actor_user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_admin_audit_events_actor_created",
        table_name="admin_audit_events",
    )
    op.drop_index(
        "ix_admin_audit_events_workspace_created",
        table_name="admin_audit_events",
    )
    op.drop_table("admin_audit_events")
