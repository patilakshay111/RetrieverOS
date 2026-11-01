"""Persist security-relevant cross-workspace access events."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import DateTime, Index, String, create_engine
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from rag_pipeline.credentials import Base


class AdminAuditEvent(Base):
    __tablename__ = "admin_audit_events"
    __table_args__ = (
        Index("ix_admin_audit_events_workspace_created", "workspace_id", "created_at"),
        Index("ix_admin_audit_events_actor_created", "actor_user_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(String(512), nullable=True)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class AdminAuditStore:
    def __init__(self, database_url: str):
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_options.update(
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            engine_options.update(pool_size=5, max_overflow=10)
        self._engine = create_engine(database_url, **engine_options)
        if database_url.startswith("sqlite"):
            Base.metadata.create_all(self._engine)

    def record_cross_workspace_access(
        self,
        actor_user_id: str,
        action: str,
        workspace_id: str | None,
        request_id: str,
    ) -> None:
        if not actor_user_id.strip() or not request_id.strip():
            raise ValueError("Audit actor and request ID must not be empty.")
        if not action or len(action) > 255:
            raise ValueError("Audit action must contain 1 to 255 characters.")
        event = AdminAuditEvent(
            id=str(uuid4()),
            actor_user_id=actor_user_id,
            action=action,
            workspace_id=workspace_id,
            request_id=request_id,
            created_at=datetime.now(timezone.utc),
        )
        with Session(self._engine) as session, session.begin():
            session.add(event)
            session.flush()

    def check(self) -> None:
        with self._engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")

    def close(self) -> None:
        self._engine.dispose()
