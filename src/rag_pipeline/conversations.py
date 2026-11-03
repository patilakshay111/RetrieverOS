"""Persist conversations with user/workspace-scoped access."""

from datetime import datetime, timezone
import json
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from sqlalchemy import DateTime, Index, String, create_engine, delete, select
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool
from pydantic import BaseModel

from rag_pipeline.credentials import Base
from rag_pipeline.models import AccessContext


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        Index(
            "ix_conversations_user_workspace_updated",
            "user_id",
            "workspace_id",
            "updated_at",
        ),
    )

    conversation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ConversationRecord(BaseModel):
    conversation_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def conversation_thread_id(
    scope: AccessContext,
    conversation_id: UUID,
    checkpoint_variant: str | None = None,
) -> str:
    identity = json.dumps(
        [
            scope.user_id,
            scope.workspace_id,
            str(conversation_id),
            checkpoint_variant,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, f"hybrid-rag-conversation:{identity}"))


class ConversationStore:
    """Create and access conversations only within the authenticated scope."""

    def __init__(self, database_url: str):
        if not database_url.strip():
            raise ValueError("DATABASE_URL must not be empty.")
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

    @staticmethod
    def _record(conversation: Conversation) -> ConversationRecord:
        return ConversationRecord(
            conversation_id=UUID(conversation.conversation_id),
            title=conversation.title,
            created_at=_utc(conversation.created_at),
            updated_at=_utc(conversation.updated_at),
        )

    def create(
        self, scope: AccessContext, title: str = "New conversation"
    ) -> ConversationRecord:
        normalized_title = title.strip()
        if not normalized_title or len(normalized_title) > 160:
            raise ValueError("Conversation titles must contain 1 to 160 characters.")
        if not scope.user_id.strip() or not scope.workspace_id.strip():
            raise ValueError("Conversation scope must include a user and workspace.")
        now = datetime.now(timezone.utc)
        conversation = Conversation(
            conversation_id=str(uuid4()),
            user_id=scope.user_id,
            workspace_id=scope.workspace_id,
            title=normalized_title,
            created_at=now,
            updated_at=now,
        )
        with Session(self._engine) as session, session.begin():
            session.add(conversation)
            record = self._record(conversation)
        return record

    def list(
        self, scope: AccessContext, limit: int = 50
    ) -> list[ConversationRecord]:
        if limit < 1 or limit > 100:
            raise ValueError("Conversation list limit must be between 1 and 100.")
        with Session(self._engine) as session:
            conversations = session.scalars(
                select(Conversation)
                .where(
                    Conversation.user_id == scope.user_id,
                    Conversation.workspace_id == scope.workspace_id,
                )
                .order_by(
                    Conversation.updated_at.desc(),
                    Conversation.conversation_id,
                )
                .limit(limit)
            )
            return [self._record(conversation) for conversation in conversations]

    def get(
        self, scope: AccessContext, conversation_id: UUID
    ) -> ConversationRecord | None:
        with Session(self._engine) as session:
            conversation = session.scalar(
                select(Conversation).where(
                    Conversation.conversation_id == str(conversation_id),
                    Conversation.user_id == scope.user_id,
                    Conversation.workspace_id == scope.workspace_id,
                )
            )
            return self._record(conversation) if conversation is not None else None

    def delete(self, scope: AccessContext, conversation_id: UUID) -> bool:
        with Session(self._engine) as session, session.begin():
            result = session.execute(
                delete(Conversation).where(
                    Conversation.conversation_id == str(conversation_id),
                    Conversation.user_id == scope.user_id,
                    Conversation.workspace_id == scope.workspace_id,
                )
            )
            return result.rowcount == 1

    def close(self) -> None:
        self._engine.dispose()
