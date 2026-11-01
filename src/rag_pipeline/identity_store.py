"""Persist workspace membership and server-managed role assignments."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    JSON,
    String,
    create_engine,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from rag_pipeline.credentials import Base


DEFAULT_ROLES = frozenset({"admin", "ceo", "employee", "hr"})
MEMBERSHIP_STATUSES = frozenset({"active", "suspended"})


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    personal_owner_id: Mapped[str | None] = mapped_column(
        String(255), unique=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class RoleDefinition(Base):
    __tablename__ = "role_definitions"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class WorkspaceMembership(Base):
    __tablename__ = "workspace_memberships"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'suspended')", name="ck_workspace_membership_status"
        ),
    )

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    roles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True)
class WorkspaceRecord:
    id: str
    name: str
    personal_owner_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class WorkspaceMembershipRecord:
    user_id: str
    workspace_id: str
    roles: frozenset[str]
    status: str


class WorkspaceStore:
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
        self._ensure_default_roles()

    def _ensure_default_roles(self) -> None:
        with Session(self._engine) as session, session.begin():
            for name in DEFAULT_ROLES:
                if session.get(RoleDefinition, name) is None:
                    session.add(RoleDefinition(name=name))

    @staticmethod
    def _membership_record(
        membership: WorkspaceMembership,
    ) -> WorkspaceMembershipRecord:
        return WorkspaceMembershipRecord(
            user_id=membership.user_id,
            workspace_id=membership.workspace_id,
            roles=frozenset(membership.roles),
            status=membership.status,
        )

    @staticmethod
    def _workspace_record(workspace: Workspace) -> WorkspaceRecord:
        return WorkspaceRecord(
            id=workspace.id,
            name=workspace.name,
            personal_owner_id=workspace.personal_owner_id,
            created_at=workspace.created_at,
        )

    def create_workspace(self, name: str, workspace_id: str | None = None) -> WorkspaceRecord:
        normalized_name = name.strip()
        if not normalized_name or len(normalized_name) > 255:
            raise ValueError("Workspace name must contain 1 to 255 characters.")
        workspace = Workspace(
            id=workspace_id or str(uuid4()),
            name=normalized_name,
        )
        with Session(self._engine) as session, session.begin():
            session.add(workspace)
            session.flush()
            return self._workspace_record(workspace)

    def create_personal_workspace(self, user_id: str) -> WorkspaceMembershipRecord:
        if not user_id.strip():
            raise ValueError("User ID must not be empty.")
        try:
            with Session(self._engine) as session, session.begin():
                workspace = session.scalar(
                    select(Workspace).where(Workspace.personal_owner_id == user_id)
                )
                if workspace is None:
                    workspace = Workspace(
                        id=str(uuid4()),
                        name="Personal workspace",
                        personal_owner_id=user_id,
                    )
                    session.add(workspace)
                    session.flush()
                membership = session.get(
                    WorkspaceMembership, (user_id, workspace.id)
                )
                if membership is None:
                    membership = WorkspaceMembership(
                        user_id=user_id,
                        workspace_id=workspace.id,
                        roles=["admin"],
                        status="active",
                    )
                    session.add(membership)
                    session.flush()
                return self._membership_record(membership)
        except IntegrityError:
            membership = self.get_membership_for_personal_workspace(user_id)
            if membership is None:
                raise
            return membership

    def get_membership_for_personal_workspace(
        self, user_id: str
    ) -> WorkspaceMembershipRecord | None:
        with Session(self._engine) as session:
            membership = session.scalar(
                select(WorkspaceMembership)
                .join(Workspace, Workspace.id == WorkspaceMembership.workspace_id)
                .where(
                    Workspace.personal_owner_id == user_id,
                    WorkspaceMembership.user_id == user_id,
                )
            )
            if membership is None:
                return None
            return self._membership_record(membership)

    def add_membership(
        self,
        user_id: str,
        workspace_id: str,
        roles: Iterable[str],
        status: str = "active",
    ) -> WorkspaceMembershipRecord:
        if not user_id.strip() or not workspace_id.strip():
            raise ValueError("User ID and workspace ID must not be empty.")
        if status not in MEMBERSHIP_STATUSES:
            raise ValueError("Membership status must be active or suspended.")
        normalized_roles = {role.strip() for role in roles}
        if any(not role for role in normalized_roles):
            raise ValueError("Role names must not be empty.")
        now = datetime.now(timezone.utc)
        with Session(self._engine) as session, session.begin():
            known_roles = set(
                session.scalars(
                    select(RoleDefinition.name).where(
                        RoleDefinition.name.in_(normalized_roles)
                    )
                )
            ) if normalized_roles else set()
            unknown_roles = normalized_roles - known_roles
            if unknown_roles:
                raise ValueError(f"Unknown role: {', '.join(sorted(unknown_roles))}")
            membership = session.get(
                WorkspaceMembership, (user_id, workspace_id)
            )
            if membership is None:
                membership = WorkspaceMembership(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    roles=sorted(normalized_roles),
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
                session.add(membership)
            else:
                membership.roles = sorted(normalized_roles)
                membership.status = status
                membership.updated_at = now
            session.flush()
            return self._membership_record(membership)

    def get_membership(
        self, user_id: str, workspace_id: str
    ) -> WorkspaceMembershipRecord | None:
        with Session(self._engine) as session:
            membership = session.get(
                WorkspaceMembership, (user_id, workspace_id)
            )
            if membership is None:
                return None
            return self._membership_record(membership)

    def set_membership_status(
        self, user_id: str, workspace_id: str, status: str
    ) -> WorkspaceMembershipRecord | None:
        if status not in MEMBERSHIP_STATUSES:
            raise ValueError("Membership status must be active or suspended.")
        with Session(self._engine) as session, session.begin():
            membership = session.get(
                WorkspaceMembership, (user_id, workspace_id)
            )
            if membership is None:
                return None
            membership.status = status
            membership.updated_at = datetime.now(timezone.utc)
            session.flush()
            return self._membership_record(membership)

    def list_memberships(self, user_id: str) -> list[WorkspaceMembershipRecord]:
        with Session(self._engine) as session:
            memberships = session.scalars(
                select(WorkspaceMembership)
                .where(WorkspaceMembership.user_id == user_id)
                .order_by(WorkspaceMembership.workspace_id)
            )
            return [self._membership_record(item) for item in memberships]

    def check(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def close(self) -> None:
        self._engine.dispose()
