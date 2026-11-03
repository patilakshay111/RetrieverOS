"""Persist workspace membership and server-managed role assignments."""

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    JSON,
    String,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from rag_pipeline.credentials import Base


DEFAULT_ROLES = frozenset({"admin", "ceo", "employee", "hr"})
DOCUMENT_PERMISSION_KEYS = frozenset(
    {
        "documents:read",
        "documents:query",
        "documents:upload",
        "documents:update",
    }
)
DEFAULT_ROLE_PERMISSIONS = {
    "admin": DOCUMENT_PERMISSION_KEYS,
    "ceo": frozenset({"documents:read", "documents:query", "documents:upload"}),
    "employee": frozenset(
        {"documents:read", "documents:query", "documents:upload"}
    ),
    "hr": frozenset({"documents:read", "documents:query", "documents:upload"}),
}
MEMBERSHIP_STATUSES = frozenset({"active", "suspended"})
ROLE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,49}$")
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}$"
)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(
        String(2000), nullable=False, default="", server_default=""
    )
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


class WorkspaceRole(Base):
    __tablename__ = "workspace_roles"

    workspace_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String(50), primary_key=True)
    description: Mapped[str] = mapped_column(
        String(500), nullable=False, default=""
    )
    permissions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
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
        String(512), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    roles: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class WorkspaceInvitation(Base):
    __tablename__ = "workspace_invitations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    email: Mapped[str | None] = mapped_column(String(254), nullable=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    roles: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    accepted_by: Mapped[str | None] = mapped_column(String(255), nullable=True)


@dataclass(frozen=True)
class WorkspaceRecord:
    id: str
    name: str
    description: str
    personal_owner_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class WorkspaceMembershipRecord:
    user_id: str
    workspace_id: str
    roles: frozenset[str]
    status: str


@dataclass(frozen=True)
class WorkspaceSummaryRecord:
    id: str
    name: str
    description: str
    roles: frozenset[str]
    status: str
    is_personal: bool


@dataclass(frozen=True)
class RoleDefinitionRecord:
    name: str
    created_by: str | None
    created_at: datetime


@dataclass(frozen=True)
class WorkspaceRoleRecord:
    workspace_id: str
    name: str
    description: str
    permissions: frozenset[str]
    created_by: str | None
    created_at: datetime


@dataclass(frozen=True)
class WorkspaceInvitationRecord:
    id: str
    workspace_id: str
    email: str
    roles: frozenset[str]
    expires_at: datetime
    created_at: datetime


class InvitationNotFound(Exception):
    pass


class InvitationExpired(Exception):
    pass


class InvitationAlreadyAccepted(Exception):
    pass


class InvitationEmailMismatch(Exception):
    pass


class InvitationEmailUnverified(Exception):
    pass


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
            description=workspace.description,
            personal_owner_id=workspace.personal_owner_id,
            created_at=workspace.created_at,
        )

    @staticmethod
    def _role_record(role: RoleDefinition) -> RoleDefinitionRecord:
        return RoleDefinitionRecord(
            name=role.name,
            created_by=role.created_by,
            created_at=role.created_at,
        )

    @staticmethod
    def _workspace_role_record(role: WorkspaceRole) -> WorkspaceRoleRecord:
        return WorkspaceRoleRecord(
            workspace_id=role.workspace_id,
            name=role.name,
            description=role.description,
            permissions=frozenset(role.permissions),
            created_by=role.created_by,
            created_at=role.created_at,
        )

    @staticmethod
    def _seed_workspace_roles(
        session: Session, workspace_id: str, created_by: str | None = None
    ) -> None:
        for name, permissions in DEFAULT_ROLE_PERMISSIONS.items():
            if session.get(WorkspaceRole, (workspace_id, name)) is None:
                session.add(
                    WorkspaceRole(
                        workspace_id=workspace_id,
                        name=name,
                        description="",
                        permissions=sorted(permissions),
                        created_by=created_by,
                    )
                )

    @staticmethod
    def _invitation_record(
        invitation: WorkspaceInvitation,
    ) -> WorkspaceInvitationRecord:
        if invitation.email is None:
            raise RuntimeError("Invitation email was erased before acceptance.")
        return WorkspaceInvitationRecord(
            id=invitation.id,
            workspace_id=invitation.workspace_id,
            email=invitation.email,
            roles=frozenset(invitation.roles),
            expires_at=invitation.expires_at,
            created_at=invitation.created_at,
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
            self._seed_workspace_roles(session, workspace.id)
            session.flush()
            return self._workspace_record(workspace)

    def create_workspace_for_user(
        self, user_id: str, name: str, description: str = ""
    ) -> tuple[WorkspaceRecord, WorkspaceMembershipRecord]:
        if not user_id.strip():
            raise ValueError("User ID must not be empty.")
        normalized_name = name.strip()
        normalized_description = description.strip()
        if not normalized_name or len(normalized_name) > 255:
            raise ValueError("Workspace name must contain 1 to 255 characters.")
        if len(normalized_description) > 2000:
            raise ValueError("Workspace description must not exceed 2000 characters.")
        workspace = Workspace(
            id=str(uuid4()),
            name=normalized_name,
            description=normalized_description,
        )
        membership = WorkspaceMembership(
            user_id=user_id,
            workspace_id=workspace.id,
            roles=["admin"],
            status="active",
        )
        with Session(self._engine) as session, session.begin():
            session.add(workspace)
            session.flush()
            self._seed_workspace_roles(session, workspace.id, user_id)
            session.add(membership)
            session.flush()
            return (
                self._workspace_record(workspace),
                self._membership_record(membership),
            )

    def update_workspace(
        self, workspace_id: str, name: str, description: str
    ) -> WorkspaceRecord | None:
        normalized_name = name.strip()
        normalized_description = description.strip()
        if not normalized_name or len(normalized_name) > 255:
            raise ValueError("Workspace name must contain 1 to 255 characters.")
        if len(normalized_description) > 2000:
            raise ValueError("Workspace description must not exceed 2000 characters.")
        with Session(self._engine) as session, session.begin():
            workspace = session.get(Workspace, workspace_id)
            if workspace is None:
                return None
            workspace.name = normalized_name
            workspace.description = normalized_description
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
                        id=f"personal:{user_id}",
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
                self._seed_workspace_roles(session, workspace.id, user_id)
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
            if session.get(Workspace, workspace_id) is None:
                raise ValueError("Workspace does not exist.")
            known_roles = set(
                session.scalars(
                    select(WorkspaceRole.name).where(
                        WorkspaceRole.workspace_id == workspace_id,
                        WorkspaceRole.name.in_(normalized_roles),
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

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        with Session(self._engine) as session:
            workspace = session.get(Workspace, workspace_id)
            if workspace is None:
                return None
            return self._workspace_record(workspace)

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

    def list_user_workspaces(self, user_id: str) -> list[WorkspaceSummaryRecord]:
        with Session(self._engine) as session:
            rows = session.execute(
                select(Workspace, WorkspaceMembership)
                .join(
                    WorkspaceMembership,
                    WorkspaceMembership.workspace_id == Workspace.id,
                )
                .where(
                    WorkspaceMembership.user_id == user_id,
                    WorkspaceMembership.status == "active",
                )
                .order_by(Workspace.name, Workspace.id)
            )
            return [
                WorkspaceSummaryRecord(
                    id=workspace.id,
                    name=workspace.name,
                    description=workspace.description,
                    roles=frozenset(membership.roles),
                    status=membership.status,
                    is_personal=workspace.personal_owner_id == user_id,
                )
                for workspace, membership in rows
            ]

    def list_workspaces(self) -> list[WorkspaceRecord]:
        with Session(self._engine) as session:
            workspaces = session.scalars(
                select(Workspace).order_by(Workspace.name, Workspace.id)
            )
            return [self._workspace_record(workspace) for workspace in workspaces]

    def list_members(
        self, workspace_id: str
    ) -> list[WorkspaceMembershipRecord]:
        with Session(self._engine) as session:
            memberships = session.scalars(
                select(WorkspaceMembership)
                .where(WorkspaceMembership.workspace_id == workspace_id)
                .order_by(WorkspaceMembership.user_id)
            )
            return [self._membership_record(item) for item in memberships]

    def remove_membership(self, user_id: str, workspace_id: str) -> bool:
        with Session(self._engine) as session, session.begin():
            result = session.execute(
                delete(WorkspaceMembership).where(
                    WorkspaceMembership.user_id == user_id,
                    WorkspaceMembership.workspace_id == workspace_id,
                )
            )
            return result.rowcount == 1

    def list_roles(self) -> list[RoleDefinitionRecord]:
        with Session(self._engine) as session:
            roles = session.scalars(
                select(RoleDefinition).order_by(RoleDefinition.name)
            )
            return [self._role_record(role) for role in roles]

    def create_role(self, name: str, created_by: str) -> RoleDefinitionRecord:
        if not ROLE_NAME_PATTERN.fullmatch(name):
            raise ValueError("Role names must be lowercase slugs of at most 50 characters.")
        role = RoleDefinition(name=name, created_by=created_by)
        with Session(self._engine) as session, session.begin():
            session.add(role)
            session.flush()
            return self._role_record(role)

    def list_workspace_roles(self, workspace_id: str) -> list[WorkspaceRoleRecord]:
        with Session(self._engine) as session:
            roles = session.scalars(
                select(WorkspaceRole)
                .where(WorkspaceRole.workspace_id == workspace_id)
                .order_by(WorkspaceRole.name)
            )
            return [self._workspace_role_record(role) for role in roles]

    def create_workspace_role(
        self,
        workspace_id: str,
        name: str,
        description: str,
        permissions: Iterable[str],
        created_by: str,
    ) -> WorkspaceRoleRecord:
        if not ROLE_NAME_PATTERN.fullmatch(name):
            raise ValueError("Role names must be lowercase slugs of at most 50 characters.")
        if name == "admin":
            raise ValueError("The admin role is reserved.")
        normalized_description = description.strip()
        if len(normalized_description) > 500:
            raise ValueError("Role description must not exceed 500 characters.")
        normalized_permissions = {permission.strip() for permission in permissions}
        unknown_permissions = normalized_permissions - DOCUMENT_PERMISSION_KEYS
        if unknown_permissions:
            raise ValueError(
                f"Unknown permission: {', '.join(sorted(unknown_permissions))}"
            )
        role = WorkspaceRole(
            workspace_id=workspace_id,
            name=name,
            description=normalized_description,
            permissions=sorted(normalized_permissions),
            created_by=created_by,
        )
        with Session(self._engine) as session, session.begin():
            if session.get(Workspace, workspace_id) is None:
                raise ValueError("Workspace does not exist.")
            session.add(role)
            session.flush()
            return self._workspace_role_record(role)

    def get_role_permissions(
        self, workspace_id: str, roles: Iterable[str]
    ) -> frozenset[str]:
        normalized_roles = {role.strip() for role in roles}
        if not normalized_roles:
            return frozenset()
        with Session(self._engine) as session:
            definitions = list(
                session.scalars(
                    select(WorkspaceRole).where(
                        WorkspaceRole.workspace_id == workspace_id,
                        WorkspaceRole.name.in_(normalized_roles),
                    )
                )
            )
        found_roles = {definition.name for definition in definitions}
        unknown_roles = normalized_roles - found_roles
        if unknown_roles:
            raise ValueError(
                f"Unknown workspace role: {', '.join(sorted(unknown_roles))}"
            )
        return frozenset(
            permission
            for definition in definitions
            for permission in definition.permissions
        )

    def create_invitation(
        self,
        workspace_id: str,
        email: str,
        roles: Iterable[str],
        token_hash: str,
        created_by: str,
        expires_at: datetime,
    ) -> WorkspaceInvitationRecord:
        normalized_email = email.strip().casefold()
        if (
            len(normalized_email) > 254
            or not EMAIL_PATTERN.fullmatch(normalized_email)
            or any(ord(character) < 32 for character in normalized_email)
        ):
            raise ValueError("A valid invitation email address is required.")
        normalized_roles = {role.strip() for role in roles}
        if not normalized_roles or any(not role for role in normalized_roles):
            raise ValueError("At least one valid role is required.")
        if not re.fullmatch(r"[a-f0-9]{64}", token_hash):
            raise ValueError("Invitation token hash is invalid.")
        invitation = WorkspaceInvitation(
            id=str(uuid4()),
            workspace_id=workspace_id,
            email=normalized_email,
            token_hash=token_hash,
            roles=sorted(normalized_roles),
            expires_at=expires_at,
            created_by=created_by,
        )
        with Session(self._engine) as session, session.begin():
            if session.get(Workspace, workspace_id) is None:
                raise ValueError("Workspace does not exist.")
            known_roles = set(
                session.scalars(
                    select(WorkspaceRole.name).where(
                        WorkspaceRole.workspace_id == workspace_id,
                        WorkspaceRole.name.in_(normalized_roles),
                    )
                )
            )
            unknown_roles = normalized_roles - known_roles
            if unknown_roles:
                raise ValueError(f"Unknown role: {', '.join(sorted(unknown_roles))}")
            session.add(invitation)
            session.flush()
            return self._invitation_record(invitation)

    def delete_invitation(self, invitation_id: str) -> bool:
        with Session(self._engine) as session, session.begin():
            result = session.execute(
                delete(WorkspaceInvitation).where(
                    WorkspaceInvitation.id == invitation_id
                )
            )
            return result.rowcount == 1

    def accept_invitation(
        self,
        token_hash: str,
        user_id: str,
        email: str | None,
        email_verified: bool,
    ) -> WorkspaceMembershipRecord:
        now = datetime.now(timezone.utc)
        invitation_expired = False
        membership_record: WorkspaceMembershipRecord | None = None
        with Session(self._engine) as session, session.begin():
            invitation = session.scalar(
                select(WorkspaceInvitation)
                .where(WorkspaceInvitation.token_hash == token_hash)
                .with_for_update()
            )
            if invitation is None:
                raise InvitationNotFound
            if invitation.accepted_at is not None:
                raise InvitationAlreadyAccepted
            expires_at = invitation.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= now:
                invitation.email = None
                invitation_expired = True
            elif not email_verified:
                raise InvitationEmailUnverified
            elif (
                email is None
                or invitation.email is None
                or email.strip().casefold() != invitation.email
            ):
                raise InvitationEmailMismatch
            else:
                membership = session.get(
                    WorkspaceMembership, (user_id, invitation.workspace_id)
                )
                roles = set(invitation.roles)
                if membership is None:
                    membership = WorkspaceMembership(
                        user_id=user_id,
                        workspace_id=invitation.workspace_id,
                        roles=sorted(roles),
                        status="active",
                    )
                    session.add(membership)
                else:
                    membership.roles = sorted(set(membership.roles) | roles)
                    membership.status = "active"
                    membership.updated_at = now
                invitation.accepted_at = now
                invitation.accepted_by = user_id
                invitation.email = None
                session.flush()
                membership_record = self._membership_record(membership)
        if invitation_expired:
            raise InvitationExpired
        if membership_record is None:
            raise RuntimeError("Invitation acceptance did not create a membership.")
        return membership_record

    def check(self) -> None:
        with self._engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    def close(self) -> None:
        self._engine.dispose()
