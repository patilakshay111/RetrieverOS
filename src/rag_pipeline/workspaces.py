"""Workspace administration, role catalog, and invitation API contracts."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from rag_pipeline.config import Settings
from rag_pipeline.email import SMTPInvitationMailer
from rag_pipeline.identity_store import (
    EMAIL_PATTERN,
    InvitationAlreadyAccepted,
    InvitationEmailMismatch,
    InvitationEmailUnverified,
    InvitationExpired,
    InvitationNotFound,
    ROLE_NAME_PATTERN,
    WorkspaceMembershipRecord,
    WorkspaceStore,
)
from rag_pipeline.models import VerifiedIdentity, WorkspaceAccessContext


class MembershipCreateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=255)
    roles: list[str] = Field(min_length=1, max_length=32)

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, values: list[str]) -> list[str]:
        roles = sorted({role.strip() for role in values})
        if not roles or any(not ROLE_NAME_PATTERN.fullmatch(role) for role in roles):
            raise ValueError("Roles must be valid lowercase role names.")
        return roles


class MembershipUpdateRequest(BaseModel):
    roles: list[str] = Field(min_length=1, max_length=32)
    status: Literal["active", "suspended"] = "active"

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, values: list[str]) -> list[str]:
        roles = sorted({role.strip() for role in values})
        if not roles or any(not ROLE_NAME_PATTERN.fullmatch(role) for role in roles):
            raise ValueError("Roles must be valid lowercase role names.")
        return roles


class InvitationCreateRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    roles: list[str] = Field(min_length=1, max_length=32)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if (
            len(normalized) > 254
            or any(ord(character) < 32 for character in normalized)
            or not EMAIL_PATTERN.fullmatch(normalized)
        ):
            raise ValueError("A valid invitation email address is required.")
        return normalized

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, values: list[str]) -> list[str]:
        roles = sorted({role.strip() for role in values})
        if not roles or any(not ROLE_NAME_PATTERN.fullmatch(role) for role in roles):
            raise ValueError("Roles must be valid lowercase role names.")
        return roles


class InvitationAcceptRequest(BaseModel):
    token: str = Field(min_length=32, max_length=128)


class WorkspaceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Workspace name must not be empty.")
        return normalized

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return value.strip()


class WorkspaceUpdateRequest(WorkspaceCreateRequest):
    pass


class RoleCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    description: str = Field(default="", max_length=500)
    permissions: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if not ROLE_NAME_PATTERN.fullmatch(name):
            raise ValueError("Role names must be lowercase slugs of at most 50 characters.")
        return name

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("permissions")
    @classmethod
    def normalize_permissions(cls, values: list[str]) -> list[str]:
        return sorted({permission.strip() for permission in values})


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    roles: list[str]
    permissions: list[str] = Field(default_factory=list)
    status: str | None
    is_personal: bool


class WorkspaceMemberResponse(BaseModel):
    user_id: str
    workspace_id: str
    roles: list[str]
    status: str


class RoleResponse(BaseModel):
    workspace_id: str
    name: str
    description: str
    permissions: list[str]
    created_by: str | None
    created_at: datetime


class InvitationResponse(BaseModel):
    id: str
    email: str
    workspace_id: str
    roles: list[str]
    expires_at: datetime


def create_workspace_router(
    store: WorkspaceStore | None,
    settings: Settings,
    authorize: Callable[..., WorkspaceAccessContext],
    authenticate: Callable[..., VerifiedIdentity],
    audit_platform_access: Callable[
        [VerifiedIdentity, Request, str, str | None], None
    ],
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def require_store() -> WorkspaceStore:
        if store is None:
            raise HTTPException(
                status_code=503, detail="Workspace membership storage is not configured."
            )
        return store

    def require_workspace_admin(
        access_context: WorkspaceAccessContext, workspace_id: str
    ) -> None:
        if access_context.platform_admin:
            return
        if (
            access_context.workspace_id != workspace_id
            or "admin" not in access_context.roles
        ):
            raise HTTPException(
                status_code=403,
                detail="Workspace administrator permission is required.",
            )

    def member_response(
        record: WorkspaceMembershipRecord,
    ) -> WorkspaceMemberResponse:
        return WorkspaceMemberResponse(
            user_id=record.user_id,
            workspace_id=record.workspace_id,
            roles=sorted(record.roles),
            status=record.status,
        )

    @router.get("/workspaces", response_model=list[WorkspaceResponse])
    def list_workspaces(
        request: Request,
        identity: Annotated[VerifiedIdentity, Depends(authenticate)],
    ) -> list[WorkspaceResponse]:
        workspace_store = require_store()
        audit_platform_access(
            identity, request, "GET /v1/workspaces", workspace_id=None
        )
        try:
            if not identity.platform_admin:
                return [
                    WorkspaceResponse(
                        id=workspace.id,
                        name=workspace.name,
                        description=workspace.description,
                        roles=sorted(workspace.roles),
                        permissions=sorted(
                            workspace_store.get_role_permissions(
                                workspace.id, workspace.roles
                            )
                        ),
                        status=workspace.status,
                        is_personal=workspace.is_personal,
                    )
                    for workspace in workspace_store.list_user_workspaces(identity.user_id)
                ]
            memberships = {
                membership.workspace_id: membership
                for membership in workspace_store.list_memberships(identity.user_id)
            }
            response: list[WorkspaceResponse] = []
            for workspace in workspace_store.list_workspaces():
                membership = memberships.get(workspace.id)
                roles = (
                    membership.roles
                    if membership is not None and membership.status == "active"
                    else frozenset()
                )
                response.append(
                    WorkspaceResponse(
                        id=workspace.id,
                        name=workspace.name,
                        description=workspace.description,
                        roles=sorted(membership.roles) if membership is not None else [],
                        permissions=sorted(
                            workspace_store.get_role_permissions(workspace.id, roles)
                        ),
                        status=membership.status if membership is not None else None,
                        is_personal=workspace.personal_owner_id == identity.user_id,
                    )
                )
            return response
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to list workspaces."
            ) from error

    @router.post(
        "/workspaces",
        response_model=WorkspaceResponse,
        status_code=201,
    )
    def create_workspace(
        request: Request,
        workspace_request: WorkspaceCreateRequest,
        identity: Annotated[VerifiedIdentity, Depends(authenticate)],
    ) -> WorkspaceResponse:
        workspace_store = require_store()
        try:
            workspace, membership = workspace_store.create_workspace_for_user(
                identity.user_id,
                workspace_request.name,
                workspace_request.description,
            )
            audit_platform_access(
                identity,
                request,
                "POST /v1/workspaces",
                workspace_id=workspace.id,
            )
            return WorkspaceResponse(
                id=workspace.id,
                name=workspace.name,
                description=workspace.description,
                roles=sorted(membership.roles),
                permissions=sorted(
                    workspace_store.get_role_permissions(
                        workspace.id, membership.roles
                    )
                ),
                status=membership.status,
                is_personal=False,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to create workspace."
            ) from error

    @router.patch(
        "/workspaces/{workspace_id}",
        response_model=WorkspaceResponse,
    )
    def update_workspace(
        workspace_id: str,
        workspace_request: WorkspaceUpdateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> WorkspaceResponse:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            workspace = workspace_store.update_workspace(
                workspace_id,
                workspace_request.name,
                workspace_request.description,
            )
            if workspace is None:
                raise HTTPException(status_code=404, detail="Workspace not found.")
            return WorkspaceResponse(
                id=workspace.id,
                name=workspace.name,
                description=workspace.description,
                roles=sorted(access_context.roles),
                permissions=sorted(access_context.permissions),
                status="active",
                is_personal=workspace.personal_owner_id == access_context.user_id,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to update workspace."
            ) from error

    @router.get(
        "/workspaces/{workspace_id}/members",
        response_model=list[WorkspaceMemberResponse],
    )
    def list_members(
        workspace_id: str,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> list[WorkspaceMemberResponse]:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            if workspace_store.get_workspace(workspace_id) is None:
                raise HTTPException(status_code=404, detail="Workspace not found.")
            return [member_response(item) for item in workspace_store.list_members(workspace_id)]
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to list workspace members."
            ) from error

    @router.post(
        "/workspaces/{workspace_id}/members",
        response_model=WorkspaceMemberResponse,
        status_code=201,
    )
    def add_member(
        workspace_id: str,
        request: MembershipCreateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> WorkspaceMemberResponse:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            record = workspace_store.add_membership(
                request.user_id, workspace_id, request.roles
            )
            return member_response(record)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to add workspace member."
            ) from error

    @router.put(
        "/workspaces/{workspace_id}/members/{user_id}",
        response_model=WorkspaceMemberResponse,
    )
    def update_member(
        workspace_id: str,
        user_id: str,
        request: MembershipUpdateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> WorkspaceMemberResponse:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            if workspace_store.get_membership(user_id, workspace_id) is None:
                raise HTTPException(status_code=404, detail="Workspace member not found.")
            record = workspace_store.add_membership(
                user_id,
                workspace_id,
                request.roles,
                status=request.status,
            )
            return member_response(record)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to update workspace member."
            ) from error

    @router.delete(
        "/workspaces/{workspace_id}/members/{user_id}",
        status_code=204,
    )
    def remove_member(
        workspace_id: str,
        user_id: str,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> None:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            workspace = workspace_store.get_workspace(workspace_id)
            if workspace is None:
                raise HTTPException(status_code=404, detail="Workspace not found.")
            if workspace.personal_owner_id == user_id:
                raise HTTPException(
                    status_code=409,
                    detail="A personal workspace owner cannot be removed.",
                )
            if not workspace_store.remove_membership(user_id, workspace_id):
                raise HTTPException(status_code=404, detail="Workspace member not found.")
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to remove workspace member."
            ) from error

    @router.get(
        "/workspaces/{workspace_id}/roles",
        response_model=list[RoleResponse],
    )
    def list_workspace_roles(
        workspace_id: str,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> list[RoleResponse]:
        if not access_context.platform_admin and access_context.workspace_id != workspace_id:
            raise HTTPException(
                status_code=403,
                detail="Workspace membership is required to list its roles.",
            )
        workspace_store = require_store()
        try:
            if workspace_store.get_workspace(workspace_id) is None:
                raise HTTPException(status_code=404, detail="Workspace not found.")
            return [
                RoleResponse(
                    workspace_id=role.workspace_id,
                    name=role.name,
                    description=role.description,
                    permissions=sorted(role.permissions),
                    created_by=role.created_by,
                    created_at=role.created_at,
                )
                for role in workspace_store.list_workspace_roles(workspace_id)
            ]
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to list role definitions."
            ) from error

    @router.post(
        "/workspaces/{workspace_id}/roles",
        response_model=RoleResponse,
        status_code=201,
    )
    def create_workspace_role(
        workspace_id: str,
        request: RoleCreateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> RoleResponse:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        try:
            role = workspace_store.create_workspace_role(
                workspace_id,
                request.name,
                request.description,
                request.permissions,
                access_context.user_id,
            )
            return RoleResponse(
                workspace_id=role.workspace_id,
                name=role.name,
                description=role.description,
                permissions=sorted(role.permissions),
                created_by=role.created_by,
                created_at=role.created_at,
            )
        except IntegrityError as error:
            raise HTTPException(status_code=409, detail="Role already exists.") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to create role definition."
            ) from error

    @router.post(
        "/workspaces/{workspace_id}/invitations",
        response_model=InvitationResponse,
        status_code=201,
    )
    def create_invitation(
        workspace_id: str,
        request: InvitationCreateRequest,
        access_context: Annotated[WorkspaceAccessContext, Depends(authorize)],
    ) -> InvitationResponse:
        require_workspace_admin(access_context, workspace_id)
        workspace_store = require_store()
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.invitation_expiry_hours
        )
        try:
            invitation = workspace_store.create_invitation(
                workspace_id=workspace_id,
                email=request.email,
                roles=request.roles,
                token_hash=token_hash,
                created_by=access_context.user_id,
                expires_at=expires_at,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to create workspace invitation."
            ) from error
        invite_url = (
            f"{settings.web_base_url}/invite#"
            f"{urlencode({'token': token})}"
        )
        try:
            SMTPInvitationMailer(settings).send_invitation(
                invitation.email, invite_url
            )
        except (OSError, RuntimeError) as error:
            try:
                deleted = workspace_store.delete_invitation(invitation.id)
            except SQLAlchemyError as cleanup_error:
                raise HTTPException(
                    status_code=503,
                    detail="Invitation delivery failed and pending cleanup failed.",
                ) from cleanup_error
            if not deleted:
                raise HTTPException(
                    status_code=503,
                    detail="Invitation delivery failed and pending cleanup failed.",
                ) from error
            raise HTTPException(
                status_code=503, detail="Unable to deliver workspace invitation."
            ) from error
        return InvitationResponse(
            id=invitation.id,
            email=invitation.email,
            workspace_id=invitation.workspace_id,
            roles=sorted(invitation.roles),
            expires_at=invitation.expires_at,
        )

    @router.post("/invitations/accept", response_model=WorkspaceMemberResponse)
    def accept_invitation(
        request: InvitationAcceptRequest,
        identity: Annotated[VerifiedIdentity, Depends(authenticate)],
    ) -> WorkspaceMemberResponse:
        if not identity.email_verified or not identity.email:
            raise HTTPException(
                status_code=403, detail="A verified email address is required."
            )
        workspace_store = require_store()
        try:
            record = workspace_store.accept_invitation(
                token_hash=hashlib.sha256(request.token.encode("utf-8")).hexdigest(),
                user_id=identity.user_id,
                email=identity.email,
                email_verified=identity.email_verified,
            )
            return member_response(record)
        except InvitationNotFound as error:
            raise HTTPException(status_code=404, detail="Invitation not found.") from error
        except InvitationExpired as error:
            raise HTTPException(status_code=410, detail="Invitation has expired.") from error
        except InvitationAlreadyAccepted as error:
            raise HTTPException(status_code=409, detail="Invitation was already accepted.") from error
        except (InvitationEmailMismatch, InvitationEmailUnverified) as error:
            raise HTTPException(
                status_code=403,
                detail="The invitation does not match the verified account email.",
            ) from error
        except SQLAlchemyError as error:
            raise HTTPException(
                status_code=503, detail="Unable to accept workspace invitation."
            ) from error

    return router
