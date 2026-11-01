# auth.py: verifies Keycloak access tokens and resolves signed access claims

from dataclasses import dataclass
from functools import cached_property

import jwt

from rag_pipeline.config import Settings
from rag_pipeline.models import AccessContext


@dataclass(frozen=True)
class OIDCVerifier:
    issuer: str
    audience: str
    jwks_url: str | None = None

    @cached_property
    def jwks(self) -> jwt.PyJWKClient:
        return jwt.PyJWKClient(
            self.jwks_url
            or f"{self.issuer.rstrip('/')}/protocol/openid-connect/certs",
            cache_jwk_set=True,
            lifespan=300,
            timeout=5,
        )

    def verify(self, token: str) -> AccessContext:
        signing_key = self.jwks.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self.audience,
            issuer=self.issuer.rstrip("/"),
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            leeway=30,
        )
        subject = claims["sub"]
        workspace_id = claims.get("workspace_id", f"personal:{subject}")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise jwt.InvalidTokenError("The token contains an invalid workspace_id claim.")
        realm_access = claims.get("realm_access")
        realm_roles = realm_access.get("roles", []) if isinstance(realm_access, dict) else []
        if not isinstance(realm_roles, list):
            realm_roles = []
        roles = claims.get("roles", realm_roles)
        if not isinstance(roles, list) or not all(
            isinstance(role, str) and role.strip() for role in roles
        ):
            raise jwt.InvalidTokenError("The token contains an invalid roles claim.")
        return AccessContext(
            user_id=subject,
            workspace_id=workspace_id,
            roles=frozenset(roles),
            platform_admin="platform-admin" in realm_roles,
        )


def build_oidc_verifier(settings: Settings) -> OIDCVerifier | None:
    if settings.oidc_issuer_url is None:
        return None
    return OIDCVerifier(
        issuer=settings.oidc_issuer_url.rstrip("/"),
        audience=settings.oidc_audience,
        jwks_url=settings.oidc_jwks_url,
    )
