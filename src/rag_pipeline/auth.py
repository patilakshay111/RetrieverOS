# auth.py: verifies Keycloak access tokens and resolves signed access claims

from dataclasses import dataclass
from functools import cached_property

import jwt

from rag_pipeline.config import Settings
from rag_pipeline.models import VerifiedIdentity


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

    def verify(self, token: str) -> VerifiedIdentity:
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
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 255:
            raise jwt.InvalidTokenError("The token contains an invalid subject claim.")
        realm_access = claims.get("realm_access")
        realm_roles = realm_access.get("roles", []) if isinstance(realm_access, dict) else []
        if not isinstance(realm_roles, list):
            realm_roles = []
        email_claim = claims.get("email")
        return VerifiedIdentity(
            user_id=subject,
            platform_admin=any(
                isinstance(role, str) and role == "platform-admin"
                for role in realm_roles
            ),
            email=email_claim.strip() if isinstance(email_claim, str) and email_claim.strip() else None,
            email_verified=claims.get("email_verified") is True,
        )


def build_oidc_verifier(settings: Settings) -> OIDCVerifier | None:
    if settings.oidc_issuer_url is None:
        return None
    return OIDCVerifier(
        issuer=settings.oidc_issuer_url.rstrip("/"),
        audience=settings.oidc_audience,
        jwks_url=settings.oidc_jwks_url,
    )
