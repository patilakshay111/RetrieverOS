"""Configure Keycloak after the stack starts.

This script is idempotent and is meant to be run once after ``docker compose up``
(or after ``docker compose down -v`` + a fresh start) so a local/development
Keycloak realm is ready for:

* email/password sign-in, sign-up and password reset (already enabled in the
  realm import),
* Google OAuth sign-in (federated identity provider), linked by email to the
  same account — only for Google identities with a verified email, via the
  custom retrieveros first broker login flow shipped in the realm import
  (see deploy/keycloak-provider/),
* outgoing mail through Gmail when Google OAuth credentials are configured,
  otherwise through Mailpit,
* a developer admin account that can be used to test the platform.

Environment variables (all optional with sensible dev defaults):
  KEYCLOAK_URL              http://localhost:8080
  KEYCLOAK_ADMIN            admin
  KEYCLOAK_ADMIN_PASSWORD   (from .env, required)
  KEYCLOAK_REALM            retrieveros
  KEYCLOAK_WEB_CLIENT_ID    retrieveros-web
  GOOGLE_OAUTH_CLIENT_ID
  GOOGLE_OAUTH_CLIENT_SECRET
  SMTP_HOST / SMTP_PORT / SMTP_USERNAME / SMTP_PASSWORD / SMTP_USE_STARTTLS / SMTP_FROM
"""

from __future__ import annotations

import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

# Load local environment values without overriding values supplied by the process.
load_dotenv()

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080").rstrip("/")
KEYCLOAK_ADMIN = os.getenv("KEYCLOAK_ADMIN", "admin")
KEYCLOAK_ADMIN_PASSWORD = os.getenv("KEYCLOAK_ADMIN_PASSWORD", "")
REALM = os.getenv("KEYCLOAK_REALM", "retrieveros")
KEYCLOAK_WEB_CLIENT_ID = os.getenv("KEYCLOAK_WEB_CLIENT_ID", "retrieveros-web")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
GOOGLE_CONFIGURED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

DEV_ADMIN_EMAIL = os.getenv("DEV_ADMIN_EMAIL", "dev@retrieveros.local")
DEV_ADMIN_PASSWORD = os.getenv("DEV_ADMIN_PASSWORD", "DevAdmin123!")
DEV_ADMIN_USERNAME = os.getenv("DEV_ADMIN_USERNAME", "dev.admin")

SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = os.getenv("SMTP_PORT", "1025").strip()
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_USE_STARTTLS = os.getenv("SMTP_USE_STARTTLS", "false").strip().lower() in {
    "true",
    "1",
    "yes",
}
SMTP_FROM = os.getenv("SMTP_FROM", "RetrieverOS <no-reply@retrieveros.local>")


class KeycloakAdmin:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._token: str | None = None

    def _admin_token(self) -> str:
        if self._token:
            return self._token
        if not KEYCLOAK_ADMIN_PASSWORD:
            raise RuntimeError(
                "KEYCLOAK_ADMIN_PASSWORD is required to configure Keycloak."
            )
        # Token endpoint requires form-encoded body, not JSON.
        headers = dict(self._session.headers)
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        resp = requests.post(
            f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
            data={
                "grant_type": "password",
                "client_id": "admin-cli",
                "username": KEYCLOAK_ADMIN,
                "password": KEYCLOAK_ADMIN_PASSWORD,
            },
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})
        return self._token

    def _realm(self) -> dict[str, Any]:
        resp = self._session.get(
            f"{KEYCLOAK_URL}/admin/realms/{REALM}", timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, payload: Any) -> requests.Response:
        resp = self._session.put(path, json=payload, timeout=30)
        resp.raise_for_status()
        return resp

    def _post(self, path: str, payload: Any) -> requests.Response:
        resp = self._session.post(path, json=payload, timeout=30)
        resp.raise_for_status()
        return resp

    def _get(self, path: str) -> requests.Response:
        resp = self._session.get(path, timeout=30)
        resp.raise_for_status()
        return resp

    def configure_smtp(self) -> None:
        # Explicit SMTP settings win; otherwise Gmail when Google OAuth is set,
        # else Mailpit (same resolution as the application mailer).
        if SMTP_HOST:
            host, port = SMTP_HOST, SMTP_PORT
        elif GOOGLE_CONFIGURED:
            host, port = "smtp.gmail.com", "587"
        else:
            host, port = "mailpit", "1025"

        username = SMTP_USERNAME or None
        password = SMTP_PASSWORD or None
        auth = bool(username and password)

        if host == "mailpit":
            ssl = "none"
            starttls = False
        else:
            ssl = "starttls" if SMTP_USE_STARTTLS else "none"
            starttls = SMTP_USE_STARTTLS

        realm = self._realm()
        realm["smtpServer"] = {
            "host": host,
            "port": port,
            "from": SMTP_FROM,
            "fromDisplayName": "RetrieverOS",
            "auth": auth,
            "ssl": ssl,
            "starttls": starttls,
            "user": username or "",
            "password": password or "",
            "replyTo": "",
            "replyToDisplayName": "",
            "envelopeFrom": "",
        }
        # Dev convenience: don't require email verification so new users can
        # sign in immediately. SMTP is still configured so password-reset
        # emails work when needed.
        realm["verifyEmail"] = False
        self._put(f"{KEYCLOAK_URL}/admin/realms/{REALM}", realm)
        print(
            f"[configure_keycloak] SMTP configured: {host}:{port} (auth={auth}), "
            f"verifyEmail=False"
        )

    def _get_web_client(self) -> dict[str, Any] | None:
        clients = self._get(
            f"{KEYCLOAK_URL}/admin/realms/{REALM}/clients?clientId={KEYCLOAK_WEB_CLIENT_ID}"
        ).json()
        return clients[0] if clients else None

    def configure_protocol_mappers(self) -> None:
        # Ensure the web client has standard OIDC protocol mappers so the ID
        # token includes sub, email, preferred_username, name, and roles.
        client = self._get_web_client()
        if not client:
            print(
                f"[configure_keycloak] Web client '{KEYCLOAK_WEB_CLIENT_ID}' not found, "
                "skipping protocol mappers."
            )
            return

        client_id = client["id"]
        existing_path = (
            f"{KEYCLOAK_URL}/admin/realms/{REALM}/clients/{client_id}"
            f"/protocol-mappers/models"
        )
        existing: list[dict[str, Any]] = self._get(existing_path).json()
        existing_names = {m.get("name") for m in existing}

        mappers = [
            {
                "name": "sub",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-sub-mapper",
                "consentRequired": False,
                "config": {
                    "id.token.claim": "true",
                    "access.token.claim": "true",
                    "userinfo.token.claim": "true",
                },
            },
            {
                "name": "email",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-usermodel-property-mapper",
                "consentRequired": False,
                "config": {
                    "user.attribute": "email",
                    "claim.name": "email",
                    "jsonType.label": "String",
                    "id.token.claim": "true",
                    "access.token.claim": "false",
                    "userinfo.token.claim": "true",
                },
            },
            {
                "name": "preferred_username",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-usermodel-property-mapper",
                "consentRequired": False,
                "config": {
                    "user.attribute": "username",
                    "claim.name": "preferred_username",
                    "jsonType.label": "String",
                    "id.token.claim": "true",
                    "access.token.claim": "false",
                    "userinfo.token.claim": "true",
                },
            },
            {
                "name": "full name",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-usermodel-property-mapper",
                "consentRequired": False,
                "config": {
                    "user.attribute": "firstName",
                    "claim.name": "name",
                    "jsonType.label": "String",
                    "id.token.claim": "true",
                    "access.token.claim": "false",
                    "userinfo.token.claim": "true",
                },
            },
            {
                "name": "realm roles",
                "protocol": "openid-connect",
                "protocolMapper": "oidc-usermodel-realm-role-mapper",
                "consentRequired": False,
                "config": {
                    "userattribute": "roles",
                    "claim.name": "realm_access.roles",
                    "jsonType.label": "String",
                    "id.token.claim": "true",
                    "access.token.claim": "true",
                    "userinfo.token.claim": "true",
                    "multivalued": "true",
                },
            },
        ]

        added = 0
        for mapper in mappers:
            if mapper["name"] in existing_names:
                continue
            self._post(existing_path, mapper)
            added += 1
        print(f"[configure_keycloak] Added {added} protocol mapper(s) to web client.")

    def configure_google_identity_provider(self) -> None:
        if not GOOGLE_CONFIGURED:
            print(
                "[configure_keycloak] GOOGLE_OAUTH_CLIENT_ID/SECRET not set, "
                "skipping Google identity provider."
            )
            return

        instances_path = (
            f"{KEYCLOAK_URL}/admin/realms/{REALM}/identity-provider/instances"
        )
        existing: list[dict[str, Any]] = self._get(instances_path).json()
        for inst in existing:
            if inst.get("providerId") == "google":
                print("[configure_keycloak] Google identity provider already present.")
                self.check_broker_flow()
                return

        config = {
            "clientId": GOOGLE_CLIENT_ID,
            "clientSecret": GOOGLE_CLIENT_SECRET,
            "alias": "google",
            "displayName": "Google",
            # Google has already verified the mailbox; the custom
            # `retrieveros first broker login` flow enforces that
            # (GoogleVerifiedEmailGateAuthenticator) before auto-linking.
            "trustEmail": "true",
            "loginHint": "",
            "guiOrder": "",
        }
        self._post(
            instances_path,
            {
                "providerId": "google",
                "config": config,
                "alias": "google",
                "displayName": "Google",
                "enabled": True,
                "trustEmail": True,
            },
        )
        print(
            "[configure_keycloak] Google identity provider added "
            "(verified-email auto-link via retrieveros first broker login)."
        )
        self.check_broker_flow()

    def check_broker_flow(self) -> None:
        """Warn when the realm does not use the RetrieverOS broker flow.

        The flow is created by the realm import (fresh volumes). An already
        imported realm keeps whatever flow it has, so call attention to it.
        """
        realm = self._get(f"{KEYCLOAK_URL}/admin/realms/{REALM}").json()
        active = realm.get("firstBrokerLoginFlow")
        if active == "retrieveros first broker login":
            print("[configure_keycloak] Realm uses retrieveros first broker login flow.")
            return
        print(
            "[configure_keycloak] WARNING: realm firstBrokerLoginFlow is "
            f"{active!r}, expected 'retrieveros first broker login'.\n"
            "  The custom flow is only created by the realm import on a fresh\n"
            "  Keycloak volume. Re-create the volume (docker compose down -v &&\n"
            "  docker compose up --build) or import deploy/keycloak/retrieveros-realm.json\n"
            "  into the existing realm to enable the verified-Google-email gate.",
            file=sys.stderr,
        )

    def _ensure_dev_admin_user(self) -> dict[str, Any]:
        users_path = f"{KEYCLOAK_URL}/admin/realms/{REALM}/users"
        existing = self._get(
            f"{users_path}?email={DEV_ADMIN_EMAIL}&exact=true"
        ).json()
        if existing:
            user = existing[0]
            print(f"[configure_keycloak] Dev admin user already exists ({user['id']}).")
            return user

        payload = {
            "email": DEV_ADMIN_EMAIL,
            "username": DEV_ADMIN_USERNAME,
            "firstName": "Dev",
            "lastName": "Admin",
            "enabled": True,
            "emailVerified": True,
            "credentials": [
                {
                    "type": "password",
                    "value": DEV_ADMIN_PASSWORD,
                    "temporary": False,
                }
            ],
        }
        resp = self._post(users_path, payload)
        # Keycloak returns 201 with the Location header pointing at the new user.
        user_id = resp.headers.get("Location", "").rsplit("/", 1)[-1]
        user = {"id": user_id}
        print(
            f"[configure_keycloak] Created dev admin user ({user_id}): "
            f"{DEV_ADMIN_EMAIL}"
        )
        return user

    def _grant_platform_admin_role(self, user: dict[str, Any]) -> None:
        roles_path = f"{KEYCLOAK_URL}/admin/realms/{REALM}/roles/platform-admin"
        role = self._get(roles_path).json()
        user_id = user["id"]
        mappings_path = (
            f"{KEYCLOAK_URL}/admin/realms/{REALM}/users/{user_id}/role-mappings/realm"
        )
        # Keycloak 26.x accepts POST to add realm role mappings to a user.
        resp = self._session.post(mappings_path, json=[role], timeout=30)
        resp.raise_for_status()
        print(
            f"[configure_keycloak] Granted realm role 'platform-admin' to dev admin."
        )

    def run(self) -> None:
        self._admin_token()
        self.configure_smtp()
        self.configure_protocol_mappers()
        self.configure_google_identity_provider()
        dev_admin = self._ensure_dev_admin_user()
        self._grant_platform_admin_role(dev_admin)
        print("[configure_keycloak] Configuration complete.")


def main() -> int:
    try:
        KeycloakAdmin().run()
    except requests.RequestException as error:
        body = ""
        if error.response is not None:
            body = error.response.text[:500]
        print(
            f"[configure_keycloak] ERROR: {error}\n{body}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
