"""Create a local-only .env file with unique service and encryption secrets."""

import base64
from pathlib import Path
import secrets


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / ".env"
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to overwrite local secrets.")

    template = (root / ".env.example").read_text(encoding="utf-8")
    postgres_password = secrets.token_urlsafe(24)
    redis_password = secrets.token_urlsafe(24)
    keycloak_password = secrets.token_urlsafe(24)
    encryption_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
    replacements = {
        "POSTGRES_PASSWORD=local-postgres-password-change-me":
            f"POSTGRES_PASSWORD={postgres_password}",
        "REDIS_PASSWORD=local-redis-password-change-me":
            f"REDIS_PASSWORD={redis_password}",
        "KEYCLOAK_ADMIN_PASSWORD=local-keycloak-password-change-me":
            f"KEYCLOAK_ADMIN_PASSWORD={keycloak_password}",
        "USER_KEY_ENCRYPTION_KEY=generate-with-scripts-init-local-env":
            f"USER_KEY_ENCRYPTION_KEY={encryption_key}",
        "DATABASE_URL=postgresql+psycopg://hybrid_rag:local-postgres-password-change-me@postgres:5432/hybrid_rag":
            "DATABASE_URL=postgresql+psycopg://hybrid_rag:"
            f"{postgres_password}@postgres:5432/hybrid_rag",
    }
    for old, new in replacements.items():
        template = template.replace(old, new)
    target.write_text(template, encoding="utf-8")
    print(f"Created {target}. Add your OPENAI_API_KEY before starting the local stack.")


if __name__ == "__main__":
    main()
