# config.py: loads validated runtime settings from environment variables and .env

from dataclasses import dataclass
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


# Load local environment values without overriding values supplied by the process.
load_dotenv()


# Parse a boolean environment setting strictly instead of silently disabling a feature.
def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be true or false.")


# Store the configuration shared by the API, indexer, evaluator, and dashboard.
@dataclass(frozen=True)
class Settings:
    app_env: str
    openai_api_key: str | None
    chat_model: str
    embedding_model: str
    api_key: str | None
    oidc_issuer_url: str | None
    oidc_jwks_url: str | None
    oidc_audience: str
    database_url: str | None
    qdrant_url: str | None
    qdrant_vector_size: int
    redis_url: str | None
    redis_cache_ttl_seconds: int
    redis_cache_max_value_bytes: int
    redis_rate_limit_requests: int
    redis_rate_limit_window_seconds: int
    user_key_encryption_key: str | None
    data_dir: Path
    chunk_size: int
    chunk_overlap: int
    semantic_similarity_threshold: float
    deduplication_threshold: float
    dense_weight: float
    sparse_weight: float
    rrf_k: int
    dense_top_k: int
    sparse_top_k: int
    rerank_top_k: int
    final_top_k: int
    reranker_enabled: bool
    reranker_model: str
    min_retrieval_confidence: float
    max_upload_bytes: int
    cors_origins: tuple[str, ...]
    web_base_url: str
    invitation_expiry_hours: int
    smtp_host: str | None
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    smtp_use_starttls: bool
    smtp_from: str | None

    # Read configuration values and reject unsafe or inconsistent settings early.
    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            app_env=os.getenv("APP_ENV", "development").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("API_KEY") or None,
            oidc_issuer_url=os.getenv("OIDC_ISSUER_URL") or None,
            oidc_jwks_url=os.getenv("OIDC_JWKS_URL") or None,
            oidc_audience=os.getenv("OIDC_AUDIENCE", "hybrid-rag-api"),
            database_url=os.getenv("DATABASE_URL") or None,
            qdrant_url=os.getenv("QDRANT_URL") or None,
            qdrant_vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "1536")),
            redis_url=os.getenv("REDIS_URL") or None,
            redis_cache_ttl_seconds=int(os.getenv("REDIS_CACHE_TTL_SECONDS", "60")),
            redis_cache_max_value_bytes=int(
                os.getenv("REDIS_CACHE_MAX_VALUE_BYTES", "262144")
            ),
            redis_rate_limit_requests=int(
                os.getenv("REDIS_RATE_LIMIT_REQUESTS", "120")
            ),
            redis_rate_limit_window_seconds=int(
                os.getenv("REDIS_RATE_LIMIT_WINDOW_SECONDS", "60")
            ),
            user_key_encryption_key=os.getenv("USER_KEY_ENCRYPTION_KEY") or None,
            data_dir=Path(os.getenv("DATA_DIR", "data/runtime")).expanduser(),
            chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            semantic_similarity_threshold=float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.72")),
            deduplication_threshold=float(os.getenv("DEDUPLICATION_THRESHOLD", "0.95")),
            dense_weight=float(os.getenv("DENSE_WEIGHT", "0.7")),
            sparse_weight=float(os.getenv("SPARSE_WEIGHT", "0.3")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "10")),
            sparse_top_k=int(os.getenv("SPARSE_TOP_K", "10")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "20")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "5")),
            reranker_enabled=_read_bool("RERANKER_ENABLED", True),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            min_retrieval_confidence=float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", "0.4")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "CORS_ORIGINS",
                    "http://localhost:5173",
                ).split(",")
                if origin.strip()
            ),
            web_base_url=os.getenv("WEB_BASE_URL", "http://localhost:5173").rstrip("/"),
            invitation_expiry_hours=int(os.getenv("INVITATION_EXPIRY_HOURS", "72")),
            smtp_host=os.getenv("SMTP_HOST") or None,
            smtp_port=int(os.getenv("SMTP_PORT", "1025")),
            smtp_username=os.getenv("SMTP_USERNAME") or None,
            smtp_password=os.getenv("SMTP_PASSWORD") or None,
            smtp_use_starttls=_read_bool("SMTP_USE_STARTTLS", False),
            smtp_from=os.getenv("SMTP_FROM") or None,
        )
        if settings.chunk_size <= 0 or not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP smaller than CHUNK_SIZE.")
        if settings.dense_weight < 0 or settings.sparse_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")
        if settings.dense_weight + settings.sparse_weight <= 0:
            raise ValueError("At least one retrieval weight must be positive.")
        numeric_settings = (
            settings.dense_weight,
            settings.sparse_weight,
            settings.semantic_similarity_threshold,
            settings.deduplication_threshold,
            settings.min_retrieval_confidence,
        )
        if not all(math.isfinite(value) for value in numeric_settings):
            raise ValueError("Retrieval weights and thresholds must be finite numbers.")
        if not 0 < settings.deduplication_threshold <= 1:
            raise ValueError("DEDUPLICATION_THRESHOLD must be in (0, 1].")
        if not 0 < settings.semantic_similarity_threshold <= 1:
            raise ValueError("SEMANTIC_SIMILARITY_THRESHOLD must be in (0, 1].")
        if not 0 <= settings.min_retrieval_confidence <= 1:
            raise ValueError("MIN_RETRIEVAL_CONFIDENCE must be in [0, 1].")
        if settings.final_top_k < 1 or settings.rerank_top_k < settings.final_top_k:
            raise ValueError("RERANK_TOP_K must be at least FINAL_TOP_K, which must be positive.")
        if settings.dense_top_k < 1 or settings.sparse_top_k < 1 or settings.rrf_k < 1:
            raise ValueError("Dense/sparse top-k and RRF_K must be positive.")
        if settings.app_env not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production.")
        if settings.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be positive.")
        if settings.qdrant_vector_size <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE must be positive.")
        if not 1 <= settings.redis_cache_ttl_seconds <= 3600:
            raise ValueError("REDIS_CACHE_TTL_SECONDS must be between 1 and 3600.")
        if not 1 <= settings.redis_cache_max_value_bytes <= 2 * 1024 * 1024:
            raise ValueError(
                "REDIS_CACHE_MAX_VALUE_BYTES must be between 1 and 2097152."
            )
        if not 1 <= settings.redis_rate_limit_requests <= 100_000:
            raise ValueError(
                "REDIS_RATE_LIMIT_REQUESTS must be between 1 and 100000."
            )
        if not 1 <= settings.redis_rate_limit_window_seconds <= 86_400:
            raise ValueError(
                "REDIS_RATE_LIMIT_WINDOW_SECONDS must be between 1 and 86400."
            )
        if settings.redis_url:
            redis_url = urlparse(settings.redis_url)
            if (
                redis_url.scheme not in {"redis", "rediss"}
                or not redis_url.hostname
                or redis_url.query
                or redis_url.fragment
            ):
                raise ValueError("REDIS_URL must be a redis:// or rediss:// URL.")
        if not 1 <= settings.invitation_expiry_hours <= 720:
            raise ValueError("INVITATION_EXPIRY_HOURS must be between 1 and 720.")
        if not 1 <= settings.smtp_port <= 65535:
            raise ValueError("SMTP_PORT must be between 1 and 65535.")
        if bool(settings.smtp_username) != bool(settings.smtp_password):
            raise ValueError("SMTP_USERNAME and SMTP_PASSWORD must be configured together.")
        web_url = urlparse(settings.web_base_url)
        if (
            web_url.scheme not in {"http", "https"}
            or not web_url.netloc
            or web_url.query
            or web_url.fragment
        ):
            raise ValueError("WEB_BASE_URL must be an HTTP(S) origin or base path.")
        if settings.app_env == "production" and "*" in settings.cors_origins:
            raise ValueError("Production CORS_ORIGINS must not contain '*'.")
        if settings.qdrant_url:
            qdrant_url = urlparse(settings.qdrant_url)
            if (
                qdrant_url.scheme not in {"http", "https"}
                or not qdrant_url.netloc
                or qdrant_url.query
                or qdrant_url.fragment
            ):
                raise ValueError("QDRANT_URL must be an HTTP(S) base URL.")
        return settings

    # Prevent an accidentally public API or unconfigured production model from starting.
    def validate_production(self) -> None:
        if self.app_env == "production" and not self.oidc_issuer_url:
            raise ValueError("OIDC_ISSUER_URL is required when APP_ENV=production.")
        if (
            self.app_env == "production"
            and self.oidc_issuer_url
            and urlparse(self.oidc_issuer_url).scheme != "https"
        ):
            raise ValueError("Production OIDC_ISSUER_URL must use HTTPS.")
        if self.app_env == "production" and self.api_key and len(self.api_key) < 32:
            raise ValueError("Production API_KEY must contain at least 32 characters.")
        if self.app_env == "production" and self.oidc_issuer_url and not self.oidc_audience.strip():
            raise ValueError("OIDC_AUDIENCE is required when OIDC authentication is enabled.")
        if self.app_env == "production" and not self.database_url:
            raise ValueError("DATABASE_URL is required when APP_ENV=production.")
        if self.app_env == "production" and not self.qdrant_url:
            raise ValueError("QDRANT_URL is required when APP_ENV=production.")
        if self.app_env == "production" and not self.redis_url:
            raise ValueError("REDIS_URL is required when APP_ENV=production.")
        if (
            self.app_env == "production"
            and self.database_url
            and not self.database_url.startswith(("postgresql://", "postgresql+"))
        ):
            raise ValueError("Production DATABASE_URL must use PostgreSQL.")
        if self.app_env == "production" and not self.user_key_encryption_key:
            raise ValueError("USER_KEY_ENCRYPTION_KEY is required when APP_ENV=production.")
        if (
            self.app_env == "production"
            and self.smtp_host
            and (not self.smtp_from or not self.smtp_use_starttls)
        ):
            raise ValueError(
                "Production SMTP requires SMTP_FROM and SMTP_USE_STARTTLS=true."
            )
        if self.app_env == "production" and urlparse(self.web_base_url).scheme != "https":
            raise ValueError("Production WEB_BASE_URL must use HTTPS.")
        if self.app_env == "production" and not self.openai_api_key and not (
            self.database_url and self.user_key_encryption_key
        ):
            raise ValueError(
                "Production requires OPENAI_API_KEY or encrypted per-user key storage."
            )
