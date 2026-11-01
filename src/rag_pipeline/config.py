# config.py: loads validated runtime settings from environment variables and .env

from dataclasses import dataclass
import math
import os
from pathlib import Path

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

    # Read configuration values and reject unsafe or inconsistent settings early.
    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            app_env=os.getenv("APP_ENV", "development").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("API_KEY") or None,
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
                for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
                if origin.strip()
            ),
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
        if settings.app_env == "production" and "*" in settings.cors_origins:
            raise ValueError("Production CORS_ORIGINS must not contain '*'.")
        return settings

    # Prevent an accidentally public API or unconfigured production model from starting.
    def validate_production(self) -> None:
        if self.app_env == "production" and not self.api_key:
            raise ValueError("API_KEY is required when APP_ENV=production.")
        if self.app_env == "production" and self.api_key and len(self.api_key) < 32:
            raise ValueError("Production API_KEY must contain at least 32 characters.")
        if self.app_env == "production" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when APP_ENV=production.")
