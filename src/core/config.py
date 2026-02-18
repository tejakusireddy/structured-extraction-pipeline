"""Application configuration via environment variables.

All settings are loaded from environment variables (or .env file) using
Pydantic BaseSettings. No hardcoded values — every tunable is here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the extraction pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    debug: bool = False

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/extraction_pipeline"
    database_pool_size: int = 20
    database_max_overflow: int = 10

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "holdings_vectors"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # --- Extraction ---
    default_extraction_model: str = "gpt-4o"
    max_extraction_retries: int = 2
    extraction_batch_size: int = 10
    confidence_review_threshold: float = 0.3

    # --- CourtListener ---
    courtlistener_api_url: str = "https://www.courtlistener.com/api/rest/v4"
    courtlistener_api_key: str = ""
    courtlistener_rate_limit: int = 5

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"
