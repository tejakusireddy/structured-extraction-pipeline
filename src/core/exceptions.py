"""Custom exception hierarchy for the extraction pipeline.

Every service-layer error inherits from PipelineError, giving the API
layer a single base class to catch and translate into structured JSON
responses. Subclasses carry domain-specific context (retry_after for
rate limits, details dict for debugging).
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(message)


class IngestionError(PipelineError):
    """Raised when document ingestion fails."""


class ExtractionError(PipelineError):
    """Raised when LLM extraction fails after retries."""


class LLMProviderError(ExtractionError):
    """Raised when an LLM provider returns an error or times out."""


class ExtractionValidationError(PipelineError):
    """Raised when extraction output fails schema or business-rule validation."""


class CitationResolutionError(PipelineError):
    """Raised when a citation cannot be resolved to a known opinion."""


class GraphQueryError(PipelineError):
    """Raised when a graph query fails."""


class DatabaseError(PipelineError):
    """Raised when a database operation fails."""


class QueueError(PipelineError):
    """Raised when a queue operation (enqueue/dequeue) fails."""


class RateLimitError(PipelineError):
    """Raised when an external rate limit is hit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after = retry_after


class NotFoundError(PipelineError):
    """Raised when a requested resource does not exist."""
