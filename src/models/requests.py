"""API request schemas.

Every inbound request body is validated through one of these models.
No raw dicts ever reach the service layer.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from src.models.domain import CourtLevel, Priority, SearchStrategy

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionRequest(BaseModel):
    """Submit opinion IDs for structured extraction."""

    model_config = ConfigDict(frozen=True)

    opinion_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="CourtListener opinion IDs to process",
    )
    priority: Priority = Priority.NORMAL
    extraction_model: str | None = Field(
        default=None,
        description="Override the default extraction model, e.g. 'gpt-4o'",
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    """Optional filters narrowing a search query."""

    model_config = ConfigDict(frozen=True)

    court_level: CourtLevel | None = None
    date_after: date | None = Field(
        default=None,
        description="Only include opinions filed on or after this date",
    )
    date_before: date | None = Field(
        default=None,
        description="Only include opinions filed on or before this date",
    )
    court_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Restrict to specific courts, e.g. ['ca9', 'ca5', 'scotus']",
    )
    jurisdiction: str | None = None


class SearchRequest(BaseModel):
    """Semantic search over extracted holdings."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language search query",
    )
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    strategy: SearchStrategy = SearchStrategy.SIMILARITY
    lambda_mult: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (0 = max diversity, 1 = max relevance)",
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestionRequest(BaseModel):
    """Submit opinions for ingestion from CourtListener."""

    model_config = ConfigDict(frozen=True)

    opinion_ids: list[int] | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
        description="Specific CourtListener opinion IDs to ingest",
    )
    court_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Ingest recent opinions from these courts",
    )
    date_after: date | None = Field(
        default=None,
        description="Only ingest opinions filed on or after this date",
    )
    date_before: date | None = Field(
        default=None,
        description="Only ingest opinions filed on or before this date",
    )
    max_opinions: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of opinions to ingest in this batch",
    )
