"""API response schemas.

Every outbound response is serialized through one of these models.
Structured error responses are included — the API never leaks raw
stack traces.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.models.domain import (
    CitationType,
    ConflictStatus,
    Disposition,
    ExtractionStatus,
    JobStatus,
)

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class DependencyHealth(BaseModel):
    """Health status of a single infrastructure dependency."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: str = Field(..., description="'healthy', 'unhealthy', or 'not_configured'")
    latency_ms: float | None = None
    details: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health check response."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    version: str
    uptime_seconds: float
    dependencies: list[DependencyHealth]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractionJobCreatedResponse(BaseModel):
    """Returned immediately after submitting an extraction job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobStatus
    total_opinions: int
    estimated_completion_seconds: float | None = None


class CitedAuthorityResponse(BaseModel):
    """A cited authority within an extraction result."""

    model_config = ConfigDict(frozen=True)

    citation_string: str
    case_name: str | None = None
    citation_context: str
    citation_type: CitationType


class ExtractionDetail(BaseModel):
    """Structured extraction output for a single opinion."""

    model_config = ConfigDict(frozen=True)

    holding: str
    holding_confidence: float = Field(..., ge=0.0, le=1.0)
    legal_standard: str | None = None
    disposition: Disposition
    disposition_confidence: float = Field(..., ge=0.0, le=1.0)
    key_authorities: list[CitedAuthorityResponse]
    legal_topics: list[str]
    dissent_present: bool
    dissent_summary: str | None = None
    concurrence_present: bool
    concurrence_summary: str | None = None


class ExtractionResultItem(BaseModel):
    """Extraction result for one opinion within a job."""

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    case_name: str
    status: ExtractionStatus
    extraction: ExtractionDetail | None = None
    error: str | None = None


class ExtractionJobDetailResponse(BaseModel):
    """Full job status with per-opinion results (GET /extract/{job_id})."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: JobStatus
    total_opinions: int
    processed: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    results: list[ExtractionResultItem]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchResultItem(BaseModel):
    """A single search hit."""

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    case_name: str
    court_id: str
    date_filed: str = Field(..., description="ISO date string")
    holding: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    legal_topics: list[str]


class SearchMetrics(BaseModel):
    """Aggregate metrics describing the search result set."""

    model_config = ConfigDict(frozen=True)

    unique_courts: int = Field(..., ge=0)
    date_range_years: float = Field(..., ge=0.0)
    avg_relevance_score: float = Field(..., ge=0.0, le=1.0)
    avg_pairwise_diversity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Only present when strategy=mmr",
    )


class SearchResponse(BaseModel):
    """Response from POST /search."""

    model_config = ConfigDict(frozen=True)

    results: list[SearchResultItem]
    metrics: SearchMetrics


# ---------------------------------------------------------------------------
# Graph — Conflicts
# ---------------------------------------------------------------------------


class ConflictOpinionSummary(BaseModel):
    """Abbreviated opinion info within a conflict."""

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    case_name: str
    holding: str
    date_filed: str = Field(..., description="ISO date string")
    court: str


class ConflictItem(BaseModel):
    """A single detected circuit split."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str
    topic: str
    court_a: str
    court_b: str
    opinion_a: ConflictOpinionSummary
    opinion_b: ConflictOpinionSummary
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    status: ConflictStatus
    detected_at: datetime


class GraphConflictResponse(BaseModel):
    """Response from GET /graph/conflicts."""

    model_config = ConfigDict(frozen=True)

    conflicts: list[ConflictItem]
    total: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Graph — Authority
# ---------------------------------------------------------------------------


class AuthorityNode(BaseModel):
    """A node in the authority citation subgraph."""

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    case_name: str
    citation_string: str
    court: str
    date_filed: str = Field(..., description="ISO date string")
    citation_count: int = Field(..., ge=0)


class AuthorityEdge(BaseModel):
    """A directed edge in the authority subgraph."""

    model_config = ConfigDict(frozen=True)

    source_id: int
    target_id: int
    citation_type: CitationType
    context: str


class AuthorityGraphResponse(BaseModel):
    """Subgraph for a citation authority query."""

    model_config = ConfigDict(frozen=True)

    anchor: AuthorityNode
    nodes: list[AuthorityNode]
    edges: list[AuthorityEdge]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class IngestionProgressResponse(BaseModel):
    """Progress report returned after an ingestion run."""

    model_config = ConfigDict(frozen=True)

    court_ids: list[str]
    total_fetched: int = Field(..., ge=0)
    total_stored: int = Field(..., ge=0)
    total_skipped: int = Field(..., ge=0, description="Duplicates skipped via ON CONFLICT")
    total_errors: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=0)
    elapsed_seconds: float = Field(..., ge=0.0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Structured error payload — the API never exposes raw stack traces."""

    model_config = ConfigDict(frozen=True)

    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable description")
    details: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    request_id: str | None = None
