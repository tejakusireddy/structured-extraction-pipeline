"""Core domain models and enumerations.

These are the canonical data shapes for the extraction pipeline. Every
service produces or consumes these types — never raw dicts. Frozen
models are used for value objects that should be immutable once created.
"""

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CourtLevel(StrEnum):
    """Federal and state court hierarchy tiers."""

    SUPREME = "supreme"
    APPELLATE = "appellate"
    DISTRICT = "district"
    STATE_SUPREME = "state_supreme"
    STATE_APPELLATE = "state_appellate"
    STATE_TRIAL = "state_trial"
    BANKRUPTCY = "bankruptcy"
    SPECIALIZED = "specialized"


class PrecedentialStatus(StrEnum):
    """CourtListener precedential status values."""

    PUBLISHED = "published"
    UNPUBLISHED = "unpublished"
    ERRATA = "errata"
    SEPARATE_OPINION = "separate_opinion"
    IN_CHAMBERS = "in_chambers"
    RELATING_TO = "relating_to"
    UNKNOWN = "unknown"


class Disposition(StrEnum):
    """How the court disposed of the case."""

    AFFIRMED = "affirmed"
    REVERSED = "reversed"
    REMANDED = "remanded"
    VACATED = "vacated"
    DISMISSED = "dismissed"
    AFFIRMED_IN_PART = "affirmed_in_part"
    REVERSED_IN_PART = "reversed_in_part"


class CitationType(StrEnum):
    """How a cited authority was used by the citing opinion."""

    FOLLOWED = "followed"
    DISTINGUISHED = "distinguished"
    OVERRULED = "overruled"
    CITED = "cited"


class ExtractionStatus(StrEnum):
    """Lifecycle status of a single extraction."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class JobStatus(StrEnum):
    """Lifecycle status of a batch extraction job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(StrEnum):
    """Extraction job priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ConflictStatus(StrEnum):
    """Lifecycle status of a detected circuit split."""

    DETECTED = "detected"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"


class SearchStrategy(StrEnum):
    """Vector search retrieval strategy."""

    SIMILARITY = "similarity"
    MMR = "mmr"


# ---------------------------------------------------------------------------
# Domain models — immutable value objects
# ---------------------------------------------------------------------------


class CitedAuthority(BaseModel):
    """A citation with context — not just the string, but WHY it was cited."""

    model_config = ConfigDict(frozen=True)

    citation_string: str = Field(
        ...,
        min_length=1,
        description="Full citation string, e.g. '554 U.S. 570 (2008)'",
    )
    case_name: str | None = Field(
        default=None,
        description="e.g. 'District of Columbia v. Heller'",
    )
    citation_context: str = Field(
        ...,
        min_length=1,
        description="How this case was used (distinguished, followed, etc.)",
    )
    citation_type: CitationType
    paragraph_context: str = Field(
        ...,
        min_length=1,
        description="The paragraph where the citation appears",
    )


class OpinionMetadata(BaseModel):
    """Metadata for a court opinion sourced from CourtListener."""

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    cluster_id: int
    court_id: str = Field(..., min_length=1)
    court_name: str = Field(..., min_length=1)
    court_level: CourtLevel
    case_name: str = Field(..., min_length=1)
    date_filed: date
    precedential_status: PrecedentialStatus
    citation_count: int = Field(default=0, ge=0)
    judges: str = ""
    jurisdiction: str = Field(
        ...,
        min_length=1,
        description="e.g. 'ca9', 'scotus'",
    )
    source_url: str = Field(..., min_length=1)


class ExtractedIntelligence(BaseModel):
    """Structured intelligence extracted from a court opinion by an LLM.

    Every field is validated — confidence scores are bounded, legal_topics
    must be non-empty, and token counts are tracked for cost analysis.
    """

    model_config = ConfigDict(frozen=True)

    holding: str = Field(..., min_length=1, description="What the court decided")
    holding_confidence: float = Field(..., ge=0.0, le=1.0)
    legal_standard: str | None = Field(
        default=None,
        description="e.g. 'strict scrutiny', 'rational basis'",
    )
    disposition: Disposition
    disposition_confidence: float = Field(..., ge=0.0, le=1.0)
    key_authorities: list[CitedAuthority] = Field(default_factory=list)
    dissent_present: bool = False
    dissent_summary: str | None = None
    concurrence_present: bool = False
    concurrence_summary: str | None = None
    legal_topics: list[str] = Field(..., min_length=1)
    extraction_model: str = Field(..., min_length=1)
    extraction_timestamp: datetime
    raw_prompt_tokens: int = Field(..., ge=0)
    raw_completion_tokens: int = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Supporting domain types
# ---------------------------------------------------------------------------


class Opinion(BaseModel):
    """Full opinion combining metadata, raw text, and optional extraction."""

    metadata: OpinionMetadata
    raw_text: str = Field(..., min_length=1)
    extraction: ExtractedIntelligence | None = None


class CitationEdge(BaseModel):
    """Directed edge in the citation graph: citing → cited."""

    model_config = ConfigDict(frozen=True)

    citing_opinion_id: int
    cited_opinion_id: int | None = Field(
        default=None,
        description="Null when cited opinion is not in our database",
    )
    citation_string: str = Field(..., min_length=1)
    cited_case_name: str | None = None
    citation_context: str = Field(..., min_length=1)
    citation_type: CitationType
    paragraph_context: str = Field(..., min_length=1)


class Conflict(BaseModel):
    """A detected circuit split between two appellate opinions."""

    model_config = ConfigDict(frozen=True)

    opinion_a_id: int
    opinion_b_id: int
    topic: str = Field(..., min_length=1)
    court_a: str = Field(..., min_length=1)
    court_b: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_at: datetime
    status: ConflictStatus = ConflictStatus.DETECTED


class DocumentChunk(BaseModel):
    """A segment of an opinion text produced by the legal-aware chunker.

    Preserves citation boundaries and tracks position within the source
    document so downstream extraction can reference back to the original.
    """

    model_config = ConfigDict(frozen=True)

    opinion_id: int
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    section_type: str | None = Field(
        default=None,
        description="e.g. 'majority', 'dissent', 'concurrence', 'syllabus'",
    )
    citation_strings: list[str] = Field(
        default_factory=list,
        description="Citations contained in this chunk, preserved during splitting",
    )
