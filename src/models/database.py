"""SQLAlchemy 2.0 ORM models for all database tables.

These map directly to the PostgreSQL schema. Domain enums are stored as
VARCHAR via their StrEnum string values. All tables use server-side
defaults for timestamps where applicable.
"""

import uuid
from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared base for all ORM models."""


# ---------------------------------------------------------------------------
# opinions
# ---------------------------------------------------------------------------


class OpinionRow(Base):
    """Raw court opinion with CourtListener metadata."""

    __tablename__ = "opinions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    courtlistener_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    court_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    court_level: Mapped[str] = mapped_column(String(30), nullable=False)
    case_name: Mapped[str] = mapped_column(Text, nullable=False)
    date_filed: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    precedential_status: Mapped[str] = mapped_column(String(30), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    citation_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    judges: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    jurisdiction: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    extractions: Mapped[list["ExtractionRow"]] = relationship(
        back_populates="opinion", cascade="all, delete-orphan"
    )
    citing_citations: Mapped[list["CitationRow"]] = relationship(
        foreign_keys="CitationRow.citing_opinion_id",
        back_populates="citing_opinion",
        cascade="all, delete-orphan",
    )

    __table_args__ = (Index("ix_opinions_court_date", "court_id", "date_filed"),)

    def __repr__(self) -> str:
        return f"<OpinionRow id={self.id} cl_id={self.courtlistener_id} case={self.case_name!r}>"


# ---------------------------------------------------------------------------
# extractions
# ---------------------------------------------------------------------------


class ExtractionRow(Base):
    """Structured extraction output for a single opinion."""

    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opinion_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("opinions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    holding: Mapped[str] = mapped_column(Text, nullable=False)
    holding_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    legal_standard: Mapped[str | None] = mapped_column(Text, nullable=True)
    disposition: Mapped[str] = mapped_column(String(30), nullable=False)
    disposition_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    dissent_present: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    dissent_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    concurrence_present: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    concurrence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    legal_topics: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    extraction_model: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending", index=True
    )

    opinion: Mapped["OpinionRow"] = relationship(back_populates="extractions")

    def __repr__(self) -> str:
        return f"<ExtractionRow id={self.id} opinion_id={self.opinion_id} status={self.status!r}>"


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


class CitationRow(Base):
    """Edge in the citation graph: citing_opinion → cited_opinion."""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    citing_opinion_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("opinions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cited_opinion_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("opinions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    citation_string: Mapped[str] = mapped_column(Text, nullable=False)
    cited_case_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_context: Mapped[str] = mapped_column(Text, nullable=False)
    citation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    paragraph_context: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    citing_opinion: Mapped["OpinionRow"] = relationship(
        foreign_keys=[citing_opinion_id], back_populates="citing_citations"
    )
    cited_opinion: Mapped["OpinionRow | None"] = relationship(foreign_keys=[cited_opinion_id])

    __table_args__ = (Index("ix_citations_citing_cited", "citing_opinion_id", "cited_opinion_id"),)

    def __repr__(self) -> str:
        return f"<CitationRow id={self.id} citing={self.citing_opinion_id}→{self.cited_opinion_id}>"


# ---------------------------------------------------------------------------
# conflicts
# ---------------------------------------------------------------------------


class ConflictRow(Base):
    """Detected circuit split between two appellate opinions."""

    __tablename__ = "conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opinion_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("opinions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    opinion_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("opinions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    court_a: Mapped[str] = mapped_column(String(100), nullable=False)
    court_b: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="detected", index=True
    )

    opinion_a: Mapped["OpinionRow"] = relationship(foreign_keys=[opinion_a_id])
    opinion_b: Mapped["OpinionRow"] = relationship(foreign_keys=[opinion_b_id])

    __table_args__ = (Index("ix_conflicts_opinions", "opinion_a_id", "opinion_b_id", unique=True),)

    def __repr__(self) -> str:
        return (
            f"<ConflictRow id={self.id} "
            f"a={self.opinion_a_id} b={self.opinion_b_id} topic={self.topic!r}>"
        )


# ---------------------------------------------------------------------------
# extraction_jobs
# ---------------------------------------------------------------------------


class ExtractionJobRow(Base):
    """Batch extraction job tracking."""

    __tablename__ = "extraction_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    opinion_ids: Mapped[list[int]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="queued", index=True
    )
    total_opinions: Mapped[int] = mapped_column(Integer, nullable=False)
    processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    failed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ExtractionJobRow id={self.id!r} "
            f"status={self.status!r} {self.processed}/{self.total_opinions}>"
        )
