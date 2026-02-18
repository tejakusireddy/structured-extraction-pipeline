"""Tests for ConflictDetector.

Covers: same circuit → no conflict, different circuits + opposing
dispositions → conflict detected, confidence scoring, min_confidence
filtering, and shared citation analysis.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from src.models.database import Base, CitationRow, ExtractionRow, OpinionRow
from src.services.graph.conflict_detector import ConflictDetector

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.integration


async def _seed_opinions_and_extractions(
    session: AsyncSession,
    *,
    court_a: str = "ca5",
    court_b: str = "ca9",
    disposition_a: str = "affirmed",
    disposition_b: str = "reversed",
    topics_a: list[str] | None = None,
    topics_b: list[str] | None = None,
) -> tuple[int, int]:
    """Seed two opinions with extractions for conflict detection tests."""
    if topics_a is None:
        topics_a = ["qualified immunity", "excessive force"]
    if topics_b is None:
        topics_b = ["qualified immunity", "excessive force"]

    op_a = OpinionRow(
        courtlistener_id=80001,
        court_id=court_a,
        court_level="appellate",
        case_name="Alpha v. Beta",
        date_filed=date(2023, 1, 15),
        precedential_status="published",
        raw_text="The Fifth Circuit held...",
        citation_count=10,
        jurisdiction=court_a,
        source_url="https://example.com/80001",
    )
    op_b = OpinionRow(
        courtlistener_id=80002,
        court_id=court_b,
        court_level="appellate",
        case_name="Gamma v. Delta",
        date_filed=date(2023, 6, 20),
        precedential_status="published",
        raw_text="The Ninth Circuit held...",
        citation_count=8,
        jurisdiction=court_b,
        source_url="https://example.com/80002",
    )
    session.add_all([op_a, op_b])
    await session.flush()

    ext_a = ExtractionRow(
        opinion_id=op_a.id,
        holding="Officers are entitled to qualified immunity.",
        holding_confidence=0.9,
        disposition=disposition_a,
        disposition_confidence=0.95,
        legal_topics=topics_a,
        extraction_model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        status="completed",
    )
    ext_b = ExtractionRow(
        opinion_id=op_b.id,
        holding="Officers are NOT entitled to qualified immunity.",
        holding_confidence=0.88,
        disposition=disposition_b,
        disposition_confidence=0.92,
        legal_topics=topics_b,
        extraction_model="gpt-4o",
        prompt_tokens=1000,
        completion_tokens=500,
        status="completed",
    )
    session.add_all([ext_a, ext_b])
    await session.flush()

    return op_a.id, op_b.id


@pytest.fixture
async def conflict_session():
    """Session for conflict detector tests with rollback isolation."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from tests.conftest import TEST_DATABASE_URL

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)

        yield session

        await session.close()
        await trans.rollback()

    await engine.dispose()


class TestConflictDetector:
    async def test_same_circuit_no_conflict(self, conflict_session: AsyncSession):
        """Opinions from the SAME circuit should never produce a conflict."""
        await _seed_opinions_and_extractions(conflict_session, court_a="ca9", court_b="ca9")
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)
        assert len(conflicts) == 0

    async def test_different_circuits_opposing_dispositions(self, conflict_session: AsyncSession):
        """Different circuits + opposing dispositions + shared topics → conflict."""
        await _seed_opinions_and_extractions(
            conflict_session,
            court_a="ca5",
            court_b="ca9",
            disposition_a="affirmed",
            disposition_b="reversed",
        )
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)

        assert len(conflicts) == 1
        c = conflicts[0]
        assert c.court_a in ("ca5", "ca9")
        assert c.court_b in ("ca5", "ca9")
        assert c.court_a != c.court_b
        assert c.confidence > 0.0

    async def test_no_topic_overlap_no_conflict(self, conflict_session: AsyncSession):
        """Without shared legal topics, no conflict should be detected."""
        await _seed_opinions_and_extractions(
            conflict_session,
            topics_a=["first amendment", "free speech"],
            topics_b=["fourth amendment", "search and seizure"],
        )
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)
        assert len(conflicts) == 0

    async def test_min_confidence_filter(self, conflict_session: AsyncSession):
        """High min_confidence should filter out weaker conflicts."""
        await _seed_opinions_and_extractions(conflict_session)
        detector = ConflictDetector(conflict_session)

        all_conflicts = await detector.detect_conflicts(min_confidence=0.0)
        high_conf = await detector.detect_conflicts(min_confidence=0.99)

        assert len(all_conflicts) >= len(high_conf)

    async def test_confidence_is_bounded(self, conflict_session: AsyncSession):
        """Confidence scores should always be in [0, 1]."""
        await _seed_opinions_and_extractions(conflict_session)
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)

        for c in conflicts:
            assert 0.0 <= c.confidence <= 1.0

    async def test_conflicting_citation_types(self, conflict_session: AsyncSession):
        """Conflicting citation types on shared authority boost confidence."""
        op_a_id, op_b_id = await _seed_opinions_and_extractions(conflict_session)

        cite_a = CitationRow(
            citing_opinion_id=op_a_id,
            cited_opinion_id=None,
            citation_string="554 U.S. 570 (2008)",
            cited_case_name="Heller",
            citation_context="Followed",
            citation_type="followed",
            paragraph_context="The court followed Heller...",
        )
        cite_b = CitationRow(
            citing_opinion_id=op_b_id,
            cited_opinion_id=None,
            citation_string="554 U.S. 570 (2008)",
            cited_case_name="Heller",
            citation_context="Distinguished",
            citation_type="distinguished",
            paragraph_context="The court distinguished Heller...",
        )
        conflict_session.add_all([cite_a, cite_b])
        await conflict_session.flush()

        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)

        assert len(conflicts) == 1
        assert conflicts[0].confidence > 0.3

    async def test_empty_database_no_conflicts(self, conflict_session: AsyncSession):
        """Empty database should return no conflicts."""
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)
        assert conflicts == []

    async def test_court_pairs_filter(self, conflict_session: AsyncSession):
        """court_pairs parameter should restrict which pairs are checked."""
        await _seed_opinions_and_extractions(conflict_session, court_a="ca5", court_b="ca9")
        detector = ConflictDetector(conflict_session)

        filtered = await detector.detect_conflicts(
            court_pairs=[("ca5", "ca11")],
            min_confidence=0.0,
        )
        assert len(filtered) == 0

        included = await detector.detect_conflicts(
            court_pairs=[("ca5", "ca9")],
            min_confidence=0.0,
        )
        assert len(included) == 1

    async def test_conflict_description_not_empty(self, conflict_session: AsyncSession):
        """Detected conflicts should have non-empty descriptions."""
        await _seed_opinions_and_extractions(conflict_session)
        detector = ConflictDetector(conflict_session)
        conflicts = await detector.detect_conflicts(min_confidence=0.0)

        for c in conflicts:
            assert len(c.description) > 0
