"""Tests for CitationResolver.

Covers exact match resolution, fuzzy match by case name, unresolvable
citations, and bulk resolution statistics. Uses a real test Postgres
database via Docker.
"""

from __future__ import annotations

from datetime import date

import pytest

from src.models.database import Base, OpinionRow
from src.models.domain import CitationType, CitedAuthority
from src.services.graph.citation_resolver import CitationResolver, ResolutionStats

pytestmark = pytest.mark.integration


def _make_authority(
    *,
    citation_string: str = "554 U.S. 570 (2008)",
    case_name: str | None = "District of Columbia v. Heller",
    citation_type: CitationType = CitationType.FOLLOWED,
) -> CitedAuthority:
    return CitedAuthority(
        citation_string=citation_string,
        case_name=case_name,
        citation_context="Followed as precedent",
        citation_type=citation_type,
        paragraph_context="The Court held in Heller...",
    )


@pytest.fixture
async def resolver_session():
    """Session with seeded opinions for resolver tests."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from tests.conftest import TEST_DATABASE_URL

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)

        op1 = OpinionRow(
            courtlistener_id=70001,
            court_id="scotus",
            court_level="supreme",
            case_name="District of Columbia v. Heller",
            date_filed=date(2008, 6, 26),
            precedential_status="published",
            raw_text=(
                "The Second Amendment protects an individual right. "
                "See 554 U.S. 570 for the full opinion."
            ),
            citation_count=1000,
            jurisdiction="scotus",
            source_url="https://example.com/70001",
        )
        op2 = OpinionRow(
            courtlistener_id=70002,
            court_id="ca9",
            court_level="appellate",
            case_name="Smith v. Jones",
            date_filed=date(2023, 3, 15),
            precedential_status="published",
            raw_text="The Ninth Circuit applied strict scrutiny.",
            citation_count=5,
            jurisdiction="ca9",
            source_url="https://example.com/70002",
        )
        session.add_all([op1, op2])
        await session.flush()

        yield session, op1.id, op2.id

        await session.close()
        await trans.rollback()

    await engine.dispose()


class TestCitationResolver:
    async def test_exact_match_resolves(self, resolver_session: tuple):
        session, heller_id, _ = resolver_session
        resolver = CitationResolver(session)

        auth = _make_authority(citation_string="554 U.S. 570 (2008)")
        edges = await resolver.resolve_citations([auth], citing_opinion_id=999)

        assert len(edges) == 1
        assert edges[0].cited_opinion_id == heller_id

    async def test_fuzzy_match_by_case_name(self, resolver_session: tuple):
        session, heller_id, _ = resolver_session
        resolver = CitationResolver(session)

        auth = _make_authority(
            citation_string="999 Fake 111",
            case_name="District of Columbia v. Heller",
        )
        edges = await resolver.resolve_citations([auth], citing_opinion_id=999)

        assert len(edges) == 1
        assert edges[0].cited_opinion_id == heller_id

    async def test_unresolvable_tracked(self, resolver_session: tuple):
        session, _heller_id, _ = resolver_session
        resolver = CitationResolver(session)

        auth = _make_authority(
            citation_string="999 Nonexistent 111",
            case_name="Totally Unknown Case",
        )
        edges = await resolver.resolve_citations([auth], citing_opinion_id=999)

        assert len(edges) == 1
        assert edges[0].cited_opinion_id is None

    async def test_multiple_citations_mixed_resolution(self, resolver_session: tuple):
        session, _heller_id, _ = resolver_session
        resolver = CitationResolver(session)

        authorities = [
            _make_authority(citation_string="554 U.S. 570 (2008)"),
            _make_authority(
                citation_string="999 Fake 111",
                case_name="Nonexistent Case",
            ),
        ]
        edges = await resolver.resolve_citations(authorities, citing_opinion_id=999)

        assert len(edges) == 2
        resolved = [e for e in edges if e.cited_opinion_id is not None]
        unresolved = [e for e in edges if e.cited_opinion_id is None]
        assert len(resolved) == 1
        assert len(unresolved) == 1

    async def test_bulk_resolve_stats(self, resolver_session: tuple):
        session, _heller_id, smith_id = resolver_session
        resolver = CitationResolver(session)

        extractions = [
            {
                "opinion_id": smith_id,
                "authorities": [
                    _make_authority(citation_string="554 U.S. 570 (2008)"),
                    _make_authority(
                        citation_string="999 Fake 111",
                        case_name="Unknown",
                    ),
                ],
            }
        ]

        stats = await resolver.bulk_resolve(extractions)

        assert isinstance(stats, ResolutionStats)
        assert stats.total == 2
        assert stats.resolved == 1
        assert stats.unresolved == 1
        assert len(stats.unresolved_citations) == 1

    async def test_bulk_resolve_empty(self, resolver_session: tuple):
        session, _, _ = resolver_session
        resolver = CitationResolver(session)

        stats = await resolver.bulk_resolve([])

        assert stats.total == 0
        assert stats.resolved == 0
        assert stats.unresolved == 0

    async def test_citation_edge_preserves_metadata(self, resolver_session: tuple):
        session, _, _ = resolver_session
        resolver = CitationResolver(session)

        auth = _make_authority(
            citation_string="554 U.S. 570 (2008)",
            case_name="Heller",
            citation_type=CitationType.DISTINGUISHED,
        )
        edges = await resolver.resolve_citations([auth], citing_opinion_id=42)

        assert edges[0].citing_opinion_id == 42
        assert edges[0].citation_string == "554 U.S. 570 (2008)"
        assert edges[0].citation_type == CitationType.DISTINGUISHED
        assert edges[0].cited_case_name == "Heller"
