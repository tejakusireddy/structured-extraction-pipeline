"""Tests for database repository classes.

Requires a running Postgres instance (Docker). Each test gets its own
transaction that is rolled back, so tests are isolated and leave no
persistent data.
"""

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.repositories import CitationRepo, ExtractionRepo, OpinionRepo
from src.models.database import CitationRow, ExtractionRow, OpinionRow

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — build OpinionRow dicts (not ORM instances) for convenience
# ---------------------------------------------------------------------------

_COUNTER = 0


def _opinion_kwargs(**overrides: object) -> dict[str, object]:
    global _COUNTER
    _COUNTER += 1
    defaults: dict[str, object] = {
        "courtlistener_id": 90_000 + _COUNTER,
        "court_id": "ca9",
        "court_level": "appellate",
        "case_name": f"Test Case #{_COUNTER}",
        "date_filed": date(2024, 1, 15),
        "precedential_status": "published",
        "raw_text": "Lorem ipsum dolor sit amet.",
        "citation_count": 0,
        "judges": "Judge A, Judge B",
        "jurisdiction": "ca9",
        "source_url": f"https://example.com/opinion/{_COUNTER}",
    }
    defaults.update(overrides)
    return defaults


def _make_opinion(**overrides: object) -> OpinionRow:
    return OpinionRow(**_opinion_kwargs(**overrides))


def _make_extraction(opinion_id: int, **overrides: object) -> ExtractionRow:
    defaults: dict[str, object] = {
        "opinion_id": opinion_id,
        "holding": "The court holds X.",
        "holding_confidence": 0.95,
        "legal_standard": "strict scrutiny",
        "disposition": "affirmed",
        "disposition_confidence": 0.90,
        "legal_topics": ["first amendment", "free speech"],
        "extraction_model": "gpt-4o",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
    }
    defaults.update(overrides)
    return ExtractionRow(**defaults)


def _make_citation(
    citing_id: int,
    cited_id: int | None,
    **overrides: object,
) -> CitationRow:
    defaults: dict[str, object] = {
        "citing_opinion_id": citing_id,
        "cited_opinion_id": cited_id,
        "citation_string": "554 U.S. 570",
        "cited_case_name": "Heller",
        "citation_context": "Followed as binding precedent",
        "citation_type": "followed",
        "paragraph_context": "In Heller, the Court held...",
    }
    defaults.update(overrides)
    return CitationRow(**defaults)


# ===========================================================================
# OpinionRepo
# ===========================================================================


class TestOpinionRepo:
    async def test_create_and_get_by_id(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        opinion = _make_opinion(case_name="Create Test v. State")

        created = await repo.create(opinion)
        assert created.id is not None
        assert created.case_name == "Create Test v. State"

        fetched = await repo.get_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.case_name == "Create Test v. State"

    async def test_get_by_id_returns_none_for_missing(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        assert await repo.get_by_id(999_999) is None

    async def test_get_by_courtlistener_id(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        opinion = _make_opinion(courtlistener_id=77777)
        await repo.create(opinion)

        found = await repo.get_by_courtlistener_id(77777)
        assert found is not None
        assert found.courtlistener_id == 77777

        assert await repo.get_by_courtlistener_id(99999) is None

    async def test_list_by_court_basic(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        await repo.create(_make_opinion(court_id="ca5", date_filed=date(2024, 3, 1)))
        await repo.create(_make_opinion(court_id="ca5", date_filed=date(2024, 1, 1)))
        await repo.create(_make_opinion(court_id="ca9", date_filed=date(2024, 2, 1)))

        ca5 = await repo.list_by_court("ca5")
        assert len(ca5) == 2
        assert ca5[0].date_filed >= ca5[1].date_filed

    async def test_list_by_court_date_filters(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        await repo.create(_make_opinion(court_id="ca11", date_filed=date(2024, 1, 1)))
        await repo.create(_make_opinion(court_id="ca11", date_filed=date(2024, 6, 15)))
        await repo.create(_make_opinion(court_id="ca11", date_filed=date(2024, 12, 1)))

        filtered = await repo.list_by_court(
            "ca11",
            date_after=date(2024, 3, 1),
            date_before=date(2024, 9, 1),
        )
        assert len(filtered) == 1
        assert filtered[0].date_filed == date(2024, 6, 15)

    async def test_list_by_court_pagination(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        for i in range(5):
            await repo.create(_make_opinion(court_id="ca3", date_filed=date(2024, 1, i + 1)))

        page1 = await repo.list_by_court("ca3", limit=2, offset=0)
        page2 = await repo.list_by_court("ca3", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id

    async def test_bulk_create(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        rows = [_opinion_kwargs(court_id="bulk") for _ in range(3)]

        inserted = await repo.bulk_create(rows)
        assert inserted == 3

    async def test_bulk_create_skips_duplicates(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        cl_id = 111_111
        await repo.create(_make_opinion(courtlistener_id=cl_id))

        rows = [
            _opinion_kwargs(courtlistener_id=cl_id),
            _opinion_kwargs(courtlistener_id=cl_id + 1),
        ]
        inserted = await repo.bulk_create(rows)
        assert inserted == 1

    async def test_bulk_create_empty_list(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        assert await repo.bulk_create([]) == 0

    async def test_count(self, db_session: AsyncSession):
        repo = OpinionRepo(db_session)
        await repo.create(_make_opinion(court_id="count_test"))
        await repo.create(_make_opinion(court_id="count_test"))
        await repo.create(_make_opinion(court_id="count_other"))

        total = await repo.count()
        assert total >= 3
        by_court = await repo.count(court_id="count_test")
        assert by_court == 2


# ===========================================================================
# ExtractionRepo
# ===========================================================================


class TestExtractionRepo:
    async def test_create_and_get_by_opinion_id(self, db_session: AsyncSession):
        o_repo = OpinionRepo(db_session)
        e_repo = ExtractionRepo(db_session)

        opinion = await o_repo.create(_make_opinion())
        ext = await e_repo.create(_make_extraction(opinion.id))

        assert ext.id is not None
        assert ext.opinion_id == opinion.id

        results = await e_repo.get_by_opinion_id(opinion.id)
        assert len(results) == 1
        assert results[0].id == ext.id

    async def test_get_by_opinion_id_empty(self, db_session: AsyncSession):
        e_repo = ExtractionRepo(db_session)
        results = await e_repo.get_by_opinion_id(999_999)
        assert results == []

    async def test_update_status(self, db_session: AsyncSession):
        o_repo = OpinionRepo(db_session)
        e_repo = ExtractionRepo(db_session)

        opinion = await o_repo.create(_make_opinion())
        ext = await e_repo.create(_make_extraction(opinion.id, status="pending"))

        updated = await e_repo.update_status(ext.id, "completed")
        assert updated is True

        await db_session.refresh(ext)
        assert ext.status == "completed"

    async def test_update_status_nonexistent(self, db_session: AsyncSession):
        e_repo = ExtractionRepo(db_session)
        updated = await e_repo.update_status(999_999, "completed")
        assert updated is False

    async def test_list_pending(self, db_session: AsyncSession):
        o_repo = OpinionRepo(db_session)
        e_repo = ExtractionRepo(db_session)

        opinion = await o_repo.create(_make_opinion())
        await e_repo.create(_make_extraction(opinion.id, status="pending"))
        await e_repo.create(_make_extraction(opinion.id, status="completed"))
        await e_repo.create(_make_extraction(opinion.id, status="pending"))

        pending = await e_repo.list_pending()
        assert len(pending) == 2
        assert all(e.status == "pending" for e in pending)

    async def test_list_pending_respects_limit(self, db_session: AsyncSession):
        o_repo = OpinionRepo(db_session)
        e_repo = ExtractionRepo(db_session)

        opinion = await o_repo.create(_make_opinion())
        for _ in range(5):
            await e_repo.create(_make_extraction(opinion.id, status="pending"))

        limited = await e_repo.list_pending(limit=2)
        assert len(limited) == 2


# ===========================================================================
# CitationRepo
# ===========================================================================


class TestCitationRepo:
    async def _create_graph_opinions(
        self, db_session: AsyncSession, count: int
    ) -> list[OpinionRow]:
        """Insert N opinions and return them."""
        repo = OpinionRepo(db_session)
        opinions = []
        for _ in range(count):
            op = await repo.create(_make_opinion())
            opinions.append(op)
        return opinions

    async def test_create_batch(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        opinions = await self._create_graph_opinions(db_session, 2)

        cits = [_make_citation(opinions[0].id, opinions[1].id)]
        result = await c_repo.create_batch(cits)
        assert len(result) == 1
        assert result[0].id is not None

    async def test_create_batch_empty(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        assert await c_repo.create_batch([]) == []

    async def test_forward_lookup(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        opinions = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(opinions[0].id, opinions[1].id),
                _make_citation(opinions[0].id, opinions[2].id),
            ]
        )

        forward = await c_repo.get_citations_for_opinion(opinions[0].id)
        assert len(forward) == 2
        cited_ids = {c.cited_opinion_id for c in forward}
        assert cited_ids == {opinions[1].id, opinions[2].id}

    async def test_reverse_lookup(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        opinions = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(opinions[0].id, opinions[2].id),
                _make_citation(opinions[1].id, opinions[2].id),
            ]
        )

        reverse = await c_repo.get_cited_by(opinions[2].id)
        assert len(reverse) == 2
        citing_ids = {c.citing_opinion_id for c in reverse}
        assert citing_ids == {opinions[0].id, opinions[1].id}

    async def test_forward_lookup_empty(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        opinions = await self._create_graph_opinions(db_session, 1)
        assert await c_repo.get_citations_for_opinion(opinions[0].id) == []

    async def test_subgraph_depth_1_outgoing(self, db_session: AsyncSession):
        """A → B, A → C  (depth 1 from A should return A, B, C)."""
        c_repo = CitationRepo(db_session)
        ops = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(ops[0].id, ops[1].id),
                _make_citation(ops[0].id, ops[2].id),
            ]
        )

        graph = await c_repo.get_citation_subgraph(ops[0].id, max_depth=1, direction="outgoing")
        ids = {n["opinion_id"] for n in graph}
        assert ops[0].id in ids
        assert ops[1].id in ids
        assert ops[2].id in ids

        root = next(n for n in graph if n["opinion_id"] == ops[0].id)
        assert root["depth"] == 0

    async def test_subgraph_depth_2_outgoing(self, db_session: AsyncSession):
        """A → B → C  (depth 2 from A should reach C)."""
        c_repo = CitationRepo(db_session)
        ops = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(ops[0].id, ops[1].id),
                _make_citation(ops[1].id, ops[2].id),
            ]
        )

        graph = await c_repo.get_citation_subgraph(ops[0].id, max_depth=2, direction="outgoing")
        ids = {n["opinion_id"] for n in graph}
        assert ids == {ops[0].id, ops[1].id, ops[2].id}

        depths = {n["opinion_id"]: n["depth"] for n in graph}
        assert depths[ops[0].id] == 0
        assert depths[ops[1].id] == 1
        assert depths[ops[2].id] == 2

    async def test_subgraph_depth_1_does_not_reach_depth_2(self, db_session: AsyncSession):
        """A → B → C  (depth 1 from A should NOT reach C)."""
        c_repo = CitationRepo(db_session)
        ops = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(ops[0].id, ops[1].id),
                _make_citation(ops[1].id, ops[2].id),
            ]
        )

        graph = await c_repo.get_citation_subgraph(ops[0].id, max_depth=1, direction="outgoing")
        ids = {n["opinion_id"] for n in graph}
        assert ops[2].id not in ids

    async def test_subgraph_incoming(self, db_session: AsyncSession):
        """B → A, C → A  (incoming from A should find B and C)."""
        c_repo = CitationRepo(db_session)
        ops = await self._create_graph_opinions(db_session, 3)

        await c_repo.create_batch(
            [
                _make_citation(ops[1].id, ops[0].id),
                _make_citation(ops[2].id, ops[0].id),
            ]
        )

        graph = await c_repo.get_citation_subgraph(ops[0].id, max_depth=1, direction="incoming")
        ids = {n["opinion_id"] for n in graph}
        assert ids == {ops[0].id, ops[1].id, ops[2].id}

    async def test_subgraph_invalid_direction(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        with pytest.raises(ValueError, match="direction must be"):
            await c_repo.get_citation_subgraph(1, direction="sideways")

    async def test_pagination_forward(self, db_session: AsyncSession):
        c_repo = CitationRepo(db_session)
        ops = await self._create_graph_opinions(db_session, 4)

        await c_repo.create_batch(
            [
                _make_citation(ops[0].id, ops[1].id),
                _make_citation(ops[0].id, ops[2].id),
                _make_citation(ops[0].id, ops[3].id),
            ]
        )

        page1 = await c_repo.get_citations_for_opinion(ops[0].id, limit=2, offset=0)
        page2 = await c_repo.get_citations_for_opinion(ops[0].id, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 1
