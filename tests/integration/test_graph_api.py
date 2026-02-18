"""Integration tests for the graph API endpoints.

GET /api/v1/graph/conflicts — circuit split detection
GET /api/v1/graph/authority/{citation} — authority subgraph

Uses a real test Postgres database. Seeds opinions, extractions, and
citations to exercise the full pipeline.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.app import create_app
from src.api.dependencies import get_db_session
from src.core.config import Settings
from src.models.database import Base, CitationRow, ExtractionRow, OpinionRow
from tests.conftest import TEST_DATABASE_URL

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        debug=True,
        database_url=TEST_DATABASE_URL,
        redis_url="redis://localhost:6379/1",
        log_format="console",
        log_level="DEBUG",
        openai_api_key="test-key",
    )


@pytest.fixture
async def app(test_settings: Settings) -> AsyncIterator[FastAPI]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    application = create_app(test_settings)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session
            await session.commit()

    application.dependency_overrides[get_db_session] = _override_session

    yield application

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM citations"))
        await conn.execute(text("DELETE FROM extractions"))
        await conn.execute(text("DELETE FROM conflicts"))
        await conn.execute(text("DELETE FROM extraction_jobs"))
        await conn.execute(text("DELETE FROM opinions"))

    await engine.dispose()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
async def seeded_conflict_data(app: FastAPI) -> dict[str, int]:
    """Seed two opinions with opposing dispositions from different circuits."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        op_a = OpinionRow(
            courtlistener_id=90001,
            court_id="ca5",
            court_level="appellate",
            case_name="Alpha v. State",
            date_filed=date(2023, 3, 15),
            precedential_status="published",
            raw_text="The Fifth Circuit held officers are protected.",
            citation_count=10,
            jurisdiction="ca5",
            source_url="https://example.com/90001",
        )
        op_b = OpinionRow(
            courtlistener_id=90002,
            court_id="ca9",
            court_level="appellate",
            case_name="Beta v. City",
            date_filed=date(2023, 7, 20),
            precedential_status="published",
            raw_text="The Ninth Circuit held officers are NOT protected.",
            citation_count=8,
            jurisdiction="ca9",
            source_url="https://example.com/90002",
        )
        session.add_all([op_a, op_b])
        await session.flush()

        ext_a = ExtractionRow(
            opinion_id=op_a.id,
            holding="Officers are entitled to qualified immunity.",
            holding_confidence=0.9,
            disposition="affirmed",
            disposition_confidence=0.95,
            legal_topics=["qualified immunity", "excessive force"],
            extraction_model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            status="completed",
        )
        ext_b = ExtractionRow(
            opinion_id=op_b.id,
            holding="Officers are NOT entitled to qualified immunity.",
            holding_confidence=0.88,
            disposition="reversed",
            disposition_confidence=0.92,
            legal_topics=["qualified immunity", "excessive force"],
            extraction_model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            status="completed",
        )
        session.add_all([ext_a, ext_b])
        await session.flush()

        cite_a = CitationRow(
            citing_opinion_id=op_a.id,
            cited_opinion_id=op_b.id,
            citation_string="Test cite",
            cited_case_name="Beta v. City",
            citation_context="Distinguished",
            citation_type="distinguished",
            paragraph_context="The court distinguished...",
        )
        session.add(cite_a)
        await session.commit()

        result = {"op_a_id": op_a.id, "op_b_id": op_b.id}

    await engine.dispose()
    return result


@pytest.fixture
async def seeded_authority_data(app: FastAPI) -> dict[str, int]:
    """Seed opinions with citation edges for authority graph tests."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        root = OpinionRow(
            courtlistener_id=91001,
            court_id="scotus",
            court_level="supreme",
            case_name="Landmark v. State",
            date_filed=date(2010, 6, 15),
            precedential_status="published",
            raw_text="554 U.S. 570 landmark ruling.",
            citation_count=100,
            jurisdiction="scotus",
            source_url="https://example.com/91001",
        )
        child = OpinionRow(
            courtlistener_id=91002,
            court_id="ca9",
            court_level="appellate",
            case_name="Follower v. Party",
            date_filed=date(2020, 3, 10),
            precedential_status="published",
            raw_text="Following Landmark v. State.",
            citation_count=5,
            jurisdiction="ca9",
            source_url="https://example.com/91002",
        )
        session.add_all([root, child])
        await session.flush()

        ext = ExtractionRow(
            opinion_id=root.id,
            holding="Landmark holding.",
            holding_confidence=0.95,
            disposition="affirmed",
            disposition_confidence=0.99,
            legal_topics=["constitutional law"],
            extraction_model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            status="completed",
        )
        session.add(ext)
        await session.flush()

        cite = CitationRow(
            citing_opinion_id=child.id,
            cited_opinion_id=root.id,
            citation_string="554 U.S. 570 (2010)",
            cited_case_name="Landmark v. State",
            citation_context="Followed as binding precedent",
            citation_type="followed",
            paragraph_context="Following Landmark...",
        )
        session.add(cite)
        await session.commit()

        result = {"root_id": root.id, "child_id": child.id}

    await engine.dispose()
    return result


# ===========================================================================
# Tests — Conflicts endpoint
# ===========================================================================


class TestConflictsEndpoint:
    async def test_conflicts_empty_db_returns_200(self, client: AsyncClient):
        response = await client.get("/api/v1/graph/conflicts")
        assert response.status_code == 200
        data = response.json()
        assert data["conflicts"] == []
        assert data["total"] == 0

    async def test_conflicts_with_seeded_data(
        self,
        client: AsyncClient,
        seeded_conflict_data: dict[str, int],
    ):
        response = await client.get("/api/v1/graph/conflicts?min_confidence=0.0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["conflicts"]) >= 1

        conflict = data["conflicts"][0]
        assert "conflict_id" in conflict
        assert "topic" in conflict
        assert "confidence" in conflict
        assert conflict["confidence"] > 0
        assert "court_a" in conflict
        assert "court_b" in conflict

    async def test_conflicts_min_confidence_filters(
        self,
        client: AsyncClient,
        seeded_conflict_data: dict[str, int],
    ):
        resp_all = await client.get("/api/v1/graph/conflicts?min_confidence=0.0")
        resp_high = await client.get("/api/v1/graph/conflicts?min_confidence=0.99")
        assert resp_all.status_code == 200
        assert resp_high.status_code == 200
        assert resp_all.json()["total"] >= resp_high.json()["total"]

    async def test_conflicts_invalid_min_confidence(self, client: AsyncClient):
        response = await client.get("/api/v1/graph/conflicts?min_confidence=1.5")
        assert response.status_code == 422


# ===========================================================================
# Tests — Authority endpoint
# ===========================================================================


class TestAuthorityEndpoint:
    async def test_authority_not_found_returns_empty(self, client: AsyncClient):
        response = await client.get("/api/v1/graph/authority/nonexistent-topic-xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["anchor"]["opinion_id"] == 0
        assert data["nodes"] == []
        assert data["edges"] == []

    async def test_authority_with_seeded_citation(
        self,
        client: AsyncClient,
        seeded_authority_data: dict[str, int],
    ):
        """Query by citation string found in the seeded opinion text."""
        response = await client.get("/api/v1/graph/authority/554 U.S. 570")
        assert response.status_code == 200
        data = response.json()
        assert data["anchor"]["opinion_id"] != 0
        assert data["anchor"]["opinion_id"] == seeded_authority_data["root_id"]

    async def test_authority_depth_parameter(
        self,
        client: AsyncClient,
        seeded_authority_data: dict[str, int],
    ):
        response = await client.get("/api/v1/graph/authority/constitutional law?depth=1")
        assert response.status_code == 200
        data = response.json()
        assert "anchor" in data
        assert "nodes" in data
        assert "edges" in data

    async def test_authority_invalid_depth(self, client: AsyncClient):
        response = await client.get("/api/v1/graph/authority/test?depth=0")
        assert response.status_code == 422

    async def test_authority_response_structure(
        self,
        client: AsyncClient,
        seeded_authority_data: dict[str, int],
    ):
        response = await client.get("/api/v1/graph/authority/constitutional law")
        assert response.status_code == 200
        data = response.json()

        anchor = data["anchor"]
        assert "opinion_id" in anchor
        assert "case_name" in anchor
        assert "court" in anchor
        assert "citation_count" in anchor

        for node in data["nodes"]:
            assert "opinion_id" in node
            assert "case_name" in node

        for edge in data["edges"]:
            assert "source_id" in edge
            assert "target_id" in edge
            assert "citation_type" in edge
