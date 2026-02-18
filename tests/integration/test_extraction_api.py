"""Integration tests for the extraction API endpoints.


POST /api/v1/extract — submit job, returns 202
GET  /api/v1/extract/{job_id} — poll status + results

The LLM client is mocked so tests never call real APIs. A real
Postgres database (via Docker) is used for job and extraction storage.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

from src.api.app import create_app
from src.api.dependencies import get_db_session, get_extraction_worker
from src.core.config import Settings
from src.models.database import Base, OpinionRow
from src.services.extraction.extractor import ExtractionService
from src.services.extraction.llm_client import LLMClient, LLMResponse
from src.services.queue.worker import ExtractionWorker
from tests.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fake LLM response that passes validation
# ---------------------------------------------------------------------------

_VALID_EXTRACTION_JSON = json.dumps(
    {
        "holding": "The court held that the statute is constitutional.",
        "holding_confidence": 0.85,
        "legal_standard": "strict scrutiny",
        "disposition": "affirmed",
        "disposition_confidence": 0.9,
        "key_authorities": [
            {
                "citation_string": "554 U.S. 570 (2008)",
                "case_name": "District of Columbia v. Heller",
                "citation_context": (
                    "The court followed this precedent because it "
                    "establishes the individual right to bear arms."
                ),
                "citation_type": "followed",
                "paragraph_context": (
                    "As established in Heller, 554 U.S. 570 (2008), "
                    "the Second Amendment confers an individual right."
                ),
            }
        ],
        "dissent_present": False,
        "dissent_summary": None,
        "concurrence_present": False,
        "concurrence_summary": None,
        "legal_topics": ["constitutional law", "second amendment"],
        "extraction_model": "gpt-4o",
        "extraction_timestamp": "2025-01-15T12:00:00Z",
        "raw_prompt_tokens": 1500,
        "raw_completion_tokens": 800,
    }
)


def _build_mock_llm_client() -> LLMClient:
    """Build a mock LLMClient that returns valid extraction JSON."""
    mock = AsyncMock(spec=LLMClient)
    mock.extract.return_value = LLMResponse(
        content=_VALID_EXTRACTION_JSON,
        prompt_tokens=1500,
        completion_tokens=800,
        model="gpt-4o",
    )
    mock.is_anthropic_model.return_value = False
    return mock


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
def mock_llm_client() -> LLMClient:
    return _build_mock_llm_client()


@pytest.fixture
async def app(
    test_settings: Settings,
    mock_llm_client: LLMClient,
) -> AsyncIterator[FastAPI]:
    """Test app with mocked LLM client and real Postgres.

    Uses real commits so data persists across POST→GET in the same
    test. Tables are truncated after the test for isolation.
    """
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import async_sessionmaker

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    application = create_app(test_settings)

    extraction_service = ExtractionService(mock_llm_client, test_settings)

    mock_redis = AsyncMock()
    mock_redis.lpush = AsyncMock(return_value=1)
    mock_redis.rpop = AsyncMock(return_value=None)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    worker = ExtractionWorker(
        redis=mock_redis,
        extraction_service=extraction_service,
        session_factory=session_factory,
    )

    application.dependency_overrides[get_extraction_worker] = lambda: worker

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session
            await session.commit()

    application.dependency_overrides[get_db_session] = _override_session

    yield application

    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM extractions"))
        await conn.execute(text("DELETE FROM citations"))
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
async def seeded_opinion_id(app: FastAPI) -> int:
    """Insert a test opinion directly and return its DB id.

    Uses a standalone engine/session so the data is committed and
    visible to the worker's independent session.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        opinion = OpinionRow(
            courtlistener_id=99999,
            court_id="ca9",
            court_level="appellate",
            case_name="Test v. Case",
            date_filed=date(2024, 6, 15),
            precedential_status="published",
            raw_text=("The court finds the statute constitutional under strict scrutiny analysis."),
            citation_count=5,
            judges="Judge A, Judge B",
            jurisdiction="ca9",
            source_url="https://example.com/opinion/99999",
        )
        session.add(opinion)
        await session.commit()
        opinion_id = opinion.id

    await engine.dispose()
    return opinion_id


# ===========================================================================
# Tests
# ===========================================================================


class TestSubmitExtractionJob:
    async def test_post_returns_202_with_job_id(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/extract",
            json={"opinion_ids": [1, 2, 3]},
        )
        assert response.status_code == 202

        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["total_opinions"] == 3
        assert data["estimated_completion_seconds"] is not None

    async def test_post_empty_opinion_ids_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/extract",
            json={"opinion_ids": []},
        )
        assert response.status_code == 422

    async def test_post_missing_body_returns_422(self, client: AsyncClient):
        response = await client.post("/api/v1/extract", json={})
        assert response.status_code == 422


class TestGetExtractionJob:
    async def test_get_nonexistent_job_returns_404(self, client: AsyncClient):
        response = await client.get("/api/v1/extract/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    async def test_get_pending_job_returns_status(self, client: AsyncClient):
        post_resp = await client.post(
            "/api/v1/extract",
            json={"opinion_ids": [1]},
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        get_resp = await client.get(f"/api/v1/extract/{job_id}")
        assert get_resp.status_code == 200

        data = get_resp.json()
        assert data["job_id"] == job_id
        assert data["total_opinions"] == 1
        assert data["processed"] >= 0
        assert data["failed"] >= 0
        assert isinstance(data["results"], list)

    async def test_get_completed_job_has_extraction_results(
        self,
        app: FastAPI,
        client: AsyncClient,
        seeded_opinion_id: int,
        mock_llm_client: LLMClient,
    ):
        """Submit a job for a real opinion, process it, then GET results."""
        post_resp = await client.post(
            "/api/v1/extract",
            json={"opinion_ids": [seeded_opinion_id]},
        )
        assert post_resp.status_code == 202
        job_id = post_resp.json()["job_id"]

        worker: ExtractionWorker = app.dependency_overrides[get_extraction_worker]()
        await worker.process_job(job_id)

        get_resp = await client.get(f"/api/v1/extract/{job_id}")
        assert get_resp.status_code == 200

        data = get_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"
        assert data["processed"] == 1
        assert data["failed"] == 0

        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["opinion_id"] == seeded_opinion_id
        assert result["status"] == "completed"
        assert result["extraction"] is not None
        assert result["extraction"]["holding"] is not None
        assert result["extraction"]["disposition"] == "affirmed"
        assert len(result["extraction"]["legal_topics"]) >= 1
