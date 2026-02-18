"""Integration tests for the POST /ingest endpoint.

Uses a mocked CourtListenerClient so tests never hit the real API.
Runs against the Docker Postgres for realistic database behaviour.
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.dependencies import get_courtlistener_client, get_db_session
from src.core.config import Settings
from src.models.database import Base
from src.services.ingestion.courtlistener import CourtListenerClient
from tests.conftest import TEST_DATABASE_URL

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fake CourtListener response payloads
# ---------------------------------------------------------------------------

_COURT_DATA: dict[str, Any] = {
    "id": "ca9",
    "full_name": "United States Court of Appeals for the Ninth Circuit",
    "short_name": "Ninth Circuit",
    "jurisdiction": "FA",
}

_CLUSTER_DATA: dict[str, Any] = {
    "id": 99001,
    "case_name": "Doe v. Roe",
    "date_filed": "2024-06-15",
    "precedential_status": "Published",
    "citation_count": 5,
    "judges": "Judge Alpha, Judge Beta",
    "sub_opinions": [77001],
}

_OPINION_DATA: dict[str, Any] = {
    "id": 77001,
    "plain_text": (
        "BACKGROUND\n\n"
        "The plaintiff filed suit alleging violations of the Fourth Amendment. "
        "The district court granted summary judgment for the defendants.\n\n"
        "ANALYSIS\n\n"
        "We review the grant of summary judgment de novo. See Heller, "
        "554 U.S. 570 (2008). The Fourth Amendment protects against "
        "unreasonable searches and seizures.\n\n"
        "CONCLUSION\n\n"
        "We AFFIRM the district court's judgment."
    ),
    "html_with_citations": "",
    "html": "",
}


def _build_mock_client() -> CourtListenerClient:
    """Build a mock CourtListenerClient that returns deterministic data."""
    mock = AsyncMock(spec=CourtListenerClient)
    mock.fetch_opinions.return_value = [_CLUSTER_DATA]
    mock.fetch_opinion_detail.return_value = _OPINION_DATA
    mock.fetch_court_detail.return_value = _COURT_DATA
    mock.close.return_value = None
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
        courtlistener_api_key="test-key",
    )


@pytest.fixture
def mock_cl_client() -> CourtListenerClient:
    return _build_mock_client()


@pytest.fixture
async def app(
    test_settings: Settings,
    mock_cl_client: CourtListenerClient,
) -> AsyncIterator[FastAPI]:
    """Create a test app with mocked CL client and real Postgres."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    application = create_app(test_settings)

    application.dependency_overrides[get_courtlistener_client] = lambda: mock_cl_client

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with engine.connect() as conn:
            trans = await conn.begin()
            session = AsyncSession(bind=conn, expire_on_commit=False)
            yield session
            await session.close()
            await trans.rollback()

    application.dependency_overrides[get_db_session] = _override_session

    yield application

    await engine.dispose()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ===========================================================================
# Tests
# ===========================================================================


class TestIngestEndpoint:
    async def test_valid_request_returns_progress(
        self, client: AsyncClient, mock_cl_client: CourtListenerClient
    ):
        response = await client.post(
            "/api/v1/ingest",
            json={
                "court_ids": ["ca9"],
                "max_opinions": 10,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["court_ids"] == ["ca9"]
        assert data["total_fetched"] >= 1
        assert data["total_stored"] >= 0
        assert data["total_skipped"] >= 0
        assert data["total_errors"] >= 0
        assert data["total_chunks"] >= 0
        assert data["elapsed_seconds"] >= 0.0

        mock_cl_client.fetch_opinions.assert_called_once()  # type: ignore[attr-defined]

    async def test_missing_court_ids_and_opinion_ids_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/ingest",
            json={"max_opinions": 10},
        )
        assert response.status_code == 422

    async def test_invalid_body_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/ingest",
            json={"max_opinions": -5},
        )
        assert response.status_code == 422

    async def test_empty_court_ids_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/ingest",
            json={"court_ids": []},
        )
        assert response.status_code == 422

    async def test_with_date_range(self, client: AsyncClient, mock_cl_client: CourtListenerClient):
        response = await client.post(
            "/api/v1/ingest",
            json={
                "court_ids": ["ca9"],
                "date_after": "2024-01-01",
                "date_before": "2024-12-31",
                "max_opinions": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["court_ids"] == ["ca9"]
        assert data["total_fetched"] >= 1

    async def test_multiple_courts(self, client: AsyncClient, mock_cl_client: CourtListenerClient):
        response = await client.post(
            "/api/v1/ingest",
            json={
                "court_ids": ["ca9", "ca5"],
                "max_opinions": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["court_ids"] == ["ca9", "ca5"]
        assert mock_cl_client.fetch_opinions.call_count == 2  # type: ignore[attr-defined]
