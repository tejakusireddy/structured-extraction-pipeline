"""Integration tests for the POST /api/v1/search endpoint.

Uses a mocked Qdrant client and mocked EmbeddingService so tests
never hit real external services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.api.dependencies import get_vector_search
from src.core.config import Settings
from src.services.search.vector_search import VectorSearchService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

pytestmark = pytest.mark.integration


def _build_mock_vector_search() -> VectorSearchService:
    """Build a VectorSearchService with mocked Qdrant + embeddings."""
    mock_qdrant = AsyncMock()

    mock_point = MagicMock()
    mock_point.payload = {
        "opinion_id": 1,
        "case_name": "Smith v. Jones",
        "court_id": "ca9",
        "court_level": "appellate",
        "date_filed": "2024-06-15",
        "holding": "The statute is constitutional under strict scrutiny.",
        "jurisdiction": "ca9",
        "legal_topics": ["constitutional law"],
    }
    mock_point.score = 0.92
    mock_point.vector = [0.1] * 1536

    mock_result = MagicMock()
    mock_result.points = [mock_point]
    mock_qdrant.query_points.return_value = mock_result

    mock_embed = AsyncMock()
    mock_embed.embed_query.return_value = [0.1] * 1536
    mock_embed.embed_batch.return_value = [[0.1] * 1536]

    return VectorSearchService(
        qdrant=mock_qdrant,
        embedding_service=mock_embed,
        collection_name="test_holdings",
        vector_size=1536,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        debug=True,
        database_url=("postgresql+asyncpg://postgres:postgres@localhost:5433/extraction_pipeline"),
        redis_url="redis://localhost:6379/1",
        log_format="console",
        log_level="DEBUG",
        openai_api_key="test-key",
    )


@pytest.fixture
def mock_search_svc() -> VectorSearchService:
    return _build_mock_vector_search()


@pytest.fixture
async def app(
    test_settings: Settings,
    mock_search_svc: VectorSearchService,
) -> AsyncIterator[FastAPI]:
    application = create_app(test_settings)
    application.dependency_overrides[get_vector_search] = lambda: mock_search_svc
    yield application


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


class TestSearchEndpoint:
    async def test_basic_search_returns_200(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={
                "query": "qualified immunity excessive force",
                "k": 5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "metrics" in data
        assert len(data["results"]) >= 1
        assert data["results"][0]["opinion_id"] == 1
        assert data["results"][0]["case_name"] == "Smith v. Jones"
        assert data["results"][0]["relevance_score"] > 0

    async def test_mmr_strategy_accepted(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={
                "query": "due process police conduct",
                "k": 3,
                "strategy": "mmr",
                "lambda_mult": 0.5,
            },
        )
        assert response.status_code == 200

    async def test_search_with_filters(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={
                "query": "fourth amendment search and seizure",
                "k": 10,
                "filters": {
                    "court_level": "appellate",
                    "court_ids": ["ca9"],
                    "date_after": "2020-01-01",
                },
            },
        )
        assert response.status_code == 200

    async def test_metrics_present(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={"query": "test query here"},
        )
        assert response.status_code == 200
        metrics = response.json()["metrics"]
        assert "unique_courts" in metrics
        assert "date_range_years" in metrics
        assert "avg_relevance_score" in metrics

    async def test_query_too_short_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={"query": "ab"},
        )
        assert response.status_code == 422

    async def test_missing_query_returns_422(self, client: AsyncClient):
        response = await client.post("/api/v1/search", json={})
        assert response.status_code == 422

    async def test_invalid_strategy_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={"query": "test query", "strategy": "invalid"},
        )
        assert response.status_code == 422

    async def test_lambda_out_of_range_returns_422(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/search",
            json={
                "query": "test query",
                "lambda_mult": 1.5,
            },
        )
        assert response.status_code == 422

    async def test_empty_results_from_qdrant(
        self, client: AsyncClient, mock_search_svc: VectorSearchService
    ):
        empty_result = MagicMock()
        empty_result.points = []
        mock_search_svc._qdrant.query_points.return_value = empty_result  # type: ignore[attr-defined]

        response = await client.post(
            "/api/v1/search",
            json={"query": "obscure topic with no results"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["metrics"]["unique_courts"] == 0
