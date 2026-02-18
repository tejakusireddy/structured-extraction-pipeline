"""Shared test fixtures and factory functions.

Factories return valid domain objects with sensible defaults. Override
any field via keyword arguments to create specific test scenarios
without repeating boilerplate.
"""

import os
from collections.abc import AsyncIterator
from datetime import date, datetime

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.api.app import create_app
from src.core.config import Settings
from src.models.database import Base
from src.models.domain import (
    CitationType,
    CitedAuthority,
    Conflict,
    ConflictStatus,
    CourtLevel,
    Disposition,
    DocumentChunk,
    ExtractedIntelligence,
    OpinionMetadata,
    PrecedentialStatus,
)

# ---------------------------------------------------------------------------
# Settings / App / Client
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5433/extraction_pipeline",
)


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing — console logs, debug enabled."""
    return Settings(
        debug=True,
        database_url=TEST_DATABASE_URL,
        redis_url="redis://localhost:6379/1",
        log_format="console",
        log_level="DEBUG",
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """FastAPI application wired with test settings."""
    return create_app(test_settings)


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Async HTTP client bound to the test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Database fixtures — test Postgres via Docker
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Per-test async session with automatic rollback for isolation.

    Creates a fresh engine and connection per test. The connection-level
    transaction is rolled back after the test, so no data persists.
    Tables are created idempotently (checkfirst=True is the default).
    """
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


# ---------------------------------------------------------------------------
# Domain model factories
# ---------------------------------------------------------------------------


def make_opinion_metadata(**overrides: object) -> OpinionMetadata:
    """Build a valid OpinionMetadata with sensible defaults."""
    defaults: dict[str, object] = {
        "opinion_id": 12345,
        "cluster_id": 67890,
        "court_id": "ca9",
        "court_name": "United States Court of Appeals for the Ninth Circuit",
        "court_level": CourtLevel.APPELLATE,
        "case_name": "Smith v. Jones",
        "date_filed": date(2023, 6, 15),
        "precedential_status": PrecedentialStatus.PUBLISHED,
        "citation_count": 42,
        "judges": "Nelson, Tallman, Rawlinson",
        "jurisdiction": "ca9",
        "source_url": "https://www.courtlistener.com/opinion/12345/smith-v-jones/",
    }
    defaults.update(overrides)
    return OpinionMetadata(**defaults)  # type: ignore[arg-type]


def make_cited_authority(**overrides: object) -> CitedAuthority:
    """Build a valid CitedAuthority with sensible defaults."""
    defaults: dict[str, object] = {
        "citation_string": "554 U.S. 570 (2008)",
        "case_name": "District of Columbia v. Heller",
        "citation_context": "Followed as precedent for individual right to bear arms",
        "citation_type": CitationType.FOLLOWED,
        "paragraph_context": "The Court held in Heller that the Second Amendment protects...",
    }
    defaults.update(overrides)
    return CitedAuthority(**defaults)  # type: ignore[arg-type]


def make_extracted_intelligence(**overrides: object) -> ExtractedIntelligence:
    """Build a valid ExtractedIntelligence with sensible defaults."""
    defaults: dict[str, object] = {
        "holding": "The defendant's Fourth Amendment rights were not violated by the traffic stop.",
        "holding_confidence": 0.92,
        "legal_standard": "reasonable suspicion",
        "disposition": Disposition.AFFIRMED,
        "disposition_confidence": 0.97,
        "key_authorities": [],
        "dissent_present": False,
        "dissent_summary": None,
        "concurrence_present": False,
        "concurrence_summary": None,
        "legal_topics": ["fourth amendment", "traffic stops", "reasonable suspicion"],
        "extraction_model": "gpt-4o",
        "extraction_timestamp": datetime(2024, 1, 15, 10, 30, 0),
        "raw_prompt_tokens": 2500,
        "raw_completion_tokens": 800,
    }
    defaults.update(overrides)
    return ExtractedIntelligence(**defaults)  # type: ignore[arg-type]


def make_conflict(**overrides: object) -> Conflict:
    """Build a valid Conflict with sensible defaults."""
    defaults: dict[str, object] = {
        "opinion_a_id": 100,
        "opinion_b_id": 200,
        "topic": "qualified immunity - excessive force",
        "court_a": "Fifth Circuit",
        "court_b": "Ninth Circuit",
        "description": "The Fifth Circuit applies qualified immunity broadly...",
        "confidence": 0.78,
        "detected_at": datetime(2024, 3, 1, 12, 0, 0),
        "status": ConflictStatus.DETECTED,
    }
    defaults.update(overrides)
    return Conflict(**defaults)  # type: ignore[arg-type]


def make_document_chunk(**overrides: object) -> DocumentChunk:
    """Build a valid DocumentChunk with sensible defaults."""
    defaults: dict[str, object] = {
        "opinion_id": 12345,
        "chunk_index": 0,
        "text": "The court finds that the officer had reasonable suspicion to initiate the stop.",
        "start_char": 0,
        "end_char": 78,
        "section_type": "majority",
        "citation_strings": ["554 U.S. 570"],
    }
    defaults.update(overrides)
    return DocumentChunk(**defaults)  # type: ignore[arg-type]
