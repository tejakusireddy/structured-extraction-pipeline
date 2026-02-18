"""FastAPI dependency injection providers.

Every external resource the API layer needs is accessed through a
Depends() callable defined here. Services are resolved from app.state,
which the lifespan populates at startup.
"""

from collections.abc import AsyncIterator
from functools import lru_cache

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import Settings
from src.db.repositories import CitationRepo, ExtractionRepo, OpinionRepo
from src.services.ingestion.courtlistener import CourtListenerClient


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    from src.core.config import Settings

    return Settings()


def get_settings_from_app(request: Request) -> Settings:
    """Retrieve settings stored on the running app instance.

    Preferred over the cached version inside route handlers since
    it respects the settings the app was actually started with
    (important for tests that override config).
    """
    settings: Settings = request.app.state.settings
    return settings


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield an async DB session scoped to the request lifecycle.

    Commits on success, rolls back on exception, always closes.
    """
    factory = request.app.state.session_factory
    session: AsyncSession = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


def get_opinion_repo(
    session: AsyncSession = Depends(get_db_session),
) -> OpinionRepo:
    """Provide an OpinionRepo bound to the current request session."""
    return OpinionRepo(session)


def get_extraction_repo(
    session: AsyncSession = Depends(get_db_session),
) -> ExtractionRepo:
    """Provide an ExtractionRepo bound to the current request session."""
    return ExtractionRepo(session)


def get_citation_repo(
    session: AsyncSession = Depends(get_db_session),
) -> CitationRepo:
    """Provide a CitationRepo bound to the current request session."""
    return CitationRepo(session)


def get_courtlistener_client(request: Request) -> CourtListenerClient:
    """Retrieve the shared CourtListener client from app state."""
    client: CourtListenerClient = request.app.state.cl_client
    return client
