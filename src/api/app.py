"""FastAPI application factory and lifespan management.

create_app() builds the fully configured application: logging,
middleware, exception handlers, and routes. The lifespan context
manager handles startup/shutdown lifecycle (connection pools,
queue workers, etc. will be added in later steps).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from src.api.dependencies import get_settings
from src.api.middleware import RequestTracingMiddleware, register_exception_handlers
from src.api.routes import api_router
from src.core.config import Settings
from src.core.logging import setup_logging

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings: Settings = app.state.settings
    logger.info("application_starting", version="0.1.0", debug=settings.debug)
    yield
    logger.info("application_shutting_down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a fully configured FastAPI application."""
    if settings is None:
        settings = get_settings()

    setup_logging(settings)

    app = FastAPI(
        title="Structured Extraction Pipeline",
        description="Production-grade document intelligence engine for court opinion analysis",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    app.state.settings = settings

    app.add_middleware(RequestTracingMiddleware)

    register_exception_handlers(app)

    app.include_router(api_router, prefix=settings.api_prefix)

    return app
