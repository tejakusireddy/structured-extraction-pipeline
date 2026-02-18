"""FastAPI dependency injection providers.

Every external resource the API layer needs is accessed through a
Depends() callable defined here. Services are resolved from app.state,
which the lifespan populates at startup.
"""

from functools import lru_cache

from fastapi import Request

from src.core.config import Settings


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
