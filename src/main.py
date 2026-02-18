"""Uvicorn entry point for the extraction pipeline API.

Run directly:        python -m src.main
Run via uvicorn:     uvicorn src.main:app --reload
"""

import uvicorn

from src.api.app import create_app
from src.core.config import Settings

app = create_app()


def main() -> None:
    """Start the API server with uvicorn."""
    settings = Settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
