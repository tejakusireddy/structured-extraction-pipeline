# =============================================================================
# Multi-stage Dockerfile for the Structured Extraction Pipeline
#
# Stages:
#   base        — installs uv + Python 3.12
#   dependencies — resolves and installs project deps (cached layer)
#   development — hot-reload server for local dev (docker-compose target)
#   production  — minimal runtime image with no dev tooling
# =============================================================================

# ---------------------------------------------------------------------------
# base: uv + Python 3.12 on slim Debian
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ---------------------------------------------------------------------------
# dependencies: install project deps into a venv (layer-cached)
# ---------------------------------------------------------------------------
FROM base AS dependencies

COPY pyproject.toml README.md uv.lock* ./
COPY src/__init__.py src/__init__.py

RUN uv sync --frozen --no-dev --no-install-project 2>/dev/null \
    || uv sync --no-dev --no-install-project

# ---------------------------------------------------------------------------
# development: hot-reload for docker-compose
# ---------------------------------------------------------------------------
FROM dependencies AS development

COPY src/ src/

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "src"]

# ---------------------------------------------------------------------------
# production: lean runtime image
# ---------------------------------------------------------------------------
FROM dependencies AS production

COPY src/ src/
COPY alembic.ini* ./
COPY scripts/ scripts/

RUN uv sync --no-dev

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8000/api/v1/health').raise_for_status()"]

CMD ["uv", "run", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4"]
