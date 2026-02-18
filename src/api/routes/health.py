"""Health check and Prometheus metrics endpoints.

/health probes every infrastructure dependency (Postgres, Redis, Qdrant),
measures per-probe latency, and reports aggregate status. Probes are
run concurrently to minimise total latency.
"""

import asyncio
import time

import httpx
import structlog
from fastapi import APIRouter, Depends
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.dependencies import get_settings_from_app
from src.core.config import Settings
from src.models.responses import DependencyHealth, HealthResponse

router = APIRouter(tags=["observability"])
logger: structlog.stdlib.BoundLogger = structlog.get_logger()

_APP_VERSION = "0.1.0"
_start_time: float = time.time()


# ---------------------------------------------------------------------------
# Dependency probes
# ---------------------------------------------------------------------------


async def _probe_postgres(database_url: str) -> DependencyHealth:
    """Attempt a lightweight asyncpg connection to Postgres."""
    start = time.perf_counter()
    try:
        import asyncpg

        raw_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncio.wait_for(asyncpg.connect(raw_url), timeout=3.0)
        await conn.execute("SELECT 1")
        await conn.close()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(name="postgresql", status="healthy", latency_ms=round(latency, 2))
    except ImportError:
        return DependencyHealth(
            name="postgresql", status="not_configured", details="asyncpg not installed"
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="postgresql",
            status="unhealthy",
            latency_ms=round(latency, 2),
            details=str(exc)[:200],
        )


async def _probe_redis(redis_url: str) -> DependencyHealth:
    """Attempt a PING against Redis."""
    start = time.perf_counter()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(redis_url, socket_connect_timeout=3.0)
        try:
            # redis-py stubs expose ping() as Awaitable[bool] | bool;
            # the async client always returns a coroutine at runtime.
            await client.ping()  # type: ignore[misc]
        finally:
            await client.aclose()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(name="redis", status="healthy", latency_ms=round(latency, 2))
    except ImportError:
        return DependencyHealth(
            name="redis", status="not_configured", details="redis-py not installed"
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="redis",
            status="unhealthy",
            latency_ms=round(latency, 2),
            details=str(exc)[:200],
        )


async def _probe_qdrant(host: str, port: int) -> DependencyHealth:
    """Hit the Qdrant REST healthz endpoint."""
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"http://{host}:{port}/healthz")
            resp.raise_for_status()
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(name="qdrant", status="healthy", latency_ms=round(latency, 2))
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return DependencyHealth(
            name="qdrant",
            status="unhealthy",
            latency_ms=round(latency, 2),
            details=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings_from_app),
) -> HealthResponse:
    """Check API and infrastructure dependency health."""
    probes = await asyncio.gather(
        _probe_postgres(settings.database_url),
        _probe_redis(settings.redis_url),
        _probe_qdrant(settings.qdrant_host, settings.qdrant_port),
    )
    dependencies = list(probes)

    has_unhealthy = any(d.status == "unhealthy" for d in dependencies)
    all_healthy = all(d.status == "healthy" for d in dependencies)

    if all_healthy:
        status = "healthy"
    elif has_unhealthy:
        status = "unhealthy"
    else:
        status = "degraded"

    return HealthResponse(
        status=status,
        version=_APP_VERSION,
        uptime_seconds=round(time.time() - _start_time, 2),
        dependencies=dependencies,
    )


@router.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
