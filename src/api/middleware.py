"""Request tracing middleware and structured exception handlers.

Every request gets a unique ID (from X-Request-ID header or generated),
which is bound to structlog contextvars so all log lines within a
request are correlated. Prometheus counters and histograms are recorded.
Exception handlers translate PipelineError subclasses into structured
ErrorResponse JSON — the API never leaks stack traces.
"""

import time
import uuid

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.core.exceptions import (
    ExtractionValidationError,
    NotFoundError,
    PipelineError,
    RateLimitError,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, bind structured log context, and record metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        logger.info(
            "request_started",
            method=request.method,
            path=str(request.url.path),
        )

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            logger.exception(
                "request_failed",
                method=request.method,
                path=str(request.url.path),
                duration_seconds=round(duration, 4),
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred.",
                    "details": {},
                    "request_id": request_id,
                },
            )

        duration = time.perf_counter() - start

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_seconds=round(duration, 4),
        )
        return response


# ---------------------------------------------------------------------------
# Exception → JSON response handlers
# ---------------------------------------------------------------------------


def _request_id() -> str | None:
    """Pull the current request ID from structlog context, if bound."""
    ctx: dict[str, str] = structlog.contextvars.get_contextvars()
    return ctx.get("request_id")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach structured error handlers to the app."""

    @app.exception_handler(NotFoundError)
    async def _not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "not_found",
                "message": exc.message,
                "details": exc.details,
                "request_id": _request_id(),
            },
        )

    @app.exception_handler(RateLimitError)
    async def _rate_limit(request: Request, exc: RateLimitError) -> JSONResponse:
        headers: dict[str, str] = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(int(exc.retry_after))
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": exc.message,
                "details": exc.details,
                "request_id": _request_id(),
            },
            headers=headers,
        )

    @app.exception_handler(ExtractionValidationError)
    async def _validation(request: Request, exc: ExtractionValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "extraction_validation_error",
                "message": exc.message,
                "details": exc.details,
                "request_id": _request_id(),
            },
        )

    @app.exception_handler(PipelineError)
    async def _pipeline(request: Request, exc: PipelineError) -> JSONResponse:
        logger.error(
            "pipeline_error",
            error_type=type(exc).__name__,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
                "request_id": _request_id(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "details": {},
                "request_id": _request_id(),
            },
        )
