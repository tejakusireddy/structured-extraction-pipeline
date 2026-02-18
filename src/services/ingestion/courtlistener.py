"""Async CourtListener API v4 client.

Rate-limited, paginated, and retry-enabled client for fetching
court opinions and their metadata. Uses an in-memory token bucket
for rate limiting (Redis-backed rate limiter can be swapped in).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import IngestionError, RateLimitError

if TYPE_CHECKING:
    from datetime import date

    from src.core.config import Settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


# ---------------------------------------------------------------------------
# In-memory token bucket rate limiter
# ---------------------------------------------------------------------------


class TokenBucket:
    """Async token bucket rate limiter.

    Refills at ``rate`` tokens per second up to ``max_tokens`` capacity.
    ``acquire()`` blocks until a token is available.
    """

    def __init__(self, rate: float, max_tokens: int | None = None) -> None:
        self._rate = rate
        self._max_tokens = float(max_tokens or int(rate * 2) or 1)
        self._tokens = self._max_tokens
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                wait_time = deficit / self._rate
                await asyncio.sleep(wait_time)
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# CourtListener client
# ---------------------------------------------------------------------------


class CourtListenerClient:
    """Async HTTP client for CourtListener REST API v4.

    Features:
    - Token-bucket rate limiting (configurable requests/sec)
    - Automatic cursor-based pagination
    - Exponential backoff retry via tenacity
    - Structured logging for all API interactions
    """

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self._base_url = settings.courtlistener_api_url.rstrip("/")
        self._api_key = settings.courtlistener_api_key
        self._rate_limiter = TokenBucket(
            rate=settings.courtlistener_rate_limit,
            max_tokens=settings.courtlistener_rate_limit * 2,
        )
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the underlying HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Token {self._api_key}"
        return headers

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Rate-limited GET with retry. Returns parsed JSON."""
        await self._rate_limiter.acquire()

        logger.debug("courtlistener_request", url=url, params=params)
        response = await self._client.get(url, headers=self._headers(), params=params)

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "5"))
            logger.warning("courtlistener_rate_limited", retry_after=retry_after)
            raise RateLimitError(
                "CourtListener rate limit hit",
                retry_after=retry_after,
            )

        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    async def _paginate(
        self,
        url: str,
        params: dict[str, Any],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Follow cursor-based pagination up to max_results items."""
        results: list[dict[str, Any]] = []
        next_url: str | None = url

        while next_url and len(results) < max_results:
            current_params = params if next_url == url else None
            data = await self._get(next_url, params=current_params)

            page_results = data.get("results", [])
            if not page_results:
                break

            remaining = max_results - len(results)
            results.extend(page_results[:remaining])
            next_url = data.get("next")

        return results

    async def fetch_opinions(
        self,
        court_id: str,
        *,
        date_after: date | None = None,
        date_before: date | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch opinion clusters from a court within an optional date range.

        Returns raw API response dicts (cluster objects) for downstream
        parsing via the parser module.
        """
        url = f"{self._base_url}/clusters/"
        params: dict[str, Any] = {
            "court": court_id,
            "order_by": "-date_filed",
            "page_size": min(max_results, 20),
        }
        if date_after:
            params["date_filed__gte"] = date_after.isoformat()
        if date_before:
            params["date_filed__lte"] = date_before.isoformat()

        logger.info(
            "fetching_opinions",
            court_id=court_id,
            date_after=str(date_after) if date_after else None,
            date_before=str(date_before) if date_before else None,
            max_results=max_results,
        )

        try:
            results = await self._paginate(url, params, max_results)
        except httpx.HTTPStatusError as exc:
            msg = f"CourtListener API error: {exc.response.status_code}"
            raise IngestionError(msg, details={"url": url}) from exc
        except httpx.TransportError as exc:
            msg = f"CourtListener connection error: {exc}"
            raise IngestionError(msg) from exc

        logger.info("fetched_opinions", count=len(results), court_id=court_id)
        return results

    async def fetch_opinion_detail(self, opinion_id: int) -> dict[str, Any]:
        """Fetch a single opinion by its CourtListener opinion ID.

        Returns the full opinion object including text fields.
        """
        url = f"{self._base_url}/opinions/{opinion_id}/"

        logger.info("fetching_opinion_detail", opinion_id=opinion_id)

        try:
            return await self._get(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                msg = f"Opinion {opinion_id} not found on CourtListener"
                raise IngestionError(msg) from exc
            msg = f"CourtListener API error: {exc.response.status_code}"
            raise IngestionError(msg, details={"opinion_id": opinion_id}) from exc
        except httpx.TransportError as exc:
            msg = f"CourtListener connection error: {exc}"
            raise IngestionError(msg) from exc

    async def fetch_cluster_detail(self, cluster_id: int) -> dict[str, Any]:
        """Fetch a single cluster by its CourtListener cluster ID."""
        url = f"{self._base_url}/clusters/{cluster_id}/"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError as exc:
            msg = f"CourtListener cluster {cluster_id} fetch failed: {exc.response.status_code}"
            raise IngestionError(msg) from exc

    async def fetch_court_detail(self, court_id: str) -> dict[str, Any]:
        """Fetch court metadata by its CourtListener court ID."""
        url = f"{self._base_url}/courts/{court_id}/"
        try:
            return await self._get(url)
        except httpx.HTTPStatusError as exc:
            msg = f"CourtListener court {court_id} fetch failed: {exc.response.status_code}"
            raise IngestionError(msg) from exc
