"""OpenAI embedding service with LRU caching and retry logic.

Wraps the text-embedding-3-large model to produce 1536-dim vectors
for semantic search. Batch embedding minimizes API round-trips;
an in-memory LRU cache avoids redundant calls for repeated queries.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.exceptions import LLMProviderError

if TYPE_CHECKING:
    from src.core.config import Settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

_CACHE_MAX_SIZE = 512


class EmbeddingService:
    """Async embedding client backed by OpenAI text-embedding-3-large."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.openai_api_key
        self._model = settings.openai_embedding_model
        self._dimensions = settings.embedding_dimensions
        self._client: object | None = None
        self._total_tokens = 0
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    @property
    def total_tokens(self) -> int:
        """Cumulative token usage across all embedding calls."""
        return self._total_tokens

    def _get_client(self) -> object:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _cache_get(self, text: str) -> list[float] | None:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        return None

    def _cache_put(self, text: str, vector: list[float]) -> None:
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > _CACHE_MAX_SIZE:
            self._cache.popitem(last=False)

    @retry(
        retry=retry_if_exception_type(LLMProviderError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=True,
    )
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single text string, returning a float vector.

        Results are cached in-memory so repeated queries are free.
        """
        cached = self._cache_get(text)
        if cached is not None:
            return cached

        vectors = await self._call_api([text])
        vec = vectors[0]
        self._cache_put(text, vec)
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []
        return await self._call_api(texts)

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API."""
        from openai import APIConnectionError, APIStatusError, AsyncOpenAI
        from openai import RateLimitError as OAIRateLimit

        client: AsyncOpenAI = self._get_client()  # type: ignore[assignment]
        try:
            response = await client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
        except OAIRateLimit as exc:
            raise LLMProviderError(f"OpenAI embedding rate limit: {exc}") from exc
        except (APIStatusError, APIConnectionError) as exc:
            raise LLMProviderError(f"OpenAI embedding error: {exc}") from exc

        self._total_tokens += response.usage.total_tokens

        logger.debug(
            "embedding_response",
            model=self._model,
            count=len(texts),
            tokens=response.usage.total_tokens,
        )

        return [item.embedding for item in response.data]
