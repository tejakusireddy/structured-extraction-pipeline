"""LLM provider abstraction layer.

Routes extraction requests to OpenAI or Anthropic based on model name.
Returns raw JSON strings and token usage for cost tracking. Retry logic
handles transient errors and rate limits with exponential backoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import LLMProviderError, RateLimitError

if TYPE_CHECKING:
    from src.core.config import Settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@dataclass(frozen=True)
class LLMResponse:
    """Raw response from an LLM provider."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLMClient:
    """Unified async client for OpenAI and Anthropic LLMs.

    Routes requests to the correct provider SDK based on the model
    name prefix. All calls return raw JSON strings — parsing and
    validation happen in the caller.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._openai_client: object | None = None
        self._anthropic_client: object | None = None

    def _get_openai(self) -> object:
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        return self._openai_client

    def _get_anthropic(self) -> object:
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(api_key=self._settings.anthropic_api_key)
        return self._anthropic_client

    @staticmethod
    def is_anthropic_model(model: str) -> bool:
        """Check whether a model name belongs to Anthropic."""
        return model.startswith("claude")

    @retry(
        retry=retry_if_exception_type((LLMProviderError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def extract(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send an extraction request and return the raw LLM response.

        Automatically routes to OpenAI or Anthropic based on model name.
        """
        if self.is_anthropic_model(model):
            return await self._call_anthropic(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return await self._call_openai(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _call_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        from openai import (
            APIConnectionError,
            APIStatusError,
            AsyncOpenAI,
        )
        from openai import (
            RateLimitError as OAIRateLimit,
        )

        client: AsyncOpenAI = self._get_openai()  # type: ignore[assignment]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except OAIRateLimit as exc:
            raise RateLimitError("OpenAI rate limit", retry_after=60.0) from exc
        except (APIStatusError, APIConnectionError) as exc:
            msg = f"OpenAI API error: {exc}"
            raise LLMProviderError(msg) from exc

        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        logger.debug(
            "openai_response",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=response.model,
        )

    async def _call_anthropic(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic
        from anthropic import RateLimitError as AnthropicRateLimit

        client: AsyncAnthropic = self._get_anthropic()  # type: ignore[assignment]
        try:
            response = await client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except AnthropicRateLimit as exc:
            raise RateLimitError("Anthropic rate limit", retry_after=60.0) from exc
        except (APIStatusError, APIConnectionError) as exc:
            msg = f"Anthropic API error: {exc}"
            raise LLMProviderError(msg) from exc

        from anthropic.types import TextBlock

        first = response.content[0] if response.content else None
        content = first.text if isinstance(first, TextBlock) else ""
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        logger.debug(
            "anthropic_response",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model,
        )
