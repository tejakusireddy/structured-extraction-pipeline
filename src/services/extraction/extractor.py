"""Extraction orchestrator.

Coordinates the full extraction cycle for a single opinion:
    build prompt → call LLM → validate → score confidence

On validation failure, retries with a corrective prompt (up to
max_retries). Tracks cumulative token usage across attempts for
cost calculation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from src.core.exceptions import ExtractionValidationError
from src.models.domain import ExtractionStatus
from src.services.extraction.prompts import build_corrective_prompt, build_extraction_prompt
from src.services.extraction.validators import determine_review_status, validate_extraction

if TYPE_CHECKING:
    from src.core.config import Settings
    from src.models.domain import ExtractedIntelligence, OpinionMetadata
    from src.services.extraction.llm_client import LLMClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


class ExtractionResult:
    """Wraps the extraction output with metadata about the attempt."""

    __slots__ = (
        "attempts",
        "error_message",
        "intelligence",
        "model",
        "status",
        "total_completion_tokens",
        "total_prompt_tokens",
    )

    def __init__(
        self,
        *,
        intelligence: ExtractedIntelligence | None,
        status: ExtractionStatus,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        attempts: int,
        model: str,
        error_message: str | None = None,
    ) -> None:
        self.intelligence = intelligence
        self.status = status
        self.total_prompt_tokens = total_prompt_tokens
        self.total_completion_tokens = total_completion_tokens
        self.attempts = attempts
        self.model = model
        self.error_message = error_message


class ExtractionService:
    """Orchestrates opinion extraction via LLM + validation loop."""

    def __init__(self, llm_client: LLMClient, settings: Settings) -> None:
        self._llm = llm_client
        self._settings = settings
        self._max_retries: int = settings.max_extraction_retries
        self._model: str = settings.default_extraction_model
        self._confidence_threshold: float = settings.confidence_review_threshold

    async def extract_opinion(
        self,
        metadata: OpinionMetadata,
        opinion_text: str,
    ) -> ExtractionResult:
        """Run the full extraction cycle for a single opinion.

        1. Build extraction prompt
        2. Call LLM
        3. Validate output
        4. On failure: build corrective prompt, retry (up to max_retries)
        5. Score confidence, determine review status
        """
        system_prompt, user_prompt = build_extraction_prompt(
            opinion_text=opinion_text,
            case_name=metadata.case_name,
            court=metadata.court_name,
        )

        total_prompt_tokens = 0
        total_completion_tokens = 0
        last_error: str | None = None
        last_raw_output: str = ""

        for attempt in range(1, self._max_retries + 2):  # +2: 1 initial + max_retries
            if attempt == 1:
                current_user_prompt = user_prompt
            else:
                current_user_prompt = build_corrective_prompt(
                    error_message=last_error or "Unknown error",
                    previous_output=last_raw_output,
                )

            try:
                llm_response = await self._llm.extract(
                    model=self._model,
                    system_prompt=system_prompt,
                    user_prompt=current_user_prompt,
                    temperature=0.0,
                )
            except Exception as exc:
                logger.error(
                    "llm_call_failed",
                    opinion_id=metadata.opinion_id,
                    attempt=attempt,
                    error=str(exc),
                )
                last_error = str(exc)
                last_raw_output = ""
                continue

            total_prompt_tokens += llm_response.prompt_tokens
            total_completion_tokens += llm_response.completion_tokens
            last_raw_output = llm_response.content

            try:
                intel = validate_extraction(llm_response.content)
            except ExtractionValidationError as exc:
                logger.warning(
                    "extraction_validation_failed",
                    opinion_id=metadata.opinion_id,
                    attempt=attempt,
                    error=exc.message,
                )
                last_error = exc.message
                continue

            intel = _patch_token_counts(
                intel,
                model=self._model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

            status = determine_review_status(intel, self._confidence_threshold)

            logger.info(
                "extraction_succeeded",
                opinion_id=metadata.opinion_id,
                attempts=attempt,
                status=status.value,
                holding_confidence=intel.holding_confidence,
            )

            return ExtractionResult(
                intelligence=intel,
                status=status,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                attempts=attempt,
                model=self._model,
            )

        logger.error(
            "extraction_exhausted_retries",
            opinion_id=metadata.opinion_id,
            max_retries=self._max_retries,
            last_error=last_error,
        )

        return ExtractionResult(
            intelligence=None,
            status=ExtractionStatus.NEEDS_REVIEW,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            attempts=self._max_retries + 1,
            model=self._model,
            error_message=last_error,
        )


def _patch_token_counts(
    intel: ExtractedIntelligence,
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> ExtractedIntelligence:
    """Return a copy of the extraction with updated token counts and metadata.

    ExtractedIntelligence is frozen, so we rebuild via model_copy.
    """
    return intel.model_copy(
        update={
            "extraction_model": model,
            "extraction_timestamp": datetime.now(UTC),
            "raw_prompt_tokens": prompt_tokens,
            "raw_completion_tokens": completion_tokens,
        },
    )
