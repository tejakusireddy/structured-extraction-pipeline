"""Extraction output validation pipeline.

Every LLM response flows through:
    JSON parse → Pydantic validation → business rules → confidence filter

Business rules catch semantic issues that Pydantic's type system cannot
(e.g., citation format plausibility, confidence bounds, date consistency).
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from src.core.exceptions import ExtractionValidationError
from src.models.domain import (
    CitationType,
    Disposition,
    ExtractedIntelligence,
    ExtractionStatus,
)
from src.utils.citation_parser import parse_citation

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

VALID_DISPOSITIONS = {d.value for d in Disposition}
VALID_CITATION_TYPES = {ct.value for ct in CitationType}


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse raw JSON string, raising ExtractionValidationError on failure."""
    try:
        data: dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ExtractionValidationError(
            f"Invalid JSON: {exc}",
            details={"raw_output": raw[:500]},
        ) from exc

    if not isinstance(data, dict):
        raise ExtractionValidationError(
            f"Expected JSON object, got {type(data).__name__}",
            details={"raw_output": raw[:500]},
        )
    return data


def _validate_pydantic(data: dict[str, Any]) -> ExtractedIntelligence:
    """Validate data against the ExtractedIntelligence Pydantic model."""
    try:
        return ExtractedIntelligence.model_validate(data)
    except Exception as exc:
        raise ExtractionValidationError(
            f"Schema validation failed: {exc}",
            details={"fields": str(exc)},
        ) from exc


def _check_business_rules(intel: ExtractedIntelligence) -> list[str]:
    """Apply domain-specific business rules. Returns list of violation messages."""
    violations: list[str] = []

    if not (0.0 <= intel.holding_confidence <= 1.0):
        violations.append(f"holding_confidence {intel.holding_confidence} not in [0.0, 1.0]")

    if not (0.0 <= intel.disposition_confidence <= 1.0):
        violations.append(
            f"disposition_confidence {intel.disposition_confidence} not in [0.0, 1.0]"
        )

    if intel.disposition.value not in VALID_DISPOSITIONS:
        violations.append(f"Invalid disposition: {intel.disposition}")

    if not intel.legal_topics:
        violations.append("legal_topics must contain at least one entry")

    for i, auth in enumerate(intel.key_authorities):
        if auth.citation_type.value not in VALID_CITATION_TYPES:
            violations.append(f"key_authorities[{i}]: invalid citation_type '{auth.citation_type}'")

        parsed = parse_citation(auth.citation_string)
        if parsed is None:
            violations.append(
                f"key_authorities[{i}]: citation_string '{auth.citation_string}' "
                "does not match legal citation regex"
            )

    return violations


def determine_review_status(
    intel: ExtractedIntelligence,
    threshold: float = 0.3,
) -> ExtractionStatus:
    """Decide if the extraction needs human review based on confidence."""
    if intel.holding_confidence < threshold:
        return ExtractionStatus.NEEDS_REVIEW
    return ExtractionStatus.COMPLETED


def validate_extraction(raw_json: str) -> ExtractedIntelligence:
    """Full validation pipeline: JSON → Pydantic → business rules.

    Raises ExtractionValidationError on any failure.
    Returns a validated ExtractedIntelligence on success.
    """
    data = _parse_json(raw_json)
    intel = _validate_pydantic(data)
    violations = _check_business_rules(intel)

    if violations:
        raise ExtractionValidationError(
            f"Business rule violations: {'; '.join(violations)}",
            details={"violations": violations},
        )

    logger.debug(
        "extraction_validated",
        holding_confidence=intel.holding_confidence,
        disposition=intel.disposition.value,
        num_authorities=len(intel.key_authorities),
        num_topics=len(intel.legal_topics),
    )

    return intel
