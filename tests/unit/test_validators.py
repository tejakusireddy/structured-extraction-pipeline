"""Tests for the extraction validation pipeline.

Covers: JSON parsing, Pydantic schema validation, business rules
(confidence bounds, citation regex, disposition values, legal topics),
and the NEEDS_REVIEW threshold.
"""

from __future__ import annotations

import json

import pytest

from src.core.exceptions import ExtractionValidationError
from src.models.domain import ExtractionStatus
from src.services.extraction.validators import (
    determine_review_status,
    validate_extraction,
)


def _make_valid_payload(**overrides: object) -> dict[str, object]:
    """Build a minimal valid extraction payload with optional overrides."""
    base = {
        "holding": "The court held that the statute is constitutional.",
        "holding_confidence": 0.85,
        "legal_standard": "strict scrutiny",
        "disposition": "affirmed",
        "disposition_confidence": 0.9,
        "key_authorities": [
            {
                "citation_string": "554 U.S. 570 (2008)",
                "case_name": "District of Columbia v. Heller",
                "citation_context": (
                    "The court followed this precedent because it "
                    "establishes the individual right to bear arms."
                ),
                "citation_type": "followed",
                "paragraph_context": (
                    "As established in District of Columbia v. Heller, "
                    "554 U.S. 570 (2008), the Second Amendment confers "
                    "an individual right."
                ),
            }
        ],
        "dissent_present": False,
        "dissent_summary": None,
        "concurrence_present": False,
        "concurrence_summary": None,
        "legal_topics": ["constitutional law", "second amendment"],
        "extraction_model": "gpt-4o",
        "extraction_timestamp": "2025-01-15T12:00:00Z",
        "raw_prompt_tokens": 1500,
        "raw_completion_tokens": 800,
    }
    base.update(overrides)
    return base


class TestValidateExtraction:
    """Tests for the full validate_extraction pipeline."""

    def test_valid_json_passes(self):
        raw = json.dumps(_make_valid_payload())
        result = validate_extraction(raw)
        assert result.holding == "The court held that the statute is constitutional."
        assert result.holding_confidence == 0.85
        assert result.disposition.value == "affirmed"
        assert len(result.key_authorities) == 1
        assert len(result.legal_topics) == 2

    def test_invalid_json_string(self):
        with pytest.raises(ExtractionValidationError, match="Invalid JSON"):
            validate_extraction("not json at all {{{")

    def test_empty_string(self):
        with pytest.raises(ExtractionValidationError, match="Invalid JSON"):
            validate_extraction("")

    def test_json_array_instead_of_object(self):
        with pytest.raises(ExtractionValidationError, match="Expected JSON object"):
            validate_extraction("[1, 2, 3]")

    def test_missing_required_field_holding(self):
        payload = _make_valid_payload()
        del payload["holding"]
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_missing_required_field_disposition(self):
        payload = _make_valid_payload()
        del payload["disposition"]
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_missing_legal_topics(self):
        payload = _make_valid_payload()
        del payload["legal_topics"]
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_empty_legal_topics(self):
        payload = _make_valid_payload(legal_topics=[])
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_invalid_disposition_value(self):
        payload = _make_valid_payload(disposition="acquitted")
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_holding_confidence_too_high(self):
        payload = _make_valid_payload(holding_confidence=1.5)
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_holding_confidence_negative(self):
        payload = _make_valid_payload(holding_confidence=-0.1)
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_disposition_confidence_bounds(self):
        payload = _make_valid_payload(disposition_confidence=2.0)
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))

    def test_invalid_citation_string_triggers_business_rule(self):
        payload = _make_valid_payload(
            key_authorities=[
                {
                    "citation_string": "some random text not a citation",
                    "case_name": None,
                    "citation_context": "The court cited this authority.",
                    "citation_type": "cited",
                    "paragraph_context": "The court relied on some random text not a citation.",
                }
            ]
        )
        with pytest.raises(ExtractionValidationError, match="does not match legal citation regex"):
            validate_extraction(json.dumps(payload))

    def test_valid_citation_string_passes_business_rule(self):
        payload = _make_valid_payload(
            key_authorities=[
                {
                    "citation_string": "410 U.S. 113 (1973)",
                    "case_name": "Roe v. Wade",
                    "citation_context": "The court followed this landmark ruling.",
                    "citation_type": "followed",
                    "paragraph_context": "As decided in Roe v. Wade, 410 U.S. 113 (1973).",
                }
            ]
        )
        result = validate_extraction(json.dumps(payload))
        assert result.key_authorities[0].citation_string == "410 U.S. 113 (1973)"

    def test_empty_authorities_is_valid(self):
        payload = _make_valid_payload(key_authorities=[])
        result = validate_extraction(json.dumps(payload))
        assert result.key_authorities == []

    def test_multiple_authorities_all_valid(self):
        payload = _make_valid_payload(
            key_authorities=[
                {
                    "citation_string": "554 U.S. 570 (2008)",
                    "case_name": "Heller",
                    "citation_context": "Followed this precedent.",
                    "citation_type": "followed",
                    "paragraph_context": "See Heller, 554 U.S. 570 (2008).",
                },
                {
                    "citation_string": "505 U.S. 833 (1992)",
                    "case_name": "Casey",
                    "citation_context": "Distinguished on procedural grounds.",
                    "citation_type": "distinguished",
                    "paragraph_context": "Unlike in Casey, 505 U.S. 833 (1992).",
                },
            ]
        )
        result = validate_extraction(json.dumps(payload))
        assert len(result.key_authorities) == 2

    def test_mixed_valid_and_invalid_authorities(self):
        payload = _make_valid_payload(
            key_authorities=[
                {
                    "citation_string": "554 U.S. 570 (2008)",
                    "case_name": "Heller",
                    "citation_context": "Followed.",
                    "citation_type": "followed",
                    "paragraph_context": "See Heller.",
                },
                {
                    "citation_string": "not a citation",
                    "case_name": None,
                    "citation_context": "Cited.",
                    "citation_type": "cited",
                    "paragraph_context": "Referenced not a citation.",
                },
            ]
        )
        with pytest.raises(ExtractionValidationError, match="does not match legal citation regex"):
            validate_extraction(json.dumps(payload))

    def test_all_disposition_values_accepted(self):
        for disp in [
            "affirmed",
            "reversed",
            "remanded",
            "vacated",
            "dismissed",
            "affirmed_in_part",
            "reversed_in_part",
        ]:
            raw = json.dumps(_make_valid_payload(disposition=disp))
            result = validate_extraction(raw)
            assert result.disposition.value == disp

    def test_all_citation_types_accepted(self):
        for ct in ["followed", "distinguished", "overruled", "cited"]:
            auth = {
                "citation_string": "554 U.S. 570 (2008)",
                "case_name": "Heller",
                "citation_context": "Context.",
                "citation_type": ct,
                "paragraph_context": "Paragraph.",
            }
            raw = json.dumps(_make_valid_payload(key_authorities=[auth]))
            result = validate_extraction(raw)
            assert result.key_authorities[0].citation_type.value == ct

    def test_null_optional_fields_accepted(self):
        payload = _make_valid_payload(
            legal_standard=None,
            dissent_summary=None,
            concurrence_summary=None,
        )
        result = validate_extraction(json.dumps(payload))
        assert result.legal_standard is None
        assert result.dissent_summary is None

    def test_dissent_and_concurrence_present(self):
        payload = _make_valid_payload(
            dissent_present=True,
            dissent_summary="The dissent argued the statute is unconstitutional.",
            concurrence_present=True,
            concurrence_summary="Concurrence agreed on result but on narrower grounds.",
        )
        result = validate_extraction(json.dumps(payload))
        assert result.dissent_present is True
        assert result.concurrence_present is True
        assert result.dissent_summary is not None
        assert result.concurrence_summary is not None

    def test_negative_token_counts_rejected(self):
        payload = _make_valid_payload(raw_prompt_tokens=-10)
        with pytest.raises(ExtractionValidationError, match="Schema validation failed"):
            validate_extraction(json.dumps(payload))


class TestDetermineReviewStatus:
    """Tests for the confidence-based review status determination."""

    def test_high_confidence_completed(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.85))
        intel = validate_extraction(raw)
        assert determine_review_status(intel) == ExtractionStatus.COMPLETED

    def test_low_confidence_needs_review(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.2))
        intel = validate_extraction(raw)
        assert determine_review_status(intel) == ExtractionStatus.NEEDS_REVIEW

    def test_boundary_value_at_threshold(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.3))
        intel = validate_extraction(raw)
        assert determine_review_status(intel) == ExtractionStatus.COMPLETED

    def test_just_below_threshold(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.29))
        intel = validate_extraction(raw)
        assert determine_review_status(intel) == ExtractionStatus.NEEDS_REVIEW

    def test_zero_confidence_needs_review(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.0))
        intel = validate_extraction(raw)
        assert determine_review_status(intel) == ExtractionStatus.NEEDS_REVIEW

    def test_custom_threshold(self):
        raw = json.dumps(_make_valid_payload(holding_confidence=0.5))
        intel = validate_extraction(raw)
        assert determine_review_status(intel, threshold=0.6) == ExtractionStatus.NEEDS_REVIEW
        assert determine_review_status(intel, threshold=0.4) == ExtractionStatus.COMPLETED
