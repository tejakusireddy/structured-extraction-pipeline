"""Tests for extraction prompt templates.

Verifies that generated prompts contain required schema fields,
instructions, and negative examples needed for reliable LLM extraction.
"""

from __future__ import annotations

from src.services.extraction.prompts import (
    EXTRACTION_SCHEMA_DESCRIPTION,
    NEGATIVE_EXAMPLES,
    build_corrective_prompt,
    build_extraction_prompt,
)


class TestBuildExtractionPrompt:
    """Tests for build_extraction_prompt."""

    def test_returns_tuple_of_two_strings(self):
        system, user = build_extraction_prompt(
            opinion_text="Some opinion text.",
            case_name="Smith v. Jones",
            court="Supreme Court of the United States",
        )
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_system_prompt_contains_schema_fields(self):
        system, _ = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        required_fields = [
            "holding",
            "holding_confidence",
            "legal_standard",
            "disposition",
            "disposition_confidence",
            "key_authorities",
            "citation_string",
            "case_name",
            "citation_context",
            "citation_type",
            "paragraph_context",
            "dissent_present",
            "dissent_summary",
            "concurrence_present",
            "concurrence_summary",
            "legal_topics",
        ]
        for field in required_fields:
            assert field in system, f"System prompt missing field: {field}"

    def test_system_prompt_contains_negative_examples(self):
        system, _ = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        assert "holding_confidence" in system and "0.0" in system
        assert "no clear holding" in system.lower() or "No clear holding" in system

    def test_system_prompt_mentions_disposition_values(self):
        system, _ = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        for disp in ["affirmed", "reversed", "remanded", "vacated", "dismissed"]:
            assert disp in system, f"System prompt missing disposition value: {disp}"

    def test_system_prompt_mentions_citation_types(self):
        system, _ = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        for ct in ["followed", "distinguished", "overruled", "cited"]:
            assert ct in system, f"System prompt missing citation_type: {ct}"

    def test_system_prompt_demands_json_only(self):
        system, _ = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        assert "json" in system.lower()
        assert "no markdown" in system.lower() or "no explanation" in system.lower()

    def test_user_prompt_contains_opinion_text(self):
        _, user = build_extraction_prompt(
            opinion_text="The defendant was found guilty of fraud.",
            case_name="USA v. Doe",
            court="Ninth Circuit",
        )
        assert "The defendant was found guilty of fraud." in user

    def test_user_prompt_contains_case_metadata(self):
        _, user = build_extraction_prompt(
            opinion_text="text",
            case_name="Miranda v. Arizona",
            court="Supreme Court of the United States",
        )
        assert "Miranda v. Arizona" in user
        assert "Supreme Court of the United States" in user

    def test_user_prompt_has_begin_end_markers(self):
        _, user = build_extraction_prompt(
            opinion_text="text",
            case_name="case",
            court="court",
        )
        assert "BEGIN OPINION TEXT" in user
        assert "END OPINION TEXT" in user

    def test_confidence_guidance_in_negative_examples(self):
        has_guidance = (
            "0.4 is better than" in NEGATIVE_EXAMPLES
            or "err on the side of LOWER" in NEGATIVE_EXAMPLES
        )
        assert has_guidance

    def test_schema_mentions_confidence_bounds(self):
        assert "0.0" in EXTRACTION_SCHEMA_DESCRIPTION
        assert "1.0" in EXTRACTION_SCHEMA_DESCRIPTION


class TestBuildCorrectivePrompt:
    """Tests for the corrective retry prompt."""

    def test_corrective_prompt_includes_error(self):
        result = build_corrective_prompt(
            error_message="Missing field: holding",
            previous_output='{"disposition": "affirmed"}',
        )
        assert "Missing field: holding" in result

    def test_corrective_prompt_includes_previous_output(self):
        result = build_corrective_prompt(
            error_message="error",
            previous_output='{"incomplete": true}',
        )
        assert '{"incomplete": true}' in result

    def test_corrective_prompt_mentions_schema_rules(self):
        result = build_corrective_prompt(
            error_message="error",
            previous_output="{}",
        )
        assert "confidence" in result.lower()
        assert "legal_topics" in result
        assert "disposition" in result

    def test_corrective_prompt_demands_json(self):
        result = build_corrective_prompt(
            error_message="error",
            previous_output="{}",
        )
        assert "json" in result.lower()
