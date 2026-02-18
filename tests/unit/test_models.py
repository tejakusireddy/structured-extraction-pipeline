"""Tests for domain models, enumerations, and request/response schemas.

Covers: enum completeness, valid/invalid construction, frozen immutability,
confidence score bounds, min_length constraints, and request validation.
"""

from datetime import date

import pytest
from pydantic import ValidationError

from src.models.domain import (
    CitationEdge,
    CitationType,
    CitedAuthority,
    ConflictStatus,
    CourtLevel,
    Disposition,
    ExtractedIntelligence,
    ExtractionStatus,
    JobStatus,
    Opinion,
    OpinionMetadata,
    PrecedentialStatus,
    Priority,
    SearchStrategy,
)
from src.models.requests import ExtractionRequest, IngestionRequest, SearchRequest
from tests.conftest import (
    make_cited_authority,
    make_conflict,
    make_document_chunk,
    make_extracted_intelligence,
    make_opinion_metadata,
)

# ===================================================================
# Enumerations
# ===================================================================


class TestEnumCompleteness:
    """Verify every enum has the expected members and string values."""

    def test_court_level_values(self):
        assert set(CourtLevel) == {
            CourtLevel.SUPREME,
            CourtLevel.APPELLATE,
            CourtLevel.DISTRICT,
            CourtLevel.STATE_SUPREME,
            CourtLevel.STATE_APPELLATE,
            CourtLevel.STATE_TRIAL,
            CourtLevel.BANKRUPTCY,
            CourtLevel.SPECIALIZED,
        }

    def test_precedential_status_values(self):
        assert set(PrecedentialStatus) == {
            PrecedentialStatus.PUBLISHED,
            PrecedentialStatus.UNPUBLISHED,
            PrecedentialStatus.ERRATA,
            PrecedentialStatus.SEPARATE_OPINION,
            PrecedentialStatus.IN_CHAMBERS,
            PrecedentialStatus.RELATING_TO,
            PrecedentialStatus.UNKNOWN,
        }

    def test_disposition_values(self):
        assert set(Disposition) == {
            Disposition.AFFIRMED,
            Disposition.REVERSED,
            Disposition.REMANDED,
            Disposition.VACATED,
            Disposition.DISMISSED,
            Disposition.AFFIRMED_IN_PART,
            Disposition.REVERSED_IN_PART,
        }

    def test_citation_type_values(self):
        assert set(CitationType) == {
            CitationType.FOLLOWED,
            CitationType.DISTINGUISHED,
            CitationType.OVERRULED,
            CitationType.CITED,
        }

    def test_extraction_status_values(self):
        assert set(ExtractionStatus) == {
            ExtractionStatus.PENDING,
            ExtractionStatus.COMPLETED,
            ExtractionStatus.FAILED,
            ExtractionStatus.NEEDS_REVIEW,
        }

    def test_job_status_values(self):
        assert set(JobStatus) == {
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        }

    def test_priority_values(self):
        assert set(Priority) == {Priority.HIGH, Priority.NORMAL, Priority.LOW}

    def test_conflict_status_values(self):
        assert set(ConflictStatus) == {
            ConflictStatus.DETECTED,
            ConflictStatus.CONFIRMED,
            ConflictStatus.DISMISSED,
        }

    def test_search_strategy_values(self):
        assert set(SearchStrategy) == {SearchStrategy.SIMILARITY, SearchStrategy.MMR}

    def test_strenum_serializes_to_string(self):
        """StrEnum members should serialize as their string value."""
        assert str(Disposition.AFFIRMED) == "affirmed"
        assert str(CourtLevel.SUPREME) == "supreme"
        assert str(CitationType.OVERRULED) == "overruled"


# ===================================================================
# CitedAuthority
# ===================================================================


class TestCitedAuthority:
    def test_valid_construction(self):
        auth = make_cited_authority()
        assert auth.citation_type == CitationType.FOLLOWED
        assert auth.case_name == "District of Columbia v. Heller"

    def test_case_name_optional(self):
        auth = make_cited_authority(case_name=None)
        assert auth.case_name is None

    def test_frozen_immutability(self):
        auth = make_cited_authority()
        with pytest.raises(ValidationError):
            auth.case_name = "Changed"  # type: ignore[misc]

    def test_empty_citation_string_rejected(self):
        with pytest.raises(ValidationError, match="citation_string"):
            make_cited_authority(citation_string="")

    def test_empty_citation_context_rejected(self):
        with pytest.raises(ValidationError, match="citation_context"):
            make_cited_authority(citation_context="")

    def test_empty_paragraph_context_rejected(self):
        with pytest.raises(ValidationError, match="paragraph_context"):
            make_cited_authority(paragraph_context="")

    def test_invalid_citation_type_rejected(self):
        with pytest.raises(ValidationError, match="citation_type"):
            make_cited_authority(citation_type="invented")

    def test_all_citation_types_accepted(self):
        for ct in CitationType:
            auth = make_cited_authority(citation_type=ct)
            assert auth.citation_type == ct

    def test_json_round_trip(self):
        auth = make_cited_authority()
        data = auth.model_dump()
        rebuilt = CitedAuthority(**data)
        assert rebuilt == auth


# ===================================================================
# OpinionMetadata
# ===================================================================


class TestOpinionMetadata:
    def test_valid_construction(self):
        meta = make_opinion_metadata()
        assert meta.court_level == CourtLevel.APPELLATE
        assert meta.date_filed == date(2023, 6, 15)

    def test_frozen_immutability(self):
        meta = make_opinion_metadata()
        with pytest.raises(ValidationError):
            meta.case_name = "Changed"  # type: ignore[misc]

    def test_citation_count_defaults_to_zero(self):
        meta = make_opinion_metadata(citation_count=0)
        assert meta.citation_count == 0

    def test_negative_citation_count_rejected(self):
        with pytest.raises(ValidationError, match="citation_count"):
            make_opinion_metadata(citation_count=-1)

    def test_empty_court_id_rejected(self):
        with pytest.raises(ValidationError, match="court_id"):
            make_opinion_metadata(court_id="")

    def test_empty_case_name_rejected(self):
        with pytest.raises(ValidationError, match="case_name"):
            make_opinion_metadata(case_name="")

    def test_empty_jurisdiction_rejected(self):
        with pytest.raises(ValidationError, match="jurisdiction"):
            make_opinion_metadata(jurisdiction="")

    def test_invalid_court_level_rejected(self):
        with pytest.raises(ValidationError, match="court_level"):
            make_opinion_metadata(court_level="galactic")

    def test_invalid_precedential_status_rejected(self):
        with pytest.raises(ValidationError, match="precedential_status"):
            make_opinion_metadata(precedential_status="classified")

    def test_all_court_levels_accepted(self):
        for level in CourtLevel:
            meta = make_opinion_metadata(court_level=level)
            assert meta.court_level == level

    def test_all_precedential_statuses_accepted(self):
        for status in PrecedentialStatus:
            meta = make_opinion_metadata(precedential_status=status)
            assert meta.precedential_status == status

    def test_json_round_trip(self):
        meta = make_opinion_metadata()
        data = meta.model_dump()
        rebuilt = OpinionMetadata(**data)
        assert rebuilt == meta


# ===================================================================
# ExtractedIntelligence
# ===================================================================


class TestExtractedIntelligence:
    def test_valid_construction(self):
        ext = make_extracted_intelligence()
        assert ext.holding_confidence == 0.92
        assert ext.disposition == Disposition.AFFIRMED
        assert len(ext.legal_topics) == 3

    def test_frozen_immutability(self):
        ext = make_extracted_intelligence()
        with pytest.raises(ValidationError):
            ext.holding = "Changed"  # type: ignore[misc]

    def test_with_key_authorities(self):
        auth = make_cited_authority()
        ext = make_extracted_intelligence(key_authorities=[auth])
        assert len(ext.key_authorities) == 1
        assert ext.key_authorities[0].citation_type == CitationType.FOLLOWED

    def test_with_dissent(self):
        ext = make_extracted_intelligence(
            dissent_present=True,
            dissent_summary="Harlan, J., dissenting: the majority overreaches...",
        )
        assert ext.dissent_present is True
        assert ext.dissent_summary is not None

    def test_with_concurrence(self):
        ext = make_extracted_intelligence(
            concurrence_present=True,
            concurrence_summary="Thomas, J., concurring: I agree but for different reasons.",
        )
        assert ext.concurrence_present is True

    # --- Confidence score bounds ---

    @pytest.mark.parametrize("field", ["holding_confidence", "disposition_confidence"])
    def test_confidence_at_zero(self, field: str) -> None:
        ext = make_extracted_intelligence(**{field: 0.0})
        assert getattr(ext, field) == 0.0

    @pytest.mark.parametrize("field", ["holding_confidence", "disposition_confidence"])
    def test_confidence_at_one(self, field: str) -> None:
        ext = make_extracted_intelligence(**{field: 1.0})
        assert getattr(ext, field) == 1.0

    @pytest.mark.parametrize("field", ["holding_confidence", "disposition_confidence"])
    def test_confidence_above_one_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            make_extracted_intelligence(**{field: 1.01})

    @pytest.mark.parametrize("field", ["holding_confidence", "disposition_confidence"])
    def test_confidence_below_zero_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match=field):
            make_extracted_intelligence(**{field: -0.01})

    # --- Required fields / min_length ---

    def test_empty_holding_rejected(self):
        with pytest.raises(ValidationError, match="holding"):
            make_extracted_intelligence(holding="")

    def test_empty_legal_topics_rejected(self):
        with pytest.raises(ValidationError, match="legal_topics"):
            make_extracted_intelligence(legal_topics=[])

    def test_empty_extraction_model_rejected(self):
        with pytest.raises(ValidationError, match="extraction_model"):
            make_extracted_intelligence(extraction_model="")

    def test_negative_prompt_tokens_rejected(self):
        with pytest.raises(ValidationError, match="raw_prompt_tokens"):
            make_extracted_intelligence(raw_prompt_tokens=-1)

    def test_negative_completion_tokens_rejected(self):
        with pytest.raises(ValidationError, match="raw_completion_tokens"):
            make_extracted_intelligence(raw_completion_tokens=-1)

    def test_invalid_disposition_rejected(self):
        with pytest.raises(ValidationError, match="disposition"):
            make_extracted_intelligence(disposition="overturned")

    def test_all_dispositions_accepted(self):
        for disp in Disposition:
            ext = make_extracted_intelligence(disposition=disp)
            assert ext.disposition == disp

    def test_json_round_trip(self):
        auth = make_cited_authority()
        ext = make_extracted_intelligence(key_authorities=[auth], dissent_present=True)
        data = ext.model_dump()
        rebuilt = ExtractedIntelligence(**data)
        assert rebuilt == ext


# ===================================================================
# Opinion (mutable wrapper)
# ===================================================================


class TestOpinion:
    def test_valid_without_extraction(self):
        meta = make_opinion_metadata()
        opinion = Opinion(metadata=meta, raw_text="Full opinion text here.")
        assert opinion.extraction is None

    def test_valid_with_extraction(self):
        meta = make_opinion_metadata()
        ext = make_extracted_intelligence()
        opinion = Opinion(metadata=meta, raw_text="Full opinion text.", extraction=ext)
        assert opinion.extraction is not None
        assert opinion.extraction.disposition == Disposition.AFFIRMED

    def test_empty_raw_text_rejected(self):
        meta = make_opinion_metadata()
        with pytest.raises(ValidationError, match="raw_text"):
            Opinion(metadata=meta, raw_text="")

    def test_opinion_is_mutable(self):
        meta = make_opinion_metadata()
        opinion = Opinion(metadata=meta, raw_text="Text here.")
        ext = make_extracted_intelligence()
        opinion.extraction = ext
        assert opinion.extraction.holding_confidence == 0.92


# ===================================================================
# CitationEdge
# ===================================================================


class TestCitationEdge:
    def test_valid_resolved_edge(self):
        edge = CitationEdge(
            citing_opinion_id=100,
            cited_opinion_id=200,
            citation_string="554 U.S. 570",
            citation_context="Followed as binding precedent",
            citation_type=CitationType.FOLLOWED,
            paragraph_context="As the Court held in 554 U.S. 570...",
        )
        assert edge.cited_opinion_id == 200

    def test_unresolved_edge(self):
        edge = CitationEdge(
            citing_opinion_id=100,
            cited_opinion_id=None,
            citation_string="999 F.3d 123",
            citation_context="Cited for general proposition",
            citation_type=CitationType.CITED,
            paragraph_context="See 999 F.3d 123 for background.",
        )
        assert edge.cited_opinion_id is None

    def test_frozen_immutability(self):
        edge = CitationEdge(
            citing_opinion_id=1,
            citation_string="test",
            citation_context="test",
            citation_type=CitationType.CITED,
            paragraph_context="test paragraph",
        )
        with pytest.raises(ValidationError):
            edge.citing_opinion_id = 999  # type: ignore[misc]


# ===================================================================
# Conflict
# ===================================================================


class TestConflict:
    def test_valid_construction(self):
        conflict = make_conflict()
        assert conflict.confidence == 0.78
        assert conflict.status == ConflictStatus.DETECTED

    def test_frozen_immutability(self):
        conflict = make_conflict()
        with pytest.raises(ValidationError):
            conflict.topic = "changed"  # type: ignore[misc]

    def test_confidence_at_bounds(self):
        assert make_conflict(confidence=0.0).confidence == 0.0
        assert make_conflict(confidence=1.0).confidence == 1.0

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError, match="confidence"):
            make_conflict(confidence=1.1)

    def test_confidence_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="confidence"):
            make_conflict(confidence=-0.1)

    def test_empty_topic_rejected(self):
        with pytest.raises(ValidationError, match="topic"):
            make_conflict(topic="")

    def test_default_status_is_detected(self):
        conflict = make_conflict()
        assert conflict.status == ConflictStatus.DETECTED

    def test_all_conflict_statuses_accepted(self):
        for status in ConflictStatus:
            conflict = make_conflict(status=status)
            assert conflict.status == status


# ===================================================================
# DocumentChunk
# ===================================================================


class TestDocumentChunk:
    def test_valid_construction(self):
        chunk = make_document_chunk()
        assert chunk.chunk_index == 0
        assert chunk.section_type == "majority"
        assert "554 U.S. 570" in chunk.citation_strings

    def test_frozen_immutability(self):
        chunk = make_document_chunk()
        with pytest.raises(ValidationError):
            chunk.text = "changed"  # type: ignore[misc]

    def test_negative_chunk_index_rejected(self):
        with pytest.raises(ValidationError, match="chunk_index"):
            make_document_chunk(chunk_index=-1)

    def test_negative_start_char_rejected(self):
        with pytest.raises(ValidationError, match="start_char"):
            make_document_chunk(start_char=-1)

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError, match="text"):
            make_document_chunk(text="")

    def test_optional_section_type(self):
        chunk = make_document_chunk(section_type=None)
        assert chunk.section_type is None

    def test_empty_citation_strings_default(self):
        chunk = make_document_chunk(citation_strings=[])
        assert chunk.citation_strings == []


# ===================================================================
# Request models
# ===================================================================


class TestExtractionRequest:
    def test_valid_request(self):
        req = ExtractionRequest(opinion_ids=[1, 2, 3])
        assert req.priority == Priority.NORMAL
        assert req.extraction_model is None

    def test_with_priority_and_model(self):
        req = ExtractionRequest(opinion_ids=[1], priority=Priority.HIGH, extraction_model="gpt-4o")
        assert req.priority == Priority.HIGH

    def test_empty_opinion_ids_rejected(self):
        with pytest.raises(ValidationError, match="opinion_ids"):
            ExtractionRequest(opinion_ids=[])

    def test_too_many_opinion_ids_rejected(self):
        with pytest.raises(ValidationError, match="opinion_ids"):
            ExtractionRequest(opinion_ids=list(range(1001)))

    def test_frozen_immutability(self):
        req = ExtractionRequest(opinion_ids=[1])
        with pytest.raises(ValidationError):
            req.priority = Priority.LOW  # type: ignore[misc]


class TestSearchRequest:
    def test_valid_defaults(self):
        req = SearchRequest(query="qualified immunity")
        assert req.k == 10
        assert req.strategy == SearchStrategy.SIMILARITY
        assert req.lambda_mult == 0.7

    def test_query_too_short_rejected(self):
        with pytest.raises(ValidationError, match="query"):
            SearchRequest(query="ab")

    def test_k_bounds(self):
        assert SearchRequest(query="test query", k=1).k == 1
        assert SearchRequest(query="test query", k=100).k == 100

    def test_k_too_high_rejected(self):
        with pytest.raises(ValidationError, match="k"):
            SearchRequest(query="test query", k=101)

    def test_k_too_low_rejected(self):
        with pytest.raises(ValidationError, match="k"):
            SearchRequest(query="test query", k=0)

    def test_lambda_mult_bounds(self):
        assert SearchRequest(query="test query", lambda_mult=0.0).lambda_mult == 0.0
        assert SearchRequest(query="test query", lambda_mult=1.0).lambda_mult == 1.0

    def test_lambda_mult_out_of_range_rejected(self):
        with pytest.raises(ValidationError, match="lambda_mult"):
            SearchRequest(query="test query", lambda_mult=1.5)


class TestIngestionRequest:
    def test_minimal_valid(self):
        req = IngestionRequest()
        assert req.max_opinions == 100

    def test_with_court_ids(self):
        req = IngestionRequest(court_ids=["ca9", "ca5"])
        assert req.court_ids == ["ca9", "ca5"]

    def test_with_date_range(self):
        req = IngestionRequest(
            date_after=date(2020, 1, 1),
            date_before=date(2024, 12, 31),
        )
        assert req.date_after == date(2020, 1, 1)

    def test_max_opinions_bounds(self):
        assert IngestionRequest(max_opinions=1).max_opinions == 1
        assert IngestionRequest(max_opinions=10000).max_opinions == 10000

    def test_max_opinions_too_high_rejected(self):
        with pytest.raises(ValidationError, match="max_opinions"):
            IngestionRequest(max_opinions=10001)

    def test_max_opinions_too_low_rejected(self):
        with pytest.raises(ValidationError, match="max_opinions"):
            IngestionRequest(max_opinions=0)
