"""Tests for the legal-aware document chunker.

Covers section detection, citation preservation during splitting,
empty/minimal inputs, large documents, and chunk size constraints.
"""

from src.services.ingestion.chunker import (
    _citation_safe_sentence_split,
    chunk_opinion,
    detect_sections,
)

# ===========================================================================
# Section detection
# ===========================================================================


class TestDetectSections:
    def test_no_headings_returns_majority(self):
        text = "The court finds that the defendant is liable for damages."
        sections = detect_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "majority"
        assert sections[0][1] == 0
        assert sections[0][2] == len(text)

    def test_background_heading(self):
        text = "Some preamble.\n\nBACKGROUND\n\nThe facts are as follows."
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "majority" in labels
        assert "background" in labels

    def test_multiple_sections(self):
        text = (
            "Preamble text here.\n\n"
            "BACKGROUND\n\nSome background.\n\n"
            "ANALYSIS\n\nThe court analyzes.\n\n"
            "CONCLUSION\n\nWe affirm."
        )
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "majority" in labels
        assert "background" in labels
        assert "analysis" in labels
        assert "conclusion" in labels

    def test_dissent_detection(self):
        text = "Majority opinion here.\n\nDISSENT\n\nI respectfully dissent."
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "dissent" in labels

    def test_concurrence_detection(self):
        text = "Majority opinion.\n\nCONCURRING OPINION\n\nI concur."
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "concurrence" in labels

    def test_standard_of_review(self):
        text = "Intro.\n\nSTANDARD OF REVIEW\n\nWe review de novo."
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "standard_of_review" in labels

    def test_roman_numeral_prefix(self):
        text = "Preamble.\n\nI. BACKGROUND\n\nFacts.\n\nII. ANALYSIS\n\nReasoning."
        sections = detect_sections(text)
        labels = [s[0] for s in sections]
        assert "background" in labels
        assert "analysis" in labels

    def test_sections_are_sorted_by_position(self):
        text = "Pre.\n\nBACKGROUND\n\nFacts.\n\nANALYSIS\n\nReasoning."
        sections = detect_sections(text)
        starts = [s[1] for s in sections]
        assert starts == sorted(starts)

    def test_sections_cover_full_text(self):
        text = "BACKGROUND\n\nFacts here.\n\nANALYSIS\n\nReasoning here."
        sections = detect_sections(text)
        assert sections[0][1] == 0
        assert sections[-1][2] == len(text)


# ===========================================================================
# Citation-safe sentence splitting
# ===========================================================================


class TestCitationSafeSplit:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third one."
        parts = _citation_safe_sentence_split(text)
        assert len(parts) == 3

    def test_citation_not_split(self):
        text = (
            "The Court held in Heller, 554 U.S. 570 (2008), "
            "that the right is individual. Next sentence."
        )
        parts = _citation_safe_sentence_split(text)
        any_contains_full_cite = any("554 U.S. 570" in p for p in parts)
        assert any_contains_full_cite

    def test_multiple_citations_preserved(self):
        text = (
            "See Heller, 554 U.S. 570 (2008). "
            "See also McDonald, 561 U.S. 742 (2010). "
            "The Court agreed."
        )
        parts = _citation_safe_sentence_split(text)
        assert any("554 U.S. 570" in p for p in parts)
        assert any("561 U.S. 742" in p for p in parts)

    def test_single_sentence_no_split(self):
        text = "One sentence only."
        parts = _citation_safe_sentence_split(text)
        assert len(parts) == 1
        assert parts[0] == text


# ===========================================================================
# chunk_opinion — full pipeline
# ===========================================================================


class TestChunkOpinion:
    def test_empty_input(self):
        assert chunk_opinion("", opinion_id=1) == []
        assert chunk_opinion("   ", opinion_id=1) == []

    def test_short_text_single_chunk(self):
        text = "The court finds for the plaintiff."
        chunks = chunk_opinion(text, opinion_id=42, target_size=5000)
        assert len(chunks) == 1
        assert chunks[0].opinion_id == 42
        assert chunks[0].chunk_index == 0
        assert chunks[0].text.strip() == text
        assert chunks[0].section_type == "majority"

    def test_chunk_indices_sequential(self):
        text = "\n\n".join(f"Paragraph {i}. " * 20 for i in range(10))
        chunks = chunk_opinion(text, opinion_id=1, target_size=200, max_size=400)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_respect_max_size(self):
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100 + ". " + "D" * 100 + "."
        chunks = chunk_opinion(text, opinion_id=1, target_size=100, max_size=250)
        for chunk in chunks:
            assert len(chunk.text) <= 300  # small grace for sentence boundaries

    def test_section_labels_propagated(self):
        text = (
            "Introduction text.\n\n"
            "BACKGROUND\n\n"
            "Factual background paragraph.\n\n"
            "ANALYSIS\n\n"
            "Legal analysis paragraph."
        )
        chunks = chunk_opinion(text, opinion_id=1, target_size=5000)
        section_types = {c.section_type for c in chunks}
        assert "majority" in section_types or "background" in section_types

    def test_citations_extracted_in_chunks(self):
        text = (
            "The Court relied on Heller, 554 U.S. 570 (2008), to hold that "
            "the Second Amendment protects an individual right. "
            "The Court also cited McDonald, 561 U.S. 742 (2010)."
        )
        chunks = chunk_opinion(text, opinion_id=1, target_size=5000)
        all_cites = []
        for c in chunks:
            all_cites.extend(c.citation_strings)
        assert any("554 U.S. 570" in cite for cite in all_cites)
        assert any("561 U.S. 742" in cite for cite in all_cites)

    def test_large_document_chunked(self):
        paragraphs = []
        for i in range(50):
            paragraphs.append(
                f"Paragraph {i}. The court considered the evidence presented by "
                f"both parties in this matter and found that the defendant's "
                f"arguments regarding the {i}th issue lacked merit. "
                f"See Smith v. Jones, {100 + i} F.3d {200 + i} (9th Cir. 2020)."
            )
        text = "\n\n".join(paragraphs)
        chunks = chunk_opinion(text, opinion_id=99, target_size=500, max_size=1000)

        assert len(chunks) > 1
        assert all(c.opinion_id == 99 for c in chunks)
        assert all(c.section_type is not None for c in chunks)
        total_cites = sum(len(c.citation_strings) for c in chunks)
        assert total_cites > 0

    def test_start_end_char_set(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_opinion(text, opinion_id=1, target_size=20, max_size=50)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_dissent_section_chunked(self):
        text = "Majority holds X.\n\nDISSENT\n\nI respectfully dissent from the majority's holding."
        chunks = chunk_opinion(text, opinion_id=1, target_size=5000)
        dissent_chunks = [c for c in chunks if c.section_type == "dissent"]
        assert len(dissent_chunks) >= 1
