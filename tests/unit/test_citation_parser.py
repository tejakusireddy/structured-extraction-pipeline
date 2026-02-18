"""Tests for the legal citation regex parser.

Covers federal reporters, state reporters, pinpoint cites, year
extraction, malformed input, and multi-citation extraction.
"""

from src.utils.citation_parser import ParsedCitation, extract_citations, parse_citation

# ===========================================================================
# parse_citation — single citation extraction
# ===========================================================================


class TestParseCitation:
    def test_us_reports(self):
        result = parse_citation("554 U.S. 570 (2008)")
        assert result is not None
        assert result.volume == 554
        assert result.reporter == "U.S."
        assert result.page == 570
        assert result.year == 2008

    def test_supreme_court_reporter(self):
        result = parse_citation("128 S. Ct. 2783 (2008)")
        assert result is not None
        assert result.volume == 128
        assert result.reporter == "S. Ct."
        assert result.page == 2783
        assert result.year == 2008

    def test_lawyers_edition(self):
        result = parse_citation("171 L. Ed. 2d 637 (2008)")
        assert result is not None
        assert result.reporter == "L. Ed. 2d"

    def test_federal_reporter_3d(self):
        result = parse_citation("789 F.3d 1034 (9th Cir. 2015)")
        assert result is not None
        assert result.volume == 789
        assert result.reporter == "F.3d"
        assert result.page == 1034

    def test_federal_reporter_2d(self):
        result = parse_citation("456 F.2d 789")
        assert result is not None
        assert result.reporter == "F.2d"
        assert result.year is None

    def test_federal_reporter_4th(self):
        result = parse_citation("12 F.4th 345 (1st Cir. 2021)")
        assert result is not None
        assert result.reporter == "F.4th"

    def test_federal_supplement_3d(self):
        result = parse_citation("567 F. Supp. 3d 890 (S.D.N.Y. 2021)")
        assert result is not None
        assert result.reporter == "F. Supp. 3d"

    def test_federal_supplement_2d(self):
        result = parse_citation("432 F. Supp. 2d 111 (D. Mass. 2006)")
        assert result is not None
        assert result.reporter == "F. Supp. 2d"

    def test_bankruptcy_reporter(self):
        result = parse_citation("123 B.R. 456 (Bankr. S.D.N.Y. 2020)")
        assert result is not None
        assert result.reporter == "B.R."

    def test_pinpoint_cite(self):
        result = parse_citation("554 U.S. 570, 573 (2008)")
        assert result is not None
        assert result.page == 570
        assert result.pinpoint == 573
        assert result.year == 2008

    def test_without_year(self):
        result = parse_citation("789 F.3d 1034")
        assert result is not None
        assert result.year is None

    # --- State reporters ---

    def test_state_ny(self):
        result = parse_citation("100 N.Y.2d 200 (2003)")
        assert result is not None
        assert result.reporter == "N.Y.2d"
        assert result.year == 2003

    def test_state_cal(self):
        result = parse_citation("50 Cal.4th 616 (2010)")
        assert result is not None
        assert result.reporter == "Cal.4th"

    def test_atlantic_reporter(self):
        result = parse_citation("234 A.3d 567 (Pa. 2020)")
        assert result is not None
        assert result.reporter == "A.3d"

    def test_pacific_reporter(self):
        result = parse_citation("456 P.3d 789 (2019)")
        assert result is not None
        assert result.reporter == "P.3d"

    def test_northeastern_reporter(self):
        result = parse_citation("123 N.E.3d 456 (Ill. 2019)")
        assert result is not None
        assert result.reporter == "N.E.3d"

    def test_southern_reporter(self):
        result = parse_citation("789 So. 3d 123 (Fla. 2022)")
        assert result is not None
        assert result.reporter == "So. 3d"

    # --- Malformed / edge cases ---

    def test_empty_string(self):
        assert parse_citation("") is None

    def test_no_citation(self):
        assert parse_citation("This is just regular text.") is None

    def test_partial_citation_no_page(self):
        assert parse_citation("554 U.S.") is None

    def test_garbage_input(self):
        assert parse_citation("!@#$%^&*()") is None

    def test_citation_embedded_in_text(self):
        text = "The Court held in Heller, 554 U.S. 570 (2008), that..."
        result = parse_citation(text)
        assert result is not None
        assert result.volume == 554

    def test_raw_text_captured(self):
        result = parse_citation("554 U.S. 570 (2008)")
        assert result is not None
        assert "554" in result.raw_text
        assert "U.S." in result.raw_text
        assert "570" in result.raw_text


# ===========================================================================
# extract_citations — multi-citation extraction
# ===========================================================================


class TestExtractCitations:
    def test_multiple_in_paragraph(self):
        text = (
            "The Second Amendment protects an individual right, see "
            "District of Columbia v. Heller, 554 U.S. 570 (2008), "
            "and applies to the states through the Fourteenth Amendment, "
            "see McDonald v. City of Chicago, 561 U.S. 742 (2010)."
        )
        cites = extract_citations(text)
        assert len(cites) == 2
        volumes = {c.volume for c in cites}
        assert volumes == {554, 561}

    def test_deduplication(self):
        text = "See Heller, 554 U.S. 570 (2008). As noted in 554 U.S. 570, the Court held..."
        cites = extract_citations(text)
        assert len(cites) == 1

    def test_mixed_reporters(self):
        text = "See 554 U.S. 570 (2008); 789 F.3d 1034 (9th Cir. 2015); 234 A.3d 567 (Pa. 2020)."
        cites = extract_citations(text)
        assert len(cites) == 3
        reporters = {c.reporter for c in cites}
        assert reporters == {"U.S.", "F.3d", "A.3d"}

    def test_empty_string(self):
        assert extract_citations("") == []

    def test_no_citations_in_text(self):
        assert extract_citations("The defendant moved for summary judgment.") == []

    def test_many_citations(self):
        text = " ".join(f"{100 + i} U.S. {200 + i} ({2000 + i})" for i in range(10))
        cites = extract_citations(text)
        assert len(cites) == 10

    def test_returns_parsed_citation_objects(self):
        cites = extract_citations("554 U.S. 570 (2008)")
        assert len(cites) == 1
        assert isinstance(cites[0], ParsedCitation)
        assert cites[0].volume == 554
