"""Regex-based legal citation parser.

Extracts structured citation information (volume, reporter, page, year)
from both federal and state reporter formats. Not exhaustive — optimized
for the most commonly cited reporters.

Citation format: {volume} {reporter} {page}[, {pinpoint}] [({year})]
Example: "554 U.S. 570, 573 (2008)" → volume=554, reporter="U.S.", page=570, year=2008
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Federal reporters in regex-escaped form (ordered longest-first to
# prevent partial matches like "F." matching before "F.3d").
FEDERAL_REPORTERS: tuple[str, ...] = (
    r"F\.\s*Supp\.\s*3d",
    r"F\.\s*Supp\.\s*2d",
    r"F\.\s*Supp\.",
    r"F\.\s*App(?:'|')x",
    r"F\.4th",
    r"F\.3d",
    r"F\.2d",
    r"L\.\s*Ed\.\s*2d",
    r"S\.\s*Ct\.",
    r"U\.S\.",
    r"B\.R\.",
)

STATE_REPORTERS: tuple[str, ...] = (
    r"N\.Y\.S\.3d",
    r"N\.Y\.S\.2d",
    r"N\.Y\.3d",
    r"N\.Y\.2d",
    r"N\.Y\.",
    r"Cal\.\s*Rptr\.\s*3d",
    r"Cal\.\s*Rptr\.\s*2d",
    r"Cal\.\s*Rptr\.",
    r"Cal\.5th",
    r"Cal\.4th",
    r"Cal\.3d",
    r"Cal\.2d",
    r"Cal\.",
    r"Ill\.2d",
    r"Ill\.\s*App\.\s*3d",
    r"Ill\.\s*App\.\s*2d",
    r"Mass\.",
    r"Pa\.\s*Super\.",
    r"Pa\.",
    r"Ohio\s*St\.\s*3d",
    r"Ohio\s*St\.\s*2d",
    r"N\.J\.\s*Super\.",
    r"N\.J\.",
    r"A\.3d",
    r"A\.2d",
    r"N\.E\.3d",
    r"N\.E\.2d",
    r"N\.W\.2d",
    r"S\.E\.2d",
    r"S\.W\.3d",
    r"S\.W\.2d",
    r"P\.3d",
    r"P\.2d",
    r"So\.\s*3d",
    r"So\.\s*2d",
)

ALL_REPORTERS: tuple[str, ...] = FEDERAL_REPORTERS + STATE_REPORTERS

_REPORTER_ALTERNATIVES = "|".join(ALL_REPORTERS)

# Full citation pattern:
#   {volume} {reporter} {page}[, {pinpoint}] [({year})]
_CITATION_RE = re.compile(
    rf"""
    (?P<volume>\d{{1,4}})       # 1-4 digit volume number
    \s+
    (?P<reporter>{_REPORTER_ALTERNATIVES})
    \s+
    (?P<page>\d{{1,5}})         # 1-5 digit starting page
    (?:                          # optional pinpoint cite
        [,\s]+
        (?P<pinpoint>\d{{1,5}})
    )?
    (?:                          # optional year in parentheses
        \s*
        \(
        (?P<year>\d{{4}})
        \)
    )?
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class ParsedCitation:
    """Structured representation of a legal citation."""

    volume: int
    reporter: str
    page: int
    pinpoint: int | None = None
    year: int | None = None
    raw_text: str = ""


def _normalize_reporter(reporter: str) -> str:
    """Collapse internal whitespace in reporter abbreviation."""
    return re.sub(r"\s+", " ", reporter).strip()


def parse_citation(text: str) -> ParsedCitation | None:
    """Parse a single citation string. Returns None if no match found."""
    match = _CITATION_RE.search(text)
    if match is None:
        return None
    return ParsedCitation(
        volume=int(match.group("volume")),
        reporter=_normalize_reporter(match.group("reporter")),
        page=int(match.group("page")),
        pinpoint=int(match.group("pinpoint")) if match.group("pinpoint") else None,
        year=int(match.group("year")) if match.group("year") else None,
        raw_text=match.group(0).strip(),
    )


def extract_citations(text: str) -> list[ParsedCitation]:
    """Find all legal citations in a block of text."""
    results: list[ParsedCitation] = []
    seen: set[tuple[int, str, int]] = set()

    for match in _CITATION_RE.finditer(text):
        reporter = _normalize_reporter(match.group("reporter"))
        key = (int(match.group("volume")), reporter, int(match.group("page")))
        if key in seen:
            continue
        seen.add(key)

        results.append(
            ParsedCitation(
                volume=key[0],
                reporter=reporter,
                page=key[2],
                pinpoint=int(match.group("pinpoint")) if match.group("pinpoint") else None,
                year=int(match.group("year")) if match.group("year") else None,
                raw_text=match.group(0).strip(),
            )
        )
    return results


# Pre-compiled pattern for protecting citations during sentence splitting.
# Matches a citation string so the splitter won't break on internal periods.
CITATION_PROTECT_RE = re.compile(
    rf"""
    \d{{1,4}}
    \s+
    (?:{_REPORTER_ALTERNATIVES})
    \s+
    \d{{1,5}}
    """,
    re.VERBOSE,
)
