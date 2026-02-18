"""Legal-aware document chunking for court opinions.

Standard text splitters destroy the structure that matters for legal
analysis. This chunker:

1. Detects opinion sections (Background, Analysis, Holding, Dissent, etc.)
2. Splits within sections at paragraph boundaries
3. Protects legal citation patterns from being broken across chunks
4. Produces DocumentChunk objects with tracked positions and section labels
"""

from __future__ import annotations

import re

from src.models.domain import DocumentChunk
from src.utils.citation_parser import CITATION_PROTECT_RE, extract_citations

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "syllabus",
        re.compile(r"^(?:SYLLABUS|Syllabus)\s*$", re.MULTILINE),
    ),
    (
        "background",
        re.compile(
            r"^(?:I+\.?\s+)?(?:BACKGROUND|FACTUAL\s+BACKGROUND|PROCEDURAL\s+(?:BACKGROUND|HISTORY))\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "standard_of_review",
        re.compile(
            r"^(?:I+\.?\s+)?STANDARD\s+OF\s+REVIEW\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "analysis",
        re.compile(
            r"^(?:I+\.?\s+)?(?:ANALYSIS|LEGAL\s+ANALYSIS|DISCUSSION)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "holding",
        re.compile(
            r"^(?:I+\.?\s+)?(?:HOLDING|ORDER|JUDGMENT)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "dissent",
        re.compile(
            r"^(?:(?:I+\.?\s+)?DISSENT(?:ING)?(?:\s+OPINION)?|.{0,40},?\s+dissenting[.:]?)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "concurrence",
        re.compile(
            r"^(?:(?:I+\.?\s+)?CONCURR(?:ENCE|ING)(?:\s+OPINION)?|.{0,40},?\s+concurring[.:]?)\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^(?:I+\.?\s+)?CONCLUSION\s*$",
            re.MULTILINE | re.IGNORECASE,
        ),
    ),
]


def detect_sections(text: str) -> list[tuple[str, int, int]]:
    """Identify opinion sections and their character spans.

    Returns a list of (section_type, start, end) sorted by position.
    Text before the first detected heading is labeled "majority".
    """
    boundaries: list[tuple[int, str]] = []

    for section_name, pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(text):
            boundaries.append((match.start(), section_name))

    boundaries.sort(key=lambda b: b[0])

    sections: list[tuple[str, int, int]] = []
    if not boundaries:
        return [("majority", 0, len(text))]

    if boundaries[0][0] > 0:
        sections.append(("majority", 0, boundaries[0][0]))

    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections.append((label, start, end))

    return sections


# ---------------------------------------------------------------------------
# Sentence splitting with citation protection
# ---------------------------------------------------------------------------

_SENTENCE_END_RE = re.compile(
    r"""
    (?<=\w[.!?])   # lookbehind: word char + sentence-ending punct
    \s+             # whitespace between sentences
    (?=[A-Z"\(])    # lookahead: next sentence starts with uppercase, quote, or paren
    """,
    re.VERBOSE,
)


def _citation_safe_sentence_split(text: str) -> list[str]:
    """Split text into sentences while protecting legal citations.

    Citations like '554 U.S. 570 (2008)' contain periods that would
    fool a naive splitter. We mask them before splitting and restore after.
    """
    placeholder_map: dict[str, str] = {}

    def _mask(match: re.Match[str]) -> str:
        token = f"\x00CITE{len(placeholder_map)}\x00"
        placeholder_map[token] = match.group(0)
        return token

    masked = CITATION_PROTECT_RE.sub(_mask, text)

    sentences = _SENTENCE_END_RE.split(masked)

    return [_restore_placeholders(s, placeholder_map).strip() for s in sentences if s.strip()]


def _restore_placeholders(text: str, mapping: dict[str, str]) -> str:
    """Replace citation placeholders back with originals."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


# ---------------------------------------------------------------------------
# Chunking engine
# ---------------------------------------------------------------------------


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on double-newline boundaries, filtering empties."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _merge_small_paragraphs(
    paragraphs: list[str],
    target_size: int,
) -> list[str]:
    """Merge consecutive paragraphs until approaching target_size."""
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer = paragraphs[0]

    for para in paragraphs[1:]:
        combined_len = len(buffer) + 1 + len(para)
        if combined_len <= target_size:
            buffer = buffer + "\n\n" + para
        else:
            merged.append(buffer)
            buffer = para

    merged.append(buffer)
    return merged


def _force_split(text: str, max_size: int) -> list[str]:
    """Split oversized text at sentence boundaries to stay under max_size."""
    sentences = _citation_safe_sentence_split(text)
    chunks: list[str] = []
    buffer = ""

    for sent in sentences:
        if not buffer:
            buffer = sent
            continue

        if len(buffer) + 1 + len(sent) <= max_size:
            buffer = buffer + " " + sent
        else:
            chunks.append(buffer)
            buffer = sent

    if buffer:
        chunks.append(buffer)

    return chunks


def chunk_opinion(
    text: str,
    opinion_id: int,
    *,
    target_size: int = 2000,
    max_size: int = 3000,
) -> list[DocumentChunk]:
    """Chunk a court opinion into DocumentChunk objects.

    Algorithm:
    1. Detect sections (Background, Analysis, Dissent, etc.)
    2. Within each section, split into paragraphs
    3. Merge small paragraphs toward target_size
    4. Force-split any chunk still above max_size at sentence boundaries
    5. Extract citation strings found in each chunk

    Parameters
    ----------
    text:
        Full opinion text (already cleaned).
    opinion_id:
        ID of the parent opinion (embedded in each chunk).
    target_size:
        Ideal character count per chunk.
    max_size:
        Hard upper bound; chunks exceeding this are force-split.
    """
    if not text or not text.strip():
        return []

    sections = detect_sections(text)
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section_type, sec_start, sec_end in sections:
        section_text = text[sec_start:sec_end].strip()
        if not section_text:
            continue

        paragraphs = _split_into_paragraphs(section_text)
        merged = _merge_small_paragraphs(paragraphs, target_size)

        for block in merged:
            sub_blocks = _force_split(block, max_size) if len(block) > max_size else [block]

            for sub in sub_blocks:
                start_char = text.find(sub[:80], sec_start)
                if start_char == -1:
                    start_char = sec_start

                cites = extract_citations(sub)
                citation_strings = [c.raw_text for c in cites]

                chunks.append(
                    DocumentChunk(
                        opinion_id=opinion_id,
                        chunk_index=chunk_index,
                        text=sub,
                        start_char=start_char,
                        end_char=start_char + len(sub),
                        section_type=section_type,
                        citation_strings=citation_strings,
                    )
                )
                chunk_index += 1

    return chunks
