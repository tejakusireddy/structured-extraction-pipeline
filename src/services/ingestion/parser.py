"""Parse CourtListener API v4 responses into domain models.

The CourtListener API returns opinions with multiple text representations
(plain_text, html_with_citations, html). This module picks the best
available source and cleans it for downstream processing.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import structlog

from src.core.exceptions import IngestionError
from src.models.domain import CourtLevel, OpinionMetadata, PrecedentialStatus
from src.utils.text_cleaning import clean_text

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

# CourtListener court_type → our CourtLevel mapping
_COURT_LEVEL_MAP: dict[str, CourtLevel] = {
    "F": CourtLevel.DISTRICT,
    "FB": CourtLevel.BANKRUPTCY,
    "FD": CourtLevel.DISTRICT,
    "FS": CourtLevel.SPECIALIZED,
    "FA": CourtLevel.APPELLATE,
    "FSC": CourtLevel.SUPREME,
    "S": CourtLevel.STATE_TRIAL,
    "SA": CourtLevel.STATE_APPELLATE,
    "SS": CourtLevel.STATE_SUPREME,
    "SAG": CourtLevel.SPECIALIZED,
    "C": CourtLevel.SPECIALIZED,
    "T": CourtLevel.SPECIALIZED,
    "I": CourtLevel.SPECIALIZED,
}

_STATUS_MAP: dict[str, PrecedentialStatus] = {
    "Published": PrecedentialStatus.PUBLISHED,
    "Unpublished": PrecedentialStatus.UNPUBLISHED,
    "Errata": PrecedentialStatus.ERRATA,
    "Separate": PrecedentialStatus.SEPARATE_OPINION,
    "In-chambers": PrecedentialStatus.IN_CHAMBERS,
    "Relating-to": PrecedentialStatus.RELATING_TO,
    "Unknown": PrecedentialStatus.UNKNOWN,
}


def _resolve_court_level(court_data: dict[str, Any]) -> CourtLevel:
    """Map CourtListener court metadata to our CourtLevel enum."""
    jurisdiction = court_data.get("jurisdiction", "")
    level = _COURT_LEVEL_MAP.get(jurisdiction, CourtLevel.SPECIALIZED)

    short_name = court_data.get("short_name", "").lower()
    if "supreme" in short_name and "state" not in jurisdiction.lower():
        level = CourtLevel.SUPREME
    elif "circuit" in short_name or "appeals" in short_name:
        level = CourtLevel.APPELLATE

    return level


def _resolve_status(status_str: str) -> PrecedentialStatus:
    """Map CourtListener precedential status strings to our enum."""
    return _STATUS_MAP.get(status_str, PrecedentialStatus.UNKNOWN)


def _parse_date(date_str: str | None) -> date:
    """Parse ISO date string from CourtListener. Raises on None or bad format."""
    if not date_str:
        msg = "Missing date_filed in opinion data"
        raise IngestionError(msg)
    try:
        return date.fromisoformat(date_str[:10])
    except ValueError as e:
        msg = f"Invalid date_filed format: {date_str!r}"
        raise IngestionError(msg) from e


def parse_opinion_response(
    opinion_data: dict[str, Any],
    cluster_data: dict[str, Any],
    court_data: dict[str, Any],
) -> OpinionMetadata:
    """Transform CourtListener API v4 response dicts into an OpinionMetadata.

    Parameters
    ----------
    opinion_data:
        The opinion object from /api/rest/v4/opinions/{id}/
    cluster_data:
        The parent cluster object from /api/rest/v4/clusters/{id}/
    court_data:
        The court object from /api/rest/v4/courts/{id}/

    Returns
    -------
    Validated OpinionMetadata domain model.
    """
    opinion_id = opinion_data.get("id")
    cluster_id = cluster_data.get("id")
    if opinion_id is None or cluster_id is None:
        msg = "Missing id in opinion or cluster data"
        raise IngestionError(msg)

    court_id = court_data.get("id", "unknown")
    court_name = court_data.get("full_name") or court_data.get("short_name", "Unknown Court")

    return OpinionMetadata(
        opinion_id=opinion_id,
        cluster_id=cluster_id,
        court_id=court_id,
        court_name=court_name,
        court_level=_resolve_court_level(court_data),
        case_name=cluster_data.get("case_name", "Unknown Case"),
        date_filed=_parse_date(cluster_data.get("date_filed")),
        precedential_status=_resolve_status(cluster_data.get("precedential_status", "Unknown")),
        citation_count=cluster_data.get("citation_count", 0) or 0,
        judges=cluster_data.get("judges", "") or "",
        jurisdiction=court_data.get("id", "unknown"),
        source_url=f"https://www.courtlistener.com/opinion/{opinion_id}/",
    )


def extract_best_text(opinion_data: dict[str, Any]) -> str:
    """Pick the best available text from a CourtListener opinion object.

    Priority: plain_text > html_with_citations > html > xml_harvard > ""

    All non-plain text formats are cleaned (HTML stripped, unicode
    normalized, whitespace collapsed).
    """
    for field in ("plain_text", "html_with_citations", "html", "xml_harvard"):
        raw = opinion_data.get(field)
        if raw and isinstance(raw, str) and raw.strip():
            if field == "plain_text":
                from src.utils.text_cleaning import normalize_unicode, normalize_whitespace

                return normalize_whitespace(normalize_unicode(raw))
            return clean_text(raw)

    logger.warning(
        "no_text_available",
        opinion_id=opinion_data.get("id"),
        fields_checked=["plain_text", "html_with_citations", "html", "xml_harvard"],
    )
    return ""
