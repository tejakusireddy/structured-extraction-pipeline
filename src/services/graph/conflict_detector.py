"""Detect circuit splits between appellate opinions.

Finds pairs of opinions from DIFFERENT circuits that cite overlapping
authorities, where one FOLLOWS a precedent and the other DISTINGUISHES
it or reaches an opposite disposition. Uses legal_topics overlap to
cluster conflicts and scores confidence based on recency, citation
overlap, disposition alignment, and court level.
"""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import combinations
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import select

from src.models.database import CitationRow, ExtractionRow, OpinionRow
from src.models.domain import CitationType, Conflict, ConflictStatus, Disposition

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

_OPPOSING_DISPOSITIONS: set[tuple[str, str]] = {
    (Disposition.AFFIRMED, Disposition.REVERSED),
    (Disposition.REVERSED, Disposition.AFFIRMED),
    (Disposition.AFFIRMED, Disposition.VACATED),
    (Disposition.VACATED, Disposition.AFFIRMED),
    (Disposition.AFFIRMED, Disposition.REVERSED_IN_PART),
    (Disposition.REVERSED_IN_PART, Disposition.AFFIRMED),
}

_CONFLICTING_CITATION_PAIRS: set[tuple[str, str]] = {
    (CitationType.FOLLOWED, CitationType.DISTINGUISHED),
    (CitationType.DISTINGUISHED, CitationType.FOLLOWED),
    (CitationType.FOLLOWED, CitationType.OVERRULED),
    (CitationType.OVERRULED, CitationType.FOLLOWED),
}

_COURT_LEVEL_WEIGHT: dict[str, float] = {
    "supreme": 1.0,
    "appellate": 0.8,
    "state_supreme": 0.7,
    "state_appellate": 0.5,
    "district": 0.4,
    "state_trial": 0.3,
    "bankruptcy": 0.2,
    "specialized": 0.3,
}


class ConflictDetector:
    """Detects circuit splits by analyzing citation and disposition patterns."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def detect_conflicts(
        self,
        *,
        court_pairs: list[tuple[str, str]] | None = None,
        min_confidence: float = 0.5,
    ) -> list[Conflict]:
        """Find circuit splits across opinions in the database.

        Parameters
        ----------
        court_pairs:
            Optional list of (court_a, court_b) tuples to restrict the
            search. If None, all cross-circuit pairs are considered.
        min_confidence:
            Minimum confidence threshold for returned conflicts.
        """
        opinions = await self._load_opinions_with_extractions()
        if len(opinions) < 2:
            return []

        pair_set = {tuple(sorted(p)) for p in court_pairs} if court_pairs is not None else None

        conflicts: list[Conflict] = []

        for op_a, op_b in combinations(opinions, 2):
            if op_a["court_id"] == op_b["court_id"]:
                continue

            if pair_set is not None:
                key = tuple(sorted((str(op_a["court_id"]), str(op_b["court_id"]))))
                if key not in pair_set:
                    continue

            conflict = await self._evaluate_pair(op_a, op_b)
            if conflict is not None and conflict.confidence >= min_confidence:
                conflicts.append(conflict)

        conflicts.sort(key=lambda c: c.confidence, reverse=True)

        logger.info("conflicts_detected", count=len(conflicts))
        return conflicts

    async def _load_opinions_with_extractions(
        self,
    ) -> list[dict[str, Any]]:
        """Load all opinions that have completed extractions."""
        stmt = (
            select(
                OpinionRow.id,
                OpinionRow.court_id,
                OpinionRow.court_level,
                OpinionRow.case_name,
                OpinionRow.date_filed,
                ExtractionRow.holding,
                ExtractionRow.disposition,
                ExtractionRow.legal_topics,
                ExtractionRow.id.label("extraction_id"),
            )
            .join(ExtractionRow, ExtractionRow.opinion_id == OpinionRow.id)
            .where(ExtractionRow.status == "completed")
            .order_by(OpinionRow.date_filed.desc())
        )
        result = await self._session.execute(stmt)
        rows = result.all()

        opinions: list[dict[str, Any]] = []
        for row in rows:
            opinions.append(
                {
                    "opinion_id": row.id,
                    "court_id": row.court_id,
                    "court_level": row.court_level,
                    "case_name": row.case_name,
                    "date_filed": row.date_filed,
                    "holding": row.holding,
                    "disposition": row.disposition,
                    "legal_topics": row.legal_topics,
                    "extraction_id": row.extraction_id,
                }
            )
        return opinions

    async def _evaluate_pair(
        self,
        op_a: dict[str, Any],
        op_b: dict[str, Any],
    ) -> Conflict | None:
        """Evaluate whether two opinions represent a circuit split."""
        topics_a: list[str] = op_a["legal_topics"]
        topics_b: list[str] = op_b["legal_topics"]
        topic_overlap = set(topics_a) & set(topics_b)
        if not topic_overlap:
            return None

        disp_a: str = op_a["disposition"]
        disp_b: str = op_b["disposition"]
        has_opposing_disposition = (disp_a, disp_b) in _OPPOSING_DISPOSITIONS

        citation_overlap = await self._find_citation_overlap(
            int(op_a["opinion_id"]),
            int(op_b["opinion_id"]),
        )
        has_conflicting_citations = any(
            (ct_a, ct_b) in _CONFLICTING_CITATION_PAIRS for _, ct_a, ct_b in citation_overlap
        )

        if not has_opposing_disposition and not has_conflicting_citations:
            return None

        confidence = self._score_confidence(
            op_a=op_a,
            op_b=op_b,
            topic_overlap_count=len(topic_overlap),
            citation_overlap_count=len(citation_overlap),
            has_opposing_disposition=has_opposing_disposition,
            has_conflicting_citations=has_conflicting_citations,
        )

        shared_topic = ", ".join(sorted(topic_overlap)[:3])

        description = self._build_description(
            op_a, op_b, has_opposing_disposition, has_conflicting_citations
        )

        return Conflict(
            opinion_a_id=int(op_a["opinion_id"]),
            opinion_b_id=int(op_b["opinion_id"]),
            topic=shared_topic,
            court_a=str(op_a["court_id"]),
            court_b=str(op_b["court_id"]),
            description=description,
            confidence=round(confidence, 4),
            detected_at=datetime.now(UTC),
            status=ConflictStatus.DETECTED,
        )

    async def _find_citation_overlap(
        self,
        opinion_a_id: int,
        opinion_b_id: int,
    ) -> list[tuple[str, str, str]]:
        """Find shared cited authorities between two opinions.

        Returns list of (citation_string, type_a, type_b) tuples where
        both opinions cite the same authority.
        """
        stmt_a = select(
            CitationRow.citation_string,
            CitationRow.citation_type,
        ).where(CitationRow.citing_opinion_id == opinion_a_id)

        stmt_b = select(
            CitationRow.citation_string,
            CitationRow.citation_type,
        ).where(CitationRow.citing_opinion_id == opinion_b_id)

        result_a = await self._session.execute(stmt_a)
        result_b = await self._session.execute(stmt_b)

        cites_a = {row.citation_string: row.citation_type for row in result_a.all()}
        cites_b = {row.citation_string: row.citation_type for row in result_b.all()}

        overlap: list[tuple[str, str, str]] = []
        for cite_str in set(cites_a.keys()) & set(cites_b.keys()):
            overlap.append((cite_str, cites_a[cite_str], cites_b[cite_str]))

        return overlap

    def _score_confidence(
        self,
        *,
        op_a: dict[str, Any],
        op_b: dict[str, Any],
        topic_overlap_count: int,
        citation_overlap_count: int,
        has_opposing_disposition: bool,
        has_conflicting_citations: bool,
    ) -> float:
        """Score conflict confidence on [0, 1] based on multiple signals."""
        score = 0.0

        if has_opposing_disposition:
            score += 0.30
        if has_conflicting_citations:
            score += 0.25

        topic_score = min(topic_overlap_count / 5.0, 1.0) * 0.15
        score += topic_score

        cite_score = min(citation_overlap_count / 3.0, 1.0) * 0.15
        score += cite_score

        level_a = _COURT_LEVEL_WEIGHT.get(str(op_a["court_level"]), 0.3)
        level_b = _COURT_LEVEL_WEIGHT.get(str(op_b["court_level"]), 0.3)
        court_score = ((level_a + level_b) / 2.0) * 0.10
        score += court_score

        from datetime import date

        date_a = op_a["date_filed"]
        date_b = op_b["date_filed"]
        if isinstance(date_a, date) and isinstance(date_b, date):
            today = date.today()
            age_a = (today - date_a).days / 365.25
            age_b = (today - date_b).days / 365.25
            avg_age = (age_a + age_b) / 2.0
            recency = max(0.0, 1.0 - avg_age / 20.0) * 0.05
            score += recency

        return min(score, 1.0)

    def _build_description(
        self,
        op_a: dict[str, Any],
        op_b: dict[str, Any],
        has_opposing_disposition: bool,
        has_conflicting_citations: bool,
    ) -> str:
        """Build a human-readable description of the conflict."""
        parts: list[str] = []
        court_a = str(op_a["court_id"])
        court_b = str(op_b["court_id"])
        name_a = str(op_a["case_name"])
        name_b = str(op_b["case_name"])

        if has_opposing_disposition:
            parts.append(
                f"{court_a} ({name_a}) reached {op_a['disposition']} "
                f"while {court_b} ({name_b}) reached {op_b['disposition']}"
            )
        if has_conflicting_citations:
            parts.append(f"{court_a} and {court_b} treat shared authorities differently")

        return "; ".join(parts) if parts else "Potential circuit split detected"
