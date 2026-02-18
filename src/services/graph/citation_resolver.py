"""Resolve extracted citation strings to opinion IDs in the database.

Given CitedAuthority objects from an extraction, this service:
1. Parses the citation string via citation_parser
2. Queries the opinions table by volume/reporter/page (exact match)
3. Falls back to fuzzy matching by case name + approximate date
4. Tracks unresolved citations for data quality reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import func, select

from src.models.database import CitationRow, OpinionRow
from src.models.domain import CitationEdge, CitedAuthority
from src.utils.citation_parser import parse_citation

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@dataclass
class ResolutionStats:
    """Aggregate stats from a bulk resolution run."""

    total: int = 0
    resolved: int = 0
    unresolved: int = 0
    unresolved_citations: list[str] = field(default_factory=list)


class CitationResolver:
    """Resolves citation strings to opinion IDs in our database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve_citations(
        self,
        authorities: list[CitedAuthority],
        citing_opinion_id: int,
    ) -> list[CitationEdge]:
        """Resolve a list of CitedAuthority to CitationEdge objects.

        For each authority, attempts exact match by parsed citation
        components, then fuzzy fallback by case name.
        """
        edges: list[CitationEdge] = []

        for auth in authorities:
            cited_id = await self._resolve_single(auth)

            edges.append(
                CitationEdge(
                    citing_opinion_id=citing_opinion_id,
                    cited_opinion_id=cited_id,
                    citation_string=auth.citation_string,
                    cited_case_name=auth.case_name,
                    citation_context=auth.citation_context,
                    citation_type=auth.citation_type,
                    paragraph_context=auth.paragraph_context,
                )
            )

        return edges

    async def bulk_resolve(
        self,
        extractions: list[dict[str, object]],
    ) -> ResolutionStats:
        """Resolve citations from multiple extractions and persist edges.

        Each dict must have: opinion_id (int), authorities (list[CitedAuthority]).
        Returns aggregate resolution statistics.
        """
        stats = ResolutionStats()

        for ext in extractions:
            opinion_id: int = ext["opinion_id"]  # type: ignore[assignment]
            authorities: list[CitedAuthority] = ext["authorities"]  # type: ignore[assignment]

            edges = await self.resolve_citations(authorities, opinion_id)

            for edge in edges:
                stats.total += 1
                if edge.cited_opinion_id is not None:
                    stats.resolved += 1
                else:
                    stats.unresolved += 1
                    stats.unresolved_citations.append(edge.citation_string)

                row = CitationRow(
                    citing_opinion_id=edge.citing_opinion_id,
                    cited_opinion_id=edge.cited_opinion_id,
                    citation_string=edge.citation_string,
                    cited_case_name=edge.cited_case_name,
                    citation_context=edge.citation_context,
                    citation_type=edge.citation_type.value,
                    paragraph_context=edge.paragraph_context,
                )
                self._session.add(row)

            await self._session.flush()

        return stats

    async def _resolve_single(self, auth: CitedAuthority) -> int | None:
        """Try to resolve a single CitedAuthority to an opinion ID."""
        parsed = parse_citation(auth.citation_string)

        if parsed is not None:
            opinion_id = await self._exact_match(parsed)
            if opinion_id is not None:
                return opinion_id

        if auth.case_name:
            opinion_id = await self._fuzzy_match(auth.case_name, auth)
            if opinion_id is not None:
                return opinion_id

        logger.debug(
            "citation_unresolved",
            citation=auth.citation_string,
            case_name=auth.case_name,
        )
        return None

    async def _exact_match(
        self,
        parsed: object,
    ) -> int | None:
        """Match by searching raw_text for volume + reporter + page pattern."""
        from src.utils.citation_parser import ParsedCitation

        if not isinstance(parsed, ParsedCitation):
            return None

        pattern = f"%{parsed.volume} % {parsed.page}%"

        stmt = select(OpinionRow.id).where(OpinionRow.raw_text.like(pattern)).limit(1)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return row

    async def _fuzzy_match(
        self,
        case_name: str,
        auth: CitedAuthority,
    ) -> int | None:
        """Fuzzy fallback: search by case name similarity."""
        search_term = f"%{case_name}%"

        stmt = (
            select(OpinionRow.id)
            .where(func.lower(OpinionRow.case_name).like(func.lower(search_term)))
            .order_by(OpinionRow.date_filed.desc())
            .limit(1)
        )

        parsed = parse_citation(auth.citation_string)
        if parsed is not None and parsed.year is not None:
            from datetime import date

            stmt = stmt.where(
                OpinionRow.date_filed.between(
                    date(parsed.year - 1, 1, 1),
                    date(parsed.year + 1, 12, 31),
                )
            )

        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            logger.debug(
                "citation_fuzzy_match",
                case_name=case_name,
                matched_id=row,
            )
        return row
