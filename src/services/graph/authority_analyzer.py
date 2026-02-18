"""Analyze citation authority subgraphs.

For a given citation string or legal topic, builds the citation subgraph
using CitationRepo, ranks authorities by citation count + court level +
recency, and identifies anchor cases (most-cited, highest court).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import cast, func, select
from sqlalchemy.dialects.postgresql import JSONB

from src.models.database import CitationRow, ExtractionRow, OpinionRow
from src.models.domain import CitationType
from src.models.responses import AuthorityEdge, AuthorityGraphResponse, AuthorityNode

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

_COURT_LEVEL_RANK: dict[str, int] = {
    "supreme": 100,
    "appellate": 80,
    "state_supreme": 70,
    "state_appellate": 50,
    "district": 40,
    "state_trial": 30,
    "bankruptcy": 20,
    "specialized": 25,
}


class AuthorityAnalyzer:
    """Builds and ranks citation authority subgraphs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def analyze_authority(
        self,
        topic_or_citation: str,
        *,
        depth: int = 2,
    ) -> AuthorityGraphResponse:
        """Build an authority graph rooted at a citation or topic.

        If the input looks like a citation (contains digits and a
        reporter abbreviation), resolve it to an opinion. Otherwise,
        search by legal topic.
        """
        root_id = await self._find_root_opinion(topic_or_citation)
        if root_id is None:
            return AuthorityGraphResponse(
                anchor=AuthorityNode(
                    opinion_id=0,
                    case_name="Not found",
                    citation_string=topic_or_citation,
                    court="unknown",
                    date_filed="",
                    citation_count=0,
                ),
                nodes=[],
                edges=[],
            )

        subgraph = await self._build_subgraph(root_id, depth=depth)
        edges = await self._build_edges(root_id, subgraph)
        nodes = await self._build_nodes(subgraph)

        nodes.sort(
            key=lambda n: self._authority_score(n),
            reverse=True,
        )

        anchor = await self._build_single_node(root_id)

        return AuthorityGraphResponse(
            anchor=anchor,
            nodes=nodes,
            edges=edges,
        )

    async def _find_root_opinion(self, query: str) -> int | None:
        """Find the root opinion for the authority graph."""
        from src.utils.citation_parser import parse_citation

        parsed = parse_citation(query)
        if parsed is not None:
            stmt = (
                select(OpinionRow.id)
                .where(OpinionRow.raw_text.like(f"%{parsed.volume}%{parsed.page}%"))
                .limit(1)
            )
            result = await self._session.execute(stmt)
            opinion_id = result.scalar_one_or_none()
            if opinion_id is not None:
                return opinion_id

        stmt = (
            select(ExtractionRow.opinion_id)
            .where(ExtractionRow.legal_topics.op("@>")(cast(f'["{query}"]', JSONB)))
            .where(ExtractionRow.status == "completed")
            .order_by(ExtractionRow.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def _build_subgraph(
        self,
        root_id: int,
        *,
        depth: int,
    ) -> list[dict[str, Any]]:
        """Build citation subgraph by merging outgoing + incoming traversals.

        PostgreSQL disallows multiple recursive references to the same
        CTE, so we run two separate queries and merge the results.
        """
        from src.db.repositories.citation_repo import CitationRepo

        repo = CitationRepo(self._session)
        outgoing = await repo.get_citation_subgraph(root_id, max_depth=depth, direction="outgoing")
        incoming = await repo.get_citation_subgraph(root_id, max_depth=depth, direction="incoming")

        seen: set[int] = set()
        merged: list[dict[str, Any]] = []
        all_nodes: list[dict[str, Any]] = [*outgoing, *incoming]
        for node in all_nodes:
            nid = int(node["opinion_id"])
            if nid not in seen:
                seen.add(nid)
                merged.append(node)
        return merged

    async def _build_edges(
        self,
        root_id: int,
        subgraph: list[dict[str, Any]],
    ) -> list[AuthorityEdge]:
        """Get citation edges between nodes in the subgraph."""
        node_ids = {int(n["opinion_id"]) for n in subgraph}
        if not node_ids:
            return []

        stmt = select(CitationRow).where(
            CitationRow.citing_opinion_id.in_(node_ids),
            CitationRow.cited_opinion_id.in_(node_ids),
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()

        edges: list[AuthorityEdge] = []
        for row in rows:
            edges.append(
                AuthorityEdge(
                    source_id=row.citing_opinion_id,
                    target_id=row.cited_opinion_id,  # type: ignore[arg-type]
                    citation_type=CitationType(row.citation_type),
                    context=row.citation_context,
                )
            )
        return edges

    async def _build_nodes(
        self,
        subgraph: list[dict[str, Any]],
    ) -> list[AuthorityNode]:
        """Build AuthorityNode objects for all nodes in the subgraph."""
        node_ids = [int(n["opinion_id"]) for n in subgraph]
        if not node_ids:
            return []

        nodes: list[AuthorityNode] = []
        for nid in node_ids:
            node = await self._build_single_node(nid)
            nodes.append(node)
        return nodes

    async def _build_single_node(self, opinion_id: int) -> AuthorityNode:
        """Build a single AuthorityNode with citation count."""
        stmt = select(OpinionRow).where(OpinionRow.id == opinion_id)
        result = await self._session.execute(stmt)
        opinion = result.scalar_one_or_none()

        if opinion is None:
            return AuthorityNode(
                opinion_id=opinion_id,
                case_name="Unknown",
                citation_string="",
                court="unknown",
                date_filed="",
                citation_count=0,
            )

        cite_count_stmt = select(func.count(CitationRow.id)).where(
            CitationRow.cited_opinion_id == opinion_id
        )
        cite_result = await self._session.execute(cite_count_stmt)
        cite_count: int = cite_result.scalar_one()

        date_str = (
            opinion.date_filed.isoformat()
            if isinstance(opinion.date_filed, date)
            else str(opinion.date_filed)
        )

        return AuthorityNode(
            opinion_id=opinion.id,
            case_name=opinion.case_name,
            citation_string=f"{opinion.case_name}",
            court=opinion.court_id,
            date_filed=date_str,
            citation_count=cite_count,
        )

    def _authority_score(self, node: AuthorityNode) -> float:
        """Score an authority for ranking: citation count + court level + recency."""
        cite_score = min(node.citation_count / 10.0, 1.0) * 0.5

        court_rank = _COURT_LEVEL_RANK.get(node.court, 25)
        court_score = (court_rank / 100.0) * 0.3

        recency_score = 0.0
        if node.date_filed:
            try:
                d = date.fromisoformat(node.date_filed)
                age = (date.today() - d).days / 365.25
                recency_score = max(0.0, 1.0 - age / 30.0) * 0.2
            except (ValueError, TypeError):
                pass

        return cite_score + court_score + recency_score
