"""Repository for citation graph operations.

Handles batch creation of citation edges, forward/reverse lookups,
and recursive CTE-based subgraph traversal for N-depth exploration.
"""

from typing import Any

from sqlalchemy import CompoundSelect, Select, and_, literal_column, select, union_all
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import CitationRow, OpinionRow


class CitationRepo:
    """Async repository for the citation graph."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_batch(self, citations: list[CitationRow]) -> list[CitationRow]:
        """Insert multiple citation edges and return them with generated fields."""
        if not citations:
            return []

        self._session.add_all(citations)
        await self._session.flush()
        return citations

    async def get_citations_for_opinion(
        self,
        opinion_id: int,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[CitationRow]:
        """Get all citations *from* a given opinion (outgoing edges)."""
        stmt = (
            select(CitationRow)
            .where(CitationRow.citing_opinion_id == opinion_id)
            .order_by(CitationRow.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_cited_by(
        self,
        opinion_id: int,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> list[CitationRow]:
        """Get all citations *to* a given opinion (incoming edges / reverse lookup)."""
        stmt = (
            select(CitationRow)
            .where(CitationRow.cited_opinion_id == opinion_id)
            .order_by(CitationRow.id)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_citation_subgraph(
        self,
        root_opinion_id: int,
        *,
        max_depth: int = 3,
        direction: str = "outgoing",
    ) -> list[dict[str, object]]:
        """Traverse the citation graph from a root node using a recursive CTE.

        Parameters
        ----------
        root_opinion_id:
            The opinion to start traversal from.
        max_depth:
            Maximum depth of the traversal (1 = direct citations only).
        direction:
            "outgoing" follows citing → cited edges.
            "incoming" follows cited → citing edges (who cites this opinion).
            "both" follows edges in both directions.

        Returns
        -------
        A list of dicts with keys: opinion_id, depth, case_name, court_id.
        The root opinion (depth=0) is included.
        """
        base_query: Select[Any] = select(
            OpinionRow.id.label("opinion_id"),
            literal_column("0").label("depth"),
        ).where(OpinionRow.id == root_opinion_id)

        graph_cte = base_query.cte(name="graph", recursive=True)

        recursive_part: Select[Any] | CompoundSelect[Any]

        if direction == "outgoing":
            recursive_part = (
                select(
                    CitationRow.cited_opinion_id.label("opinion_id"),
                    (graph_cte.c.depth + 1).label("depth"),
                )
                .join(graph_cte, CitationRow.citing_opinion_id == graph_cte.c.opinion_id)
                .where(
                    and_(
                        graph_cte.c.depth < max_depth,
                        CitationRow.cited_opinion_id.is_not(None),
                    )
                )
            )
        elif direction == "incoming":
            recursive_part = (
                select(
                    CitationRow.citing_opinion_id.label("opinion_id"),
                    (graph_cte.c.depth + 1).label("depth"),
                )
                .join(graph_cte, CitationRow.cited_opinion_id == graph_cte.c.opinion_id)
                .where(graph_cte.c.depth < max_depth)
            )
        elif direction == "both":
            outgoing = (
                select(
                    CitationRow.cited_opinion_id.label("opinion_id"),
                    (graph_cte.c.depth + 1).label("depth"),
                )
                .join(graph_cte, CitationRow.citing_opinion_id == graph_cte.c.opinion_id)
                .where(
                    and_(
                        graph_cte.c.depth < max_depth,
                        CitationRow.cited_opinion_id.is_not(None),
                    )
                )
            )
            incoming = (
                select(
                    CitationRow.citing_opinion_id.label("opinion_id"),
                    (graph_cte.c.depth + 1).label("depth"),
                )
                .join(graph_cte, CitationRow.cited_opinion_id == graph_cte.c.opinion_id)
                .where(graph_cte.c.depth < max_depth)
            )
            recursive_part = union_all(outgoing, incoming)
        else:
            msg = f"direction must be 'outgoing', 'incoming', or 'both', got {direction!r}"
            raise ValueError(msg)

        graph_cte = graph_cte.union_all(recursive_part)

        final_stmt = (
            select(
                graph_cte.c.opinion_id,
                graph_cte.c.depth,
                OpinionRow.case_name,
                OpinionRow.court_id,
            )
            .join(OpinionRow, OpinionRow.id == graph_cte.c.opinion_id)
            .distinct(graph_cte.c.opinion_id)
            .order_by(graph_cte.c.opinion_id, graph_cte.c.depth)
        )

        result = await self._session.execute(final_stmt)
        return [
            {
                "opinion_id": row.opinion_id,
                "depth": row.depth,
                "case_name": row.case_name,
                "court_id": row.court_id,
            }
            for row in result.all()
        ]
