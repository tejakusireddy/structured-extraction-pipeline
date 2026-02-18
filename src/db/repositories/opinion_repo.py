"""Repository for opinion CRUD operations.

All database access for the opinions table is encapsulated here.
Services never execute raw SQL — they call repository methods.
"""

from datetime import date

from sqlalchemy import CursorResult, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import OpinionRow


class OpinionRepo:
    """Async repository for court opinions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, opinion: OpinionRow) -> OpinionRow:
        """Insert a single opinion and return it with generated fields populated."""
        self._session.add(opinion)
        await self._session.flush()
        return opinion

    async def get_by_id(self, opinion_id: int) -> OpinionRow | None:
        """Fetch an opinion by its primary key."""
        stmt = select(OpinionRow).where(OpinionRow.id == opinion_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_courtlistener_id(self, cl_id: int) -> OpinionRow | None:
        """Fetch an opinion by its CourtListener ID."""
        stmt = select(OpinionRow).where(OpinionRow.courtlistener_id == cl_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_court(
        self,
        court_id: str,
        *,
        date_after: date | None = None,
        date_before: date | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[OpinionRow]:
        """List opinions for a court with optional date range filtering."""
        stmt = select(OpinionRow).where(OpinionRow.court_id == court_id)

        if date_after is not None:
            stmt = stmt.where(OpinionRow.date_filed >= date_after)
        if date_before is not None:
            stmt = stmt.where(OpinionRow.date_filed <= date_before)

        stmt = stmt.order_by(OpinionRow.date_filed.desc()).limit(limit).offset(offset)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def bulk_create(self, opinions: list[dict[str, object]]) -> int:
        """Upsert opinions in bulk using ON CONFLICT DO NOTHING.

        Returns the number of rows actually inserted (excludes conflicts).
        Each dict must contain all required OpinionRow column values.
        """
        if not opinions:
            return 0

        stmt = pg_insert(OpinionRow).values(opinions)
        stmt = stmt.on_conflict_do_nothing(index_elements=["courtlistener_id"])
        cursor: CursorResult[tuple[()]] = await self._session.execute(stmt)  # type: ignore[assignment]
        await self._session.flush()
        return cursor.rowcount

    async def count(self, *, court_id: str | None = None) -> int:
        """Count opinions, optionally filtered by court."""
        from sqlalchemy import func

        stmt = select(func.count(OpinionRow.id))
        if court_id is not None:
            stmt = stmt.where(OpinionRow.court_id == court_id)
        result = await self._session.execute(stmt)
        return result.scalar_one()
