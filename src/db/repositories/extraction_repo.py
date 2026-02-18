"""Repository for extraction CRUD operations.

Handles creating extraction results, querying by opinion, updating
extraction status, and listing pending extractions for processing.
"""

from datetime import UTC, datetime

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import ExtractionRow


class ExtractionRepo:
    """Async repository for extraction results."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, extraction: ExtractionRow) -> ExtractionRow:
        """Insert a single extraction and return it with generated fields populated."""
        self._session.add(extraction)
        await self._session.flush()
        return extraction

    async def get_by_opinion_id(self, opinion_id: int) -> list[ExtractionRow]:
        """Fetch all extractions for a given opinion, newest first."""
        stmt = (
            select(ExtractionRow)
            .where(ExtractionRow.opinion_id == opinion_id)
            .order_by(ExtractionRow.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        extraction_id: int,
        status: str,
        *,
        error_message: str | None = None,
    ) -> bool:
        """Update the status of an extraction. Returns True if the row existed."""
        values: dict[str, object] = {"status": status}
        if error_message is not None:
            values["updated_at"] = datetime.now(UTC)

        stmt = update(ExtractionRow).where(ExtractionRow.id == extraction_id).values(**values)
        cursor: CursorResult[tuple[()]] = await self._session.execute(stmt)  # type: ignore[assignment]
        return cursor.rowcount > 0

    async def list_pending(self, *, limit: int = 50) -> list[ExtractionRow]:
        """List extractions with status 'pending', oldest first."""
        stmt = (
            select(ExtractionRow)
            .where(ExtractionRow.status == "pending")
            .order_by(ExtractionRow.created_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
