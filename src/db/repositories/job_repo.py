"""Repository for extraction job CRUD operations.

Handles creating, updating, and querying batch extraction jobs.
"""

from datetime import UTC, datetime

from sqlalchemy import CursorResult, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import ExtractionJobRow


class JobRepo:
    """Async repository for extraction jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        opinion_ids: list[int],
        total_opinions: int,
    ) -> ExtractionJobRow:
        """Create a new extraction job and return it."""
        job = ExtractionJobRow(
            opinion_ids=opinion_ids,
            total_opinions=total_opinions,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get_by_id(self, job_id: str) -> ExtractionJobRow | None:
        """Fetch a job by its primary key."""
        stmt = select(ExtractionJobRow).where(ExtractionJobRow.id == job_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        job_id: str,
        status: str,
        *,
        processed: int | None = None,
        failed: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update job status and optional counters. Returns True if row existed."""
        values: dict[str, object] = {"status": status}
        if processed is not None:
            values["processed"] = processed
        if failed is not None:
            values["failed"] = failed
        if error_message is not None:
            values["error_message"] = error_message

        if status == "running" and "started_at" not in values:
            values["started_at"] = datetime.now(UTC)
        if status in ("completed", "failed"):
            values["completed_at"] = datetime.now(UTC)

        stmt = update(ExtractionJobRow).where(ExtractionJobRow.id == job_id).values(**values)
        cursor: CursorResult[tuple[()]] = await self._session.execute(stmt)  # type: ignore[assignment]
        return cursor.rowcount > 0

    async def increment_counters(
        self,
        job_id: str,
        *,
        processed_delta: int = 0,
        failed_delta: int = 0,
    ) -> bool:
        """Atomically increment processed/failed counters."""
        stmt = (
            update(ExtractionJobRow)
            .where(ExtractionJobRow.id == job_id)
            .values(
                processed=ExtractionJobRow.processed + processed_delta,
                failed=ExtractionJobRow.failed + failed_delta,
            )
        )
        cursor: CursorResult[tuple[()]] = await self._session.execute(stmt)  # type: ignore[assignment]
        return cursor.rowcount > 0
