"""Redis-based async extraction job processor.

ExtractionWorker manages the lifecycle of batch extraction jobs:
  1. submit_job — creates a DB record, pushes job_id to Redis queue
  2. process_job — pulls opinion text, runs extraction, stores results
  3. Handles partial failures (some opinions fail, job still completes)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.db.repositories import ExtractionRepo, JobRepo, OpinionRepo
from src.models.database import ExtractionRow
from src.models.domain import CourtLevel, ExtractionStatus, OpinionMetadata, PrecedentialStatus

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.services.extraction.extractor import ExtractionService

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

QUEUE_KEY = "extraction:jobs"


class ExtractionWorker:
    """Async worker for processing extraction jobs via Redis queue."""

    def __init__(
        self,
        *,
        redis: Redis,
        extraction_service: ExtractionService,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._redis = redis
        self._extraction_service = extraction_service
        self._session_factory = session_factory

    async def submit_job(
        self,
        *,
        opinion_ids: list[int],
        session: AsyncSession,
    ) -> str:
        """Create a job record in Postgres and enqueue in Redis.

        Returns the job_id for the caller to track progress.
        Uses the provided session (from the request scope) so the
        job record is visible to subsequent GET requests.
        """
        from src.db.repositories import JobRepo

        job_repo = JobRepo(session)
        job = await job_repo.create(
            opinion_ids=opinion_ids,
            total_opinions=len(opinion_ids),
        )
        await session.flush()

        job_id: str = job.id

        await self._redis.lpush(QUEUE_KEY, job_id)  # type: ignore[misc]

        logger.info(
            "job_submitted",
            job_id=job_id,
            total_opinions=len(opinion_ids),
        )
        return job_id

    async def process_job(self, job_id: str) -> None:
        """Process a single extraction job end-to-end.

        Opens its own DB session so it's independent of the request
        lifecycle. Updates counters after each opinion is processed.
        """
        async with self._session_factory() as session:
            job_repo = JobRepo(session)
            opinion_repo = OpinionRepo(session)
            extraction_repo = ExtractionRepo(session)

            job = await job_repo.get_by_id(job_id)
            if job is None:
                logger.error("job_not_found", job_id=job_id)
                return

            await job_repo.update_status(job_id, "running")
            await session.commit()

            processed = 0
            failed = 0

            for opinion_id in job.opinion_ids:
                try:
                    await self._process_single_opinion(
                        opinion_id=opinion_id,
                        opinion_repo=opinion_repo,
                        extraction_repo=extraction_repo,
                    )
                    processed += 1
                except Exception:
                    logger.exception(
                        "opinion_extraction_failed",
                        job_id=job_id,
                        opinion_id=opinion_id,
                    )
                    failed += 1

                await job_repo.update_status(
                    job_id,
                    "running",
                    processed=processed,
                    failed=failed,
                )
                await session.commit()

            final_status = "completed" if failed < len(job.opinion_ids) else "failed"
            await job_repo.update_status(
                job_id,
                final_status,
                processed=processed,
                failed=failed,
            )
            await session.commit()

            logger.info(
                "job_completed",
                job_id=job_id,
                status=final_status,
                processed=processed,
                failed=failed,
            )

    async def _process_single_opinion(
        self,
        *,
        opinion_id: int,
        opinion_repo: OpinionRepo,
        extraction_repo: ExtractionRepo,
    ) -> None:
        """Extract structured intelligence from a single opinion."""
        opinion = await opinion_repo.get_by_id(opinion_id)
        if opinion is None:
            msg = f"Opinion {opinion_id} not found in database"
            raise ValueError(msg)

        metadata = OpinionMetadata(
            opinion_id=opinion.courtlistener_id,
            cluster_id=0,
            court_id=opinion.court_id,
            court_name=opinion.case_name,
            court_level=CourtLevel(opinion.court_level),
            case_name=opinion.case_name,
            date_filed=opinion.date_filed,
            precedential_status=PrecedentialStatus(opinion.precedential_status),
            citation_count=opinion.citation_count,
            judges=opinion.judges,
            jurisdiction=opinion.jurisdiction,
            source_url=opinion.source_url,
        )

        result = await self._extraction_service.extract_opinion(
            metadata=metadata,
            opinion_text=opinion.raw_text,
        )

        if result.intelligence is not None:
            intel = result.intelligence
            extraction_row = ExtractionRow(
                opinion_id=opinion.id,
                holding=intel.holding,
                holding_confidence=intel.holding_confidence,
                legal_standard=intel.legal_standard,
                disposition=intel.disposition.value,
                disposition_confidence=intel.disposition_confidence,
                dissent_present=intel.dissent_present,
                dissent_summary=intel.dissent_summary,
                concurrence_present=intel.concurrence_present,
                concurrence_summary=intel.concurrence_summary,
                legal_topics=list(intel.legal_topics),
                extraction_model=result.model,
                prompt_tokens=result.total_prompt_tokens,
                completion_tokens=result.total_completion_tokens,
                status=result.status.value,
            )
            await extraction_repo.create(extraction_row)
        else:
            extraction_row = ExtractionRow(
                opinion_id=opinion.id,
                holding="Extraction failed",
                holding_confidence=0.0,
                disposition="dismissed",
                disposition_confidence=0.0,
                legal_topics=["unknown"],
                extraction_model=result.model,
                prompt_tokens=result.total_prompt_tokens,
                completion_tokens=result.total_completion_tokens,
                status=ExtractionStatus.FAILED.value,
            )
            await extraction_repo.create(extraction_row)

    async def process_next(self) -> bool:
        """Pop and process the next job from the Redis queue.

        Returns True if a job was processed, False if the queue was empty.
        """
        raw = await self._redis.rpop(QUEUE_KEY)  # type: ignore[misc]
        if raw is None:
            return False

        job_id = raw if isinstance(raw, str) else raw.decode("utf-8")
        await self.process_job(job_id)
        return True
