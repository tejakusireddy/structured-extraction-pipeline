"""Extraction API endpoints.

POST /extract — submit an extraction job (returns 202 with job_id)
GET  /extract/{job_id} — poll job status and results
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from src.api.dependencies import get_db_session, get_extraction_worker
from src.db.repositories import ExtractionRepo, JobRepo
from src.models.domain import Disposition, ExtractionStatus, JobStatus
from src.models.requests import ExtractionRequest  # noqa: TC001
from src.models.responses import (
    ExtractionDetail,
    ExtractionJobCreatedResponse,
    ExtractionJobDetailResponse,
    ExtractionResultItem,
)

if TYPE_CHECKING:
    from src.services.queue.worker import ExtractionWorker

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

router = APIRouter(prefix="/extract", tags=["extraction"])


@router.post(
    "",
    response_model=ExtractionJobCreatedResponse,
    status_code=202,
    summary="Submit extraction job",
)
async def submit_extraction_job(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    worker: ExtractionWorker = Depends(get_extraction_worker),
) -> ExtractionJobCreatedResponse:
    """Accept opinion IDs for extraction and return a job tracking ID."""
    job_id = await worker.submit_job(
        opinion_ids=list(request.opinion_ids),
        session=session,
    )

    background_tasks.add_task(_run_job, worker, job_id)

    estimated_seconds = len(request.opinion_ids) * 5.0

    return ExtractionJobCreatedResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        total_opinions=len(request.opinion_ids),
        estimated_completion_seconds=estimated_seconds,
    )


async def _run_job(worker: ExtractionWorker, job_id: str) -> None:
    """Background task that processes the extraction job."""
    try:
        await worker.process_job(job_id)
    except Exception:
        logger.exception("background_job_failed", job_id=job_id)


@router.get(
    "/{job_id}",
    response_model=ExtractionJobDetailResponse,
    summary="Get extraction job status and results",
)
async def get_extraction_job(
    job_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> ExtractionJobDetailResponse:
    """Return current job status and any completed extraction results."""
    job_repo = JobRepo(session)
    job = await job_repo.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    extraction_repo = ExtractionRepo(session)

    results: list[ExtractionResultItem] = []
    for opinion_id in job.opinion_ids:
        extractions = await extraction_repo.get_by_opinion_id(opinion_id)
        if extractions:
            ext = extractions[0]
            detail = ExtractionDetail(
                holding=ext.holding,
                holding_confidence=ext.holding_confidence,
                legal_standard=ext.legal_standard,
                disposition=Disposition(ext.disposition),
                disposition_confidence=ext.disposition_confidence,
                key_authorities=[],
                legal_topics=ext.legal_topics if ext.legal_topics else [],
                dissent_present=ext.dissent_present,
                dissent_summary=ext.dissent_summary,
                concurrence_present=ext.concurrence_present,
                concurrence_summary=ext.concurrence_summary,
            )
            results.append(
                ExtractionResultItem(
                    opinion_id=opinion_id,
                    case_name=ext.holding[:80],
                    status=ExtractionStatus(ext.status),
                    extraction=detail,
                )
            )
        else:
            results.append(
                ExtractionResultItem(
                    opinion_id=opinion_id,
                    case_name="pending",
                    status=ExtractionStatus.PENDING,
                )
            )

    return ExtractionJobDetailResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        total_opinions=job.total_opinions,
        processed=job.processed,
        failed=job.failed,
        results=results,
    )
