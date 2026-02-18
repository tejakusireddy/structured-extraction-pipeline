"""Ingestion endpoints — trigger opinion fetching from CourtListener."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_courtlistener_client, get_opinion_repo
from src.db.repositories import OpinionRepo
from src.models.requests import IngestionRequest
from src.models.responses import IngestionProgressResponse
from src.services.ingestion.bulk_loader import BulkLoader
from src.services.ingestion.courtlistener import CourtListenerClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestionProgressResponse)
async def ingest_opinions(
    request: IngestionRequest,
    cl_client: CourtListenerClient = Depends(get_courtlistener_client),
    opinion_repo: OpinionRepo = Depends(get_opinion_repo),
) -> IngestionProgressResponse:
    """Ingest opinions from CourtListener into the pipeline database.

    Requires at least one of ``opinion_ids`` or ``court_ids``.
    """
    if not request.court_ids and not request.opinion_ids:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'court_ids' or 'opinion_ids' must be provided.",
        )

    court_ids = request.court_ids or []

    loader = BulkLoader(cl_client, opinion_repo)

    logger.info(
        "ingestion_request",
        court_ids=court_ids,
        max_opinions=request.max_opinions,
    )

    return await loader.ingest(
        court_ids,
        date_after=request.date_after,
        date_before=request.date_before,
        max_opinions=request.max_opinions,
    )
