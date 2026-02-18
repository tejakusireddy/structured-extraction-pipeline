"""Search API endpoint.

POST /search — semantic search over extracted holdings via Qdrant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends

from src.api.dependencies import get_vector_search
from src.models.requests import SearchRequest  # noqa: TC001
from src.models.responses import SearchResponse

if TYPE_CHECKING:
    from src.services.search.vector_search import VectorSearchService

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic search over holdings",
)
async def search_holdings(
    request: SearchRequest,
    vector_search: VectorSearchService = Depends(get_vector_search),
) -> SearchResponse:
    """Search extracted holdings using similarity or MMR strategy."""
    return await vector_search.search(
        query=request.query,
        k=request.k,
        strategy=request.strategy,
        lambda_mult=request.lambda_mult,
        filters=request.filters,
    )
