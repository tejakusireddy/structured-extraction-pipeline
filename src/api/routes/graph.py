"""Graph API endpoints.

GET /graph/conflicts — detect and return circuit splits.
GET /graph/authority/{citation} — build authority subgraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_db_session
from src.models.responses import (
    AuthorityGraphResponse,
    ConflictItem,
    ConflictOpinionSummary,
    GraphConflictResponse,
)
from src.services.graph.authority_analyzer import AuthorityAnalyzer
from src.services.graph.conflict_detector import ConflictDetector

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get(
    "/conflicts",
    response_model=GraphConflictResponse,
    summary="Detect circuit splits",
)
async def get_conflicts(
    min_confidence: float = Query(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    session: AsyncSession = Depends(get_db_session),
) -> GraphConflictResponse:
    """Detect circuit splits across opinions in the database."""
    detector = ConflictDetector(session)
    conflicts = await detector.detect_conflicts(min_confidence=min_confidence)

    items: list[ConflictItem] = []
    for c in conflicts:
        items.append(
            ConflictItem(
                conflict_id=f"{c.opinion_a_id}-{c.opinion_b_id}",
                topic=c.topic,
                court_a=c.court_a,
                court_b=c.court_b,
                opinion_a=ConflictOpinionSummary(
                    opinion_id=c.opinion_a_id,
                    case_name="",
                    holding="",
                    date_filed="",
                    court=c.court_a,
                ),
                opinion_b=ConflictOpinionSummary(
                    opinion_id=c.opinion_b_id,
                    case_name="",
                    holding="",
                    date_filed="",
                    court=c.court_b,
                ),
                description=c.description,
                confidence=c.confidence,
                status=c.status,
                detected_at=c.detected_at,
            )
        )

    return GraphConflictResponse(conflicts=items, total=len(items))


@router.get(
    "/authority/{citation:path}",
    response_model=AuthorityGraphResponse,
    summary="Build authority subgraph",
)
async def get_authority_graph(
    citation: str,
    depth: int = Query(default=2, ge=1, le=5, description="Traversal depth"),
    session: AsyncSession = Depends(get_db_session),
) -> AuthorityGraphResponse:
    """Build a citation authority subgraph for a citation or topic."""
    analyzer = AuthorityAnalyzer(session)
    return await analyzer.analyze_authority(citation, depth=depth)
