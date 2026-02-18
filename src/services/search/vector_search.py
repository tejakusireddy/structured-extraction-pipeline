"""Vector search service backed by Qdrant.

Manages a Qdrant collection for holding embeddings and provides
similarity and MMR (Maximal Marginal Relevance) search strategies.
MMR reranking uses numpy for efficient pairwise cosine similarity.
"""

from __future__ import annotations

import contextlib
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from qdrant_client import AsyncQdrantClient, models

from src.models.domain import SearchStrategy
from src.models.responses import SearchMetrics, SearchResponse, SearchResultItem

if TYPE_CHECKING:
    from src.models.requests import SearchFilters
    from src.services.search.embeddings import EmbeddingService

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

MMR_FETCH_K = 50


class VectorSearchService:
    """Manages Qdrant collection and provides similarity + MMR search."""

    def __init__(
        self,
        *,
        qdrant: AsyncQdrantClient,
        embedding_service: EmbeddingService,
        collection_name: str = "holdings_vectors",
        vector_size: int = 1536,
    ) -> None:
        self._qdrant = qdrant
        self._embeddings = embedding_service
        self._collection = collection_name
        self._vector_size = vector_size

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        collections = await self._qdrant.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection in existing:
            return

        await self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._vector_size,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
            ),
        )
        logger.info("qdrant_collection_created", name=self._collection)

    async def index_holdings(
        self,
        extractions: list[dict[str, Any]],
    ) -> int:
        """Embed extracted holdings and upsert into Qdrant.

        Each dict must have: opinion_id, holding, court_id, court_level,
        date_filed, case_name, jurisdiction, legal_topics.
        Returns the number of points upserted.
        """
        if not extractions:
            return 0

        texts = [e["holding"] for e in extractions]
        vectors = await self._embeddings.embed_batch(texts)

        points: list[models.PointStruct] = []
        for ext, vec in zip(extractions, vectors, strict=True):
            date_filed = ext["date_filed"]
            date_str = date_filed.isoformat() if isinstance(date_filed, date) else str(date_filed)
            points.append(
                models.PointStruct(
                    id=ext["opinion_id"],
                    vector=vec,
                    payload={
                        "opinion_id": ext["opinion_id"],
                        "court_id": ext["court_id"],
                        "court_level": ext["court_level"],
                        "date_filed": date_str,
                        "case_name": ext["case_name"],
                        "jurisdiction": ext.get("jurisdiction", ""),
                        "holding": ext["holding"],
                        "legal_topics": ext.get("legal_topics", []),
                    },
                )
            )

        await self._qdrant.upsert(
            collection_name=self._collection,
            points=points,
        )

        logger.info("holdings_indexed", count=len(points))
        return len(points)

    async def search(
        self,
        *,
        query: str,
        k: int = 10,
        strategy: SearchStrategy = SearchStrategy.SIMILARITY,
        lambda_mult: float = 0.7,
        filters: SearchFilters | None = None,
    ) -> SearchResponse:
        """Run a vector search with optional MMR reranking."""
        query_vec = await self._embeddings.embed_query(query)
        qdrant_filter = _build_filter(filters) if filters else None

        if strategy == SearchStrategy.MMR:
            return await self._search_mmr(
                query_vec=query_vec,
                k=k,
                lambda_mult=lambda_mult,
                qdrant_filter=qdrant_filter,
            )
        return await self._search_similarity(
            query_vec=query_vec,
            k=k,
            qdrant_filter=qdrant_filter,
        )

    async def _search_similarity(
        self,
        *,
        query_vec: list[float],
        k: int,
        qdrant_filter: models.Filter | None,
    ) -> SearchResponse:
        hits = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = _hits_to_results(hits.points)
        metrics = _compute_metrics(results, diversity=None)
        return SearchResponse(results=results, metrics=metrics)

    async def _search_mmr(
        self,
        *,
        query_vec: list[float],
        k: int,
        lambda_mult: float,
        qdrant_filter: models.Filter | None,
    ) -> SearchResponse:
        fetch_k = max(k, MMR_FETCH_K)
        hits = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=fetch_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True,
        )

        if not hits.points:
            return SearchResponse(
                results=[],
                metrics=_compute_metrics([], diversity=0.0),
            )

        candidates = hits.points
        candidate_vecs = _extract_vectors(candidates)
        query_arr = np.array(query_vec, dtype=np.float64)

        selected_indices = mmr_rerank(
            query_vec=query_arr,
            candidate_vecs=candidate_vecs,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected_points = [candidates[i] for i in selected_indices]
        results = _hits_to_results(selected_points)

        diversity = (
            _avg_pairwise_diversity(candidate_vecs[selected_indices])
            if len(selected_indices) > 1
            else 0.0
        )

        metrics = _compute_metrics(results, diversity=diversity)
        return SearchResponse(results=results, metrics=metrics)


# ---------------------------------------------------------------------------
# MMR algorithm
# ---------------------------------------------------------------------------


def mmr_rerank(
    *,
    query_vec: np.ndarray[Any, np.dtype[np.float64]],
    candidate_vecs: np.ndarray[Any, np.dtype[np.float64]],
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Maximal Marginal Relevance greedy selection.

    Picks documents that balance relevance to the query against
    diversity from already-selected documents.

    Score_i = λ * sim(d_i, query) - (1-λ) * max_{j ∈ selected} sim(d_i, d_j)

    Returns indices into the candidate array.
    """
    n = len(candidate_vecs)
    if n == 0:
        return []
    k = min(k, n)

    query_sims = _cosine_similarities(query_vec.reshape(1, -1), candidate_vecs)[0]

    selected: list[int] = [int(np.argmax(query_sims))]
    remaining = set(range(n)) - set(selected)

    for _ in range(k - 1):
        if not remaining:
            break

        best_idx = -1
        best_score = -float("inf")

        selected_vecs = candidate_vecs[selected]
        for idx in remaining:
            cand_vec = candidate_vecs[idx : idx + 1]
            inter_sims = _cosine_similarities(cand_vec, selected_vecs)[0]
            max_inter_sim = float(np.max(inter_sims))

            score = lambda_mult * float(query_sims[idx]) - (1.0 - lambda_mult) * max_inter_sim
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _cosine_similarities(
    a: np.ndarray[Any, np.dtype[np.float64]],
    b: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute cosine similarity matrix between rows of a and b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    result: np.ndarray[Any, np.dtype[np.float64]] = a_norm @ b_norm.T
    return result


def _avg_pairwise_diversity(
    vecs: np.ndarray[Any, np.dtype[np.float64]],
) -> float:
    """Average pairwise diversity (1 - cosine_sim) among a set of vectors."""
    if len(vecs) < 2:
        return 0.0
    sim_matrix = _cosine_similarities(vecs, vecs)
    n = len(vecs)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim_matrix[i, j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_vectors(
    points: list[Any],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Extract float vectors from Qdrant ScoredPoint objects."""
    vecs: list[list[float]] = []
    for p in points:
        v = p.vector
        if isinstance(v, dict):
            v = next(iter(v.values()))
        vecs.append(v)
    return np.array(vecs, dtype=np.float64)


def _hits_to_results(points: list[Any]) -> list[SearchResultItem]:
    """Convert Qdrant scored points to SearchResultItem list."""
    results: list[SearchResultItem] = []
    for p in points:
        payload = p.payload or {}
        score = getattr(p, "score", 0.0) or 0.0
        clamped_score = max(0.0, min(1.0, float(score)))
        results.append(
            SearchResultItem(
                opinion_id=payload.get("opinion_id", 0),
                case_name=payload.get("case_name", ""),
                court_id=payload.get("court_id", ""),
                date_filed=payload.get("date_filed", ""),
                holding=payload.get("holding", ""),
                relevance_score=clamped_score,
                legal_topics=payload.get("legal_topics", []),
            )
        )
    return results


def _compute_metrics(
    results: list[SearchResultItem],
    *,
    diversity: float | None,
) -> SearchMetrics:
    """Build SearchMetrics from a result set."""
    if not results:
        return SearchMetrics(
            unique_courts=0,
            date_range_years=0.0,
            avg_relevance_score=0.0,
            avg_pairwise_diversity=diversity,
        )

    courts = {r.court_id for r in results}
    scores = [r.relevance_score for r in results]

    dates: list[date] = []
    for r in results:
        with contextlib.suppress(ValueError, TypeError):
            dates.append(date.fromisoformat(r.date_filed))

    date_range = 0.0
    if len(dates) >= 2:
        span = max(dates) - min(dates)
        date_range = span.days / 365.25

    return SearchMetrics(
        unique_courts=len(courts),
        date_range_years=round(date_range, 2),
        avg_relevance_score=round(sum(scores) / len(scores), 4),
        avg_pairwise_diversity=(round(diversity, 4) if diversity is not None else None),
    )


def _build_filter(filters: SearchFilters) -> models.Filter | None:
    """Translate SearchFilters to a Qdrant Filter."""
    conditions: list[models.Condition] = []

    if filters.court_level is not None:
        conditions.append(
            models.FieldCondition(
                key="court_level",
                match=models.MatchValue(value=filters.court_level.value),
            )
        )

    if filters.court_ids:
        conditions.append(
            models.FieldCondition(
                key="court_id",
                match=models.MatchAny(any=filters.court_ids),
            )
        )

    if filters.jurisdiction is not None:
        conditions.append(
            models.FieldCondition(
                key="jurisdiction",
                match=models.MatchValue(value=filters.jurisdiction),
            )
        )

    if filters.date_after is not None:
        conditions.append(
            models.FieldCondition(
                key="date_filed",
                range=models.DatetimeRange(gte=filters.date_after),
            )
        )

    if filters.date_before is not None:
        conditions.append(
            models.FieldCondition(
                key="date_filed",
                range=models.DatetimeRange(lte=filters.date_before),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)
