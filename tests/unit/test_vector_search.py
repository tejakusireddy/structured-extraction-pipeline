"""Tests for the MMR reranking algorithm and search helpers.

Uses synthetic vectors to verify correctness of the hand-rolled MMR
implementation: pure relevance, pure diversity, balanced selection,
empty inputs, and filter construction.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from src.models.domain import CourtLevel
from src.models.requests import SearchFilters
from src.services.search.vector_search import (
    _avg_pairwise_diversity,
    _build_filter,
    _compute_metrics,
    _cosine_similarities,
    mmr_rerank,
)


class TestMMRRerank:
    """Tests for the MMR greedy selection algorithm."""

    def _make_vecs(self) -> np.ndarray:
        """Build 5 synthetic candidate vectors with known relationships.

        v0: very close to query (high relevance)
        v1: near-duplicate of v0
        v2: moderately relevant, different direction
        v3: less relevant, very different direction
        v4: low relevance, orthogonal
        """
        return np.array(
            [
                [0.9, 0.1, 0.0],  # v0: close to query [1,0,0]
                [0.88, 0.12, 0.0],  # v1: near-duplicate of v0
                [0.5, 0.5, 0.0],  # v2: 45 degrees from query
                [0.1, 0.9, 0.1],  # v3: mostly orthogonal
                [0.0, 0.0, 1.0],  # v4: fully orthogonal
            ],
            dtype=np.float64,
        )

    def test_pure_relevance_matches_similarity_order(self):
        """λ=1.0 means no diversity penalty — just pick by relevance."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=3,
            lambda_mult=1.0,
        )

        assert len(selected) == 3
        assert selected[0] == 0, "Most relevant should be first"
        assert selected[1] == 1, "Second most relevant (near-dup) second"

    def test_pure_diversity_avoids_near_duplicates(self):
        """λ=0.0 means maximize diversity — should pick most different second."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=3,
            lambda_mult=0.0,
        )

        assert selected[0] == 0, "First pick is always most relevant"
        assert selected[1] != 1, "Near-duplicate should NOT be second"
        assert selected[1] == 4, "Orthogonal vector should be second"

    def test_balanced_mmr_avoids_near_duplicates(self):
        """λ=0.5 should balance — pick v0 first but avoid v1 (near-dup)."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=3,
            lambda_mult=0.5,
        )

        assert selected[0] == 0, "Most relevant first"
        assert 1 not in selected[:2], "Near-duplicate of first pick should not be second"

    def test_k_greater_than_candidates(self):
        """k > n should return all candidates."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=100,
            lambda_mult=0.7,
        )

        assert len(selected) == 5
        assert set(selected) == {0, 1, 2, 3, 4}

    def test_k_equals_one(self):
        """k=1 should return just the most relevant."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=1,
            lambda_mult=0.7,
        )

        assert selected == [0]

    def test_empty_candidates(self):
        """Empty candidate set returns empty selection."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = np.array([], dtype=np.float64).reshape(0, 3)

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=5,
            lambda_mult=0.7,
        )

        assert selected == []

    def test_no_duplicate_selections(self):
        """Selected indices should all be unique."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        candidates = self._make_vecs()

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=5,
            lambda_mult=0.7,
        )

        assert len(selected) == len(set(selected))

    def test_identical_vectors(self):
        """All identical vectors — should still select k unique indices."""
        query = np.array([1.0, 0.0], dtype=np.float64)
        candidates = np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            dtype=np.float64,
        )

        selected = mmr_rerank(
            query_vec=query,
            candidate_vecs=candidates,
            k=3,
            lambda_mult=0.5,
        )

        assert len(selected) == 3
        assert len(set(selected)) == 3


class TestCosineSimilarities:
    def test_identical_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float64)
        b = np.array([[1.0, 0.0]], dtype=np.float64)
        sim = _cosine_similarities(a, b)
        assert sim.shape == (1, 1)
        assert abs(sim[0, 0] - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float64)
        b = np.array([[0.0, 1.0]], dtype=np.float64)
        sim = _cosine_similarities(a, b)
        assert abs(sim[0, 0]) < 1e-6

    def test_matrix_shape(self):
        a = np.array([[1, 0], [0, 1]], dtype=np.float64)
        b = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
        sim = _cosine_similarities(a, b)
        assert sim.shape == (2, 3)


class TestAvgPairwiseDiversity:
    def test_identical_vectors_zero_diversity(self):
        vecs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        div = _avg_pairwise_diversity(vecs)
        assert abs(div) < 1e-6

    def test_orthogonal_vectors_high_diversity(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        div = _avg_pairwise_diversity(vecs)
        assert abs(div - 1.0) < 1e-6

    def test_single_vector_zero(self):
        vecs = np.array([[1.0, 0.0]], dtype=np.float64)
        assert _avg_pairwise_diversity(vecs) == 0.0


class TestComputeMetrics:
    def test_empty_results(self):
        metrics = _compute_metrics([], diversity=None)
        assert metrics.unique_courts == 0
        assert metrics.avg_relevance_score == 0.0
        assert metrics.avg_pairwise_diversity is None

    def test_single_result(self):
        from src.models.responses import SearchResultItem

        results = [
            SearchResultItem(
                opinion_id=1,
                case_name="Case A",
                court_id="ca9",
                date_filed="2024-01-15",
                holding="Test holding",
                relevance_score=0.85,
                legal_topics=["topic1"],
            )
        ]
        metrics = _compute_metrics(results, diversity=0.5)
        assert metrics.unique_courts == 1
        assert metrics.avg_relevance_score == 0.85
        assert metrics.avg_pairwise_diversity == 0.5

    def test_date_range_calculation(self):
        from src.models.responses import SearchResultItem

        results = [
            SearchResultItem(
                opinion_id=1,
                case_name="A",
                court_id="ca9",
                date_filed="2020-01-01",
                holding="h",
                relevance_score=0.8,
                legal_topics=["t"],
            ),
            SearchResultItem(
                opinion_id=2,
                case_name="B",
                court_id="ca5",
                date_filed="2024-01-01",
                holding="h",
                relevance_score=0.7,
                legal_topics=["t"],
            ),
        ]
        metrics = _compute_metrics(results, diversity=None)
        assert metrics.unique_courts == 2
        assert metrics.date_range_years > 3.5


class TestBuildFilter:
    def test_no_filters_returns_none(self):
        f = SearchFilters()
        assert _build_filter(f) is None

    def test_court_level_filter(self):
        f = SearchFilters(court_level=CourtLevel.APPELLATE)
        result = _build_filter(f)
        assert result is not None
        assert len(result.must) == 1  # type: ignore[arg-type]

    def test_court_ids_filter(self):
        f = SearchFilters(court_ids=["ca9", "ca5"])
        result = _build_filter(f)
        assert result is not None
        assert len(result.must) == 1  # type: ignore[arg-type]

    def test_date_range_filter(self):
        f = SearchFilters(
            date_after=date(2020, 1, 1),
            date_before=date(2024, 12, 31),
        )
        result = _build_filter(f)
        assert result is not None
        assert len(result.must) == 2  # type: ignore[arg-type]

    def test_combined_filters(self):
        f = SearchFilters(
            court_level=CourtLevel.SUPREME,
            court_ids=["scotus"],
            jurisdiction="federal",
            date_after=date(2020, 1, 1),
        )
        result = _build_filter(f)
        assert result is not None
        assert len(result.must) == 4  # type: ignore[arg-type]
