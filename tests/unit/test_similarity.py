"""Tests for similarity module."""

import numpy as np
import pytest

from taxonomise.similarity import (
    compute_cosine_similarity,
    search_top_n,
    search_batch,
)


class TestComputeCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        sim = compute_cosine_similarity(embeddings, embeddings, normalized=False)
        assert sim.shape == (1, 1)
        assert np.isclose(sim[0, 0], 1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        query = np.array([[1.0, 0.0, 0.0]])
        target = np.array([[0.0, 1.0, 0.0]])
        sim = compute_cosine_similarity(query, target, normalized=False)
        assert np.isclose(sim[0, 0], 0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        query = np.array([[1.0, 0.0, 0.0]])
        target = np.array([[-1.0, 0.0, 0.0]])
        sim = compute_cosine_similarity(query, target, normalized=False)
        assert np.isclose(sim[0, 0], -1.0)

    def test_normalized_input(self):
        """Test with pre-normalized input."""
        # Normalized vectors
        query = np.array([[0.6, 0.8, 0.0]])
        target = np.array([[0.6, 0.8, 0.0], [1.0, 0.0, 0.0]])

        sim = compute_cosine_similarity(query, target, normalized=True)
        assert np.isclose(sim[0, 0], 1.0)  # Same vector

    def test_batch_shape(self):
        """Test output shape for batched input."""
        query = np.random.randn(5, 10)
        target = np.random.randn(8, 10)

        sim = compute_cosine_similarity(query, target, normalized=False)
        assert sim.shape == (5, 8)


class TestSearchTopN:
    """Tests for top-N search."""

    def test_basic_search(self):
        """Test basic top-N search."""
        query = np.array([[1.0, 0.0, 0.0]])
        target = np.array([
            [1.0, 0.0, 0.0],  # Most similar
            [0.7, 0.7, 0.0],  # Partially similar
            [0.0, 1.0, 0.0],  # Orthogonal
        ])

        result = search_top_n(
            query, ["q1"], target, ["t1", "t2", "t3"], top_n=2, normalized=False
        )

        assert len(result) == 2
        # First result should be t1 (identical vector)
        assert result.target_ids[0] == "t1"
        assert np.isclose(result.similarities[0], 1.0, atol=0.01)

    def test_top_n_limit(self):
        """Test that top_n correctly limits results."""
        query = np.random.randn(2, 10)
        target = np.random.randn(100, 10)

        result = search_top_n(
            query, ["q1", "q2"],
            target, [f"t{i}" for i in range(100)],
            top_n=5,
            normalized=False,
        )

        # Should have 5 results per query = 10 total
        assert len(result) == 10


class TestSearchBatch:
    """Tests for batched search."""

    def test_batch_processing(self):
        """Test that batch processing produces same results as single search."""
        np.random.seed(42)
        query = np.random.randn(10, 8)
        target = np.random.randn(20, 8)

        query_ids = [f"q{i}" for i in range(10)]
        target_ids = [f"t{i}" for i in range(20)]

        # Single batch
        result1 = search_top_n(query, query_ids, target, target_ids, top_n=3)

        # Multiple batches
        result2 = search_batch(
            query, query_ids, target, target_ids, batch_size=3, top_n=3
        )

        # Should have same number of results
        assert len(result1) == len(result2)

        # Same query-target pairs should appear
        pairs1 = set(zip(result1.query_ids, result1.target_ids))
        pairs2 = set(zip(result2.query_ids, result2.target_ids))
        assert pairs1 == pairs2


class TestSimilarityInvariants:
    """Tests for similarity score invariants."""

    def test_scores_in_range(self):
        """Test that cosine similarity scores are in [-1, 1]."""
        np.random.seed(42)
        query = np.random.randn(10, 32)
        target = np.random.randn(20, 32)

        result = search_batch(
            query, [f"q{i}" for i in range(10)],
            target, [f"t{i}" for i in range(20)],
            top_n=5,
            normalized=False,
        )

        assert np.all(result.similarities >= -1.0)
        assert np.all(result.similarities <= 1.0)

    def test_normalized_scores_positive(self):
        """Test that normalized positive embeddings produce positive similarities."""
        # All positive embeddings
        query = np.abs(np.random.randn(5, 16))
        target = np.abs(np.random.randn(10, 16))

        # Normalize
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        target = target / np.linalg.norm(target, axis=1, keepdims=True)

        result = search_batch(
            query, [f"q{i}" for i in range(5)],
            target, [f"t{i}" for i in range(10)],
            top_n=3,
            normalized=True,
        )

        # All similarities should be non-negative
        assert np.all(result.similarities >= 0)
