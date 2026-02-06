"""Tests for confidence module."""

import numpy as np
import pytest

from taxonomise.confidence import (
    assign_global_bins,
    assign_local_bins,
    combine_bins,
    _compute_dropoff_bins,
)


class TestAssignGlobalBins:
    """Tests for global binning."""

    def test_basic_binning(self):
        """Test basic global binning."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bins = assign_global_bins(scores, q2_threshold=0.5, q3_threshold=0.75)

        assert bins[0] == "low"  # 0.1 is below median
        assert bins[4] == "high"  # 0.9 is above q3

    def test_uniform_scores(self):
        """Test binning with uniform scores."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        bins = assign_global_bins(scores, q2_threshold=0.5, q3_threshold=0.75)

        # All scores are equal to q2, so should be "low" (not > q2)
        assert all(b == "low" for b in bins)


class TestAssignLocalBins:
    """Tests for local binning."""

    def test_single_label(self):
        """Test that single label always gets 'high'."""
        doc_ids = np.array(["doc1"])
        scores = np.array([0.5])
        bins = assign_local_bins(doc_ids, scores)

        assert bins[0] == "high"

    def test_two_labels(self):
        """Test that two labels get 'high' and 'medium'."""
        doc_ids = np.array(["doc1", "doc1"])
        scores = np.array([0.8, 0.3])
        bins = assign_local_bins(doc_ids, scores)

        assert bins[0] == "high"  # Higher score
        assert bins[1] == "medium"  # Lower score

    def test_multiple_documents(self):
        """Test binning across multiple documents."""
        doc_ids = np.array(["doc1", "doc1", "doc2", "doc2"])
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        bins = assign_local_bins(doc_ids, scores)

        # Each document should be binned independently
        assert bins[0] == "high"  # doc1 high score
        assert bins[2] == "high"  # doc2 high score


class TestDropoffBins:
    """Tests for dropoff-based binning."""

    def test_clear_gaps(self):
        """Test binning with clear score gaps."""
        scores = np.array([0.9, 0.85, 0.4, 0.35, 0.1])  # Sorted descending
        bins = _compute_dropoff_bins(scores)

        # Gap between 0.85 and 0.4 should create boundary
        # Gap between 0.35 and 0.1 should create another boundary
        assert bins[0] == "high"
        assert bins[1] == "high"

    def test_no_gaps(self):
        """Test binning with uniform scores."""
        scores = np.array([0.5, 0.49, 0.48, 0.47, 0.46])
        bins = _compute_dropoff_bins(scores)

        # Should still produce valid bins
        assert all(b in ["high", "medium", "low"] for b in bins)


class TestCombineBins:
    """Tests for combining bins."""

    def test_min_combination(self):
        """Test that combination takes minimum."""
        global_bins = np.array(["high", "high", "medium", "low"])
        local_bins = np.array(["high", "medium", "medium", "high"])

        combined = combine_bins(global_bins, local_bins)

        assert combined[0] == "high"   # min(high, high)
        assert combined[1] == "medium" # min(high, medium)
        assert combined[2] == "medium" # min(medium, medium)
        assert combined[3] == "low"    # min(low, high)


class TestBinningInvariants:
    """Tests for binning invariants."""

    def test_monotonicity(self):
        """Test that higher scores never get lower bins."""
        np.random.seed(42)

        # Generate random scores
        doc_ids = np.array(["doc1"] * 10)
        scores = np.random.rand(10)

        bins = assign_local_bins(doc_ids, scores)

        # Sort by score
        sorted_idx = np.argsort(-scores)
        sorted_bins = bins[sorted_idx]

        # Convert to numeric
        bin_order = {"high": 3, "medium": 2, "low": 1}
        numeric_bins = [bin_order[b] for b in sorted_bins]

        # Bins should be non-increasing (higher score = higher or equal bin)
        for i in range(len(numeric_bins) - 1):
            assert numeric_bins[i] >= numeric_bins[i + 1]

    def test_valid_bins(self):
        """Test that all bins are valid values."""
        doc_ids = np.array(["doc1"] * 5 + ["doc2"] * 5)
        scores = np.random.rand(10)

        bins = assign_local_bins(doc_ids, scores)

        valid_bins = {"high", "medium", "low"}
        for b in bins:
            assert b in valid_bins
