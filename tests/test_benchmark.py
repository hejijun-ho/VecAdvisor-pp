"""Tests for benchmark metrics and workload generation."""

import numpy as np
import pytest

from src.benchmark.metrics import (
    compute_latency_percentiles,
    compute_recall,
    compute_topk_completion_rate,
)
from src.benchmark.workload import (
    build_filter_mask,
    generate_pure_queries,
    generate_filtered_queries,
)


class TestRecall:
    def test_perfect_recall(self):
        """When returned IDs exactly match ground truth."""
        # Ground truth is 0-indexed, result IDs are 1-indexed (PostgreSQL)
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        results = [[1, 2, 3], [4, 5, 6]]  # 1-indexed versions
        recall = compute_recall(results, gt, k=3)
        assert recall == pytest.approx(1.0)

    def test_zero_recall(self):
        """When none of the returned IDs match."""
        gt = np.array([[0, 1, 2]])
        results = [[10, 11, 12]]  # 1-indexed, none match
        recall = compute_recall(results, gt, k=3)
        assert recall == pytest.approx(0.0)

    def test_partial_recall(self):
        """When some IDs match."""
        gt = np.array([[0, 1, 2, 3]])
        results = [[1, 2, 10, 11]]  # matches 0 and 1
        recall = compute_recall(results, gt, k=4)
        assert recall == pytest.approx(0.5)

    def test_empty_results(self):
        """When no results returned."""
        gt = np.array([[0, 1, 2]])
        results = [[]]
        recall = compute_recall(results, gt, k=3)
        assert recall == pytest.approx(0.0)


class TestLatencyPercentiles:
    def test_basic_percentiles(self):
        latencies = list(range(1, 101))  # 1 to 100
        p = compute_latency_percentiles(latencies)
        assert p["p50"] == pytest.approx(50.5)
        assert p["p95"] == pytest.approx(95.05, abs=1)
        assert p["mean"] == pytest.approx(50.5)

    def test_single_value(self):
        p = compute_latency_percentiles([5.0])
        assert p["p50"] == pytest.approx(5.0)
        assert p["p95"] == pytest.approx(5.0)
        assert p["mean"] == pytest.approx(5.0)


class TestCompletionRate:
    def test_all_complete(self):
        results = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        rate = compute_topk_completion_rate(results, k=3)
        assert rate == pytest.approx(1.0)

    def test_none_complete(self):
        results = [[1], [2], [3]]
        rate = compute_topk_completion_rate(results, k=3)
        assert rate == pytest.approx(0.0)

    def test_partial_complete(self):
        results = [[1, 2, 3], [4], [7, 8, 9]]
        rate = compute_topk_completion_rate(results, k=3)
        assert rate == pytest.approx(2 / 3)

    def test_empty(self):
        rate = compute_topk_completion_rate([], k=3)
        assert rate == pytest.approx(0.0)


class TestFilterMask:
    def test_equality_filter(self):
        attrs = {"category_10": np.array([0, 1, 2, 0, 1])}
        mask = build_filter_mask(attrs, "category_10", 0)
        np.testing.assert_array_equal(mask, [True, False, False, True, False])

    def test_range_filter(self):
        attrs = {"price": np.array([10.0, 50.0, 100.0, 200.0])}
        mask = build_filter_mask(attrs, "price", 100.0, "<")
        np.testing.assert_array_equal(mask, [True, True, False, False])


class TestWorkloadGeneration:
    def test_pure_queries(self):
        vecs = np.random.rand(10, 4).astype(np.float32)
        queries = generate_pure_queries(vecs, "test_table", k=5, num_queries=3)
        assert len(queries) == 3
        assert "LIMIT" in queries[0].sql
        assert "WHERE" not in queries[0].sql

    def test_filtered_queries(self):
        vecs = np.random.rand(10, 4).astype(np.float32)
        mask = np.array([True] * 5 + [False] * 5)
        queries = generate_filtered_queries(
            vecs, "test_table", k=5, filter_column="cat", filter_value=1,
            filter_mask=mask, num_queries=3,
        )
        assert len(queries) == 3
        assert "WHERE" in queries[0].sql
        assert queries[0].filter_mask is not None
