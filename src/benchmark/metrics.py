"""Benchmark metric computation for VecAdvisor++.

Computes recall@k, latency percentiles, top-k completion rate,
and other evaluation metrics.
"""

from dataclasses import dataclass

import numpy as np

try:
    import vecadvisor_rs as _rs
    # Guard against the vecadvisor_rs/ source directory being imported as a
    # namespace package (e.g. on servers where the .so hasn't been built yet).
    _RUST_AVAILABLE = hasattr(_rs, "compute_recall")
except ImportError:
    _RUST_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a single configuration.

    When num_runs > 1, the primary fields (recall, latency_*, completion_rate)
    hold the mean across runs and the *_std fields hold the standard deviation.
    """

    config_name: str
    index_type: str
    index_params: dict
    query_params: dict
    recall: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    build_time_s: float
    memory_mb: float
    disk_mb: float
    completion_rate: float
    num_queries: int
    k: int
    filter_selectivity: float | None = None
    # Multi-run standard deviations (0.0 when num_runs == 1)
    recall_std: float = 0.0
    latency_p50_std: float = 0.0
    latency_p95_std: float = 0.0
    latency_p99_std: float = 0.0
    latency_mean_std: float = 0.0
    completion_rate_std: float = 0.0
    num_runs: int = 1


def compute_recall(
    result_ids: list[list[int]],
    ground_truth_ids: np.ndarray,
    k: int,
) -> float:
    """Compute recall@k: fraction of true top-k neighbors found.

    Args:
        result_ids: List of lists of returned IDs (1-indexed from PostgreSQL).
        ground_truth_ids: (nq, k) array of true neighbor IDs (0-indexed).
        k: Number of neighbors to evaluate.

    Returns:
        Mean recall@k across all queries (0.0 to 1.0).
    """
    if _RUST_AVAILABLE:
        return _rs.compute_recall(
            result_ids,
            np.asarray(ground_truth_ids, dtype=np.int64),
            k,
        )
    total_recall = 0.0
    n_queries = len(result_ids)

    for i in range(n_queries):
        # PostgreSQL IDs are 1-indexed, ground truth is 0-indexed
        # Convert result IDs to 0-indexed: subtract 1
        returned = set(rid - 1 for rid in result_ids[i][:k])
        true_nn = set(ground_truth_ids[i][:k])
        # Remove -1 padding from ground truth
        true_nn.discard(-1)

        if len(true_nn) == 0:
            total_recall += 1.0  # No valid neighbors to find
        else:
            total_recall += len(returned & true_nn) / len(true_nn)

    return total_recall / n_queries if n_queries > 0 else 0.0


def compute_latency_percentiles(
    latencies_ms: list[float],
) -> dict[str, float]:
    """Compute latency percentiles from per-query measurements.

    Args:
        latencies_ms: List of per-query latencies in milliseconds.

    Returns:
        Dict with p50, p95, p99, and mean latency.
    """
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def compute_topk_completion_rate(
    result_ids: list[list[int]], k: int
) -> float:
    """Compute fraction of queries that returned exactly k results.

    This is critical for filtered queries where the index may not find
    enough candidates that pass the filter.

    Args:
        result_ids: List of lists of returned IDs per query.
        k: Expected number of results.

    Returns:
        Fraction of queries with exactly k results (0.0 to 1.0).
    """
    if not result_ids:
        return 0.0
    complete = sum(1 for ids in result_ids if len(ids) >= k)
    return complete / len(result_ids)
