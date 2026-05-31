"""Comparison framework for evaluating VecAdvisor++ against baselines.

Defines baseline configurations and runs side-by-side benchmarks.
"""

from dataclasses import dataclass

import numpy as np

from src.advisor.advisor import VecAdvisor
from src.benchmark.metrics import BenchmarkResult
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.workload import Query
from src.data.schema import IndexConfig
from src.profiler.workload_profiler import WorkloadProfile


@dataclass
class BaselineConfig:
    """A baseline index configuration for comparison."""

    name: str
    index_config: IndexConfig
    query_params: dict


def get_default_baselines(n_vectors: int = 1_000_000) -> list[BaselineConfig]:
    """Get the standard baseline configurations.

    Args:
        n_vectors: Dataset size (used to set IVFFlat lists appropriately).

    Returns:
        List of baseline configs:
        1. pgvector_default      — HNSW with pgvector defaults (m=16, ef_search=40)
        2. heuristic_hnsw        — HNSW with common hand-tuned settings
        3. hnsw_aggressive       — HNSW with very high ef_search (shows HNSW ceiling)
        4. pgvector_ivfflat_default — IVFFlat with sqrt(n) lists, sqrt(lists) probes
        5. ivfflat_full_probes   — IVFFlat scanning all partitions (IVFFlat ceiling)
        6. sequential_scan       — No index, brute-force sequential scan (recall upper bound)
    """
    import math

    ivfflat_lists = max(10, min(int(math.sqrt(n_vectors)), 1000))
    ivfflat_probes = max(1, int(math.sqrt(ivfflat_lists)))

    return [
        BaselineConfig(
            name="pgvector_default",
            index_config=IndexConfig(
                index_type="hnsw",
                params={"m": 16, "ef_construction": 64},
            ),
            query_params={"ef_search": 40},
        ),
        BaselineConfig(
            name="heuristic_hnsw",
            index_config=IndexConfig(
                index_type="hnsw",
                params={"m": 16, "ef_construction": 128},
            ),
            query_params={"ef_search": 100},
        ),
        BaselineConfig(
            name="hnsw_aggressive",
            index_config=IndexConfig(
                index_type="hnsw",
                params={"m": 16, "ef_construction": 128},
            ),
            query_params={"ef_search": 500},
        ),
        BaselineConfig(
            name="pgvector_ivfflat_default",
            index_config=IndexConfig(
                index_type="ivfflat",
                params={"lists": ivfflat_lists},
            ),
            query_params={"probes": ivfflat_probes},
        ),
        BaselineConfig(
            name="ivfflat_full_probes",
            index_config=IndexConfig(
                index_type="ivfflat",
                params={"lists": ivfflat_lists},
            ),
            query_params={"probes": ivfflat_lists},
        ),
        BaselineConfig(
            name="sequential_scan",
            index_config=IndexConfig(
                index_type="none",
                params={},
            ),
            query_params={},
        ),
    ]


def run_comparison(
    conn_params: dict,
    table_name: str,
    queries: list[Query],
    ground_truth_ids: np.ndarray,
    k: int,
    profile: WorkloadProfile,
    baselines: list[BaselineConfig] | None = None,
    cache_mode: str = "warm",
    filter_selectivity: float | None = None,
    num_runs: int = 1,
) -> list[BenchmarkResult]:
    """Run comparison benchmarks for baselines and advisor recommendation.

    Args:
        conn_params: Database connection parameters.
        table_name: Table to benchmark.
        queries: Query workload.
        ground_truth_ids: Ground truth for recall.
        k: Top-k value.
        profile: WorkloadProfile for advisor.
        baselines: List of baseline configs (None = use defaults).
        cache_mode: "cold" or "warm".
        filter_selectivity: Selectivity for reporting.
        num_runs: Number of query-phase repetitions per config (for error bars).

    Returns:
        List of BenchmarkResults, last one is VecAdvisor++ recommendation.
    """
    if baselines is None:
        baselines = get_default_baselines(n_vectors=profile.n_vectors)

    runner = BenchmarkRunner(conn_params)
    results = []

    # Run baselines
    for baseline in baselines:
        print(f"\nRunning baseline: {baseline.name}")
        result = runner.run_full_benchmark(
            table_name=table_name,
            queries=queries,
            index_config=baseline.index_config,
            query_params=baseline.query_params,
            ground_truth_ids=ground_truth_ids,
            k=k,
            config_name=baseline.name,
            cache_mode=cache_mode,
            filter_selectivity=filter_selectivity,
            num_runs=num_runs,
        )
        results.append(result)
        std_str = f" ±{result.recall_std:.4f}" if result.num_runs > 1 else ""
        print(
            f"  Recall: {result.recall:.4f}{std_str}, "
            f"p95: {result.latency_p95_ms:.2f}ms, "
            f"Completion: {result.completion_rate:.2%}"
        )

    # Run VecAdvisor++ recommendation
    advisor = VecAdvisor()
    recommendation = advisor.analyze(profile)

    advisor_index_config = IndexConfig(
        index_type=recommendation.index_type,
        params=recommendation.build_params,
    )

    print(f"\nRunning VecAdvisor++ ({recommendation.index_type})")
    advisor_result = runner.run_full_benchmark(
        table_name=table_name,
        queries=queries,
        index_config=advisor_index_config,
        query_params=recommendation.query_params,
        ground_truth_ids=ground_truth_ids,
        k=k,
        config_name="vecadvisor++",
        cache_mode=cache_mode,
        filter_selectivity=filter_selectivity,
        num_runs=num_runs,
    )
    results.append(advisor_result)
    std_str = f" ±{advisor_result.recall_std:.4f}" if advisor_result.num_runs > 1 else ""
    print(
        f"  Recall: {advisor_result.recall:.4f}{std_str}, "
        f"p95: {advisor_result.latency_p95_ms:.2f}ms, "
        f"Completion: {advisor_result.completion_rate:.2%}"
    )

    return results


def summarize_comparison(results: list[BenchmarkResult]) -> str:
    """Generate a text summary of comparison results.

    Args:
        results: List of BenchmarkResults.

    Returns:
        Formatted summary string.
    """
    lines = [
        "",
        "=" * 90,
        f"{'Config':<25} {'Index':<8} {'Recall':>8} {'p50 ms':>8} "
        f"{'p95 ms':>8} {'Build s':>8} {'Idx MB':>8} {'Compl':>8}",
        "-" * 90,
    ]
    for r in results:
        lines.append(
            f"{r.config_name:<25} {r.index_type:<8} {r.recall:>8.4f} "
            f"{r.latency_p50_ms:>8.2f} {r.latency_p95_ms:>8.2f} "
            f"{r.build_time_s:>8.2f} {r.memory_mb:>8.1f} "
            f"{r.completion_rate:>7.1%}"
        )
    lines.append("=" * 90)

    # Highlight winner
    best_recall = max(r.recall for r in results)
    best_latency = min(r.latency_p95_ms for r in results)
    best_completion = max(r.completion_rate for r in results)

    winners = []
    advisor_result = next((r for r in results if r.config_name == "vecadvisor++"), None)
    if advisor_result:
        if advisor_result.recall == best_recall:
            winners.append("best recall")
        if advisor_result.latency_p95_ms == best_latency:
            winners.append("lowest p95 latency")
        if advisor_result.completion_rate == best_completion:
            winners.append("best completion rate")

    if winners:
        lines.append(f"\nVecAdvisor++ achieves: {', '.join(winners)}")
    else:
        lines.append("\nVecAdvisor++ did not lead any metric in this workload.")

    return "\n".join(lines)
