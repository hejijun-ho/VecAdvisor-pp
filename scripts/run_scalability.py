#!/usr/bin/env python3
"""Scalability study for VecAdvisor++.

Benchmarks all configurations across a range of dataset sizes
(n_base = 10K → 1M) to show how performance scales with n.

Outputs one JSON file per scale and a combined scalability plot.

Usage:
    python scripts/run_scalability.py --config config/remote.yaml \
        --n-queries 500 --k 10 --num-runs 3 \
        --output-dir results/scalability
"""

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.loader import download_sift1m, load_sift1m, load_subset
from src.data.schema import (
    create_vector_table,
    generate_synthetic_attributes,
    get_connection,
    insert_vectors,
)
from src.ground_truth.compute import compute_filtered_ground_truth, compute_ground_truth
from src.benchmark.workload import build_filter_mask, generate_filtered_queries, generate_pure_queries
from src.evaluation.compare import run_comparison, summarize_comparison
from src.evaluation.visualize import plot_scalability
from src.profiler.workload_profiler import profile_from_params

# Sizes to sweep (vectors)
SCALE_VALUES = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

# Representative selectivity config for filtered benchmarks
FILTER_COL, FILTER_VAL, FILTER_LABEL, FILTER_SEL = "category_100", 0, "~1%", 0.01


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _result_to_dict(r, n: int) -> dict:
    return {
        "n_base": n,
        "config_name": r.config_name,
        "index_type": r.index_type,
        "recall": r.recall,
        "recall_std": r.recall_std,
        "latency_p95_ms": r.latency_p95_ms,
        "latency_p95_std": r.latency_p95_std,
        "build_time_s": r.build_time_s,
        "memory_mb": r.memory_mb,
        "disk_mb": r.disk_mb,
        "completion_rate": r.completion_rate,
        "completion_rate_std": r.completion_rate_std,
        "num_runs": r.num_runs,
        "k": r.k,
        "filter_selectivity": r.filter_selectivity,
    }


def main():
    parser = argparse.ArgumentParser(description="VecAdvisor++ Scalability Study")
    parser.add_argument("--config", default="config/remote.yaml")
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Query-phase repetitions per config")
    parser.add_argument("--table", default="vectors_scale")
    parser.add_argument("--output-dir", default="results/scalability")
    parser.add_argument("--scales", type=int, nargs="+", default=SCALE_VALUES,
                        help="List of n_base values to benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    conn_params = config["database"]
    data_dir = config["dataset"]["data_dir"]
    batch_size = config.get("benchmark", {}).get("batch_size", 1000)
    k = args.k
    num_runs = args.num_runs

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    # Load full SIFT1M once
    print("Loading SIFT1M dataset...")
    download_sift1m(data_dir)
    base_all, query_all, _ = load_sift1m(data_dir)

    all_records = []
    # results_by_n[n] = list[BenchmarkResult] for pure queries
    results_by_n_pure: dict[int, list] = {}
    results_by_n_filt: dict[int, list] = {}

    for n in sorted(args.scales):
        print("\n" + "=" * 60)
        print(f"Scale: n_base = {n:,}")
        print("=" * 60)

        base, queries = load_subset(base_all, query_all, n, args.n_queries)
        attrs = generate_synthetic_attributes(n)

        # Load into DB
        conn = get_connection(conn_params)
        try:
            create_vector_table(conn, args.table, base.shape[1], with_attributes=True)
            insert_vectors(conn, args.table, base, attrs, batch_size=batch_size)
        finally:
            conn.close()

        # -- Pure benchmark --
        _, gt_pure = compute_ground_truth(base, queries, k)
        pure_qs = generate_pure_queries(queries, args.table, k, args.n_queries)
        pure_profile = profile_from_params(n_vectors=n, dim=base.shape[1],
                                           k=k, has_filters=False)
        pure_results = run_comparison(
            conn_params, args.table, pure_qs, gt_pure,
            k, pure_profile, cache_mode="warm", num_runs=num_runs,
        )
        print(summarize_comparison(pure_results))
        results_by_n_pure[n] = pure_results
        for r in pure_results:
            all_records.append(_result_to_dict(r, n))

        # -- Filtered benchmark (1% selectivity) --
        mask = build_filter_mask(attrs, FILTER_COL, FILTER_VAL)
        _, gt_filt = compute_filtered_ground_truth(base, queries, mask, k)
        filt_qs = generate_filtered_queries(
            queries, args.table, k, FILTER_COL, FILTER_VAL,
            filter_mask=mask, num_queries=args.n_queries,
        )
        filt_profile = profile_from_params(
            n_vectors=n, dim=base.shape[1], k=k,
            has_filters=True, filter_selectivity=FILTER_SEL, filter_columns=[FILTER_COL],
        )
        filt_results = run_comparison(
            conn_params, args.table, filt_qs, gt_filt,
            k, filt_profile, cache_mode="warm",
            filter_selectivity=FILTER_SEL, num_runs=num_runs,
        )
        print(summarize_comparison(filt_results))
        results_by_n_filt[n] = filt_results
        for r in filt_results:
            all_records.append(_result_to_dict(r, n))

        # Save per-scale JSON
        scale_file = os.path.join(args.output_dir, f"scale_{n}.json")
        with open(scale_file, "w") as f:
            json.dump(
                [_result_to_dict(r, n) for r in pure_results + filt_results],
                f, indent=2,
            )
        print(f"  Saved: {scale_file}")

    # Save combined JSON
    combined_file = os.path.join(args.output_dir, "scalability_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nCombined results: {combined_file}")

    # Generate scalability plots
    plot_dir = os.path.join(args.output_dir, "plots")
    for metric, label, by_n in [
        ("recall", "pure", results_by_n_pure),
        ("latency_p95_ms", "pure_latency", results_by_n_pure),
        ("build_time_s", "pure_build", results_by_n_pure),
        ("recall", "filtered", results_by_n_filt),
        ("completion_rate", "filtered_completion", results_by_n_filt),
    ]:
        p = plot_scalability(
            by_n, metric=metric,
            output_dir=plot_dir,
            filename=f"scalability_{label}_{metric}.png",
        )
        if p:
            print(f"  Saved plot: {p}")


if __name__ == "__main__":
    main()
