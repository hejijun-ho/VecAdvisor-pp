#!/usr/bin/env python3
"""CLI entry point for VecAdvisor++ benchmark suite.

Runs end-to-end benchmarks comparing baseline configurations
against VecAdvisor++ recommendations across multiple selectivity
levels and k values.

Usage:
    # Local quick test
    python scripts/run_benchmark.py --config config/default.yaml \
        --n-base 10000 --n-queries 100 --k 10

    # Full SIFT1M on remote server (5 runs for error bars)
    python scripts/run_benchmark.py --config config/remote.yaml \
        --n-base 1000000 --n-queries 1000 --full-sweep --num-runs 5 \
        --dataset sift1m --output-dir results/sift1m_full

    # GIST1M on remote server
    python scripts/run_benchmark.py --config config/remote.yaml \
        --n-base 1000000 --n-queries 1000 --full-sweep --num-runs 5 \
        --dataset gist1m --output-dir results/gist1m_full
"""

import argparse
import json
import os
import sys

# Limit Faiss/OpenBLAS thread count to avoid resource spikes that can kill
# SSH sessions on shared servers. The default (all cores) causes issues on
# 32-core machines. Set before importing numpy/faiss.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from src.data.loader import (
    download_gist1m,
    download_sift1m,
    load_gist1m,
    load_sift1m,
    load_subset,
)
from src.data.schema import (
    create_vector_table,
    generate_synthetic_attributes,
    get_connection,
    insert_vectors,
)
from src.ground_truth.compute import (
    compute_filtered_ground_truth,
    compute_ground_truth,
)
from src.benchmark.workload import (
    build_filter_mask,
    generate_filtered_queries,
    generate_pure_queries,
)
from src.evaluation.compare import (
    run_comparison,
    summarize_comparison,
)
from src.evaluation.visualize import (
    generate_all_plots,
    plot_selectivity_heatmap,
)
from src.profiler.workload_profiler import profile_from_params


# Selectivity configurations: (column, filter_value, label, approx_selectivity)
SELECTIVITY_CONFIGS = [
    ("category_10",   0, "~10%",  0.10),
    ("category_100",  0, "~1%",   0.01),
    ("category_1000", 0, "~0.1%", 0.001),
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _result_to_dict(r) -> dict:
    return {
        "config_name": r.config_name,
        "index_type": r.index_type,
        "index_params": r.index_params,
        "query_params": r.query_params,
        "recall": r.recall,
        "recall_std": r.recall_std,
        "latency_p50_ms": r.latency_p50_ms,
        "latency_p50_std": r.latency_p50_std,
        "latency_p95_ms": r.latency_p95_ms,
        "latency_p95_std": r.latency_p95_std,
        "latency_p99_ms": r.latency_p99_ms,
        "latency_p99_std": r.latency_p99_std,
        "latency_mean_ms": r.latency_mean_ms,
        "latency_mean_std": r.latency_mean_std,
        "build_time_s": r.build_time_s,
        "memory_mb": r.memory_mb,
        "disk_mb": r.disk_mb,
        "completion_rate": r.completion_rate,
        "completion_rate_std": r.completion_rate_std,
        "num_queries": r.num_queries,
        "num_runs": r.num_runs,
        "k": r.k,
        "filter_selectivity": r.filter_selectivity,
    }


def run_selectivity_benchmark(
    conn_params, table_name, query_vectors, base_vectors,
    attributes, k, n_queries, cache_mode, num_runs,
    col, val, label, sel,
):
    """Run a single selectivity-level benchmark."""
    mask = build_filter_mask(attributes, col, val)
    actual_sel = mask.sum() / len(mask)
    print(f"\n  Filter: {col} = {val} | Label: {label} | "
          f"Actual selectivity: {actual_sel:.2%} ({mask.sum()}/{len(mask)})")

    _, gt_ids = compute_filtered_ground_truth(base_vectors, query_vectors, mask, k)

    queries = generate_filtered_queries(
        query_vectors, table_name, k, col, val,
        filter_mask=mask, num_queries=n_queries,
    )

    profile = profile_from_params(
        n_vectors=len(base_vectors), dim=base_vectors.shape[1],
        k=k, has_filters=True,
        filter_selectivity=sel, filter_columns=[col],
    )

    results = run_comparison(
        conn_params, table_name, queries, gt_ids,
        k, profile, cache_mode=cache_mode,
        filter_selectivity=sel, num_runs=num_runs,
    )
    print(summarize_comparison(results))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="VecAdvisor++ Benchmark Suite"
    )
    parser.add_argument("--config", default="config/default.yaml",
                        help="Config file path")
    parser.add_argument("--dataset", choices=["sift1m", "gist1m"], default="sift1m",
                        help="Dataset to use (default: sift1m)")
    parser.add_argument("--n-base", type=int, default=10000,
                        help="Number of base vectors to use")
    parser.add_argument("--n-queries", type=int, default=100,
                        help="Number of query vectors to use")
    parser.add_argument("--k", type=int, default=10,
                        help="Top-k parameter (used when --full-sweep is off)")
    parser.add_argument("--table", default="vectors",
                        help="Table name")
    parser.add_argument("--cache-mode", choices=["cold", "warm"],
                        default="warm", help="Cache mode")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of query-phase repetitions per config (for error bars)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip data loading (use existing table)")
    parser.add_argument("--full-sweep", action="store_true",
                        help="Run full sweep across k values and selectivities")

    args = parser.parse_args()
    config = load_config(args.config)
    conn_params = config["database"]

    # num_runs: CLI flag takes precedence; fall back to config file value
    num_runs = args.num_runs
    if num_runs == 1 and config.get("benchmark", {}).get("num_runs", 1) > 1:
        num_runs = config["benchmark"]["num_runs"]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    if args.full_sweep:
        k_values = config["benchmark"].get("k_values", [1, 10, 50, 100])
    else:
        k_values = [args.k]

    # ================================================================
    # Step 1: Load dataset
    # ================================================================
    print("=" * 60)
    print(f"Step 1: Loading {args.dataset.upper()} dataset")
    print("=" * 60)

    dataset_cfg = config.get("dataset", {})

    if args.dataset == "sift1m":
        data_dir = dataset_cfg.get("data_dir", "data/sift1m")
        download_sift1m(data_dir)
        base_vectors, query_vectors, _ = load_sift1m(data_dir)
    elif args.dataset == "gist1m":
        # Allow overriding data_dir via config; default to sibling gist1m dir
        default_gist_dir = os.path.join(
            os.path.dirname(dataset_cfg.get("data_dir", "data/sift1m")), "gist1m"
        )
        data_dir = dataset_cfg.get("gist_data_dir", default_gist_dir)
        download_gist1m(data_dir)
        base_vectors, query_vectors, _ = load_gist1m(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    base_vectors, query_vectors = load_subset(
        base_vectors, query_vectors, args.n_base, args.n_queries
    )
    print(f"Using {len(base_vectors)} base vectors ({base_vectors.shape[1]}-dim), "
          f"{len(query_vectors)} queries")

    # ================================================================
    # Step 2: Load into PostgreSQL
    # ================================================================
    if not args.skip_load:
        print("\n" + "=" * 60)
        print("Step 2: Loading data into PostgreSQL")
        print("=" * 60)
        dim = base_vectors.shape[1]
        conn = get_connection(conn_params)
        try:
            create_vector_table(conn, args.table, dim, with_attributes=True)
            attributes = generate_synthetic_attributes(len(base_vectors))
            batch_size = config.get("benchmark", {}).get("batch_size", 1000)
            insert_vectors(conn, args.table, base_vectors, attributes,
                           batch_size=batch_size)
            print(f"Inserted {len(base_vectors)} vectors into '{args.table}'")
        finally:
            conn.close()
    else:
        attributes = generate_synthetic_attributes(len(base_vectors))

    if num_runs > 1:
        print(f"\n(Each config will run {num_runs}x for error bars)")

    all_results = []
    results_by_selectivity = {}

    for k in k_values:
        print("\n" + "#" * 60)
        print(f"# Benchmarking with k={k}")
        print("#" * 60)

        # ============================================================
        # Step 3: Ground truth
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Step 3: Computing ground truth (k={k})")
        print("=" * 60)
        _, gt_pure_ids = compute_ground_truth(base_vectors, query_vectors, k)
        print(f"  Pure ground truth: {gt_pure_ids.shape}")

        # ============================================================
        # Step 4: Pure query benchmark
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Step 4: Pure query benchmark (k={k})")
        print("=" * 60)
        pure_queries = generate_pure_queries(
            query_vectors, args.table, k, args.n_queries
        )
        pure_profile = profile_from_params(
            n_vectors=len(base_vectors), dim=base_vectors.shape[1],
            k=k, has_filters=False,
        )
        pure_results = run_comparison(
            conn_params, args.table, pure_queries, gt_pure_ids,
            k, pure_profile, cache_mode=args.cache_mode, num_runs=num_runs,
        )
        all_results.extend(pure_results)
        results_by_selectivity[f"pure_k{k}"] = pure_results
        print(summarize_comparison(pure_results))

        # ============================================================
        # Step 5: Filtered query benchmarks
        # ============================================================
        print("\n" + "=" * 60)
        print(f"Step 5: Filtered query benchmarks (k={k})")
        print("=" * 60)

        for col, val, label, sel in SELECTIVITY_CONFIGS:
            results = run_selectivity_benchmark(
                conn_params, args.table, query_vectors, base_vectors,
                attributes, k, args.n_queries, args.cache_mode, num_runs,
                col, val, label, sel,
            )
            all_results.extend(results)
            results_by_selectivity[f"sel_{label}_k{k}"] = results

    # ================================================================
    # Step 6: Generate plots
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 6: Generating plots")
    print("=" * 60)
    plot_dir = os.path.join(args.output_dir, "plots")
    plots = generate_all_plots(all_results, plot_dir)
    for p in plots:
        print(f"  Saved: {p}")

    filtered_sel_keys = {
        k: v for k, v in results_by_selectivity.items()
        if k.startswith("sel_")
    }
    if filtered_sel_keys:
        for metric in ["recall", "completion_rate"]:
            p = plot_selectivity_heatmap(
                filtered_sel_keys, metric=metric, output_dir=plot_dir,
                filename=f"selectivity_{metric}_heatmap.png",
            )
            if p:
                print(f"  Saved: {p}")

    # ================================================================
    # Step 7: Save results (includes stddev fields)
    # ================================================================
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump([_result_to_dict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_file}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}, n={len(base_vectors)}, "
          f"dim={base_vectors.shape[1]}")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"K values: {k_values}")
    print(f"Num runs per config: {num_runs}")
    print(f"Selectivity levels: pure, {', '.join(s[2] for s in SELECTIVITY_CONFIGS)}")
    print(f"Results: {results_file}")
    print(f"Plots: {plot_dir}/")


if __name__ == "__main__":
    main()
