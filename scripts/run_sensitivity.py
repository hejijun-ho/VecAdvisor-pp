#!/usr/bin/env python3
"""Parameter sensitivity analysis for VecAdvisor++.

Sweeps individual HNSW and IVFFlat parameters across a range of values
at each selectivity level, producing Pareto curves of recall vs latency.
The advisor's chosen value is marked on each curve.

Fixed configuration: SIFT1M, 1M vectors, k=10, warm-cache.

Usage:
    python scripts/run_sensitivity.py --config config/remote.yaml \
        --n-base 1000000 --n-queries 500 --num-runs 3 \
        --output-dir results/sensitivity
"""

import argparse
import json
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.advisor.advisor import VecAdvisor
from src.benchmark.runner import BenchmarkRunner
from src.benchmark.workload import build_filter_mask, generate_filtered_queries, generate_pure_queries
from src.data.loader import download_sift1m, load_sift1m, load_subset
from src.data.schema import IndexConfig, create_vector_table, generate_synthetic_attributes, get_connection, insert_vectors
from src.evaluation.visualize import plot_pareto_sweep
from src.ground_truth.compute import compute_filtered_ground_truth, compute_ground_truth
from src.profiler.workload_profiler import profile_from_params

# Sweep values
HNSW_EF_SEARCH_VALUES = [10, 20, 40, 80, 160, 320, 500, 1000]
IVFFLAT_PROBES_VALUES  = [1, 5, 10, 20, 50, 100, 200, 316, 500, 1000]
IVFFLAT_LISTS_VALUES   = [100, 250, 500, 1000, 2000, 4000]

# Selectivity configs to test (column, value, label, selectivity)
SELECTIVITY_CONFIGS = [
    (None,          None, "pure",  0.0),
    ("category_10",   0,  "10pct", 0.10),
    ("category_100",  0,  "1pct",  0.01),
    ("category_1000", 0,  "0.1pct",0.001),
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="VecAdvisor++ Parameter Sensitivity")
    parser.add_argument("--config", default="config/remote.yaml")
    parser.add_argument("--n-base", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--table", default="vectors_sensitivity")
    parser.add_argument("--output-dir", default="results/sensitivity")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--start-sweep", type=int, default=1, choices=[1, 2, 3],
                        help="Resume from this sweep number (1=HNSW ef_search, 2=IVFFlat probes, 3=IVFFlat lists)")
    args = parser.parse_args()

    config = load_config(args.config)
    conn_params = config["database"]
    data_dir = config["dataset"]["data_dir"]
    batch_size = config.get("benchmark", {}).get("batch_size", 1000)
    k = args.k

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    print(f"Loading SIFT1M (n={args.n_base:,}) ...")
    download_sift1m(data_dir)
    base_all, query_all, _ = load_sift1m(data_dir)
    base, queries = load_subset(base_all, query_all, args.n_base, args.n_queries)
    attrs = generate_synthetic_attributes(len(base))

    if not args.skip_load:
        conn = get_connection(conn_params)
        try:
            create_vector_table(conn, args.table, base.shape[1], with_attributes=True)
            insert_vectors(conn, args.table, base, attrs, batch_size=batch_size)
            print(f"Inserted {len(base):,} vectors into '{args.table}'")
        finally:
            conn.close()

    runner = BenchmarkRunner(conn_params)
    advisor = VecAdvisor()

    # Load any previously saved records so incremental saves don't lose old sweeps
    out_file = os.path.join(args.output_dir, "sensitivity_results.json")
    if os.path.exists(out_file):
        with open(out_file) as f:
            all_records: list = json.load(f)
        print(f"Loaded {len(all_records)} existing records from {out_file}")
    else:
        all_records = []

    def _save_records() -> None:
        with open(out_file, "w") as f:
            json.dump(all_records, f, indent=2)

    # ------------------------------------------------------------------
    # Pre-compute ground truth for each selectivity
    # ------------------------------------------------------------------
    gt_cache: dict[str, object] = {}
    query_cache: dict[str, list] = {}
    mask_cache: dict[str, object] = {}

    for col, val, label, sel in SELECTIVITY_CONFIGS:
        if col is None:
            _, gt_cache[label] = compute_ground_truth(base, queries, k)
            query_cache[label] = generate_pure_queries(queries, args.table, k, args.n_queries)
        else:
            mask = build_filter_mask(attrs, col, val)
            mask_cache[label] = mask
            _, gt_cache[label] = compute_filtered_ground_truth(base, queries, mask, k)
            query_cache[label] = generate_filtered_queries(
                queries, args.table, k, col, val,
                filter_mask=mask, num_queries=args.n_queries,
            )

    def _get_advisor_value(index_type: str, param_name: str, sel: float, col) -> float | None:
        """Ask the advisor what value it would choose for this knob."""
        has_filters = col is not None
        profile = profile_from_params(
            n_vectors=len(base), dim=base.shape[1], k=k,
            has_filters=has_filters, filter_selectivity=sel,
            filter_columns=[col] if col else [],
        )
        rec = advisor.analyze(profile)
        if index_type == "hnsw" and param_name == "ef_search":
            return rec.query_params.get("ef_search")
        if index_type == "ivfflat" and param_name == "probes":
            return rec.query_params.get("probes")
        if index_type == "ivfflat" and param_name == "lists":
            return rec.build_params.get("lists")
        return None

    ivfflat_lists_fixed = max(10, min(int(math.sqrt(len(base))), 1000))

    # ------------------------------------------------------------------
    # Sweep 1: HNSW ef_search
    # ------------------------------------------------------------------
    if args.start_sweep > 1:
        print("\n=== Sweep 1: HNSW ef_search (SKIPPED) ===")
    else:
        print("\n=== Sweep 1: HNSW ef_search ===")
        hnsw_index = IndexConfig("hnsw", {"m": 16, "ef_construction": 128})

        for col, val, label, sel in SELECTIVITY_CONFIGS:
            print(f"\n  Selectivity: {label}")
            sweep_data = []
            gt = gt_cache[label]
            qs = query_cache[label]

            for ef in HNSW_EF_SEARCH_VALUES:
                result = runner.run_full_benchmark(
                    args.table, qs, hnsw_index,
                    query_params={"ef_search": ef},
                    ground_truth_ids=gt, k=k,
                    config_name=f"hnsw_ef{ef}",
                    cache_mode="warm", num_runs=args.num_runs,
                    filter_selectivity=sel if col else None,
                )
                sweep_data.append((float(ef), result))
                all_records.append({
                    "sweep": "hnsw_ef_search", "selectivity": label,
                    "param_value": ef, "recall": result.recall, "recall_std": result.recall_std,
                    "latency_p95_ms": result.latency_p95_ms, "latency_p95_std": result.latency_p95_std,
                })
                print(f"    ef_search={ef:5d}  recall={result.recall:.4f}  p95={result.latency_p95_ms:.2f}ms")

            advisor_val = _get_advisor_value("hnsw", "ef_search", sel, col)
            p = plot_pareto_sweep(
                sweep_data, param_name="ef_search",
                advisor_value=advisor_val,
                output_dir=os.path.join(args.output_dir, "plots"),
                filename=f"hnsw_ef_search_{label}.png",
            )
            if p:
                print(f"    Saved: {p}")
        _save_records()
        print(f"  Sweep 1 records saved to {out_file}")

    # ------------------------------------------------------------------
    # Sweep 2: IVFFlat probes  (fixed lists = sqrt(n))
    # ------------------------------------------------------------------
    if args.start_sweep > 2:
        print(f"\n=== Sweep 2: IVFFlat probes (SKIPPED) ===")
    else:
        print(f"\n=== Sweep 2: IVFFlat probes (lists={ivfflat_lists_fixed}) ===")
        ivfflat_index = IndexConfig("ivfflat", {"lists": ivfflat_lists_fixed})

        for col, val, label, sel in SELECTIVITY_CONFIGS:
            print(f"\n  Selectivity: {label}")
            sweep_data = []
            gt = gt_cache[label]
            qs = query_cache[label]

            for probes in IVFFLAT_PROBES_VALUES:
                probes_capped = min(probes, ivfflat_lists_fixed)
                result = runner.run_full_benchmark(
                    args.table, qs, ivfflat_index,
                    query_params={"probes": probes_capped},
                    ground_truth_ids=gt, k=k,
                    config_name=f"ivf_probes{probes_capped}",
                    cache_mode="warm", num_runs=args.num_runs,
                    filter_selectivity=sel if col else None,
                )
                sweep_data.append((float(probes_capped), result))
                all_records.append({
                    "sweep": "ivfflat_probes", "selectivity": label,
                    "param_value": probes_capped, "recall": result.recall, "recall_std": result.recall_std,
                    "latency_p95_ms": result.latency_p95_ms, "latency_p95_std": result.latency_p95_std,
                })
                print(f"    probes={probes_capped:5d}  recall={result.recall:.4f}  p95={result.latency_p95_ms:.2f}ms")

            advisor_val = _get_advisor_value("ivfflat", "probes", sel, col)
            p = plot_pareto_sweep(
                sweep_data, param_name="probes",
                advisor_value=advisor_val,
                output_dir=os.path.join(args.output_dir, "plots"),
                filename=f"ivfflat_probes_{label}.png",
            )
            if p:
                print(f"    Saved: {p}")
        _save_records()
        print(f"  Sweep 2 records saved to {out_file}")

    # ------------------------------------------------------------------
    # Sweep 3: IVFFlat lists  (probes = sqrt(lists) each time)
    # (Uses pure queries only — the lists parameter doesn't interact
    #  with filter selectivity in the same way probes does)
    # ------------------------------------------------------------------
    print("\n=== Sweep 3: IVFFlat lists (probes = sqrt(lists)) ===")
    gt = gt_cache["pure"]
    qs = query_cache["pure"]
    sweep_data = []

    for lists in IVFFLAT_LISTS_VALUES:
        probes = max(1, int(math.sqrt(lists)))
        idx_cfg = IndexConfig("ivfflat", {"lists": lists})
        result = runner.run_full_benchmark(
            args.table, qs, idx_cfg,
            query_params={"probes": probes},
            ground_truth_ids=gt, k=k,
            config_name=f"ivf_lists{lists}",
            cache_mode="warm", num_runs=args.num_runs,
        )
        sweep_data.append((float(lists), result))
        all_records.append({
            "sweep": "ivfflat_lists", "selectivity": "pure",
            "param_value": lists, "recall": result.recall, "recall_std": result.recall_std,
            "latency_p95_ms": result.latency_p95_ms, "latency_p95_std": result.latency_p95_std,
        })
        print(f"  lists={lists:5d}  probes={probes:4d}  recall={result.recall:.4f}  p95={result.latency_p95_ms:.2f}ms")

    advisor_val = _get_advisor_value("ivfflat", "lists", 0.0, None)
    p = plot_pareto_sweep(
        sweep_data, param_name="lists",
        advisor_value=advisor_val,
        output_dir=os.path.join(args.output_dir, "plots"),
        filename="ivfflat_lists_pure.png",
    )
    if p:
        print(f"  Saved: {p}")

    # ------------------------------------------------------------------
    # Save all records (final save after Sweep 3)
    # ------------------------------------------------------------------
    _save_records()
    print(f"\nAll sensitivity results saved to {out_file}")


if __name__ == "__main__":
    main()
