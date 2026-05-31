#!/usr/bin/env python3
"""VecAdvisor++ Interactive Demo

Demonstrates the advisor's filter-aware index selection across
multiple workload scenarios. No database connection required.

Usage:
    source venv/bin/activate
    python scripts/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.advisor.advisor import VecAdvisor


SCENARIOS = [
    {
        "name": "Scenario 1: Pure Vector Search (1M vectors, 128-dim)",
        "desc": "Standard ANN workload, no filters. Typical for embedding retrieval.",
        "params": dict(n_vectors=1_000_000, dim=128, k=10),
    },
    {
        "name": "Scenario 2: Moderate Filter (10% selectivity, 1M vectors)",
        "desc": "Filtered query with broad filter. E.g., 'category = electronics'.",
        "params": dict(
            n_vectors=1_000_000, dim=128, k=10,
            has_filters=True, filter_selectivity=0.10,
            filter_columns=["category_10"],
        ),
    },
    {
        "name": "Scenario 3: Selective Filter (1% selectivity, 1M vectors)",
        "desc": "Narrow filter. E.g., 'subcategory = vintage_cameras'. "
                "This is where HNSW defaults catastrophically fail.",
        "params": dict(
            n_vectors=1_000_000, dim=128, k=10,
            has_filters=True, filter_selectivity=0.01,
            filter_columns=["category_100"],
        ),
    },
    {
        "name": "Scenario 4: Very Selective Filter (0.1%, 1M vectors)",
        "desc": "Highly selective filter. E.g., 'sku = ABC123'. "
                "Requires maximum probe/search amplification.",
        "params": dict(
            n_vectors=1_000_000, dim=128, k=10,
            has_filters=True, filter_selectivity=0.001,
            filter_columns=["category_1000"],
        ),
    },
    {
        "name": "Scenario 5: High-Dimensional Vectors (512-dim)",
        "desc": "Text embeddings (e.g., sentence-transformers). "
                "Higher dimensionality requires more graph connections.",
        "params": dict(n_vectors=500_000, dim=512, k=10),
    },
    {
        "name": "Scenario 6: Small Dataset (5K vectors)",
        "desc": "Small collection. Sequential scan is faster than maintaining an index.",
        "params": dict(n_vectors=5_000, dim=128, k=10),
    },
    {
        "name": "Scenario 7: Write-Heavy Workload (30% updates)",
        "desc": "Frequent inserts/updates. Needs an index that is cheap to rebuild.",
        "params": dict(
            n_vectors=500_000, dim=128, k=10,
            update_rate=0.3,
        ),
    },
    {
        "name": "Scenario 8: Memory-Constrained (512 MB budget, 1M vectors)",
        "desc": "Limited memory for indexes. Must choose the smaller index type.",
        "params": dict(
            n_vectors=1_000_000, dim=128, k=10,
            memory_budget_mb=512,
        ),
    },
    {
        "name": "Scenario 9: Latency-Critical Warm Cache (< 5ms target)",
        "desc": "Real-time serving requirement. Needs the fastest query path.",
        "params": dict(
            n_vectors=500_000, dim=128, k=10,
            latency_budget_ms=5.0, cache_regime="warm",
        ),
    },
]


def main():
    print()
    print("=" * 70)
    print("  VecAdvisor++: Filter-Aware Vector Index Advisor for PostgreSQL")
    print("=" * 70)
    print()
    print("This demo shows how VecAdvisor++ adapts its index recommendations")
    print("based on workload characteristics. Each scenario demonstrates a")
    print("different real-world use case.")
    print()
    print(f"Total scenarios: {len(SCENARIOS)}")
    print()

    advisor = VecAdvisor()

    summary_rows = []

    for i, scenario in enumerate(SCENARIOS):
        print("-" * 70)
        print(f"\n{scenario['name']}")
        print(f"  {scenario['desc']}")
        print()

        rec = advisor.analyze_from_params(**scenario["params"])

        # Print reasoning
        print("  Advisor Reasoning:")
        for j, exp in enumerate(rec.explanation, 1):
            print(f"    {j}. {exp}")

        # Print recommendation summary
        print()
        print(f"  >>> Index Type: {rec.index_type.upper()}")
        if rec.build_params:
            print(f"  >>> Build Params: {rec.build_params}")
        if rec.query_params:
            print(f"  >>> Query Params: {rec.query_params}")

        # Print SQL
        if rec.index_type != "none":
            sql = advisor.get_sql(rec, "vectors")
            sql_lines = [l for l in sql.split("\n") if l.strip() and not l.startswith("--")]
            print()
            print("  SQL:")
            for line in sql_lines:
                print(f"    {line}")

        if rec.auxiliary_indexes:
            print()
            print("  Auxiliary Indexes:")
            for aux in rec.auxiliary_indexes:
                print(f"    - B-tree on '{aux['column']}'")

        summary_rows.append({
            "scenario": scenario["name"].split(":")[0],
            "index": rec.index_type.upper(),
            "build": str(rec.build_params) if rec.build_params else "-",
            "query": str(rec.query_params) if rec.query_params else "-",
        })
        print()

    # Print summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Scenario':<14} {'Index':<9} {'Build Params':<35} {'Query Params'}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['scenario']:<14} {row['index']:<9} {row['build']:<35} {row['query']}")

    print()
    print("=" * 70)
    print("KEY INSIGHT: The advisor switches from HNSW to IVFFlat for filtered")
    print("queries with selectivity <= 10% on datasets >= 50K vectors. This is")
    print("because HNSW graph traversal is blind to attribute filters and can")
    print("achieve as low as 4% recall at 1% selectivity, while IVFFlat with")
    print("increased probes achieves 99.7% recall.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
