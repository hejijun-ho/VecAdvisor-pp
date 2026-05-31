#!/usr/bin/env python3
"""CLI entry point for VecAdvisor++ advisor.

Generates vector index recommendations for PostgreSQL with pgvector.

Usage:
    # From parameters (no DB connection needed):
    python scripts/run_advisor.py --n 1000000 --dim 128 --k 10 \
        --filter-selectivity 0.01 --filter-columns category_100

    # From a live table:
    python scripts/run_advisor.py --table vectors --k 10 \
        --filter-columns category_100 --filter-clauses "category_100 = 0"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.advisor.advisor import VecAdvisor


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="VecAdvisor++: Vector index recommendation for PostgreSQL"
    )
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to config file")

    # Parameter-based mode
    parser.add_argument("--n", type=int, help="Number of vectors")
    parser.add_argument("--dim", type=int, help="Vector dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Top-k parameter")
    parser.add_argument("--filter-selectivity", type=float, default=1.0,
                        help="Filter selectivity (0.0-1.0)")
    parser.add_argument("--filter-columns", nargs="+", default=None,
                        help="Filter column names")
    parser.add_argument("--update-rate", type=float, default=0.0,
                        help="Write operation fraction (0.0-1.0)")
    parser.add_argument("--memory-budget", type=int, default=2048,
                        help="Memory budget in MB")
    parser.add_argument("--latency-budget", type=float, default=50.0,
                        help="Target p95 latency in ms")
    parser.add_argument("--cache-regime", choices=["cold", "warm"],
                        default="warm", help="Cache regime")

    # DB-based mode
    parser.add_argument("--table", type=str, help="Table name (DB mode)")
    parser.add_argument("--filter-clauses", nargs="+", default=None,
                        help="WHERE clauses for selectivity estimation")
    parser.add_argument("--output-table", type=str, default="vectors",
                        help="Table name for SQL output")

    args = parser.parse_args()

    advisor = VecAdvisor()

    if args.table:
        # DB-based mode
        config = load_config(args.config)
        from src.data.schema import get_connection
        conn = get_connection(config["database"])
        try:
            recommendation = advisor.analyze_from_db(
                conn=conn,
                table_name=args.table,
                k=args.k,
                filter_columns=args.filter_columns,
                filter_clauses=args.filter_clauses,
                update_rate=args.update_rate,
                memory_budget_mb=args.memory_budget,
                latency_budget_ms=args.latency_budget,
                cache_regime=args.cache_regime,
            )
        finally:
            conn.close()
        table_name = args.table
    elif args.n and args.dim:
        # Parameter-based mode
        has_filters = (
            args.filter_columns is not None or args.filter_selectivity < 1.0
        )
        recommendation = advisor.analyze_from_params(
            n_vectors=args.n,
            dim=args.dim,
            k=args.k,
            has_filters=has_filters,
            filter_selectivity=args.filter_selectivity,
            filter_columns=args.filter_columns,
            update_rate=args.update_rate,
            memory_budget_mb=args.memory_budget,
            latency_budget_ms=args.latency_budget,
            cache_regime=args.cache_regime,
        )
        table_name = args.output_table
    else:
        parser.error("Provide either --table (DB mode) or --n and --dim (parameter mode)")
        return

    advisor.print_recommendation(recommendation, table_name)


if __name__ == "__main__":
    main()
