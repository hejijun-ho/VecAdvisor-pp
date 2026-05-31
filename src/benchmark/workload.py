"""Workload generation for VecAdvisor++ benchmarks.

Generates pure vector similarity queries and filtered vector queries
at various selectivity levels.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkloadConfig:
    """Configuration for a benchmark workload."""

    k: int = 10
    num_queries: int = 1000
    filter_type: str | None = None        # column name, e.g., "category_10"
    filter_selectivity: float | None = None  # target selectivity (0.0–1.0)
    cache_mode: str = "warm"               # "cold" or "warm"


@dataclass
class Query:
    """A single benchmark query."""

    query_vector: np.ndarray
    sql: str
    params: tuple
    filter_mask: np.ndarray | None = None  # boolean mask for ground-truth


def generate_pure_queries(
    query_vectors: np.ndarray,
    table_name: str,
    k: int,
    num_queries: int | None = None,
) -> list[Query]:
    """Generate pure ANN queries (no filter).

    Args:
        query_vectors: (nq, dim) query vectors.
        table_name: Name of the table to query.
        k: Number of nearest neighbors.
        num_queries: Limit number of queries (None = use all).

    Returns:
        List of Query objects.
    """
    n = num_queries if num_queries else len(query_vectors)
    n = min(n, len(query_vectors))

    queries = []
    for i in range(n):
        vec = query_vectors[i]
        vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
        sql = (
            f"SELECT id, embedding <-> %s::vector AS distance "
            f"FROM {table_name} "
            f"ORDER BY embedding <-> %s::vector "
            f"LIMIT %s;"
        )
        queries.append(Query(
            query_vector=vec,
            sql=sql,
            params=(vec_str, vec_str, k),
            filter_mask=None,
        ))
    return queries


def generate_filtered_queries(
    query_vectors: np.ndarray,
    table_name: str,
    k: int,
    filter_column: str,
    filter_value: int | float | bool,
    filter_op: str = "=",
    filter_mask: np.ndarray | None = None,
    num_queries: int | None = None,
) -> list[Query]:
    """Generate filtered ANN queries with a WHERE clause.

    Args:
        query_vectors: (nq, dim) query vectors.
        table_name: Table to query.
        k: Number of nearest neighbors.
        filter_column: Column name for the WHERE clause.
        filter_value: Value to filter on.
        filter_op: Comparison operator ("=", "<", ">", "<=", ">=").
        filter_mask: Precomputed boolean mask over base vectors for ground-truth.
        num_queries: Limit number of queries.

    Returns:
        List of Query objects with filter information.
    """
    n = num_queries if num_queries else len(query_vectors)
    n = min(n, len(query_vectors))

    queries = []
    for i in range(n):
        vec = query_vectors[i]
        vec_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
        sql = (
            f"SELECT id, embedding <-> %s::vector AS distance "
            f"FROM {table_name} "
            f"WHERE {filter_column} {filter_op} %s "
            f"ORDER BY embedding <-> %s::vector "
            f"LIMIT %s;"
        )
        queries.append(Query(
            query_vector=vec,
            sql=sql,
            params=(vec_str, filter_value, vec_str, k),
            filter_mask=filter_mask,
        ))
    return queries


def build_filter_mask(
    attributes: dict,
    filter_column: str,
    filter_value: int | float | bool,
    filter_op: str = "=",
) -> np.ndarray:
    """Build a boolean mask for ground-truth filtered evaluation.

    Args:
        attributes: Dict of column_name -> values array.
        filter_column: Column to filter on.
        filter_value: Target value.
        filter_op: Comparison operator.

    Returns:
        Boolean numpy array where True means the row passes the filter.
    """
    col_values = attributes[filter_column]
    ops = {
        "=": lambda a, b: a == b,
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
    }
    if filter_op not in ops:
        raise ValueError(f"Unsupported filter operator: {filter_op}")
    return ops[filter_op](col_values, filter_value)


def generate_selectivity_workloads(
    query_vectors: np.ndarray,
    table_name: str,
    k: int,
    attributes: dict,
    num_queries: int | None = None,
) -> dict[str, list[Query]]:
    """Generate workloads at multiple selectivity levels using category columns.

    Uses category_10 (10% per value), category_100 (1%), category_1000 (0.1%).

    Args:
        query_vectors: Query vectors.
        table_name: Table name.
        k: Top-k.
        attributes: Synthetic attributes dict.
        num_queries: Limit queries.

    Returns:
        Dict mapping selectivity label to list of queries.
        Keys: "pure", "sel_10pct", "sel_1pct", "sel_0.1pct"
    """
    workloads = {}

    # Pure (no filter)
    workloads["pure"] = generate_pure_queries(
        query_vectors, table_name, k, num_queries
    )

    # ~10% selectivity
    mask_10 = build_filter_mask(attributes, "category_10", 0)
    workloads["sel_10pct"] = generate_filtered_queries(
        query_vectors, table_name, k, "category_10", 0,
        filter_mask=mask_10, num_queries=num_queries,
    )

    # ~1% selectivity
    mask_1 = build_filter_mask(attributes, "category_100", 0)
    workloads["sel_1pct"] = generate_filtered_queries(
        query_vectors, table_name, k, "category_100", 0,
        filter_mask=mask_1, num_queries=num_queries,
    )

    # ~0.1% selectivity
    mask_01 = build_filter_mask(attributes, "category_1000", 0)
    workloads["sel_0.1pct"] = generate_filtered_queries(
        query_vectors, table_name, k, "category_1000", 0,
        filter_mask=mask_01, num_queries=num_queries,
    )

    return workloads
