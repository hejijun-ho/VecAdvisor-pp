"""Workload profiler for VecAdvisor++.

Extracts workload features from PostgreSQL table statistics and query patterns
to inform the advisor's index selection and tuning decisions.
"""

from dataclasses import dataclass

import psycopg2


@dataclass
class WorkloadProfile:
    """Captured workload characteristics used by the advisor."""

    n_vectors: int
    dim: int
    k: int
    has_filters: bool
    filter_selectivity: float  # 0.0–1.0, fraction of rows passing filter
    filter_columns: list[str]  # columns used in WHERE clauses
    update_rate: float         # 0.0 = read-only, 1.0 = write-only
    memory_budget_mb: int
    latency_budget_ms: float   # target p95 latency
    cache_regime: str          # "cold" or "warm"


def profile_from_table(
    conn,
    table_name: str,
    vector_column: str = "embedding",
    k: int = 10,
    filter_columns: list[str] | None = None,
    filter_clauses: list[str] | None = None,
    update_rate: float = 0.0,
    memory_budget_mb: int = 2048,
    latency_budget_ms: float = 50.0,
    cache_regime: str = "warm",
) -> WorkloadProfile:
    """Profile a workload by inspecting PostgreSQL table metadata.

    Args:
        conn: psycopg2 connection.
        table_name: Table to profile.
        vector_column: Name of the vector column.
        k: Typical top-k query parameter.
        filter_columns: Columns used in WHERE clauses.
        filter_clauses: SQL WHERE clauses for selectivity estimation.
        update_rate: Fraction of operations that are writes (0.0–1.0).
        memory_budget_mb: Available memory for indexes.
        latency_budget_ms: Target p95 latency.
        cache_regime: "cold" or "warm".

    Returns:
        WorkloadProfile with extracted features.
    """
    with conn.cursor() as cur:
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        n_vectors = cur.fetchone()[0]

        # Get vector dimension
        cur.execute(
            f"SELECT vector_dims({vector_column}) FROM {table_name} LIMIT 1;"
        )
        dim = cur.fetchone()[0]

    # Estimate filter selectivity
    selectivity = 1.0  # default: no filter
    if filter_clauses:
        selectivities = [
            estimate_selectivity(conn, table_name, clause)
            for clause in filter_clauses
        ]
        selectivity = min(selectivities)  # most selective filter

    has_filters = filter_columns is not None and len(filter_columns) > 0

    return WorkloadProfile(
        n_vectors=n_vectors,
        dim=dim,
        k=k,
        has_filters=has_filters,
        filter_selectivity=selectivity if has_filters else 1.0,
        filter_columns=filter_columns or [],
        update_rate=update_rate,
        memory_budget_mb=memory_budget_mb,
        latency_budget_ms=latency_budget_ms,
        cache_regime=cache_regime,
    )


def profile_from_params(
    n_vectors: int,
    dim: int,
    k: int = 10,
    has_filters: bool = False,
    filter_selectivity: float = 1.0,
    filter_columns: list[str] | None = None,
    update_rate: float = 0.0,
    memory_budget_mb: int = 2048,
    latency_budget_ms: float = 50.0,
    cache_regime: str = "warm",
) -> WorkloadProfile:
    """Create a WorkloadProfile directly from parameters (no DB connection needed).

    Useful for testing and CLI usage.
    """
    return WorkloadProfile(
        n_vectors=n_vectors,
        dim=dim,
        k=k,
        has_filters=has_filters,
        filter_selectivity=filter_selectivity,
        filter_columns=filter_columns or [],
        update_rate=update_rate,
        memory_budget_mb=memory_budget_mb,
        latency_budget_ms=latency_budget_ms,
        cache_regime=cache_regime,
    )


def estimate_selectivity(
    conn, table_name: str, where_clause: str
) -> float:
    """Estimate filter selectivity using PostgreSQL EXPLAIN.

    Args:
        conn: psycopg2 connection.
        table_name: Table name.
        where_clause: WHERE clause without the 'WHERE' keyword.

    Returns:
        Estimated selectivity (0.0–1.0).
    """
    with conn.cursor() as cur:
        # Get total row count
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        total = cur.fetchone()[0]
        if total == 0:
            return 0.0

        # Use EXPLAIN to get estimated rows
        cur.execute(
            f"EXPLAIN (FORMAT JSON) SELECT * FROM {table_name} "
            f"WHERE {where_clause};"
        )
        plan = cur.fetchone()[0]
        estimated_rows = plan[0]["Plan"]["Plan Rows"]

        return estimated_rows / total
