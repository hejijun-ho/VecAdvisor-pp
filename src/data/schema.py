"""PostgreSQL schema management for VecAdvisor++.

Handles table creation, vector data insertion, synthetic attribute generation,
and index creation/management for pgvector.
"""

import time
from dataclasses import dataclass

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

try:
    import vecadvisor_rs as _rs
    # Guard against the vecadvisor_rs/ source directory being imported as a
    # namespace package (e.g. on servers where the .so hasn't been built yet).
    _RUST_AVAILABLE = hasattr(_rs, "build_insert_rows")
except ImportError:
    _RUST_AVAILABLE = False


@dataclass
class IndexConfig:
    """Configuration for a vector index."""

    index_type: str  # "hnsw" or "ivfflat"
    params: dict  # e.g., {"m": 16, "ef_construction": 128} or {"lists": 1000}


def get_connection(conn_params: dict):
    """Create a PostgreSQL connection and register pgvector types.

    Args:
        conn_params: Dict with host, port, dbname, user, password.

    Returns:
        psycopg2 connection with pgvector types registered.
    """
    conn = psycopg2.connect(**conn_params)
    register_vector(conn)
    return conn


def create_vector_table(
    conn, table_name: str, dim: int, with_attributes: bool = True
) -> None:
    """Create the vector table with optional filter attributes.

    Args:
        conn: psycopg2 connection.
        table_name: Name of the table to create.
        dim: Vector dimensionality.
        with_attributes: If True, add synthetic filter columns.
    """
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

        if with_attributes:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({dim}),
                    category_10 INT,
                    category_100 INT,
                    category_1000 INT,
                    price FLOAT,
                    is_active BOOLEAN
                );
            """)
        else:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    embedding vector({dim})
                );
            """)
    conn.commit()


def generate_synthetic_attributes(n: int, seed: int = 42) -> dict:
    """Generate synthetic filter attributes with known selectivity distributions.

    Args:
        n: Number of rows.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping column name to numpy array of values.
    """
    rng = np.random.RandomState(seed)
    return {
        "category_10": rng.randint(0, 10, size=n),       # ~10% selectivity per value
        "category_100": rng.randint(0, 100, size=n),     # ~1% selectivity per value
        "category_1000": rng.randint(0, 1000, size=n),   # ~0.1% selectivity per value
        "price": rng.uniform(0, 1000, size=n).round(2),  # range filter
        "is_active": rng.choice([True, False], size=n, p=[0.7, 0.3]),
    }


def insert_vectors(
    conn,
    table_name: str,
    vectors: np.ndarray,
    attributes: dict | None = None,
    batch_size: int = 1000,
) -> None:
    """Bulk insert vectors and optional attributes into the table.

    Args:
        conn: psycopg2 connection.
        table_name: Target table name.
        vectors: (n, dim) float32 array.
        attributes: Optional dict of column_name -> values array.
        batch_size: Number of rows per INSERT batch.
    """
    n = len(vectors)
    with conn.cursor() as cur:
        if attributes:
            cols = ["embedding"] + list(attributes.keys())
            col_str = ", ".join(cols)
            template = "(" + ", ".join(["%s"] * len(cols)) + ")"

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                if _RUST_AVAILABLE:
                    # Rust: build all rows for this batch in one call, avoiding
                    # per-element Python object allocation (128M floats for 1M vecs)
                    rows = _rs.build_insert_rows(
                        np.ascontiguousarray(vectors[start:end], dtype=np.float32),
                        np.asarray(attributes["category_10"][start:end], dtype=np.int64),
                        np.asarray(attributes["category_100"][start:end], dtype=np.int64),
                        np.asarray(attributes["category_1000"][start:end], dtype=np.int64),
                        np.asarray(attributes["price"][start:end], dtype=np.float64),
                        np.asarray(attributes["is_active"][start:end], dtype=np.uint8),
                    )
                else:
                    rows = []
                    for i in range(start, end):
                        row = [vectors[i].tolist()]
                        for col in attributes:
                            val = attributes[col][i]
                            # Convert numpy types to Python native types
                            if isinstance(val, (np.integer,)):
                                val = int(val)
                            elif isinstance(val, (np.floating,)):
                                val = float(val)
                            elif isinstance(val, (np.bool_,)):
                                val = bool(val)
                            row.append(val)
                        rows.append(tuple(row))

                execute_values(
                    cur,
                    f"INSERT INTO {table_name} ({col_str}) VALUES %s",
                    rows,
                    template=template,
                )
        else:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                rows = [(vectors[i].tolist(),) for i in range(start, end)]
                execute_values(
                    cur,
                    f"INSERT INTO {table_name} (embedding) VALUES %s",
                    rows,
                    template="(%s)",
                )
    conn.commit()


def create_index(
    conn, table_name: str, config: IndexConfig, column: str = "embedding"
) -> tuple[str, float]:
    """Create a vector index on the specified table.

    Args:
        conn: psycopg2 connection.
        table_name: Table containing the vector column.
        config: IndexConfig specifying type and parameters.
        column: Name of the vector column.

    Returns:
        Tuple of (index_name, build_time_seconds).
    """
    index_name = f"idx_{table_name}_{config.index_type}"

    if config.index_type == "hnsw":
        m = config.params.get("m", 16)
        ef_construction = config.params.get("ef_construction", 128)
        sql = (
            f"CREATE INDEX {index_name} ON {table_name} "
            f"USING hnsw ({column} vector_l2_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction});"
        )
    elif config.index_type == "ivfflat":
        lists = config.params.get("lists", 1000)
        sql = (
            f"CREATE INDEX {index_name} ON {table_name} "
            f"USING ivfflat ({column} vector_l2_ops) "
            f"WITH (lists = {lists});"
        )
    else:
        raise ValueError(f"Unknown index type: {config.index_type}")

    with conn.cursor() as cur:
        # Drop ALL vector indexes on this table to ensure clean state
        # (prevents PostgreSQL from picking a stale index from a prior run)
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s "
            "AND (indexdef LIKE '%%hnsw%%' OR indexdef LIKE '%%ivfflat%%');",
            (table_name,),
        )
        for row in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {row[0]};")
        conn.commit()

        # Ensure sufficient maintenance_work_mem for index builds
        # Use 8GB to handle large IVFFlat builds (lists=4000 needs ~3.2 GB)
        cur.execute("SET maintenance_work_mem = '8GB';")

        start = time.perf_counter()
        cur.execute(sql)
        conn.commit()
        build_time = time.perf_counter() - start

    return index_name, build_time


def drop_index(conn, index_name: str) -> None:
    """Drop an index by name."""
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")
    conn.commit()


def drop_all_vector_indexes(conn, table_name: str) -> None:
    """Drop all HNSW and IVFFlat vector indexes on a table.

    Used by the sequential_scan baseline to ensure no vector index
    is present, forcing PostgreSQL to perform a full sequential scan.

    Args:
        conn: psycopg2 connection.
        table_name: Table whose vector indexes should be dropped.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = %s "
            "AND (indexdef LIKE '%%hnsw%%' OR indexdef LIKE '%%ivfflat%%');",
            (table_name,),
        )
        for row in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {row[0]};")
    conn.commit()


def create_btree_index(conn, table_name: str, column: str) -> str:
    """Create a B-tree index on a filter attribute column.

    Args:
        conn: psycopg2 connection.
        table_name: Table name.
        column: Column to index.

    Returns:
        Index name.
    """
    index_name = f"idx_{table_name}_{column}_btree"
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")
        cur.execute(f"CREATE INDEX {index_name} ON {table_name} ({column});")
    conn.commit()
    return index_name


def create_partitioned_vector_table(
    conn,
    table_name: str,
    dim: int,
    partition_column: str,
    partition_values: list[int],
    with_attributes: bool = True,
) -> None:
    """Create a LIST-partitioned vector table with one child per partition value.

    The parent table is partitioned by ``partition_column`` using
    ``PARTITION BY LIST``.  One child table is created for each value in
    ``partition_values``.  Data inserted into the parent is automatically
    routed to the correct child.

    Note: partitioned tables cannot have a SERIAL primary key on the parent
    (the partition key must be included in any unique constraint).  Use an
    INT column for ``id`` and assign IDs before insertion.

    Args:
        conn: psycopg2 connection.
        table_name: Name of the parent (partitioned) table.
        dim: Vector dimensionality.
        partition_column: Column used as the LIST partition key.
        partition_values: Distinct integer values; one child table per value.
        with_attributes: If True, include synthetic filter columns.
    """
    with conn.cursor() as cur:
        # Drop the whole hierarchy (CASCADE removes child tables too)
        cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")

        if with_attributes:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id            INT,
                    embedding     vector({dim}),
                    category_10   INT,
                    category_100  INT,
                    category_1000 INT,
                    price         FLOAT,
                    is_active     BOOLEAN
                ) PARTITION BY LIST ({partition_column});
            """)
        else:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id            INT,
                    embedding     vector({dim}),
                    {partition_column} INT
                ) PARTITION BY LIST ({partition_column});
            """)

        for val in partition_values:
            child = f"{table_name}_p{val}"
            cur.execute(
                f"CREATE TABLE {child} PARTITION OF {table_name} "
                f"FOR VALUES IN ({val});"
            )
    conn.commit()


def create_partition_vector_indexes(
    conn,
    table_name: str,
    partition_values: list[int],
    index_type: str,
    params: dict,
    column: str = "embedding",
) -> float:
    """Create vector indexes on each child partition of a partitioned table.

    Args:
        conn: psycopg2 connection.
        table_name: Parent table name (child names inferred as ``{table}_p{val}``).
        partition_values: Partition key values (one child per value).
        index_type: "hnsw" or "ivfflat".
        params: Index build parameters (m/ef_construction for HNSW, lists for IVFFlat).
        column: Vector column name.

    Returns:
        Total index build time in seconds (sum across all partitions).
    """
    total_build_time = 0.0
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '2GB';")

    for val in partition_values:
        child = f"{table_name}_p{val}"
        index_name = f"idx_{child}_{index_type}"

        if index_type == "hnsw":
            m = params.get("m", 16)
            ef_construction = params.get("ef_construction", 128)
            sql = (
                f"CREATE INDEX {index_name} ON {child} "
                f"USING hnsw ({column} vector_l2_ops) "
                f"WITH (m = {m}, ef_construction = {ef_construction});"
            )
        elif index_type == "ivfflat":
            lists = params.get("lists", 100)
            sql = (
                f"CREATE INDEX {index_name} ON {child} "
                f"USING ivfflat ({column} vector_l2_ops) "
                f"WITH (lists = {lists});"
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(sql)
            conn.commit()
            total_build_time += time.perf_counter() - t0

    return total_build_time


def get_table_size(conn, table_name: str) -> dict:
    """Get table and index sizes in MB.

    Args:
        conn: psycopg2 connection.
        table_name: Table to inspect.

    Returns:
        Dict with table_size_mb and index_size_mb.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_total_relation_size(%s), pg_table_size(%s), "
            "pg_indexes_size(%s);",
            (table_name, table_name, table_name),
        )
        total, table, indexes = cur.fetchone()
    return {
        "total_size_mb": round(total / (1024 * 1024), 2),
        "table_size_mb": round(table / (1024 * 1024), 2),
        "index_size_mb": round(indexes / (1024 * 1024), 2),
    }
