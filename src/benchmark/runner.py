"""Benchmark execution engine for VecAdvisor++.

Runs query workloads against PostgreSQL, measures latency,
and collects system metrics.
"""

import os
import platform
import subprocess
import time

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from src.benchmark.metrics import (
    BenchmarkResult,
    compute_latency_percentiles,
    compute_recall,
    compute_topk_completion_rate,
)
from src.benchmark.workload import Query
from src.data.schema import IndexConfig, create_index, drop_all_vector_indexes, get_table_size


class BenchmarkRunner:
    """Executes benchmark workloads and collects performance metrics."""

    def __init__(self, conn_params: dict):
        """Initialize with database connection parameters.

        Args:
            conn_params: Dict with host, port, dbname, user, password.
        """
        self.conn_params = conn_params

    def _get_connection(self):
        conn = psycopg2.connect(**self.conn_params)
        register_vector(conn)
        return conn

    def set_query_params(self, conn, index_type: str, params: dict) -> None:
        """Set session-level query parameters for vector search.

        Args:
            conn: psycopg2 connection.
            index_type: "hnsw", "ivfflat", or "none".
            params: Query-time parameters (e.g., ef_search, probes).
        """
        with conn.cursor() as cur:
            if index_type == "hnsw":
                ef_search = params.get("ef_search", 40)
                cur.execute(f"SET hnsw.ef_search = {ef_search};")
            elif index_type == "ivfflat":
                probes = params.get("probes", 10)
                cur.execute(f"SET ivfflat.probes = {probes};")
            # "none" (sequential scan) — no special params needed

    def run_build_benchmark(
        self, table_name: str, config: IndexConfig
    ) -> tuple[str, float]:
        """Build an index and measure build time.

        For index_type="none", drops all vector indexes and returns immediately.

        Args:
            table_name: Table to index.
            config: Index configuration.

        Returns:
            Tuple of (index_name_or_empty, build_time_seconds).
        """
        conn = self._get_connection()
        try:
            if config.index_type == "none":
                # Sequential scan baseline: drop all vector indexes
                drop_all_vector_indexes(conn, table_name)
                return ("", 0.0)
            index_name, build_time = create_index(conn, table_name, config)
            return index_name, build_time
        finally:
            conn.close()

    def run_query_benchmark(
        self,
        queries: list[Query],
        index_type: str,
        query_params: dict,
        warmup_queries: int = 100,
        cache_mode: str = "warm",
    ) -> tuple[list[list[int]], list[float]]:
        """Execute queries and measure per-query latency.

        Args:
            queries: List of Query objects.
            index_type: "hnsw", "ivfflat", or "none".
            query_params: Session parameters (ef_search, probes).
            warmup_queries: Number of warmup queries before measurement.
            cache_mode: "cold" or "warm".

        Returns:
            Tuple of (result_ids, latencies_ms):
                - result_ids: List of lists of returned row IDs per query.
                - latencies_ms: Per-query latency in ms.
        """
        if cache_mode == "cold":
            self._clear_caches()

        conn = self._get_connection()
        try:
            self.set_query_params(conn, index_type, query_params)

            # Warmup
            if cache_mode == "warm":
                warmup_n = min(warmup_queries, len(queries))
                with conn.cursor() as cur:
                    for q in queries[:warmup_n]:
                        cur.execute(q.sql, q.params)
                        cur.fetchall()

            # Measured queries
            result_ids = []
            latencies_ms = []

            with conn.cursor() as cur:
                for q in queries:
                    start = time.perf_counter()
                    cur.execute(q.sql, q.params)
                    rows = cur.fetchall()
                    elapsed = (time.perf_counter() - start) * 1000

                    ids = [row[0] for row in rows]
                    result_ids.append(ids)
                    latencies_ms.append(elapsed)

            return result_ids, latencies_ms
        finally:
            conn.close()

    def run_full_benchmark(
        self,
        table_name: str,
        queries: list[Query],
        index_config: IndexConfig,
        query_params: dict,
        ground_truth_ids,
        k: int,
        config_name: str = "",
        cache_mode: str = "warm",
        filter_selectivity: float | None = None,
        num_runs: int = 1,
    ) -> BenchmarkResult:
        """Run a complete benchmark: build index, run queries (num_runs times), compute metrics.

        The index is built once. When num_runs > 1 the query phase is repeated
        num_runs times; metrics are reported as mean ± std across runs.

        Args:
            table_name: Table to benchmark.
            queries: Query workload.
            index_config: Index build configuration.
            query_params: Query-time parameters.
            ground_truth_ids: Ground truth neighbour indices for recall.
            k: Top-k value.
            config_name: Label for this configuration.
            cache_mode: "cold" or "warm".
            filter_selectivity: Optional selectivity for reporting.
            num_runs: Number of query-phase repetitions (default 1).

        Returns:
            BenchmarkResult with mean metrics (and stddev fields when num_runs > 1).
        """
        # Build index (once)
        _, build_time = self.run_build_benchmark(table_name, index_config)

        # Get sizes
        conn = self._get_connection()
        try:
            sizes = get_table_size(conn, table_name)
        finally:
            conn.close()

        # Run query phase num_runs times
        all_recalls: list[float] = []
        all_p50: list[float] = []
        all_p95: list[float] = []
        all_p99: list[float] = []
        all_mean: list[float] = []
        all_completion: list[float] = []

        for _ in range(max(1, num_runs)):
            result_ids, latencies_ms = self.run_query_benchmark(
                queries, index_config.index_type, query_params,
                cache_mode=cache_mode,
            )
            recall = compute_recall(result_ids, ground_truth_ids, k)
            lat = compute_latency_percentiles(latencies_ms)
            completion = compute_topk_completion_rate(result_ids, k)

            all_recalls.append(recall)
            all_p50.append(lat["p50"])
            all_p95.append(lat["p95"])
            all_p99.append(lat["p99"])
            all_mean.append(lat["mean"])
            all_completion.append(completion)

        def _mean(lst): return float(np.mean(lst))
        def _std(lst): return float(np.std(lst, ddof=0)) if len(lst) > 1 else 0.0

        return BenchmarkResult(
            config_name=config_name,
            index_type=index_config.index_type,
            index_params=index_config.params,
            query_params=query_params,
            recall=_mean(all_recalls),
            latency_p50_ms=_mean(all_p50),
            latency_p95_ms=_mean(all_p95),
            latency_p99_ms=_mean(all_p99),
            latency_mean_ms=_mean(all_mean),
            build_time_s=build_time,
            memory_mb=sizes["index_size_mb"],
            disk_mb=sizes["total_size_mb"],
            completion_rate=_mean(all_completion),
            num_queries=len(queries),
            k=k,
            filter_selectivity=filter_selectivity,
            recall_std=_std(all_recalls),
            latency_p50_std=_std(all_p50),
            latency_p95_std=_std(all_p95),
            latency_p99_std=_std(all_p99),
            latency_mean_std=_std(all_mean),
            completion_rate_std=_std(all_completion),
            num_runs=num_runs,
        )

    def _clear_caches(self) -> None:
        """Attempt to clear caches for cold-cache benchmarks.

        macOS: uses 'sudo purge'.
        Linux: tries kernel drop_caches (needs sudo); falls back to
               restarting the personal PG cluster via pg_ctl (clears
               shared_buffers) when PGDATA env var is set.
        Always issues DISCARD ALL to clear session state.
        """
        system = platform.system()
        if system == "Darwin":
            try:
                subprocess.run(["sudo", "purge"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("Warning: Could not run 'sudo purge'. Cold-cache results may be inaccurate.")
        elif system == "Linux":
            # Try kernel page-cache drop (needs sudo)
            try:
                subprocess.run(
                    ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
                    check=True, capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: restart personal PG cluster to flush shared_buffers
                pgdata = os.environ.get("PGDATA", "")
                if pgdata and os.path.isdir(pgdata):
                    try:
                        subprocess.run(
                            ["pg_ctl", "-D", pgdata, "restart", "-w", "-t", "60"],
                            check=True, capture_output=True,
                        )
                        print(f"Restarted PostgreSQL at {pgdata} to clear shared_buffers.")
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print("Warning: Could not restart PostgreSQL. Cold-cache may be inaccurate.")
                else:
                    print("Warning: PGDATA not set; cannot clear caches on Linux without sudo.")

        # Clear session state
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("DISCARD ALL;")
            conn.close()
        except Exception:
            pass
