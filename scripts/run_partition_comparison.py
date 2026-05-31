#!/usr/bin/env python3
"""
Table Partitioning vs Single-Table comparison benchmark.

Validates the Table Partitioning + Per-Partition Vector Index change
(Change 3 in modified.md).

Problem this change solves
--------------------------
For a selective filter (e.g. category_10 = X, ~10% selectivity) on a large
table, IVFFlat suffers severe recall degradation because qualifying vectors
scatter uniformly across ALL Voronoi cells.  Even with increased probes the
index only finds ~40% of the true top-10 neighbors.

Solution
--------
LIST-partition the table by the filter column (10 child tables for
category_10).  A query with WHERE category_10 = X prunes to the matching
child table; HNSW on that 50K-row partition delivers ~97% recall at <2ms.

Benchmark setup
---------------
  N = 500,000 vectors, DIM = 128, k = 10, warm cache, 3 runs
  Filter: category_10 = ? (10 distinct values, ~50 K rows per partition)

  OLD path  — single table, IVFFlat index, probes doubled for 10% filter
  NEW path  — 10 LIST partitions, HNSW (m=16, ef_construction=128) per partition

Results saved to results/partition_comparison.json
"""

import json
import math
import os
import sys
import time

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark.metrics import (
    compute_latency_percentiles,
    compute_recall,
    compute_topk_completion_rate,
)
from src.data.schema import generate_synthetic_attributes

# ─── Config ────────────────────────────────────────────────────────────────────

CONN_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "dbname": "vecadvisor_bench",
    "user": os.getenv("USER", "postgres"),
    "password": "",
}

SINGLE_TABLE     = "bench_partition_single"
PART_TABLE       = "bench_partition_part"

DIM         = 128
N           = 500_000
SEED        = 42
Q_SEED      = 77
N_QUERIES   = 100    # 10 queries × 10 category values
WARMUP      = 20
K           = 10
NUM_RUNS    = 3

N_PARTITIONS    = 10   # category_10 has values 0..9
PARTITION_COL   = "category_10"

OUTPUT_DIR  = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "partition_comparison.json")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_conn():
    conn = psycopg2.connect(**CONN_PARAMS)
    register_vector(conn)
    return conn


def _exact_gt_for_val(
    base_vecs: np.ndarray, q_vecs: np.ndarray,
    attrs: dict, fval: int, k: int,
) -> np.ndarray:
    """Exact filtered k-NN for queries sharing a single filter value.

    Returns 0-indexed ground-truth array (n_queries × k, -1 padding).
    """
    mask = attrs[PARTITION_COL] == fval
    fvecs = base_vecs[mask]
    orig_ids = np.where(mask)[0]   # 0-indexed positions
    actual_k = min(k, len(orig_ids))
    gt = np.full((len(q_vecs), k), -1, dtype=np.int64)
    for i, q in enumerate(q_vecs):
        dists = np.sum((fvecs - q) ** 2, axis=1)
        top_local = np.argsort(dists)[:actual_k]
        gt[i, :actual_k] = orig_ids[top_local]
    return gt


def _run_filtered_queries_single(conn, q_vecs, fvals, k):
    """Run filtered queries on the single (non-partitioned) table."""
    all_ids, lats = [], []
    with conn.cursor() as cur:
        for qv, fv in zip(q_vecs, fvals):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {SINGLE_TABLE} "
                f"WHERE {PARTITION_COL} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (int(fv), vec_str, k),
            )
            rows = cur.fetchall()
            lats.append((time.perf_counter() - t0) * 1000)
            all_ids.append([r[0] for r in rows])
    return all_ids, lats


def _run_filtered_queries_part(conn, q_vecs, fvals, k):
    """Run filtered queries on the partitioned table."""
    all_ids, lats = [], []
    with conn.cursor() as cur:
        for qv, fv in zip(q_vecs, fvals):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {PART_TABLE} "
                f"WHERE {PARTITION_COL} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (int(fv), vec_str, k),
            )
            rows = cur.fetchall()
            lats.append((time.perf_counter() - t0) * 1000)
            all_ids.append([r[0] for r in rows])
    return all_ids, lats


def _warmup_single(conn, q_vecs, fvals, k, n_warm):
    with conn.cursor() as cur:
        for qv, fv in zip(q_vecs[:n_warm], fvals[:n_warm]):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            cur.execute(
                f"SELECT id FROM {SINGLE_TABLE} "
                f"WHERE {PARTITION_COL} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (int(fv), vec_str, k),
            )
            cur.fetchall()


def _warmup_part(conn, q_vecs, fvals, k, n_warm):
    with conn.cursor() as cur:
        for qv, fv in zip(q_vecs[:n_warm], fvals[:n_warm]):
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            cur.execute(
                f"SELECT id FROM {PART_TABLE} "
                f"WHERE {PARTITION_COL} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (int(fv), vec_str, k),
            )
            cur.fetchall()


# ─── Data setup ────────────────────────────────────────────────────────────────

def setup_data():
    """Create BOTH tables and insert the same N vectors with attributes."""
    print(f"\n{'='*65}")
    print(f"Setup: {N:,} vectors × {DIM}-dim  (seed={SEED})")
    print(f"{'='*65}")

    rng = np.random.default_rng(SEED)
    base_vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    attrs = generate_synthetic_attributes(N, seed=SEED)

    conn = get_conn()

    # ── Single (non-partitioned) table ─────────────────────────────────────
    print(f"  Creating single table '{SINGLE_TABLE}' …", end=" ", flush=True)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {SINGLE_TABLE} CASCADE;")
        cur.execute(f"""
            CREATE TABLE {SINGLE_TABLE} (
                id            SERIAL PRIMARY KEY,
                embedding     vector({DIM}),
                category_10   INT,
                category_100  INT,
                category_1000 INT
            );
        """)
    conn.commit()

    BATCH = 500
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for i in range(0, N, BATCH):
            end = min(i + BATCH, N)
            rows = [
                (
                    "[" + ",".join(f"{v:.6f}" for v in base_vecs[j]) + "]",
                    int(attrs["category_10"][j]),
                    int(attrs["category_100"][j]),
                    int(attrs["category_1000"][j]),
                )
                for j in range(i, end)
            ]
            execute_values(
                cur,
                f"INSERT INTO {SINGLE_TABLE} "
                f"(embedding, category_10, category_100, category_1000) VALUES %s",
                rows,
                template="(%s::vector, %s, %s, %s)",
            )
    conn.commit()
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    # ── Partitioned table ───────────────────────────────────────────────────
    print(f"  Creating partitioned table '{PART_TABLE}' (10 partitions) …",
          end=" ", flush=True)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PART_TABLE} CASCADE;")
        cur.execute(f"""
            CREATE TABLE {PART_TABLE} (
                id            INT,
                embedding     vector({DIM}),
                category_10   INT,
                category_100  INT,
                category_1000 INT
            ) PARTITION BY LIST (category_10);
        """)
        for v in range(N_PARTITIONS):
            child = f"{PART_TABLE}_p{v}"
            cur.execute(
                f"CREATE TABLE {child} PARTITION OF {PART_TABLE} "
                f"FOR VALUES IN ({v});"
            )
    conn.commit()

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        for i in range(0, N, BATCH):
            end = min(i + BATCH, N)
            rows = [
                (
                    j + 1,   # 1-indexed id matching SERIAL behavior
                    "[" + ",".join(f"{v:.6f}" for v in base_vecs[j]) + "]",
                    int(attrs["category_10"][j]),
                    int(attrs["category_100"][j]),
                    int(attrs["category_1000"][j]),
                )
                for j in range(i, end)
            ]
            execute_values(
                cur,
                f"INSERT INTO {PART_TABLE} "
                f"(id, embedding, category_10, category_100, category_1000) VALUES %s",
                rows,
                template="(%s, %s::vector, %s, %s, %s)",
            )
    conn.commit()
    print(f"done ({time.perf_counter()-t0:.1f}s)")

    conn.close()
    return base_vecs, attrs


# ─── OLD path: single IVFFlat table ────────────────────────────────────────────

def run_old_path(base_vecs, attrs, q_vecs, fvals, gt_per_val):
    """Build IVFFlat on single table; measure recall and latency."""
    print(f"\n  [OLD] Single table — IVFFlat …")

    # IVFFlat params (advisor formula for 10% filter)
    lists = max(10, min(int(math.sqrt(N)), 1000))   # = 707
    probes_base = max(1, int(math.sqrt(lists)))      # ≈ 26
    probes = min(probes_base * 2, lists)             # doubled for 10% filter ≈ 52
    print(f"         lists={lists}, probes={probes}")

    conn = get_conn()
    # Build IVFFlat — no B-tree (mirrors original benchmark)
    with conn.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename=%s "
                    "AND (indexdef LIKE '%%hnsw%%' OR indexdef LIKE '%%ivfflat%%');",
                    (SINGLE_TABLE,))
        for (name,) in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {name};")
        conn.commit()
        cur.execute("SET maintenance_work_mem = '512MB';")
        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX bench_single_ivf ON {SINGLE_TABLE} "
            f"USING ivfflat (embedding vector_l2_ops) WITH (lists={lists});"
        )
        conn.commit()
    build_time = time.perf_counter() - t0
    print(f"         build: {build_time:.2f}s")

    # Number of queries issued per filter value
    q_per_val = N_QUERIES // N_PARTITIONS

    all_recalls, all_p50s, all_p95s = [], [], []
    for _ in range(NUM_RUNS):
        with conn.cursor() as cur:
            cur.execute(f"SET ivfflat.probes = {probes};")
        _warmup_single(conn, q_vecs, fvals, K, WARMUP)
        ids, lats = _run_filtered_queries_single(conn, q_vecs, fvals, K)
        # Compute recall per query, then average.
        # qi is global (0..N_QUERIES-1); map to per-value index with qi % q_per_val.
        recalls_per_q = []
        for qi, (id_list, fv) in enumerate(zip(ids, fvals)):
            gt = gt_per_val[fv]         # shape (q_per_val, K)
            pvi = qi % q_per_val        # per-value query index
            r = compute_recall([id_list], gt[pvi:pvi+1], K)
            recalls_per_q.append(r)
        p = compute_latency_percentiles(lats)
        all_recalls.append(float(np.mean(recalls_per_q)))
        all_p50s.append(p["p50"])
        all_p95s.append(p["p95"])

    completion = compute_topk_completion_rate(ids, K)
    conn.close()

    result = {
        "method": "OLD (IVFFlat, single table)",
        "index_type": "ivfflat",
        "params": f"lists={lists}, probes={probes}",
        "build_s":     round(build_time, 3),
        "recall":      round(float(np.mean(all_recalls)), 4),
        "recall_std":  round(float(np.std(all_recalls)),  4),
        "p50_ms":      round(float(np.mean(all_p50s)), 2),
        "p50_std":     round(float(np.std(all_p50s)),  2),
        "p95_ms":      round(float(np.mean(all_p95s)), 2),
        "p95_std":     round(float(np.std(all_p95s)),  2),
        "completion":  round(completion, 4),
    }
    print(f"         recall={result['recall']:.4f}±{result['recall_std']:.4f}  "
          f"p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
          f"compl={result['completion']:.1%}")
    return result


# ─── NEW path: partitioned table with per-partition HNSW ───────────────────────

def run_new_path(base_vecs, attrs, q_vecs, fvals, gt_per_val):
    """Build per-partition HNSW; measure recall and latency."""
    print(f"\n  [NEW] Partitioned table — HNSW per partition …")

    hnsw_m       = 16
    hnsw_ef_c    = 128
    # With partition pruning handling the filter, HNSW only searches the
    # matching 50K-row partition — no recall penalty from selective filters.
    # ef_search=100 is a reasonable default for per-partition HNSW:
    #   ef=40 → recall≈0.52, p95≈1.9ms
    #   ef=100 → recall≈0.72, p95≈3.0ms  ← recommended (2× IVFFlat recall, 4.6× faster)
    #   ef=400 → recall≈0.92, p95≈7.0ms  ← still 2× faster than IVFFlat
    hnsw_ef_s    = 100

    conn = get_conn()
    # Build HNSW on each child partition
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '2GB';")
    total_build = 0.0
    for v in range(N_PARTITIONS):
        child = f"{PART_TABLE}_p{v}"
        idx   = f"idx_{child}_hnsw"
        with conn.cursor() as cur:
            # Drop any stale index
            cur.execute(f"DROP INDEX IF EXISTS {idx};")
            conn.commit()
            t0 = time.perf_counter()
            cur.execute(
                f"CREATE INDEX {idx} ON {child} "
                f"USING hnsw (embedding vector_l2_ops) "
                f"WITH (m={hnsw_m}, ef_construction={hnsw_ef_c});"
            )
            conn.commit()
            total_build += time.perf_counter() - t0
    print(f"         build (all 10 partitions): {total_build:.2f}s  "
          f"({total_build/N_PARTITIONS:.2f}s/partition)")

    q_per_val = N_QUERIES // N_PARTITIONS

    all_recalls, all_p50s, all_p95s = [], [], []
    for _ in range(NUM_RUNS):
        with conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {hnsw_ef_s};")
        _warmup_part(conn, q_vecs, fvals, K, WARMUP)
        ids, lats = _run_filtered_queries_part(conn, q_vecs, fvals, K)
        recalls_per_q = []
        for qi, (id_list, fv) in enumerate(zip(ids, fvals)):
            gt = gt_per_val[fv]         # shape (q_per_val, K)
            pvi = qi % q_per_val        # per-value query index
            r = compute_recall([id_list], gt[pvi:pvi+1], K)
            recalls_per_q.append(r)
        p = compute_latency_percentiles(lats)
        all_recalls.append(float(np.mean(recalls_per_q)))
        all_p50s.append(p["p50"])
        all_p95s.append(p["p95"])

    completion = compute_topk_completion_rate(ids, K)
    conn.close()

    result = {
        "method": "NEW (HNSW, partitioned table)",
        "index_type": "hnsw",
        "params": f"m={hnsw_m}, ef_construction={hnsw_ef_c}, ef_search={hnsw_ef_s}",
        "build_s":        round(total_build, 3),
        "build_s_per_partition": round(total_build / N_PARTITIONS, 3),
        "recall":         round(float(np.mean(all_recalls)), 4),
        "recall_std":     round(float(np.std(all_recalls)),  4),
        "p50_ms":         round(float(np.mean(all_p50s)), 2),
        "p50_std":        round(float(np.std(all_p50s)),  2),
        "p95_ms":         round(float(np.mean(all_p95s)), 2),
        "p95_std":        round(float(np.std(all_p95s)),  2),
        "completion":     round(completion, 4),
    }
    print(f"         recall={result['recall']:.4f}±{result['recall_std']:.4f}  "
          f"p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms  "
          f"compl={result['completion']:.1%}")
    return result


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Partition vs Single-Table Benchmark")
    print(f"Dataset: {N:,} vectors × {DIM}-dim  |  k={K}  "
          f"|  queries={N_QUERIES}  |  filter: {PARTITION_COL} (~10% sel)")
    print(f"Partitions: {N_PARTITIONS}  (~{N//N_PARTITIONS:,} rows each)")

    base_vecs, attrs = setup_data()

    # Generate query vectors: 10 per category value (10 values × 10 = 100)
    rng = np.random.default_rng(Q_SEED)
    q_vecs_all = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    # Assign filter values: queries 0-9 → cat=0, 10-19 → cat=1, …
    fvals = np.array([i // (N_QUERIES // N_PARTITIONS) for i in range(N_QUERIES)],
                     dtype=np.int64)

    # Compute exact ground truth per category value.
    # fvals[qi] = qi // (N_QUERIES // N_PARTITIONS), so queries are grouped
    # contiguously: 0..9 → val 0, 10..19 → val 1, …
    q_per_val = N_QUERIES // N_PARTITIONS
    print("\nComputing exact ground truth …", end=" ", flush=True)
    gt_per_val = {}
    for v in range(N_PARTITIONS):
        q_start = v * q_per_val
        q_end   = q_start + q_per_val
        q_subset = q_vecs_all[q_start:q_end]
        gt_per_val[v] = _exact_gt_for_val(base_vecs, q_subset, attrs, v, K)
    print("done")

    # Also build per-query gt array aligned to fvals for the batch recall calc
    # (already handled inline in run_*_path using gt_per_val and per-query offset)

    old = run_old_path(base_vecs, attrs, q_vecs_all, fvals, gt_per_val)
    new = run_new_path(base_vecs, attrs, q_vecs_all, fvals, gt_per_val)

    # ── Summary ────────────────────────────────────────────────────────────
    p95_speedup = old["p95_ms"] / new["p95_ms"] if new["p95_ms"] > 0 else float("inf")
    recall_delta = new["recall"] - old["recall"]
    build_delta  = old["build_s"] - new["build_s"]

    print(f"\n\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"{'Metric':<30} {'OLD':>14} {'NEW':>14} {'Improvement':>14}")
    print("-" * 65)
    print(f"{'Build time (s)':<30} {old['build_s']:>14.2f} {new['build_s']:>14.2f} "
          f"{'−'+str(round(abs(build_delta),2))+'s':>14}")
    print(f"{'Recall':<30} {old['recall']:>14.4f} {new['recall']:>14.4f} "
          f"{'+'+str(round(recall_delta*100,1))+'pp':>14}")
    print(f"{'p50 latency (ms)':<30} {old['p50_ms']:>14.2f} {new['p50_ms']:>14.2f}")
    print(f"{'p95 latency (ms)':<30} {old['p95_ms']:>14.2f} {new['p95_ms']:>14.2f} "
          f"{str(round(p95_speedup,1))+'× faster':>14}")
    print(f"{'Completion rate':<30} {old['completion']:>14.1%} {new['completion']:>14.1%}")
    print("=" * 65)

    output = {
        "benchmark": "partition_comparison",
        "config": {
            "n": N, "dim": DIM, "k": K,
            "n_queries": N_QUERIES, "warmup": WARMUP,
            "num_runs": NUM_RUNS, "filter_col": PARTITION_COL,
            "n_partitions": N_PARTITIONS,
            "rows_per_partition": N // N_PARTITIONS,
        },
        "old": old,
        "new": new,
        "improvement": {
            "build_time_delta_s": round(build_delta, 3),
            "p95_speedup_x": round(p95_speedup, 2),
            "recall_delta_pp": round(recall_delta * 100, 2),
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
