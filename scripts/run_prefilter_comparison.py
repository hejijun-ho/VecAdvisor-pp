#!/usr/bin/env python3
"""
Pre-filter vs IVFFlat comparison benchmark.

Verifies that the Pre-filter Threshold change (Change 1 in modified.md) delivers
real performance improvement over the old IVFFlat-for-everything approach.

- Uses synthetic random float32 vectors (no SIFT1M download required).
- DIM=128 to match SIFT1M dimensionality.
- Tested scenarios:
    A. N=200K, sel≈0.1%  →  M≈200 rows   (pre-filter triggers)
    B. N=200K, sel≈1%    →  M≈2K rows    (pre-filter triggers)
    C. N=200K, sel≈10%   →  M≈20K rows   (control: pre-filter does NOT trigger)

For each scenario:
  OLD = IVFFlat + probes (what advisor recommended before the change)
  NEW = B-tree + exact scan (current advisor after the change)

Results saved to results/prefilter_comparison.json
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

from src.advisor.rules import PREFILTER_ROW_THRESHOLD
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

TABLE   = "bench_prefilter"
DIM     = 128       # same as SIFT1M
N       = 500_000   # large enough to show latency difference
SEED    = 42        # reproducible vectors
Q_SEED  = 99        # reproducible queries
N_QUERIES = 200
WARMUP    = 30
K         = 10
NUM_RUNS  = 3       # repeat query phase, report mean ± std

# OLD path: IVFFlat index built, NO B-tree on filter column.
#   This mirrors the original benchmark setup (compare.py never builds B-trees).
#   PostgreSQL is forced to use IVFFlat for the vector search.
#
# NEW path: No vector index at all, B-tree on filter column only.
#   PostgreSQL uses the B-tree to fetch M matching rows, then does exact NN.
#   This is the pre-filter strategy the updated advisor recommends.

SCENARIOS = [
    # (filter_col,    filter_val, approx_sel,  label,       pre_filter_expected)
    ("category_1000", 0,          0.001,       "~0.1% sel", True),   # M≈500
    ("category_100",  0,          0.010,       "~1.0% sel", True),   # M≈5K
    ("category_10",   0,          0.100,       "~10%  sel", False),  # M≈50K (control)
]

OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prefilter_comparison.json")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_conn():
    conn = psycopg2.connect(**CONN_PARAMS)
    register_vector(conn)
    return conn


def _ivfflat_params(n: int, sel: float) -> tuple[int, int]:
    """Compute IVFFlat (lists, probes) as the OLD advisor would have for a
    filtered workload with selectivity ``sel`` on a dataset of size ``n``."""
    lists = max(10, min(int(math.sqrt(n)), 1000))
    probes_base = max(1, int(math.sqrt(lists)))
    if sel < 0.01:
        probes = min(probes_base * 10, lists)
    elif sel < 0.05:
        probes = min(probes_base * 5, lists)
    elif sel < 0.10:
        probes = min(probes_base * 3, lists)
    else:
        probes = min(probes_base * 2, lists)
    return lists, probes


def _exact_gt(base_vecs: np.ndarray, q_vecs: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    """Exact filtered k-NN via numpy (no Faiss required).

    Returns a (n_queries, k) int64 array of 0-indexed base-vector positions,
    matching the convention that ``compute_recall`` expects (it converts
    PostgreSQL 1-indexed IDs to 0-indexed internally before comparing).
    """
    fvecs = base_vecs[mask]
    orig_ids = np.where(mask)[0]   # 0-indexed positions in base_vecs
    n_q = len(q_vecs)
    actual_k = min(k, len(orig_ids))
    gt = np.full((n_q, k), -1, dtype=np.int64)  # -1 = padding (ignored by compute_recall)
    for i, q in enumerate(q_vecs):
        dists = np.sum((fvecs - q) ** 2, axis=1)
        top_k_local = np.argsort(dists)[:actual_k]
        gt[i, :actual_k] = orig_ids[top_k_local]
    return gt


def _run_filtered_queries(conn, q_vecs: np.ndarray, fcol: str, fval: int, k: int):
    """Execute filtered vector queries; return (result_id_lists, latencies_ms)."""
    all_ids, lats = [], []
    with conn.cursor() as cur:
        for qv in q_vecs:
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {TABLE} "
                f"WHERE {fcol} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (fval, vec_str, k),
            )
            rows = cur.fetchall()
            lats.append((time.perf_counter() - t0) * 1000)
            all_ids.append([r[0] for r in rows])
    return all_ids, lats


def _warmup(conn, q_vecs: np.ndarray, fcol: str, fval: int, k: int, n_warm: int):
    with conn.cursor() as cur:
        for qv in q_vecs[:n_warm]:
            vec_str = "[" + ",".join(f"{v:.6f}" for v in qv) + "]"
            cur.execute(
                f"SELECT id FROM {TABLE} "
                f"WHERE {fcol} = %s "
                f"ORDER BY embedding <-> %s::vector LIMIT %s;",
                (fval, vec_str, k),
            )
            cur.fetchall()


def _drop_vector_indexes(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename=%s "
            "AND (indexdef LIKE '%%hnsw%%' OR indexdef LIKE '%%ivfflat%%');",
            (TABLE,),
        )
        for (name,) in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {name};")
    conn.commit()


def _build_ivfflat(conn, lists: int) -> float:
    _drop_vector_indexes(conn)
    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '512MB';")
        t0 = time.perf_counter()
        cur.execute(
            f"CREATE INDEX bench_ivf ON {TABLE} "
            f"USING ivfflat (embedding vector_l2_ops) WITH (lists={lists});"
        )
        conn.commit()
    return time.perf_counter() - t0


def _build_btree(conn, col: str) -> float:
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS bench_btree_{col};")
        t0 = time.perf_counter()
        cur.execute(f"CREATE INDEX bench_btree_{col} ON {TABLE} ({col});")
        conn.commit()
    return time.perf_counter() - t0


def _drop_btrees(conn, col: str):
    """Drop B-tree index on filter column so the planner cannot use it."""
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS bench_btree_{col};")
    conn.commit()


# ─── Data setup ────────────────────────────────────────────────────────────────

def setup_data():
    """Create the table and insert random vectors + synthetic attributes."""
    print(f"\n{'='*60}")
    print(f"Setting up: {N:,} vectors × {DIM} dims  (seed={SEED})")
    print(f"{'='*60}")

    rng = np.random.default_rng(SEED)
    base_vecs = rng.standard_normal((N, DIM)).astype(np.float32)
    attrs = generate_synthetic_attributes(N, seed=SEED)

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE;")
        cur.execute(f"""
            CREATE TABLE {TABLE} (
                id            SERIAL PRIMARY KEY,
                embedding     vector({DIM}),
                category_10   INT,
                category_100  INT,
                category_1000 INT
            );
        """)
    conn.commit()

    t0 = time.perf_counter()
    BATCH = 500
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
                f"INSERT INTO {TABLE} (embedding, category_10, category_100, category_1000) VALUES %s",
                rows,
                template="(%s::vector, %s, %s, %s)",
            )
    conn.commit()
    load_time = time.perf_counter() - t0
    print(f"  Inserted {N:,} rows in {load_time:.1f}s")
    conn.close()
    return base_vecs, attrs


# ─── Per-scenario benchmark ────────────────────────────────────────────────────

def run_scenario(
    base_vecs: np.ndarray,
    attrs: dict,
    fcol: str,
    fval: int,
    approx_sel: float,
    label: str,
    pre_filter_expected: bool,
):
    rng = np.random.default_rng(Q_SEED)
    q_vecs = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)

    mask = attrs[fcol] == fval
    m = int(mask.sum())
    actual_sel = m / N

    print(f"\n{'-'*60}")
    print(f"Scenario: {label}  |  {fcol}={fval}  |  M={m:,} rows  "
          f"(sel={actual_sel:.3%})  |  pre-filter: {pre_filter_expected}")
    print(f"  PREFILTER_ROW_THRESHOLD = {PREFILTER_ROW_THRESHOLD:,}  |  "
          f"triggers = {m < PREFILTER_ROW_THRESHOLD}")

    gt = _exact_gt(base_vecs, q_vecs, mask, K)
    lists, probes = _ivfflat_params(N, actual_sel)
    print(f"  IVFFlat params: lists={lists}, probes={probes}")

    # ── OLD: IVFFlat only, NO B-tree on filter column ──────────────────────
    # This mirrors the original benchmark (compare.py never builds B-trees).
    # PostgreSQL is forced to use IVFFlat for the ORDER BY embedding <-> query.
    conn = get_conn()
    ivfflat_time = _build_ivfflat(conn, lists)      # vector index only
    _drop_btrees(conn, fcol)                        # ensure no B-tree
    old_build_total = ivfflat_time

    old_recalls, old_p50s, old_p95s = [], [], []
    for _ in range(NUM_RUNS):
        with conn.cursor() as cur:
            cur.execute(f"SET ivfflat.probes = {probes};")
        _warmup(conn, q_vecs, fcol, fval, K, WARMUP)
        ids, lats = _run_filtered_queries(conn, q_vecs, fcol, fval, K)
        p = compute_latency_percentiles(lats)
        old_recalls.append(compute_recall(ids, gt, K))
        old_p50s.append(p["p50"])
        old_p95s.append(p["p95"])
    old_completion = compute_topk_completion_rate(ids, K)
    conn.close()

    old = {
        "method": "OLD (IVFFlat, no B-tree)",
        "index_type": "ivfflat",
        "params": f"lists={lists}, probes={probes}",
        "vector_index_build_s": round(ivfflat_time, 3),
        "total_build_s": round(old_build_total, 3),
        "recall":      round(float(np.mean(old_recalls)), 4),
        "recall_std":  round(float(np.std(old_recalls)),  4),
        "p50_ms":      round(float(np.mean(old_p50s)), 2),
        "p50_std":     round(float(np.std(old_p50s)),  2),
        "p95_ms":      round(float(np.mean(old_p95s)), 2),
        "p95_std":     round(float(np.std(old_p95s)),  2),
        "completion":  round(old_completion, 4),
    }
    print(f"  OLD  build={old['total_build_s']:.2f}s  "
          f"recall={old['recall']:.4f}±{old['recall_std']:.4f}  "
          f"p50={old['p50_ms']:.2f}ms  p95={old['p95_ms']:.2f}ms  "
          f"compl={old['completion']:.1%}")

    # ── NEW: B-tree only (pre-filter exact scan), NO vector index ──────────
    # PostgreSQL uses B-tree to fetch M matching rows, then computes exact NN.
    conn = get_conn()
    btree_time = _build_btree(conn, fcol)
    _drop_vector_indexes(conn)          # ensure no vector index is present
    new_build_total = btree_time

    new_recalls, new_p50s, new_p95s = [], [], []
    for _ in range(NUM_RUNS):
        _warmup(conn, q_vecs, fcol, fval, K, WARMUP)
        ids, lats = _run_filtered_queries(conn, q_vecs, fcol, fval, K)
        p = compute_latency_percentiles(lats)
        new_recalls.append(compute_recall(ids, gt, K))
        new_p50s.append(p["p50"])
        new_p95s.append(p["p95"])
    new_completion = compute_topk_completion_rate(ids, K)
    conn.close()

    new = {
        "method": "NEW (B-tree pre-filter)",
        "index_type": "none (B-tree pre-filter)",
        "params": f"B-tree on {fcol}, exact NN scan",
        "btree_build_s": round(btree_time, 3),
        "vector_index_build_s": 0.0,
        "total_build_s": round(new_build_total, 3),
        "recall":      round(float(np.mean(new_recalls)), 4),
        "recall_std":  round(float(np.std(new_recalls)),  4),
        "p50_ms":      round(float(np.mean(new_p50s)), 2),
        "p50_std":     round(float(np.std(new_p50s)),  2),
        "p95_ms":      round(float(np.mean(new_p95s)), 2),
        "p95_std":     round(float(np.std(new_p95s)),  2),
        "completion":  round(new_completion, 4),
    }
    print(f"  NEW  build={new['total_build_s']:.2f}s  "
          f"recall={new['recall']:.4f}±{new['recall_std']:.4f}  "
          f"p50={new['p50_ms']:.2f}ms  p95={new['p95_ms']:.2f}ms  "
          f"compl={new['completion']:.1%}")

    # ── Improvement summary ─────────────────────────────────────────────────
    build_saved = old["total_build_s"] - new["total_build_s"]
    latency_ratio = old["p95_ms"] / new["p95_ms"] if new["p95_ms"] > 0 else float("inf")
    print(f"  => Build time saved: {build_saved:.2f}s  "
          f"| p95 speedup: {latency_ratio:.1f}×  "
          f"| Recall change: {new['recall'] - old['recall']:+.4f}")

    return {
        "scenario": label,
        "n": N,
        "filter": f"{fcol}={fval}",
        "m_rows": m,
        "selectivity": round(actual_sel, 5),
        "pre_filter_expected": pre_filter_expected,
        "pre_filter_triggered": m < PREFILTER_ROW_THRESHOLD,
        "old": old,
        "new": new,
        "improvement": {
            "build_time_saved_s": round(build_saved, 3),
            "p95_speedup_x": round(latency_ratio, 2),
            "recall_delta": round(new["recall"] - old["recall"], 4),
        },
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Pre-filter vs IVFFlat Benchmark")
    print(f"Dataset: {N:,} vectors × {DIM}-dim  |  k={K}  "
          f"|  queries={N_QUERIES}  |  runs={NUM_RUNS}")
    print(f"PREFILTER_ROW_THRESHOLD = {PREFILTER_ROW_THRESHOLD:,}")

    base_vecs, attrs = setup_data()

    all_results = []
    for fcol, fval, approx_sel, label, pf_expected in SCENARIOS:
        result = run_scenario(
            base_vecs, attrs, fcol, fval, approx_sel, label, pf_expected
        )
        all_results.append(result)

    # ── Print summary table ─────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    hdr = (f"{'Scenario':<14} {'M rows':>7} {'Triggers':>9} "
           f"{'Method':<20} {'Build(s)':>9} {'Recall':>7} {'p50(ms)':>9} "
           f"{'p95(ms)':>9} {'Compl':>7}")
    print(hdr)
    print("-" * 90)
    for r in all_results:
        for which in ("old", "new"):
            d = r[which]
            trig = "YES" if r["pre_filter_triggered"] else "no"
            print(f"{r['scenario']:<14} {r['m_rows']:>7,} {trig:>9} "
                  f"{d['method']:<20} {d['total_build_s']:>9.2f} "
                  f"{d['recall']:>7.4f} {d['p50_ms']:>9.2f} "
                  f"{d['p95_ms']:>9.2f} {d['completion']:>7.1%}")
        imp = r["improvement"]
        print(f"  => saved {imp['build_time_saved_s']:.2f}s build  "
              f"| p95 {imp['p95_speedup_x']:.1f}× faster  "
              f"| recall {imp['recall_delta']:+.4f}")
        print()
    print("=" * 90)

    # ── Save JSON ───────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
