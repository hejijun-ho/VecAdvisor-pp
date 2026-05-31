# VecAdvisor++

**Filter-Aware Vector Index Selection and Tuning for PostgreSQL**

VecAdvisor++ is a workload-aware vector index advisor for PostgreSQL with pgvector. It selects between HNSW and IVFFlat indexes, recommends build-time and query-time parameters, and explicitly accounts for filtered vector queries — a major pain point in practice where ANN indexes often return too few results.

## Research Foundations

This project is grounded in modern DBMS research:

- **ICDE 2024** — *Are There Fundamental Limitations in Supporting Vector Data Management in Relational Databases?* Demonstrates that performance gaps between PostgreSQL and specialized ANN systems stem from engineering/tuning issues, not theoretical limits.
- **PVLDB 2025** — *Turbocharging Vector Databases Using Modern SSDs.* Shows disk-resident ANN performance is dominated by cache behavior and I/O efficiency.
- **VLDB Journal 2024** — *Vector Databases: A Survey.* Defines the design space of vector DBMSs and discusses hybrid workload tradeoffs.
- **pgvector Documentation** — Provides the real DBMS knobs (HNSW, IVFFlat, ef_search, lists, memory limits).

## Quick Start

### Prerequisites

- Python 3.11.2
- PostgreSQL (with pgvector extension installed)

### Setup

```bash
# 1. Create virtual environment with uv
cd /path/to/VecAdvisor++
uv venv --python 3.11
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Set up the database
bash scripts/setup_db.sh

# 4. Run tests
python -m pytest tests/ -v
```

### Get Index Recommendations (No Database Required)

```bash
source venv/bin/activate

# Pure vector workload (1M vectors, 128-dim)
python scripts/run_advisor.py --n 1000000 --dim 128 --k 10

# Filtered workload with 1% selectivity
python scripts/run_advisor.py --n 1000000 --dim 128 --k 10 \
    --filter-selectivity 0.01 --filter-columns category_100

# Very selective filter (0.5%)
python scripts/run_advisor.py --n 1000000 --dim 128 --k 10 \
    --filter-selectivity 0.005 --filter-columns category_1000

# High write-rate workload
python scripts/run_advisor.py --n 500000 --dim 128 --k 10 --update-rate 0.3
```

### Run End-to-End Benchmark

```bash
source venv/bin/activate

# Small scale (fast, for validation)
python scripts/run_benchmark.py --n-base 10000 --n-queries 100 --k 10

# Medium scale
python scripts/run_benchmark.py --n-base 100000 --n-queries 1000 --k 10

# Full SIFT1M
python scripts/run_benchmark.py --n-base 1000000 --n-queries 10000 --k 10

# Cold-cache benchmark
python scripts/run_benchmark.py --n-base 10000 --n-queries 100 --k 10 --cache-mode cold
```

## Project Structure

```
VecAdvisor++/
├── src/
│   ├── data/               # Dataset loading & PostgreSQL schema
│   │   ├── loader.py       # SIFT1M download/parsing (fvecs/ivecs)
│   │   └── schema.py       # Table creation, insertion, index management
│   ├── ground_truth/       # Exact NN computation
│   │   └── compute.py      # Faiss IndexFlatL2 (pure + filtered)
│   ├── benchmark/          # Benchmark harness
│   │   ├── workload.py     # Query generation (pure + filtered)
│   │   ├── runner.py       # Query execution (cold/warm cache)
│   │   └── metrics.py      # Recall@k, latency, completion rate
│   ├── profiler/           # Workload characterization
│   │   └── workload_profiler.py  # Feature extraction + EXPLAIN selectivity
│   ├── advisor/            # Core advisor (main contribution)
│   │   ├── rules.py        # Decision tree for index selection & tuning
│   │   ├── advisor.py      # Orchestrator
│   │   └── sql_generator.py # Executable SQL output
│   └── evaluation/         # Comparison framework
│       ├── compare.py      # Baseline vs advisor benchmarks
│       └── visualize.py    # Matplotlib plots
├── tests/                  # 53 unit tests (all pure Python)
├── scripts/
│   ├── run_advisor.py      # Advisor CLI
│   ├── run_benchmark.py    # Full benchmark CLI
│   └── setup_db.sh         # Database setup
├── config/
│   └── default.yaml        # Configuration
└── results/                # Benchmark outputs (plots, logs, JSON)
```

## Advisor Decision Logic

### Index Type Selection

| Condition | Recommendation | Rationale |
|-----------|---------------|-----------|
| n < 10,000 | Sequential scan | Index overhead not justified |
| update_rate > 20% | IVFFlat | Cheaper index rebuilds |
| filter_selectivity < 1% | IVFFlat | HNSW graph traversal struggles with very selective filters |
| filter_selectivity > 10%, n > 100K | HNSW | Good recall with moderate filters |
| Latency-critical, warm cache | HNSW | Faster query processing |
| Default | HNSW | Better recall-latency tradeoff |

### Filtered Query Mitigation

- **Selectivity < 5%**: Increase ef_search/probes 4x
- **Selectivity < 1%**: Increase ef_search/probes 8-10x
- **Always**: Recommend B-tree indexes on filter columns
- **Completion rate < 100%**: Iterative widening strategy

### Output

VecAdvisor++ produces concrete, executable SQL:

```sql
-- Session Settings
SET hnsw.ef_search = 160;
SET work_mem = '256MB';
SET maintenance_work_mem = '512MB';

-- Vector Index
CREATE INDEX idx_vectors_hnsw ON vectors
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 128);

-- Auxiliary Indexes (for filter columns)
CREATE INDEX IF NOT EXISTS idx_vectors_category_100_btree
ON vectors (category_100);
```

## Evaluation Metrics

- **Recall@k** — Fraction of true top-k neighbors found
- **p50/p95 Latency** — Query latency percentiles
- **Top-k Completion Rate** — Fraction of queries returning exactly k results (critical for filtered queries)
- **Index Build Time** — Time to construct the index
- **Memory/Disk Usage** — Index footprint

## Baselines

1. **pgvector defaults** — HNSW with m=16, ef_construction=64, ef_search=40
2. **Common heuristics** — HNSW with m=16, ef_construction=128, ef_search=100
3. **IVFFlat defaults** — IVFFlat with lists=1000, probes=10

## Configuration

Edit `config/default.yaml` to configure database connection, dataset parameters, and benchmark settings.

## Testing

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

All 53 tests are pure Python (no database dependency required).
