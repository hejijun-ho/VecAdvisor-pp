"""Tests for VecAdvisor++ advisor logic."""

import pytest

from src.advisor.advisor import VecAdvisor
from src.advisor.rules import (
    Recommendation,
    generate_recommendation,
    recommend_hnsw_params,
    recommend_ivfflat_params,
    select_index_type,
)
from src.profiler.workload_profiler import WorkloadProfile, profile_from_params


def _make_profile(**kwargs) -> WorkloadProfile:
    """Create a WorkloadProfile with sensible defaults, overriding with kwargs."""
    defaults = {
        "n_vectors": 1_000_000,
        "dim": 128,
        "k": 10,
        "has_filters": False,
        "filter_selectivity": 1.0,
        "filter_columns": [],
        "update_rate": 0.0,
        "memory_budget_mb": 2048,
        "latency_budget_ms": 50.0,
        "cache_regime": "warm",
    }
    defaults.update(kwargs)
    return WorkloadProfile(**defaults)


class TestIndexTypeSelection:
    def test_small_dataset_no_index(self):
        profile = _make_profile(n_vectors=5000)
        idx_type, _ = select_index_type(profile)
        assert idx_type == "none"

    def test_high_update_rate_ivfflat(self):
        profile = _make_profile(update_rate=0.3)
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_very_selective_filter_prefilter(self):
        # selectivity=0.5%  n=1 M  →  estimated_rows=5 000 < PREFILTER_ROW_THRESHOLD
        # Pre-filter exact scan is faster than any ANN index traversal.
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.005,
            filter_columns=["category_1000"]
        )
        idx_type, reason = select_index_type(profile)
        assert idx_type == "none"
        assert "pre-filter" in reason.lower()

    def test_moderate_filter_hnsw(self):
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.2,
            filter_columns=["category_10"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"

    def test_selective_filter_small_subset_prefilter(self):
        """1% selectivity on 100 K  →  estimated_rows=1 000 < threshold → pre-filter."""
        profile = _make_profile(
            n_vectors=100_000, has_filters=True,
            filter_selectivity=0.01, filter_columns=["category_100"]
        )
        idx_type, reason = select_index_type(profile)
        assert idx_type == "none"
        assert "pre-filter" in reason.lower()

    def test_selective_filter_large_subset_ivfflat(self):
        """1% selectivity on 1 M  →  estimated_rows=10 000 = threshold (boundary, not below) → IVFFlat."""
        profile = _make_profile(
            n_vectors=1_000_000, has_filters=True,
            filter_selectivity=0.01, filter_columns=["category_100"]
        )
        idx_type, _ = select_index_type(profile)
        # 10 000 is NOT below the threshold (strict <), so IVFFlat should be chosen.
        assert idx_type == "ivfflat"

    def test_5pct_selectivity_large_dataset_ivfflat(self):
        """At 5% selectivity on 1M dataset, should pick IVFFlat."""
        profile = _make_profile(
            n_vectors=1_000_000, has_filters=True,
            filter_selectivity=0.05, filter_columns=["category_100"]
        )
        idx_type, _ = select_index_type(profile)
        assert idx_type == "ivfflat"

    def test_default_hnsw(self):
        profile = _make_profile()
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"

    def test_latency_critical_warm_cache_hnsw(self):
        profile = _make_profile(latency_budget_ms=5.0, cache_regime="warm")
        idx_type, _ = select_index_type(profile)
        assert idx_type == "hnsw"


class TestHNSWParams:
    def test_default_params(self):
        profile = _make_profile()
        build, query, _ = recommend_hnsw_params(profile)
        assert build["m"] == 16
        assert build["ef_construction"] == 128
        assert query["ef_search"] >= 40

    def test_high_dim_increases_m(self):
        profile = _make_profile(dim=512)
        build, _, _ = recommend_hnsw_params(profile)
        assert build["m"] == 32

    def test_tight_memory_decreases_m(self):
        profile = _make_profile(memory_budget_mb=512, n_vectors=1_000_000)
        build, _, _ = recommend_hnsw_params(profile)
        assert build["m"] == 8

    def test_filter_increases_ef_search(self):
        profile_no_filter = _make_profile()
        profile_filtered = _make_profile(
            has_filters=True, filter_selectivity=0.05
        )
        _, query_nf, _ = recommend_hnsw_params(profile_no_filter)
        _, query_f, _ = recommend_hnsw_params(profile_filtered)
        assert query_f["ef_search"] > query_nf["ef_search"]

    def test_very_selective_filter_large_ef_search(self):
        profile = _make_profile(has_filters=True, filter_selectivity=0.005)
        _, query, _ = recommend_hnsw_params(profile)
        assert query["ef_search"] >= 400


class TestIVFFlatParams:
    def test_default_params(self):
        profile = _make_profile()
        build, query, _ = recommend_ivfflat_params(profile)
        assert build["lists"] >= 100
        assert query["probes"] >= 1

    def test_large_dataset_more_lists(self):
        profile_small = _make_profile(n_vectors=100_000)
        profile_large = _make_profile(n_vectors=5_000_000)
        build_s, _, _ = recommend_ivfflat_params(profile_small)
        build_l, _, _ = recommend_ivfflat_params(profile_large)
        assert build_l["lists"] > build_s["lists"]

    def test_filter_increases_probes(self):
        profile_nf = _make_profile()
        profile_f = _make_profile(has_filters=True, filter_selectivity=0.01)
        _, query_nf, _ = recommend_ivfflat_params(profile_nf)
        _, query_f, _ = recommend_ivfflat_params(profile_f)
        assert query_f["probes"] > query_nf["probes"]


class TestFullRecommendation:
    def test_generates_recommendation(self):
        profile = _make_profile()
        rec = generate_recommendation(profile)
        assert isinstance(rec, Recommendation)
        assert rec.index_type in ("hnsw", "ivfflat", "none")

    def test_no_index_for_small_dataset(self):
        profile = _make_profile(n_vectors=1000)
        rec = generate_recommendation(profile)
        assert rec.index_type == "none"
        assert len(rec.build_params) == 0

    def test_prefilter_none_with_btree(self):
        """Pre-filter path: index_type=none but B-tree auxiliary index is included."""
        profile = _make_profile(
            n_vectors=500_000, has_filters=True,
            filter_selectivity=0.005,      # 500K × 0.005 = 2 500 rows < 10 000
            filter_columns=["category_1000"],
        )
        rec = generate_recommendation(profile)
        assert rec.index_type == "none"
        assert len(rec.build_params) == 0
        # B-tree on the filter column must be present (it is the primary access path)
        assert len(rec.auxiliary_indexes) == 1
        assert rec.auxiliary_indexes[0]["column"] == "category_1000"
        # Session settings should still be generated
        assert "work_mem" in rec.session_settings

    def test_prefilter_boundary_is_exclusive(self):
        """Exactly PREFILTER_ROW_THRESHOLD rows should NOT trigger pre-filter."""
        from src.advisor.rules import PREFILTER_ROW_THRESHOLD
        # Construct a profile where n * sel == threshold exactly (boundary)
        profile = _make_profile(
            n_vectors=PREFILTER_ROW_THRESHOLD * 100,
            has_filters=True,
            filter_selectivity=1 / 100,   # estimated_rows == PREFILTER_ROW_THRESHOLD
            filter_columns=["category_100"],
        )
        rec = generate_recommendation(profile)
        # Boundary is exclusive (strict <), so ANN index should be chosen
        assert rec.index_type != "none"

    def test_filtered_includes_auxiliary(self):
        profile = _make_profile(
            has_filters=True, filter_selectivity=0.1,
            filter_columns=["category_10"]
        )
        rec = generate_recommendation(profile)
        assert len(rec.auxiliary_indexes) > 0
        assert rec.auxiliary_indexes[0]["column"] == "category_10"

    def test_session_settings_present(self):
        profile = _make_profile(n_vectors=1_000_000)
        rec = generate_recommendation(profile)
        assert "work_mem" in rec.session_settings
        assert "maintenance_work_mem" in rec.session_settings


class TestPartitioning:
    """Tests for the TABLE PARTITIONING + PER-PARTITION HNSW feature."""

    def test_partition_recommended_for_categorical_filter(self):
        """10% selectivity on 500K → 10 partitions of 50K → partitioning triggered."""
        from src.advisor.rules import recommend_partitioning
        profile = _make_profile(
            n_vectors=500_000, has_filters=True,
            filter_selectivity=0.10, filter_columns=["category_10"]
        )
        use_part, col, n_parts = recommend_partitioning(profile)
        assert use_part is True
        assert col == "category_10"
        assert n_parts == 10

    def test_partition_not_triggered_when_prefilter_applies(self):
        """When pre-filter handles M < PREFILTER_ROW_THRESHOLD, no partitioning."""
        from src.advisor.rules import recommend_partitioning
        profile = _make_profile(
            n_vectors=500_000, has_filters=True,
            filter_selectivity=0.005,  # M = 2,500 < 10,000
            filter_columns=["category_1000"]
        )
        use_part, _, _ = recommend_partitioning(profile)
        assert use_part is False

    def test_partition_not_triggered_without_filter(self):
        from src.advisor.rules import recommend_partitioning
        profile = _make_profile(n_vectors=1_000_000)
        use_part, _, _ = recommend_partitioning(profile)
        assert use_part is False

    def test_partition_not_triggered_too_many_partitions(self):
        """0.5% selectivity → round(1/0.005) = 200 > PARTITION_MAX_PARTITIONS."""
        from src.advisor.rules import recommend_partitioning, PARTITION_MAX_PARTITIONS
        profile = _make_profile(
            n_vectors=1_000_000, has_filters=True,
            filter_selectivity=0.005, filter_columns=["category_1000"]
        )
        # M = 5,000 which is < PREFILTER_ROW_THRESHOLD → pre-filter wins anyway
        # (Testing the n_parts > MAX branch with a profile where M >= threshold)
        profile2 = _make_profile(
            n_vectors=10_000_000, has_filters=True,
            filter_selectivity=0.004, filter_columns=["x"]
        )
        # n_parts = round(1/0.004) = 250 > PARTITION_MAX_PARTITIONS
        use_part, _, n_parts = recommend_partitioning(profile2)
        assert use_part is False

    def test_full_recommendation_uses_hnsw_when_partitioned(self):
        """generate_recommendation() on 10%-selectivity filtered workload → HNSW + partitioning."""
        profile = _make_profile(
            n_vectors=500_000, has_filters=True,
            filter_selectivity=0.10, filter_columns=["category_10"]
        )
        rec = generate_recommendation(profile)
        assert rec.use_partitioning is True
        assert rec.partition_column == "category_10"
        assert rec.n_partitions == 10
        assert rec.index_type == "hnsw"

    def test_ivfflat_upgraded_to_hnsw_when_partitioned(self):
        """Workload that would normally pick IVFFlat (selective filter) is upgraded to HNSW
        when partitioning is recommended."""
        # 5% selectivity on 1M → would be IVFFlat (sel ≤ 0.10, n ≥ 50K) without partitioning,
        # but n_partitions = round(1/0.05) = 20, within limits → partitioned HNSW
        profile = _make_profile(
            n_vectors=1_000_000, has_filters=True,
            filter_selectivity=0.05, filter_columns=["category_20"]
        )
        rec = generate_recommendation(profile)
        assert rec.use_partitioning is True
        assert rec.n_partitions == 20
        assert rec.index_type == "hnsw"

    def test_partition_sql_ddl_contains_child_tables(self):
        """generate_partition_ddl_sql() should output CREATE TABLE for each partition."""
        from src.advisor.sql_generator import generate_partition_ddl_sql
        profile = _make_profile(
            n_vectors=500_000, has_filters=True,
            filter_selectivity=0.10, filter_columns=["category_10"]
        )
        rec = generate_recommendation(profile)
        assert rec.use_partitioning is True
        ddl = generate_partition_ddl_sql(rec, "my_table", 128)
        for v in range(10):
            assert f"my_table_p{v}" in ddl
        assert "PARTITION BY LIST" in ddl
        assert "hnsw" in ddl.lower()

    def test_no_partitioning_fields_when_not_recommended(self):
        """Recommendations without partitioning should have default (False/empty/0) fields."""
        profile = _make_profile()  # no filters → no partitioning
        rec = generate_recommendation(profile)
        assert rec.use_partitioning is False
        assert rec.partition_column == ""
        assert rec.n_partitions == 0


class TestVecAdvisorInterface:
    def test_analyze_from_params(self):
        advisor = VecAdvisor()
        rec = advisor.analyze_from_params(
            n_vectors=500_000, dim=128, k=10,
            has_filters=True, filter_selectivity=0.05,
            filter_columns=["category_100"],
        )
        assert rec.index_type in ("hnsw", "ivfflat")
        assert len(rec.explanation) > 0

    def test_get_sql(self):
        advisor = VecAdvisor()
        rec = advisor.analyze_from_params(
            n_vectors=500_000, dim=128, k=10,
        )
        sql = advisor.get_sql(rec, "test_table")
        assert "CREATE INDEX" in sql or "No vector index" in sql
