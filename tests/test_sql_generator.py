"""Tests for SQL generator."""

import pytest

from src.advisor.rules import Recommendation
from src.advisor.sql_generator import (
    generate_auxiliary_index_sql,
    generate_create_index_sql,
    generate_full_recommendation_sql,
    generate_query_sql,
    generate_session_settings_sql,
)


def test_generate_hnsw_index_sql():
    rec = Recommendation(
        index_type="hnsw",
        build_params={"m": 32, "ef_construction": 256},
    )
    sql = generate_create_index_sql(rec, "vectors")
    assert "USING hnsw" in sql
    assert "m = 32" in sql
    assert "ef_construction = 256" in sql
    assert "vector_l2_ops" in sql


def test_generate_ivfflat_index_sql():
    rec = Recommendation(
        index_type="ivfflat",
        build_params={"lists": 500},
    )
    sql = generate_create_index_sql(rec, "vectors")
    assert "USING ivfflat" in sql
    assert "lists = 500" in sql


def test_generate_no_index():
    rec = Recommendation(index_type="none")
    sql = generate_create_index_sql(rec, "vectors")
    assert sql is None


def test_session_settings_hnsw():
    rec = Recommendation(
        index_type="hnsw",
        query_params={"ef_search": 200},
        session_settings={"work_mem": "256MB"},
    )
    stmts = generate_session_settings_sql(rec)
    assert any("hnsw.ef_search = 200" in s for s in stmts)
    assert any("work_mem" in s for s in stmts)


def test_session_settings_ivfflat():
    rec = Recommendation(
        index_type="ivfflat",
        query_params={"probes": 50},
    )
    stmts = generate_session_settings_sql(rec)
    assert any("ivfflat.probes = 50" in s for s in stmts)


def test_auxiliary_index_sql():
    rec = Recommendation(
        index_type="hnsw",
        auxiliary_indexes=[
            {"column": "category_100", "index_type": "btree", "reason": "test"},
        ],
    )
    stmts = generate_auxiliary_index_sql(rec, "vectors")
    assert len(stmts) == 1
    assert "category_100" in stmts[0]
    assert "USING btree" not in stmts[0]  # default index type, no explicit USING
    assert "CREATE INDEX" in stmts[0]


def test_generate_query_sql_pure():
    sql = generate_query_sql("vectors", "'[1,2,3]'", None, 10)
    assert "ORDER BY" in sql
    assert "LIMIT 10" in sql
    assert "WHERE" not in sql


def test_generate_query_sql_filtered():
    sql = generate_query_sql("vectors", "'[1,2,3]'", "category_10 = 5", 10)
    assert "WHERE category_10 = 5" in sql
    assert "ORDER BY" in sql
    assert "LIMIT 10" in sql


def test_full_recommendation_sql():
    rec = Recommendation(
        index_type="hnsw",
        build_params={"m": 16, "ef_construction": 128},
        query_params={"ef_search": 100},
        auxiliary_indexes=[
            {"column": "category", "index_type": "btree", "reason": "test"},
        ],
        session_settings={"work_mem": "128MB"},
    )
    sql = generate_full_recommendation_sql(rec, "vectors")
    assert "VecAdvisor++" in sql
    assert "CREATE INDEX" in sql
    assert "hnsw.ef_search = 100" in sql
    assert "work_mem" in sql
    assert "category" in sql
