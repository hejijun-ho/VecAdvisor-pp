"""Tests for vecadvisor_rs Rust extension.

All tests compare Rust output against the pure-Python reference implementations
and are skipped if vecadvisor_rs is not built.

Build the extension first with:
    .venv/bin/maturin develop --release --manifest-path vecadvisor_rs/Cargo.toml
"""

import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import vecadvisor_rs as rs
    RUST_AVAILABLE = hasattr(rs, "read_fvecs")
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="vecadvisor_rs not built — run: .venv/bin/maturin develop --release",
)


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_loader.py)
# ---------------------------------------------------------------------------

def _write_fvecs(filepath: str, vectors: np.ndarray):
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}f", *vec))


def _write_ivecs(filepath: str, vectors: np.ndarray):
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}i", *vec))


def _py_read_fvecs(filepath: str) -> np.ndarray:
    """Pure Python reference for fvecs parsing."""
    vectors = []
    with open(filepath, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}f", f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def _py_read_ivecs(filepath: str) -> np.ndarray:
    """Pure Python reference for ivecs parsing."""
    vectors = []
    with open(filepath, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack("i", dim_bytes)[0]
            vec = struct.unpack(f"{dim}i", f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)


def _py_compute_recall(result_ids, ground_truth_ids, k):
    """Pure Python reference for recall computation."""
    total_recall = 0.0
    n_queries = len(result_ids)
    for i in range(n_queries):
        returned = set(rid - 1 for rid in result_ids[i][:k])
        true_nn = set(ground_truth_ids[i][:k])
        true_nn.discard(-1)
        if len(true_nn) == 0:
            total_recall += 1.0
        else:
            total_recall += len(returned & true_nn) / len(true_nn)
    return total_recall / n_queries if n_queries > 0 else 0.0


# ---------------------------------------------------------------------------
# TestReadFvecs
# ---------------------------------------------------------------------------

class TestReadFvecs:
    def test_matches_pure_python(self):
        """Rust output must be byte-identical to Python _read_fvecs."""
        rng = np.random.RandomState(0)
        expected = rng.rand(100, 128).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            fpath = f.name
        try:
            _write_fvecs(fpath, expected)
            rust_result = rs.read_fvecs(fpath)
            py_result = _py_read_fvecs(fpath)
            assert rust_result.dtype == np.float32
            assert rust_result.shape == (100, 128)
            np.testing.assert_array_equal(rust_result, py_result)
        finally:
            os.unlink(fpath)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            fpath = f.name
        try:
            result = rs.read_fvecs(fpath)
            assert len(result) == 0
        finally:
            os.unlink(fpath)

    def test_single_vector(self):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            fpath = f.name
        try:
            _write_fvecs(fpath, vec.reshape(1, -1))
            result = rs.read_fvecs(fpath)
            assert result.shape == (1, 3)
            np.testing.assert_array_equal(result[0], vec)
        finally:
            os.unlink(fpath)

    def test_multiple_dimensions(self):
        """Test with non-128 dimension vectors."""
        rng = np.random.RandomState(42)
        for dim in [4, 64, 256]:
            vecs = rng.rand(10, dim).astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
                fpath = f.name
            try:
                _write_fvecs(fpath, vecs)
                result = rs.read_fvecs(fpath)
                np.testing.assert_array_equal(result, vecs)
            finally:
                os.unlink(fpath)


# ---------------------------------------------------------------------------
# TestReadIvecs
# ---------------------------------------------------------------------------

class TestReadIvecs:
    def test_matches_pure_python(self):
        rng = np.random.RandomState(0)
        expected = rng.randint(0, 1000000, (10000, 100)).astype(np.int32)
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            fpath = f.name
        try:
            _write_ivecs(fpath, expected)
            rust_result = rs.read_ivecs(fpath)
            py_result = _py_read_ivecs(fpath)
            assert rust_result.dtype == np.int32
            assert rust_result.shape == (10000, 100)
            np.testing.assert_array_equal(rust_result, py_result)
        finally:
            os.unlink(fpath)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            fpath = f.name
        try:
            result = rs.read_ivecs(fpath)
            assert len(result) == 0
        finally:
            os.unlink(fpath)

    def test_negative_values(self):
        """Ivecs can contain any int32 including negatives in general."""
        vecs = np.array([[0, 100, 999], [1, 50, 200]], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            fpath = f.name
        try:
            _write_ivecs(fpath, vecs)
            result = rs.read_ivecs(fpath)
            np.testing.assert_array_equal(result, vecs)
        finally:
            os.unlink(fpath)


# ---------------------------------------------------------------------------
# TestBuildInsertRows
# ---------------------------------------------------------------------------

class TestBuildInsertRows:
    def _make_inputs(self, n=50, dim=128, seed=42):
        rng = np.random.RandomState(seed)
        vectors = rng.rand(n, dim).astype(np.float32)
        attrs = {
            "category_10":   rng.randint(0, 10, n).astype(np.int64),
            "category_100":  rng.randint(0, 100, n).astype(np.int64),
            "category_1000": rng.randint(0, 1000, n).astype(np.int64),
            "price":         rng.uniform(0, 1000, n).astype(np.float64),
            "is_active":     rng.randint(0, 2, n).astype(np.uint8),
        }
        return vectors, attrs

    def _call_rust(self, vectors, attrs):
        return rs.build_insert_rows(
            np.ascontiguousarray(vectors, dtype=np.float32),
            attrs["category_10"].astype(np.int64),
            attrs["category_100"].astype(np.int64),
            attrs["category_1000"].astype(np.int64),
            attrs["price"].astype(np.float64),
            attrs["is_active"].astype(np.uint8),
        )

    def test_row_count(self):
        vectors, attrs = self._make_inputs(n=50)
        rows = self._call_rust(vectors, attrs)
        assert len(rows) == 50

    def test_vector_is_python_list(self):
        vectors, attrs = self._make_inputs(n=5)
        rows = self._call_rust(vectors, attrs)
        for row in rows:
            assert isinstance(row[0], list), f"Expected list, got {type(row[0])}"

    def test_vector_length_matches_dim(self):
        vectors, attrs = self._make_inputs(n=5, dim=128)
        rows = self._call_rust(vectors, attrs)
        for row in rows:
            assert len(row[0]) == 128

    def test_vector_values_match_tolist(self):
        vectors, attrs = self._make_inputs(n=20, dim=16)
        rows = self._call_rust(vectors, attrs)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(row[0], vectors[i].tolist(), rtol=1e-6)

    def test_categorical_attrs_are_ints(self):
        vectors, attrs = self._make_inputs(n=5)
        rows = self._call_rust(vectors, attrs)
        for i, row in enumerate(rows):
            assert row[1] == int(attrs["category_10"][i])
            assert row[2] == int(attrs["category_100"][i])
            assert row[3] == int(attrs["category_1000"][i])

    def test_price_is_float(self):
        vectors, attrs = self._make_inputs(n=5)
        rows = self._call_rust(vectors, attrs)
        for i, row in enumerate(rows):
            assert abs(row[4] - float(attrs["price"][i])) < 1e-9

    def test_is_active_is_bool(self):
        vectors, attrs = self._make_inputs(n=20)
        rows = self._call_rust(vectors, attrs)
        for i, row in enumerate(rows):
            assert isinstance(row[5], bool), f"Expected bool, got {type(row[5])}"
            expected = bool(attrs["is_active"][i])
            assert row[5] == expected

    def test_tuple_length(self):
        """Each row must have 6 elements: vec + 5 attributes."""
        vectors, attrs = self._make_inputs(n=5)
        rows = self._call_rust(vectors, attrs)
        for row in rows:
            assert len(row) == 6


# ---------------------------------------------------------------------------
# TestComputeRecall
# ---------------------------------------------------------------------------

class TestComputeRecall:
    def test_perfect_recall(self):
        """All returned IDs match ground truth → recall=1.0."""
        gt = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        result_ids = [[1, 2, 3], [4, 5, 6]]  # 1-indexed: subtract 1 → [0,1,2],[3,4,5]
        rust_val = rs.compute_recall(result_ids, gt, k=3)
        py_val = _py_compute_recall(result_ids, gt, k=3)
        assert abs(rust_val - 1.0) < 1e-12
        assert abs(rust_val - py_val) < 1e-12

    def test_zero_recall(self):
        """No overlap → recall=0.0."""
        gt = np.array([[0, 1, 2]], dtype=np.int64)
        result_ids = [[10, 11, 12]]  # 0-indexed: 9,10,11 — no overlap with 0,1,2
        rust_val = rs.compute_recall(result_ids, gt, k=3)
        py_val = _py_compute_recall(result_ids, gt, k=3)
        assert abs(rust_val - 0.0) < 1e-12
        assert abs(rust_val - py_val) < 1e-12

    def test_partial_recall(self):
        """Partial overlap → matches Python computation."""
        gt = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        # Return 3 correct (IDs 1,2,3 → 0-indexed 0,1,2) + 2 wrong
        result_ids = [[1, 2, 3, 10, 11]]
        rust_val = rs.compute_recall(result_ids, gt, k=5)
        py_val = _py_compute_recall(result_ids, gt, k=5)
        assert abs(rust_val - py_val) < 1e-12
        assert abs(rust_val - 3.0 / 5.0) < 1e-12

    def test_padded_ground_truth(self):
        """Ground truth rows with -1 padding are handled correctly."""
        gt = np.array([[0, 1, -1], [3, -1, -1]], dtype=np.int64)
        result_ids = [[1, 2], [4]]
        rust_val = rs.compute_recall(result_ids, gt, k=3)
        py_val = _py_compute_recall(result_ids, gt, k=3)
        assert abs(rust_val - py_val) < 1e-12

    def test_large_random_matches_python(self):
        """Large random case: Rust and Python must agree to float precision."""
        rng = np.random.RandomState(99)
        n_queries, k = 1000, 10
        gt = rng.randint(0, 10000, (n_queries, k)).astype(np.int64)
        result_ids = [
            [int(x) + 1 for x in rng.choice(10000, k, replace=False)]
            for _ in range(n_queries)
        ]
        rust_val = rs.compute_recall(result_ids, gt, k=k)
        py_val = _py_compute_recall(result_ids, gt, k=k)
        assert abs(rust_val - py_val) < 1e-10

    def test_empty_result_ids(self):
        gt = np.array([[0, 1, 2]], dtype=np.int64)
        result_ids = [[]]
        rust_val = rs.compute_recall(result_ids, gt, k=3)
        py_val = _py_compute_recall(result_ids, gt, k=3)
        assert abs(rust_val - py_val) < 1e-12

    def test_empty_queries(self):
        gt = np.zeros((0, 10), dtype=np.int64)
        result_ids: list = []
        rust_val = rs.compute_recall(result_ids, gt, k=10)
        assert rust_val == 0.0
