"""Tests for data loader utilities."""

import os
import struct
import tempfile

import numpy as np
import pytest

from src.data.loader import _read_fvecs, _read_ivecs, load_subset


def _write_fvecs(filepath: str, vectors: np.ndarray):
    """Write vectors in fvecs format for testing."""
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}f", *vec))


def _write_ivecs(filepath: str, vectors: np.ndarray):
    """Write vectors in ivecs format for testing."""
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("i", dim))
            f.write(struct.pack(f"{dim}i", *vec))


class TestFvecsReader:
    def test_read_fvecs(self):
        """Test reading fvecs format."""
        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            _write_fvecs(f.name, expected)
            result = _read_fvecs(f.name)

        np.testing.assert_array_almost_equal(result, expected)
        os.unlink(f.name)

    def test_read_empty_fvecs(self):
        """Test reading empty fvecs file."""
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            pass  # empty file
        result = _read_fvecs(f.name)
        assert len(result) == 0
        os.unlink(f.name)


class TestIvecsReader:
    def test_read_ivecs(self):
        """Test reading ivecs format."""
        expected = np.array([
            [10, 20, 30],
            [40, 50, 60],
        ], dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            _write_ivecs(f.name, expected)
            result = _read_ivecs(f.name)

        np.testing.assert_array_equal(result, expected)
        os.unlink(f.name)


class TestLoadSubset:
    def test_full_subset(self):
        base = np.random.rand(100, 8).astype(np.float32)
        queries = np.random.rand(10, 8).astype(np.float32)
        b, q = load_subset(base, queries)
        assert len(b) == 100
        assert len(q) == 10

    def test_limited_subset(self):
        base = np.random.rand(100, 8).astype(np.float32)
        queries = np.random.rand(10, 8).astype(np.float32)
        b, q = load_subset(base, queries, n_base=50, n_queries=5)
        assert len(b) == 50
        assert len(q) == 5
