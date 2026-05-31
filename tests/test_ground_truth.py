"""Tests for ground-truth computation."""

import os
import tempfile

import numpy as np
import pytest

from src.ground_truth.compute import (
    compute_filtered_ground_truth,
    compute_ground_truth,
    load_ground_truth,
    save_ground_truth,
)


def test_compute_ground_truth_basic():
    """Test exact NN computation with known vectors."""
    # Create simple vectors where ground truth is obvious
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ], dtype=np.float32)

    query = np.array([[0.0, 0.0]], dtype=np.float32)

    distances, indices = compute_ground_truth(base, query, k=3)

    assert indices.shape == (1, 3)
    assert indices[0, 0] == 0  # closest is itself
    assert set(indices[0, :2]) == {0, 1} or set(indices[0, :2]) == {0, 2}


def test_compute_ground_truth_k():
    """Test that k parameter is respected."""
    rng = np.random.RandomState(42)
    base = rng.rand(100, 8).astype(np.float32)
    query = rng.rand(5, 8).astype(np.float32)

    for k in [1, 5, 10]:
        distances, indices = compute_ground_truth(base, query, k=k)
        assert indices.shape == (5, k)
        assert distances.shape == (5, k)


def test_compute_filtered_ground_truth():
    """Test filtered NN computation."""
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ], dtype=np.float32)

    query = np.array([[0.0, 0.0]], dtype=np.float32)

    # Only include vectors at indices 2, 3, 4
    mask = np.array([False, False, True, True, True])

    distances, indices = compute_filtered_ground_truth(base, query, mask, k=2)

    assert indices.shape == (1, 2)
    # Closest filtered vector to [0,0] is [0,1] at index 2
    assert indices[0, 0] == 2


def test_compute_filtered_empty_filter():
    """Test filtered NN when no vectors pass the filter."""
    base = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    mask = np.array([False, False])

    distances, indices = compute_filtered_ground_truth(base, query, mask, k=1)
    assert indices[0, 0] == -1
    assert distances[0, 0] == np.inf


def test_save_load_ground_truth():
    """Test ground truth persistence."""
    rng = np.random.RandomState(42)
    distances = rng.rand(10, 5).astype(np.float32)
    indices = rng.randint(0, 100, (10, 5)).astype(np.int64)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_ground_truth(distances, indices, tmpdir)
        loaded_dist, loaded_idx = load_ground_truth(tmpdir)

        np.testing.assert_array_almost_equal(loaded_dist, distances)
        np.testing.assert_array_equal(loaded_idx, indices)
