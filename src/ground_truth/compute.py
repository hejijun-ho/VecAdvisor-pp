"""Ground-truth computation using exact nearest-neighbor search via Faiss.

Provides brute-force exact NN for both unfiltered and filtered queries,
which is required for accurate recall measurement.
"""

import os

import faiss
import numpy as np


def compute_ground_truth(
    base_vectors: np.ndarray, query_vectors: np.ndarray, k: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact k-nearest neighbors using Faiss IndexFlatL2.

    Args:
        base_vectors: (n, dim) float32 database vectors.
        query_vectors: (nq, dim) float32 query vectors.
        k: Number of nearest neighbors to retrieve.

    Returns:
        Tuple of (distances, indices):
            - distances: (nq, k) float32 L2 distances.
            - indices: (nq, k) int64 neighbor indices into base_vectors.
    """
    base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    dim = base_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(base_vectors)

    distances, indices = index.search(query_vectors, k)
    return distances, indices


def compute_filtered_ground_truth(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    filter_mask: np.ndarray,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact k-NN restricted to vectors passing a filter.

    Args:
        base_vectors: (n, dim) float32 database vectors.
        query_vectors: (nq, dim) float32 query vectors.
        filter_mask: (n,) boolean array. True = vector passes the filter.
        k: Number of nearest neighbors.

    Returns:
        Tuple of (distances, indices):
            - distances: (nq, k) float32. Padded with inf if fewer than k pass.
            - indices: (nq, k) int64. Padded with -1 if fewer than k pass.
              Indices refer to positions in the ORIGINAL base_vectors array.
    """
    base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    # Get indices of vectors that pass the filter
    passing_indices = np.where(filter_mask)[0]
    if len(passing_indices) == 0:
        nq = query_vectors.shape[0]
        return (
            np.full((nq, k), np.inf, dtype=np.float32),
            np.full((nq, k), -1, dtype=np.int64),
        )

    filtered_vectors = base_vectors[passing_indices]
    actual_k = min(k, len(passing_indices))

    dim = base_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(filtered_vectors)

    distances, local_indices = index.search(query_vectors, actual_k)

    # Map local indices back to original base_vectors indices (vectorized)
    nq = query_vectors.shape[0]
    global_indices = np.full((nq, k), -1, dtype=np.int64)
    global_distances = np.full((nq, k), np.inf, dtype=np.float32)

    valid = local_indices[:, :actual_k] >= 0
    global_indices[:, :actual_k][valid] = passing_indices[local_indices[:, :actual_k][valid]]
    global_distances[:, :actual_k][valid] = distances[:, :actual_k][valid]

    return global_distances, global_indices


def save_ground_truth(
    distances: np.ndarray, indices: np.ndarray, path: str
) -> None:
    """Save ground-truth results to disk.

    Args:
        distances: (nq, k) distance array.
        indices: (nq, k) index array.
        path: Directory to save to.
    """
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "gt_distances.npy"), distances)
    np.save(os.path.join(path, "gt_indices.npy"), indices)


def load_ground_truth(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load ground-truth results from disk.

    Args:
        path: Directory containing gt_distances.npy and gt_indices.npy.

    Returns:
        Tuple of (distances, indices).
    """
    distances = np.load(os.path.join(path, "gt_distances.npy"))
    indices = np.load(os.path.join(path, "gt_indices.npy"))
    return distances, indices
