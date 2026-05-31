"""Dataset download and loading utilities for VecAdvisor++.

Supports SIFT1M and other standard ANN benchmark datasets in fvecs/ivecs format.
"""

import os
import struct
import tarfile
import urllib.request

import numpy as np

try:
    import vecadvisor_rs as _rs
    # Guard against the vecadvisor_rs/ source directory being imported as a
    # namespace package (e.g. on servers where the .so hasn't been built yet).
    _RUST_AVAILABLE = hasattr(_rs, "read_fvecs")
except ImportError:
    _RUST_AVAILABLE = False


SIFT1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_FILENAME = "sift.tar.gz"

GIST1M_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
GIST1M_FILENAME = "gist.tar.gz"


def _read_fvecs(filepath: str) -> np.ndarray:
    """Read vectors from fvecs format file.

    fvecs format: each vector is stored as [dim (4 bytes int)] [dim floats (4 bytes each)].
    Uses Rust-accelerated parser if vecadvisor_rs is available.
    """
    if _RUST_AVAILABLE:
        return _rs.read_fvecs(filepath)
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


def _read_ivecs(filepath: str) -> np.ndarray:
    """Read integer vectors from ivecs format file (used for ground truth indices).

    Uses Rust-accelerated parser if vecadvisor_rs is available.
    """
    if _RUST_AVAILABLE:
        return _rs.read_ivecs(filepath)
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


def download_sift1m(data_dir: str) -> None:
    """Download and extract the SIFT1M dataset if not already present.

    Args:
        data_dir: Directory to store the dataset files.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Check if already extracted
    sift_dir = os.path.join(data_dir, "sift")
    if os.path.isdir(sift_dir) and os.path.isfile(
        os.path.join(sift_dir, "sift_base.fvecs")
    ):
        print(f"SIFT1M dataset already exists at {sift_dir}")
        return

    archive_path = os.path.join(data_dir, SIFT1M_FILENAME)
    if not os.path.isfile(archive_path):
        print(f"Downloading SIFT1M from {SIFT1M_URL} ...")
        urllib.request.urlretrieve(SIFT1M_URL, archive_path)
        print("Download complete.")

    print("Extracting SIFT1M archive ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"Extraction complete. Files at {sift_dir}")


def load_sift1m(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the SIFT1M dataset from disk.

    Args:
        data_dir: Root directory containing the extracted 'sift/' folder.

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth_ids):
            - base_vectors: (1_000_000, 128) float32
            - query_vectors: (10_000, 128) float32
            - ground_truth_ids: (10_000, 100) int32 — true NN indices
    """
    sift_dir = os.path.join(data_dir, "sift")

    base_vectors = _read_fvecs(os.path.join(sift_dir, "sift_base.fvecs"))
    query_vectors = _read_fvecs(os.path.join(sift_dir, "sift_query.fvecs"))
    ground_truth_ids = _read_ivecs(os.path.join(sift_dir, "sift_groundtruth.ivecs"))

    print(
        f"Loaded SIFT1M: base={base_vectors.shape}, "
        f"queries={query_vectors.shape}, gt={ground_truth_ids.shape}"
    )
    return base_vectors, query_vectors, ground_truth_ids


def load_fvecs_dataset(
    data_dir: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic loader for any fvecs-format ANN benchmark dataset.

    Expects files inside data_dir named:
        {prefix}_base.fvecs, {prefix}_query.fvecs, {prefix}_groundtruth.ivecs

    Args:
        data_dir: Directory containing the dataset files.
        prefix: Filename prefix (e.g. "sift", "gist").

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth_ids).
    """
    base_vectors = _read_fvecs(os.path.join(data_dir, f"{prefix}_base.fvecs"))
    query_vectors = _read_fvecs(os.path.join(data_dir, f"{prefix}_query.fvecs"))
    ground_truth_ids = _read_ivecs(os.path.join(data_dir, f"{prefix}_groundtruth.ivecs"))
    print(
        f"Loaded {prefix}: base={base_vectors.shape}, "
        f"queries={query_vectors.shape}, gt={ground_truth_ids.shape}"
    )
    return base_vectors, query_vectors, ground_truth_ids


def download_gist1m(data_dir: str) -> None:
    """Download and extract the GIST1M dataset if not already present.

    GIST1M: 1,000,000 base vectors × 960 dimensions (GIST descriptors).
    1,000 query vectors, 100 ground-truth neighbours each.
    ~3.6 GB on disk after extraction.

    Args:
        data_dir: Directory to store the dataset files.
    """
    os.makedirs(data_dir, exist_ok=True)

    gist_dir = os.path.join(data_dir, "gist")
    if os.path.isdir(gist_dir) and os.path.isfile(
        os.path.join(gist_dir, "gist_base.fvecs")
    ):
        print(f"GIST1M dataset already exists at {gist_dir}")
        return

    archive_path = os.path.join(data_dir, GIST1M_FILENAME)
    if not os.path.isfile(archive_path):
        print(f"Downloading GIST1M from {GIST1M_URL} (~3.6 GB, this may take a while)...")
        urllib.request.urlretrieve(GIST1M_URL, archive_path)
        print("Download complete.")

    print("Extracting GIST1M archive ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"Extraction complete. Files at {gist_dir}")


def load_gist1m(
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the GIST1M dataset from disk.

    Args:
        data_dir: Root directory containing the extracted 'gist/' folder.

    Returns:
        Tuple of (base_vectors, query_vectors, ground_truth_ids):
            - base_vectors: (1_000_000, 960) float32
            - query_vectors: (1_000, 960) float32
            - ground_truth_ids: (1_000, 100) int32 — true NN indices
    """
    gist_dir = os.path.join(data_dir, "gist")
    return load_fvecs_dataset(gist_dir, "gist")


def load_subset(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    n_base: int | None = None,
    n_queries: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a subset of the dataset for faster experimentation.

    Args:
        base_vectors: Full base vectors array.
        query_vectors: Full query vectors array.
        n_base: Number of base vectors to use (None = all).
        n_queries: Number of query vectors to use (None = all).

    Returns:
        Tuple of (base_subset, query_subset).
    """
    base = base_vectors[:n_base] if n_base else base_vectors
    queries = query_vectors[:n_queries] if n_queries else query_vectors
    return base, queries
