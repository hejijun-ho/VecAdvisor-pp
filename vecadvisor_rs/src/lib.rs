/// vecadvisor_rs — Rust acceleration for VecAdvisor++
///
/// Provides PyO3-based Python extension functions for the four pure-Python
/// bottlenecks identified in the codebase:
///   1. read_fvecs  — binary fvecs file parsing (replaces struct.unpack loop)
///   2. read_ivecs  — binary ivecs file parsing (same, for int32)
///   3. build_insert_rows — numpy→psycopg2 row serialization (replaces per-row Python loop)
///   4. compute_recall — set-intersection recall@k (releases GIL during computation)
use std::collections::HashSet;
use std::fs::File;

use memmap2::MmapOptions;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyTuple};

// ---------------------------------------------------------------------------
// 1. read_fvecs
// ---------------------------------------------------------------------------

/// Parse an fvecs file and return a (n_vectors, dim) float32 ndarray.
///
/// fvecs format: each vector = [dim: i32 LE][dim × f32 LE]
/// All vectors in a well-formed file share the same dim.
/// Empty file → returns shape (0, 0) float32 array.
#[pyfunction]
fn read_fvecs<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let file =
        File::open(path).map_err(|e| PyIOError::new_err(format!("Cannot open {path}: {e}")))?;

    let metadata = file
        .metadata()
        .map_err(|e| PyIOError::new_err(format!("Cannot stat {path}: {e}")))?;
    let file_len = metadata.len() as usize;

    if file_len == 0 {
        let empty = Array2::<f32>::zeros((0, 0));
        return Ok(empty.into_pyarray(py));
    }

    // Read dim from first 4 bytes
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| PyIOError::new_err(format!("Cannot mmap {path}: {e}")))?
    };

    let dim = i32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
    if dim == 0 {
        return Err(PyValueError::new_err("fvecs dim must be > 0"));
    }

    let record_bytes = (dim + 1) * 4; // 4-byte header + dim * 4-byte floats
    if file_len % record_bytes != 0 {
        return Err(PyValueError::new_err(format!(
            "fvecs file size {file_len} is not a multiple of record size {record_bytes}"
        )));
    }
    let n = file_len / record_bytes;

    let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4; // skip 4-byte dim header
        for j in 0..dim {
            let byte_pos = offset + j * 4;
            let val = f32::from_le_bytes(mmap[byte_pos..byte_pos + 4].try_into().unwrap());
            flat.push(val);
        }
    }

    let arr = Array2::from_shape_vec((n, dim), flat)
        .map_err(|e| PyValueError::new_err(format!("Shape error: {e}")))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// 2. read_ivecs
// ---------------------------------------------------------------------------

/// Parse an ivecs file and return a (n_vectors, dim) int32 ndarray.
///
/// ivecs format: identical to fvecs but elements are i32 LE.
/// Empty file → returns shape (0, 0) int32 array.
#[pyfunction]
fn read_ivecs<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let file =
        File::open(path).map_err(|e| PyIOError::new_err(format!("Cannot open {path}: {e}")))?;

    let metadata = file
        .metadata()
        .map_err(|e| PyIOError::new_err(format!("Cannot stat {path}: {e}")))?;
    let file_len = metadata.len() as usize;

    if file_len == 0 {
        let empty = Array2::<i32>::zeros((0, 0));
        return Ok(empty.into_pyarray(py));
    }

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| PyIOError::new_err(format!("Cannot mmap {path}: {e}")))?
    };

    let dim = i32::from_le_bytes(mmap[0..4].try_into().unwrap()) as usize;
    if dim == 0 {
        return Err(PyValueError::new_err("ivecs dim must be > 0"));
    }

    let record_bytes = (dim + 1) * 4;
    if file_len % record_bytes != 0 {
        return Err(PyValueError::new_err(format!(
            "ivecs file size {file_len} is not a multiple of record size {record_bytes}"
        )));
    }
    let n = file_len / record_bytes;

    let mut flat: Vec<i32> = Vec::with_capacity(n * dim);
    for i in 0..n {
        let offset = i * record_bytes + 4;
        for j in 0..dim {
            let byte_pos = offset + j * 4;
            let val = i32::from_le_bytes(mmap[byte_pos..byte_pos + 4].try_into().unwrap());
            flat.push(val);
        }
    }

    let arr = Array2::from_shape_vec((n, dim), flat)
        .map_err(|e| PyValueError::new_err(format!("Shape error: {e}")))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// 3. build_insert_rows
// ---------------------------------------------------------------------------

/// Convert numpy arrays into a Python list-of-tuples for psycopg2 execute_values.
///
/// Each output tuple: (vec_as_python_list, cat10, cat100, cat1000, price, is_active_bool)
///
/// Column order is a contract matching generate_synthetic_attributes() key order:
///   category_10, category_100, category_1000, price, is_active
///
/// is_active must be passed as uint8 (numpy bool is stored as u8 at C level).
#[pyfunction]
fn build_insert_rows<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'py, f32>,
    cat10: PyReadonlyArray1<'py, i64>,
    cat100: PyReadonlyArray1<'py, i64>,
    cat1000: PyReadonlyArray1<'py, i64>,
    price: PyReadonlyArray1<'py, f64>,
    is_active: PyReadonlyArray1<'py, u8>,
) -> PyResult<Vec<PyObject>> {
    let vecs = vectors
        .as_slice()
        .map_err(|_| PyValueError::new_err("vectors array must be C-contiguous"))?;
    let cat10_s = cat10
        .as_slice()
        .map_err(|_| PyValueError::new_err("cat10 must be C-contiguous"))?;
    let cat100_s = cat100
        .as_slice()
        .map_err(|_| PyValueError::new_err("cat100 must be C-contiguous"))?;
    let cat1000_s = cat1000
        .as_slice()
        .map_err(|_| PyValueError::new_err("cat1000 must be C-contiguous"))?;
    let price_s = price
        .as_slice()
        .map_err(|_| PyValueError::new_err("price must be C-contiguous"))?;
    let active_s = is_active
        .as_slice()
        .map_err(|_| PyValueError::new_err("is_active must be C-contiguous"))?;

    let shape = vectors.shape();
    let n = shape[0];
    let dim = shape[1];

    let mut rows: Vec<PyObject> = Vec::with_capacity(n);

    for i in 0..n {
        // Build vector as Python list of floats (psycopg2 + pgvector requires Python list)
        let vec_slice = &vecs[i * dim..(i + 1) * dim];
        let vec_list = PyList::new(py, vec_slice.iter().map(|&v| v as f64))?;

        // Build the full row tuple
        let active_bool = active_s[i] != 0u8;
        let row = PyTuple::new(
            py,
            &[
                vec_list.into_any().unbind(),
                cat10_s[i].into_pyobject(py)?.into_any().unbind(),
                cat100_s[i].into_pyobject(py)?.into_any().unbind(),
                cat1000_s[i].into_pyobject(py)?.into_any().unbind(),
                price_s[i].into_pyobject(py)?.into_any().unbind(),
                PyBool::new(py, active_bool).to_owned().into_any().unbind(),
            ],
        )?;
        rows.push(row.into_any().unbind());
    }

    Ok(rows)
}

// ---------------------------------------------------------------------------
// 4. compute_recall
// ---------------------------------------------------------------------------

/// Compute mean recall@k.
///
/// result_ids: list of lists of PostgreSQL row IDs (1-indexed).
///   Converted to 0-indexed by subtracting 1 before comparison.
/// ground_truth_ids: (nq, k) int64 array (0-indexed). -1 values are padding.
/// k: number of neighbors.
///
/// Returns mean recall as f64, matching the Python implementation exactly.
/// GIL is released during the computation loop.
#[pyfunction]
fn compute_recall(
    py: Python<'_>,
    result_ids: Vec<Vec<i64>>,
    ground_truth_ids: PyReadonlyArray2<'_, i64>,
    k: usize,
) -> PyResult<f64> {
    let gt = ground_truth_ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("ground_truth_ids must be C-contiguous"))?;

    let n_queries = result_ids.len();
    if n_queries == 0 {
        return Ok(0.0);
    }

    let gt_cols = ground_truth_ids.shape()[1];

    // Release the GIL for the pure-Rust computation
    let total_recall = py.allow_threads(|| {
        let mut total = 0.0f64;

        for i in 0..n_queries {
            // Build set of returned IDs, converted from 1-indexed to 0-indexed
            let mut returned: HashSet<i64> = HashSet::new();
            let take = result_ids[i].len().min(k);
            for &rid in &result_ids[i][..take] {
                returned.insert(rid - 1);
            }

            // Build set of true nearest neighbors, skipping -1 padding
            let gt_row_start = i * gt_cols;
            let gt_row_end = gt_row_start + k.min(gt_cols);
            let mut true_nn: HashSet<i64> = HashSet::new();
            for &gid in &gt[gt_row_start..gt_row_end] {
                if gid >= 0 {
                    true_nn.insert(gid);
                }
            }

            if true_nn.is_empty() {
                total += 1.0; // No valid neighbors to find — matches Python behavior
            } else {
                let intersection = returned.intersection(&true_nn).count() as f64;
                total += intersection / true_nn.len() as f64;
            }
        }

        total
    });

    Ok(total_recall / n_queries as f64)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn vecadvisor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_fvecs, m)?)?;
    m.add_function(wrap_pyfunction!(read_ivecs, m)?)?;
    m.add_function(wrap_pyfunction!(build_insert_rows, m)?)?;
    m.add_function(wrap_pyfunction!(compute_recall, m)?)?;
    Ok(())
}
