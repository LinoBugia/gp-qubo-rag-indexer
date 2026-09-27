"""
gram_matrix.py — Sparse Gram Matrix Builder
============================================

Computes a sparse Gram matrix G where G[i,j] = metric(v_i, v_j) for m
n-dimensional input vectors.  Entries with |G[i,j]| <= threshold are
treated as zero and are *not* stored.

Storage format  (JSON):
  {
    "_meta": { "m": 4, "n": 3, "metric": "cosine", ... },   # optional
    "(0,0)": 1.0,
    "(0,1)": 0.3241,
    ...
  }

Public API
----------
  compute_sparse_gram_matrix(vectors, metric, threshold, upper_triangle_only, **kw)
  save_to_json(sparse, filepath, metadata, compact)
  load_from_json(filepath)  -> (sparse_dict, metadata_dict)
  to_dense(sparse, m)       -> np.ndarray  (m × m)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Union

import numpy as np

# ── Built-in metric functions ────────────────────────────────────────────────

def _dot(vi: np.ndarray, vj: np.ndarray) -> float:
    """Standard dot product (inner product)."""
    return float(np.dot(vi, vj))


def _cosine(vi: np.ndarray, vj: np.ndarray) -> float:
    """Cosine similarity ∈ [-1, 1].  Returns 0 for zero vectors."""
    ni, nj = float(np.linalg.norm(vi)), float(np.linalg.norm(vj))
    if ni == 0.0 or nj == 0.0:
        return 0.0
    return float(np.dot(vi, vj) / (ni * nj))


def _rbf(vi: np.ndarray, vj: np.ndarray, gamma: float = 1.0) -> float:
    """Radial Basis Function (Gaussian) kernel: exp(-γ ||vi - vj||²)."""
    diff = vi - vj
    return float(np.exp(-gamma * float(np.dot(diff, diff))))


def _polynomial(vi: np.ndarray, vj: np.ndarray, degree: int = 2, c: float = 1.0) -> float:
    """Polynomial kernel: (⟨vi, vj⟩ + c)^degree."""
    return float((float(np.dot(vi, vj)) + c) ** degree)


def _laplacian(vi: np.ndarray, vj: np.ndarray, gamma: float = 1.0) -> float:
    """Laplacian kernel: exp(-γ ||vi - vj||₁)."""
    return float(np.exp(-gamma * float(np.sum(np.abs(vi - vj)))))


BUILTIN_METRICS: Dict[str, Callable] = {
    "dot":        _dot,
    "cosine":     _cosine,
    "rbf":        _rbf,
    "polynomial": _polynomial,
    "laplacian":  _laplacian,
}


# ── Core computation ─────────────────────────────────────────────────────────

def compute_sparse_gram_matrix(
    vectors: Union[list, np.ndarray],
    metric: Union[str, Callable] = "cosine",
    threshold: float = 0.0,
    upper_triangle_only: bool = False,
    **metric_kwargs,
) -> Dict[str, float]:
    """
    Compute a sparse Gram matrix from m n-dimensional vectors.

    Parameters
    ----------
    vectors : array-like of shape (m, n)
        The input vectors.  Each row is one vector.
    metric : str or callable
        Similarity function.  Built-ins: "dot", "cosine", "rbf",
        "polynomial", "laplacian".  Pass any callable f(vi, vj) -> float.
    threshold : float
        Sparsity cutoff.  Entries with |G[i,j]| <= threshold are dropped.
        Use threshold=0.0 to keep every non-zero entry.
    upper_triangle_only : bool
        If True, only entries with i <= j are computed and stored.
        Halves the work for symmetric metrics (dot, cosine, rbf, …).
    **metric_kwargs
        Extra keyword arguments forwarded to the metric function.
        E.g. gamma=0.5 for "rbf", degree=3 for "polynomial".

    Returns
    -------
    Dict[str, float]
        Keys are "(i,j)" strings, values are the similarity scores.
    """
    vecs = np.asarray(vectors, dtype=float)
    if vecs.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array (m vectors × n dims), got shape {vecs.shape}."
        )
    m = vecs.shape[0]

    # Resolve metric
    if callable(metric):
        sim_fn = metric
    elif isinstance(metric, str):
        if metric not in BUILTIN_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'.  "
                f"Available built-ins: {sorted(BUILTIN_METRICS)}"
            )
        base = BUILTIN_METRICS[metric]
        sim_fn = (lambda vi, vj: base(vi, vj, **metric_kwargs)) if metric_kwargs else base
    else:
        raise TypeError("'metric' must be a string identifier or a callable.")

    sparse: Dict[str, float] = {}
    for i in range(m):
        j_start = i if upper_triangle_only else 0
        for j in range(j_start, m):
            val = sim_fn(vecs[i], vecs[j])
            if abs(val) > threshold:
                sparse[f"({i},{j})"] = round(val, 10)

    return sparse


# ── I/O ───────────────────────────��──────────────────────────────────────────

def save_to_json(
    sparse_matrix: Dict[str, float],
    filepath: str,
    metadata: dict | None = None,
    compact: bool = False,
) -> None:
    """
    Save a sparse Gram matrix to a JSON file.

    The optional *metadata* dict is stored under the reserved key "_meta".
    Use compact=True for a single-line file (much smaller for large matrices).

    Parameters
    ----------
    sparse_matrix : dict
        Output of compute_sparse_gram_matrix().
    filepath : str or Path
        Destination file path.  Parent directories are created as needed.
    metadata : dict, optional
        Arbitrary key/value information (m, n, metric, threshold, …).
    compact : bool
        If True, write minified JSON without whitespace.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    if metadata:
        payload["_meta"] = metadata
    payload.update(sparse_matrix)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=None if compact else 2, ensure_ascii=False)

    size_kb = path.stat().st_size / 1024
    print(f"✓ Saved {len(sparse_matrix):,} entries → {path}  ({size_kb:.1f} KB)")


def load_from_json(filepath: str) -> tuple[Dict[str, float], dict]:
    """
    Load a sparse Gram matrix from a JSON file.

    Returns
    -------
    (sparse_matrix, metadata)
        *sparse_matrix* is the dict of "(i,j)" → float entries.
        *metadata* is the "_meta" dict (empty dict if not present).
    """
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as fh:
        payload: dict = json.load(fh)

    meta = payload.pop("_meta", {})
    return payload, meta


# ── Dense reconstruction ─────────────────────────────────────────────────────

def to_dense(sparse_matrix: Dict[str, float], m: int) -> np.ndarray:
    """
    Reconstruct a dense m×m numpy matrix from a sparse dict.

    If the matrix was stored upper-triangle-only, mirror it to produce the
    full symmetric matrix automatically.

    Parameters
    ----------
    sparse_matrix : dict
        Keys "(i,j)" → float values.
    m : int
        Matrix dimension (number of original vectors).

    Returns
    -------
    np.ndarray of shape (m, m)
    """
    G = np.zeros((m, m))
    for key, val in sparse_matrix.items():
        i, j = map(int, key.strip("()").split(","))
        G[i, j] = val
        G[j, i] = val  # mirrors upper-triangle entries; no-op for full storage
    return G
