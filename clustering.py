"""
clustering.py — Normalized Gram Matrix for Binary / QUBO Clustering
====================================================================

Implements the normalized Gram matrix construction used in binary clustering
formulations (e.g. Pseudo-Boolean Functions / QUBO problems):

    1.  Center    : x_i  =  v_i − μ             (zero-mean per coordinate)
    2.  Scale     : x̃_i  =  x_i · n / Σ‖x_i‖   (avg L2-norm → 1)
    3.  Gram      : Q[i,j] = ⟨x̃_i, x̃_j⟩        (scaled inner product)

The scaling ensures that the average vector has unit L2-norm.
This is *not* cosine similarity (no per-vector normalisation);
the global scale factor preserves relative distances.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── External dependency: annealing-cop-approximator ────────────────────────────
# Adds ../annealing-cop-approximator/Code to sys.path so Funcs_Optimizers can
# be imported without installation.  Path is resolved relative to this file,
# so it works regardless of where you call the script from.
_COP_CODE = Path(__file__).resolve().parent.parent / "annealing-cop-approximator" / "Code"
if str(_COP_CODE) not in sys.path:
    sys.path.insert(0, str(_COP_CODE))

import Funcs_Optimizers  # noqa: E402  (import after sys.path manipulation)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Union

import numpy as np


# ── Core ──────────────────────────────────────────────��──────────────────────

def normalize_coords(coords: Union[list, np.ndarray]) -> np.ndarray:
    """
    Apply the binary-clustering normalisation to a coordinate array.

    Steps
    -----
    1. Subtract centroid (zero-mean each coordinate axis).
    2. Scale so that Σ‖x_i‖ = n  ↔  average L2-norm equals 1.

    Parameters
    ----------
    coords : array-like of shape (n, d)
        n points in d-dimensional space.

    Returns
    -------
    np.ndarray of shape (n, d)
        Normalized coordinate matrix x̃.
    """
    x = np.asarray(coords, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D array (n points × d dims), got shape {x.shape}.")

    # 1. Zero-mean (vectorized, no Python loop)
    x = x - x.mean(axis=0)

    # 2. Scale: n / Σ‖x_i‖  so that average L2-norm = 1
    norms = np.linalg.norm(x, axis=1)   # shape (n,)
    mag = norms.sum()
    if mag == 0.0:
        raise ValueError("All vectors are identical (zero after centering) — cannot normalize.")
    x = x * (len(x) / mag)

    return x


def clustering_gram_matrix(coords: Union[list, np.ndarray]) -> np.ndarray:
    """
    Compute the dense normalized Gram matrix Q for binary clustering.

    Q[i,j] = ⟨x̃_i, x̃_j⟩   where x̃ = normalize_coords(coords)

    Equivalent to the double-loop version in the original code but fully
    vectorized via a single matrix multiplication.

    Parameters
    ----------
    coords : array-like of shape (n, d)
        n points in d-dimensional space.

    Returns
    -------
    np.ndarray of shape (n, n)
        Dense symmetric Gram matrix.
    """
    x = normalize_coords(coords)
    return x @ x.T          # equivalent to Q[i,j] = dot(x[i], x[j]) for all i,j


def sparse_gram_entries(
    coords: Union[list, np.ndarray],
    threshold: float,
    max_block_elems: int = 4_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse Gram-Berechnung: Einträge |Q[i,j]| < threshold werden schon
    WÄHREND der Berechnung verworfen — die dichte n×n-Matrix entsteht nie.

    Blockweise: pro Schritt liegt nur ein (b × n)-Streifen x̃[s:e] @ x̃.T
    im Speicher (b so gewählt, dass der Streifen ≤ max_block_elems Werte
    hat, default ~32 MB), darauf die Schwelle, Überlebende wandern als
    COO-Tripel in die Ausgabe.  Die Zeilen-/Spaltensummen (Basis der
    linearen PBF-Terme) werden ausschließlich aus den ÜBERLEBENDEN
    Einträgen akkumuliert — ein Wert unter der Schwelle existiert nirgends,
    weder als quadratischer Term noch in einer Summe.

    Returns
    -------
    (rows, cols, vals, Q_row, Q_col)
      rows, cols, vals : COO-Tripel der überlebenden Einträge (row-major,
                         gleiche Reihenfolge wie die dichte Konstruktion)
      Q_row            : Spaltensummen  = np.sum(Q_sparse, axis=0)
      Q_col            : Zeilensummen   = np.sum(Q_sparse, axis=1)
    """
    x = normalize_coords(coords)
    n = len(x)
    block = max(1, int(max_block_elems // max(n, 1)))
    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []
    Q_row = np.zeros(n)
    Q_col = np.zeros(n)
    for s in range(0, n, block):
        e = min(s + block, n)
        B = x[s:e] @ x.T                      # ein Streifen, nie die volle Matrix
        B[np.abs(B) < threshold] = 0.0
        bi, bj = np.nonzero(B)                # row-major innerhalb des Streifens
        rows_l.append((bi + s).astype(np.int64))
        cols_l.append(bj.astype(np.int64))
        vals_l.append(B[bi, bj])
        Q_col[s:e] += B.sum(axis=1)           # Zeilensummen der Überlebenden
        Q_row += B.sum(axis=0)                # Spaltensummen der Überlebenden
    rows = np.concatenate(rows_l) if rows_l else np.zeros(0, np.int64)
    cols = np.concatenate(cols_l) if cols_l else np.zeros(0, np.int64)
    vals = np.concatenate(vals_l) if vals_l else np.zeros(0)
    return rows, cols, vals, Q_row, Q_col


def sparse_clustering_gram_matrix(
    coords: Union[list, np.ndarray],
    threshold: float = 0.0,
    upper_triangle_only: bool = False,
) -> dict[str, float]:
    """
    Compute the normalized Gram matrix and return it in sparse JSON format.

    Entries with |Q[i,j]| <= threshold are dropped.

    Parameters
    ----------
    coords : array-like of shape (n, d)
    threshold : float
        Sparsity cutoff.
    upper_triangle_only : bool
        If True, store only i <= j entries (matrix is symmetric).

    Returns
    -------
    dict  "(i,j)" → float
    """
    Q = clustering_gram_matrix(coords)
    n = Q.shape[0]
    sparse: dict[str, float] = {}

    for i in range(n):
        j_start = i if upper_triangle_only else 0
        for j in range(j_start, n):
            val = float(Q[i, j])
            if abs(val) > threshold:
                sparse[f"({i},{j})"] = round(val, 10)

    return sparse


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_clustering_gram(
    coords: Union[list, np.ndarray],
    filepath: str,
    threshold: float = 0.0,
    upper_triangle_only: bool = False,
    compact: bool = False,
) -> dict[str, float]:
    """
    Convenience wrapper: normalize → compute sparse → save as JSON.

    Returns the sparse dict (e.g. to pass into gram_matrix.to_dense).
    """
    from gram_matrix import save_to_json   # lazy import to keep modules independent

    coords_arr = np.asarray(coords, dtype=float)
    n, d = coords_arr.shape

    sparse = sparse_clustering_gram_matrix(coords_arr, threshold, upper_triangle_only)

    metadata = {
        "n": n,
        "d": d,
        "metric": "clustering_normalized_dot",
        "threshold": threshold,
        "upper_triangle_only": upper_triangle_only,
    }
    save_to_json(sparse, filepath, metadata=metadata, compact=compact)
    return sparse


# ── Comparison helper ─────────────────────────────────────────────────────────

def compare_with_original(coords: Union[list, np.ndarray]) -> float:
    """
    Verify that the vectorized version matches the original double-loop.
    Returns the max absolute difference (should be < 1e-12).
    """
    coords_arr = np.asarray(coords, dtype=float)
    n = len(coords_arr)

    # ── Original (your code, transcribed 1:1) ──
    coords_mean = []
    mean_x, mean_y = 0.0, 0.0
    for i in range(n):
        mean_x += coords_arr[i][0]
        mean_y += coords_arr[i][1]
    mean_x /= n
    mean_y /= n
    mag = 0.0
    for i in range(n):
        coords_mean.append([coords_arr[i][0] - mean_x, coords_arr[i][1] - mean_y])
        vector = np.array([coords_arr[i][0] - mean_x, coords_arr[i][1] - mean_y])
        mag += np.sqrt(vector.dot(vector))
    x_orig = np.asarray(coords_mean, dtype=float) * n / mag
    Q_orig = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Q_orig[i, j] = np.dot(x_orig[i], x_orig[j])

    # ── Vectorized ──
    Q_new = clustering_gram_matrix(coords_arr)

    return float(np.max(np.abs(Q_orig - Q_new)))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib

    # Example 2-D coords (could be cities, particles, graph nodes, ...)
    coords = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 1.0],
        [2.0, 6.0],
        [4.0, 3.0],
    ]

    print("── Normalized coords ──────────────────────────────────────")
    x_norm = normalize_coords(coords)
    norms = np.linalg.norm(x_norm, axis=1)
    print(x_norm.round(4))
    print(f"L2-norms: {norms.round(4)}  (avg = {norms.mean():.4f})")

    print("\n── Dense Gram matrix ──────────────────────────────────────")
    Q = clustering_gram_matrix(coords)
    print(Q.round(4))

    print("\n── Sparse Gram matrix (threshold=0.5) ─────────────────────")
    sparse = sparse_clustering_gram_matrix(coords, threshold=0.5)
    for k, v in sparse.items():
        print(f"  Q{k} = {v:+.6f}")

    print("\n── Save to JSON ────────────────────────────────────────────")
    pathlib.Path("output").mkdir(exist_ok=True)
    save_clustering_gram(coords, "output/clustering_gram.json", threshold=0.5, compact=False)

    print("\n── Equivalence check vs. original double-loop ─────────────")
    max_err = compare_with_original(coords)
    print(f"  Max |Q_orig - Q_new| = {max_err:.2e}  {'✓ identical' if max_err < 1e-10 else '✗ mismatch'}")
