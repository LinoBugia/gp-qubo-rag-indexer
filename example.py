"""
example.py — Usage examples for gram_matrix.py
===============================================

Run with:  python example.py
"""

import numpy as np
from gram_matrix import (
    compute_sparse_gram_matrix,
    save_to_json,
    load_from_json,
    to_dense,
)

SEP = "─" * 60

# ── Test vectors ──────────────────────────────────────────────────────────────
# 5 three-dimensional vectors
vectors = np.array([
    [1.0,  0.0,  0.0],   # unit x
    [0.9,  0.1,  0.0],   # almost unit x
    [0.0,  1.0,  0.0],   # unit y
    [0.0,  0.0,  1.0],   # unit z
    [1.0,  1.0,  0.0],   # diagonal xy (not normalised)
])


# ── 1. Cosine similarity ──────────────────────────────────────────────────────
print(SEP)
print("Example 1 — Cosine similarity  (threshold = 0.1)")
print(SEP)

sparse_cos = compute_sparse_gram_matrix(vectors, metric="cosine", threshold=0.1)
print(f"Non-zero entries: {len(sparse_cos)}")
for key, val in sparse_cos.items():
    print(f"  G{key} = {val:+.6f}")

save_to_json(
    sparse_cos,
    "output/cosine_gram.json",
    metadata={"metric": "cosine", "threshold": 0.1, "m": 5, "n": 3},
)


# ── 2. Dot product ────────────────────────────────────────────────────────────
print(SEP)
print("Example 2 — Dot product  (threshold = 0.05)")
print(SEP)

sparse_dot = compute_sparse_gram_matrix(vectors, metric="dot", threshold=0.05)
print(f"Non-zero entries: {len(sparse_dot)}")
for key, val in sparse_dot.items():
    print(f"  G{key} = {val:+.6f}")


# ── 3. Upper-triangle storage (symmetric matrices) ────────────────────────────
print(SEP)
print("Example 3 — Upper triangle only  (halves storage for symmetric metrics)")
print(SEP)

sparse_upper = compute_sparse_gram_matrix(
    vectors, metric="cosine", threshold=0.1, upper_triangle_only=True
)
print(f"Full entries : {len(sparse_cos)}")
print(f"Upper entries: {len(sparse_upper)}  (i <= j only)")
save_to_json(
    sparse_upper,
    "output/cosine_gram_upper.json",
    metadata={"metric": "cosine", "threshold": 0.1, "upper_triangle_only": True},
    compact=True,          # minified JSON — good for large matrices
)


# ── 4. RBF kernel ─────────────────────────────────────────────────────────────
print(SEP)
print("Example 4 — RBF (Gaussian) kernel  γ=2.0,  threshold=0.5")
print(SEP)

sparse_rbf = compute_sparse_gram_matrix(
    vectors, metric="rbf", threshold=0.5, gamma=2.0
)
print(f"Non-zero entries: {len(sparse_rbf)}")
for key, val in sparse_rbf.items():
    print(f"  G{key} = {val:+.6f}")


# ── 5. Polynomial kernel ──────────────────────────────────────────────────────
print(SEP)
print("Example 5 — Polynomial kernel  degree=2, c=1,  threshold=0.1")
print(SEP)

sparse_poly = compute_sparse_gram_matrix(
    vectors, metric="polynomial", threshold=0.1, degree=2, c=1.0
)
print(f"Non-zero entries: {len(sparse_poly)}")


# ── 6. Custom metric (callable) ─────────────────────��─────────────────────────
print(SEP)
print("Example 6 — Custom metric: |cosine|  (absolute cosine similarity)")
print(SEP)

def abs_cosine(vi, vj):
    ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
    if ni == 0.0 or nj == 0.0:
        return 0.0
    return abs(float(np.dot(vi, vj) / (ni * nj)))

sparse_custom = compute_sparse_gram_matrix(vectors, metric=abs_cosine, threshold=0.1)
print(f"Non-zero entries: {len(sparse_custom)}")


# ── 7. Load from JSON & reconstruct dense matrix ──────────────────────────────
print(SEP)
print("Example 7 — Load from JSON and reconstruct dense matrix")
print(SEP)

loaded_sparse, meta = load_from_json("output/cosine_gram.json")
print(f"Metadata  : {meta}")
print(f"Entries   : {len(loaded_sparse)}")

dense = to_dense(loaded_sparse, m=len(vectors))
print("Dense Gram matrix (cosine, threshold=0.1):")
print(np.round(dense, 4))


# ── 8. Large random example ───────────────────────────────────────────────────
print(SEP)
print("Example 8 — Large random matrix  (m=200, n=128, cosine, threshold=0.3)")
print(SEP)

rng = np.random.default_rng(42)
big_vectors = rng.standard_normal((200, 128))

sparse_big = compute_sparse_gram_matrix(
    big_vectors, metric="cosine", threshold=0.3, upper_triangle_only=True
)
total = 200 * 201 // 2
print(f"Non-zero entries : {len(sparse_big):,} / {total:,}  "
      f"(sparsity {1 - len(sparse_big)/total:.1%})")

save_to_json(
    sparse_big,
    "output/big_gram.json",
    metadata={"m": 200, "n": 128, "metric": "cosine", "threshold": 0.3},
    compact=True,
)

print(SEP)
print("All examples complete.  Output files written to ./output/")
