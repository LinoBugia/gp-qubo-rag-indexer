# The sparse Gram matrix builder

The standalone component under the partitioner: turning m vectors into a
sparse Gram matrix, its CLI, its Python API and its storage format.

- [Storage format](#storage-format)
- [Metrics](#metrics)
- [Command-line usage](#command-line-usage)
- [Python API](#python-api)
- [Performance notes](#performance-notes)
- [Run examples](#run-examples)

---

## Storage format

```json
{
  "_meta": {
    "m": 4,
    "n": 3,
    "metric": "cosine",
    "threshold": 0.1,
    "upper_triangle_only": false
  },
  "(0,0)": 1.0,
  "(0,1)": 0.9938,
  "(1,0)": 0.9938,
  "(1,1)": 1.0,
  "(4,4)": 1.0
}
```

* Keys are `"(i,j)"` strings (row, column of the Gram matrix).
* The optional `"_meta"` block stores parameters for reproducibility.
* Use `--upper-triangle` to store only `i ≤ j` — halves the file for symmetric metrics.
* Use `--compact` for minified JSON (no whitespace) — much smaller for large matrices.

---

## Metrics

| Key           | Formula                                    | Range      |
|---------------|--------------------------------------------|------------|
| `cosine`      | `⟨vi, vj⟩ / (‖vi‖ ‖vj‖)`                 | [−1, 1]    |
| `dot`         | `⟨vi, vj⟩`                                | ℝ          |
| `rbf`         | `exp(−γ ‖vi − vj‖²)`                      | (0, 1]     |
| `polynomial`  | `(⟨vi, vj⟩ + c)^degree`                   | ℝ          |
| `laplacian`   | `exp(−γ ‖vi − vj‖₁)`                      | (0, 1]     |
| *callable*    | any `f(vi, vj) → float` you pass in        | —          |

---

## Command-line usage

```bash
python3 cli.py <input> [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--metric`, `-m` | `cosine` | Similarity metric |
| `--threshold`, `-t` | `0.0` | Drop entries with `\|val\| ≤ threshold` |
| `--output`, `-o` | `gram_matrix.json` | Output file path |
| `--upper-triangle` | off | Store only `i ≤ j` entries |
| `--compact` | off | Minified JSON |
| `--show-dense` | off | Print dense matrix to stdout |
| `--rbf-gamma` | `1.0` | γ for RBF kernel |
| `--poly-degree` | `2` | Degree for polynomial kernel |
| `--poly-c` | `1.0` | Constant c for polynomial kernel |
| `--lap-gamma` | `1.0` | γ for Laplacian kernel |

### Input formats

| Extension | Content |
|---|---|
| `.csv` | One vector per row, comma-separated floats (no header) |
| `.json` | 2-D JSON array `[[v1_1, v1_2, ...], [v2_1, ...], ...]` |
| `.npy` | NumPy array of shape `(m, n)` |

### Examples

```bash
# Cosine similarity, threshold 0.1
python3 cli.py vectors.csv --metric cosine --threshold 0.1 -o gram.json

# RBF kernel with custom γ, upper-triangle storage, compact output
python3 cli.py vectors.json --metric rbf --rbf-gamma 0.5 \
    --upper-triangle --compact -o gram_rbf.json

# Dot product, print dense matrix (small matrices only)
python3 cli.py vectors.npy --metric dot --show-dense

# Polynomial kernel degree 3
python3 cli.py vectors.csv --metric polynomial --poly-degree 3 --poly-c 0 \
    --threshold 0.01 -o gram_poly.json
```

---

## Python API

```python
import numpy as np
from gram_matrix import (
    compute_sparse_gram_matrix,
    save_to_json,
    load_from_json,
    to_dense,
)

vectors = np.random.randn(100, 64)

# Compute
sparse = compute_sparse_gram_matrix(
    vectors,
    metric="cosine",
    threshold=0.2,
    upper_triangle_only=True,
)

# Save
save_to_json(sparse, "gram.json", metadata={"m": 100, "n": 64})

# Load
sparse_loaded, meta = load_from_json("gram.json")

# Reconstruct dense matrix
dense = to_dense(sparse_loaded, m=100)   # shape (100, 100)
```

### Custom metric

Pass any callable `f(vi, vj) -> float`:

```python
def my_kernel(vi, vj):
    return float(np.exp(-np.sum(np.abs(vi - vj))))   # Laplacian variant

sparse = compute_sparse_gram_matrix(vectors, metric=my_kernel, threshold=0.5)
```

### Upper-triangle + dense reconstruction

Symmetric metrics (cosine, RBF, …) only need the upper triangle.
`to_dense()` mirrors entries automatically:

```python
sparse = compute_sparse_gram_matrix(vectors, metric="cosine",
                                    upper_triangle_only=True)
dense = to_dense(sparse, m=len(vectors))   # full symmetric matrix
```

---

## Performance notes

These concern the **standalone Gram matrix builder** (`gram_matrix.py` /
`cli.py`), not the RAG indexer — that one partitions tens of thousands of
chunks and has its own budget rules (see the DA parameter table).

* The loop is pure Python + NumPy — sufficient for `m ≤ ~2 000`.
* For very large `m` (tens of thousands), consider vectorised batch computation
  with `np.einsum` or `sklearn.metrics.pairwise` and sparse filtering as a
  post-processing step.
* Use `--compact` for large outputs; minified JSON is 30–50 % smaller.
* `upper_triangle_only=True` halves both compute time and output size.

---

## Run examples

```bash
python3 example.py
```
