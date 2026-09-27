"""
cli.py — Command-line interface for the Sparse Gram Matrix Builder
==================================================================

Usage
-----
    python cli.py <input> [options]

Input file formats: .csv (rows = vectors), .json (2-D list), .npy (numpy array)

Examples
--------
    python cli.py vectors.csv --metric cosine --threshold 0.1 -o gram.json
    python cli.py vectors.json --metric rbf --rbf-gamma 0.5 -o gram.json
    python cli.py vectors.csv --metric dot --upper-triangle --compact
    python cli.py vectors.npy --metric polynomial --poly-degree 3 --show-dense
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from gram_matrix import (
    BUILTIN_METRICS,
    compute_sparse_gram_matrix,
    load_from_json,
    save_to_json,
    to_dense,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_vectors(filepath: str) -> np.ndarray:
    """Load vectors from a .csv, .json, or .npy file."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = [list(map(float, row)) for row in reader if row]
        return np.array(rows)

    elif suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return np.array(data, dtype=float)

    elif suffix == ".npy":
        return np.load(path)

    else:
        raise ValueError(
            f"Unsupported input format '{suffix}'.  Use .csv, .json, or .npy."
        )


def print_stats(sparse: dict, m: int, upper_only: bool) -> None:
    total = m * (m + 1) // 2 if upper_only else m * m
    nz = len(sparse)
    sparsity = 1.0 - nz / total if total else 0.0
    print(f"  Non-zero entries : {nz:,} / {total:,}")
    print(f"  Sparsity         : {sparsity:.2%}")


# ── CLI ──────────���────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python cli.py",
        description="Compute a sparse Gram matrix from m n-dimensional vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("input", help="Path to vectors file (.csv, .json, .npy)")

    p.add_argument(
        "--metric", "-m",
        choices=sorted(BUILTIN_METRICS),
        default="cosine",
        help="Similarity metric  (default: cosine)",
    )
    p.add_argument(
        "--threshold", "-t",
        type=float, default=0.0,
        help="Sparsity cutoff — entries with |val| <= threshold are dropped  (default: 0.0)",
    )
    p.add_argument(
        "--output", "-o",
        default="gram_matrix.json",
        help="Output JSON path  (default: gram_matrix.json)",
    )
    p.add_argument(
        "--upper-triangle",
        action="store_true",
        help="Store only i <= j entries — halves file size for symmetric metrics",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Write compact (minified) JSON — much smaller for large matrices",
    )

    # Metric-specific params
    rbf = p.add_argument_group("RBF kernel options")
    rbf.add_argument("--rbf-gamma", type=float, default=1.0,
                     help="γ parameter  (default: 1.0)")

    poly = p.add_argument_group("Polynomial kernel options")
    poly.add_argument("--poly-degree", type=int, default=2,
                      help="Degree  (default: 2)")
    poly.add_argument("--poly-c", type=float, default=1.0,
                      help="Additive constant c  (default: 1.0)")

    lap = p.add_argument_group("Laplacian kernel options")
    lap.add_argument("--lap-gamma", type=float, default=1.0,
                     help="γ parameter  (default: 1.0)")

    # Output extras
    p.add_argument(
        "--show-dense",
        action="store_true",
        help="Print the reconstructed dense matrix to stdout (for small matrices)",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Load input ─────────────────────────────────────────────────────────
    print(f"\nLoading vectors from '{args.input}' …")
    try:
        vectors = load_vectors(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    m, n = vectors.shape
    print(f"  {m} vectors × {n} dimensions")

    # ── Resolve metric kwargs ──────────────────────────────────────────────
    metric_kwargs: dict = {}
    if args.metric == "rbf":
        metric_kwargs["gamma"] = args.rbf_gamma
    elif args.metric == "polynomial":
        metric_kwargs["degree"] = args.poly_degree
        metric_kwargs["c"] = args.poly_c
    elif args.metric == "laplacian":
        metric_kwargs["gamma"] = args.lap_gamma

    # ── Compute ────────────────────────────────────────────────────────────
    print(
        f"\nComputing Gram matrix …\n"
        f"  metric    = {args.metric}"
        + (f"  {metric_kwargs}" if metric_kwargs else "")
        + f"\n  threshold = {args.threshold}"
        + f"\n  storage   = {'upper triangle' if args.upper_triangle else 'full'}"
    )

    sparse = compute_sparse_gram_matrix(
        vectors,
        metric=args.metric,
        threshold=args.threshold,
        upper_triangle_only=args.upper_triangle,
        **metric_kwargs,
    )

    print_stats(sparse, m, args.upper_triangle)

    # ── Save ───────────────────────────────────────────────────────────────
    metadata = {
        "m": m,
        "n": n,
        "metric": args.metric,
        "threshold": args.threshold,
        "upper_triangle_only": args.upper_triangle,
        **{f"metric_{k}": v for k, v in metric_kwargs.items()},
    }

    save_to_json(
        sparse, args.output,
        metadata=metadata,
        compact=args.compact,
    )

    # ── Optional dense print ───────────────────────────────────────────────
    if args.show_dense:
        dense = to_dense(sparse, m)
        print("\nDense Gram matrix:")
        print(np.round(dense, 4))

    print("\nDone.")


if __name__ == "__main__":
    main()
