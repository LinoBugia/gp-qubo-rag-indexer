"""
Anker nachträglich an einen fertigen Index hängen — ohne Neubau.

    python3 benchmarks/add_anchors.py <index_dir> --k 4 16

Die Anker sind k-Means-Zentren der FERTIGEN Gruppen.  Sie verändern weder
Baum noch Trennebenen, sie beschreiben die Knoten nur anders, damit der Beam
`max_j cos(q, aⱼ)` statt `cos(q, mean)` fragen kann.  Deshalb muss ein Baum
für k = 1, 4, 16 nicht dreimal gebaut werden — einmal bauen, dreimal
beschriften.

Geschrieben wird `anchors_k<K>.npz` neben den Index (Vektoren + Offsets in
PREORDER-Reihenfolge der Knoten); `ann_tree.json` bleibt unangetastet, damit
der Index ohne diese Datei exakt weiterläuft wie bisher.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import graph_partitioning as gp                              # noqa: E402


def preorder(node: dict):
    """Knoten in fester Reihenfolge — die Zuordnung Anker↔Knoten hängt daran."""
    yield node
    for c in node.get("children") or []:
        yield from preorder(c)


def leaf_rows(node: dict) -> list[int]:
    if node.get("children") is None:
        return node["rows"]
    return [r for c in node["children"] for r in leaf_rows(c)]


def build(index_dir: Path, k: int) -> Path:
    tree = json.loads((index_dir / "ann_tree.json").read_text())
    E = np.load(index_dir / "embeddings.npy").astype(float)
    vecs, offs = [], []
    pos = 0
    for nd in preorder(tree):
        rows = leaf_rows(nd)
        A = gp.node_anchors(E[rows], k, seed=len(rows) * 7919 + k)
        vecs.append(A.astype(np.float32))
        offs.append((pos, len(A)))
        pos += len(A)
    out = index_dir / f"anchors_k{k}.npz"
    np.savez_compressed(out, vectors=np.concatenate(vecs),
                        offsets=np.asarray(offs, dtype=np.int64))
    return out


def attach(tree: dict, npz_path: Path) -> np.ndarray:
    """Anker aus der Datei an die Knoten hängen; gibt die Matrix zurück."""
    z = np.load(npz_path)
    V, offs = z["vectors"].astype(float), z["offsets"]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)
    for nd, (o, c) in zip(preorder(tree), offs):
        nd["_S"] = V[o:o + c]
    return V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("--k", nargs="+", type=int, default=[4, 16])
    a = ap.parse_args()
    d = Path(a.index_dir)
    for k in a.k:
        p = build(d, k)
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
