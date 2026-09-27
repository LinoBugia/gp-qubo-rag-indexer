#!/usr/bin/env python3
"""
Jeden gebauten Index gegen denselben Query-Satz messen.

    python3 benchmarks/evaluate.py              # alle bench_*-Indexe
    python3 benchmarks/evaluate.py --beam 1 2 4

Gemessen wird gegen die EXAKTE Suche: für jede Query ist die Wahrheit die
Top-k nach Cosine über alle Chunk-Embeddings.  Der Baum darf davon abweichen
— genau diese Abweichung ist der Preis der Approximation, und genau den will
der Benchmark beziffern.

Kennzahlen je Index und Beam:
  found     Anteil der Queries, die ihren Ursprungs-Chunk in den Top-5 haben
            (die Query ist ein Ausschnitt genau dieses Chunks)
  recall@5  Überlappung der Baum-Top-5 mit der exakten Top-5
  ms        Zeit für einen Abstieg + exaktes Nachranken der Kandidaten
  visited   besuchte Blatt-Einträge je Query (was der Baum anfassen musste)

Dazu die Bau-Kennzahlen aus index_meta.json / indexing_stats.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rag_query as rq                                      # noqa: E402

QDIR = Path(__file__).resolve().parent / "queries"


def unit(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-30)


def leaf_rows(node: dict) -> list[int]:
    return node.get("rows") or node.get("indices") or []


def tree_stats(tree: dict) -> dict:
    leaves, depths, entries = 0, [], 0
    stack = [(tree, 0)]
    while stack:
        nd, d = stack.pop()
        if nd.get("children"):
            for c in nd["children"]:
                stack.append((c, d + 1))
        else:
            leaves += 1
            depths.append(d)
            entries += len(leaf_rows(nd))
    return {"leaves": leaves, "depth_max": max(depths),
            "depth_mean": round(float(np.mean(depths)), 2),
            "leaf_entries": entries}


def evaluate(index_dir: Path, Q: np.ndarray, gold_rows: np.ndarray,
             E: np.ndarray, beams: list[int], k: int = 5) -> list[dict]:
    tree = json.loads((index_dir / "ann_tree.json").read_text())
    meta = json.loads((index_dir / "index_meta.json").read_text())
    st = tree_stats(tree)

    En = unit(E)
    Qn = unit(Q)
    # Wahrheit: exakte Top-k über die ganze Wissensbasis
    exact = np.argsort(-(Qn @ En.T), axis=1)[:, :k]

    out = []
    for beam in beams:
        hits = rec = 0
        visited = 0
        t0 = time.perf_counter()
        for i, q in enumerate(Qn):
            fr = rq.descend(tree, q, beam)
            rows: list[int] = []
            for nd in fr:
                rows.extend(leaf_rows(nd))
            rows = list(dict.fromkeys(rows))
            visited += len(rows)
            if not rows:
                continue
            cand = np.asarray(rows)
            sims = En[cand] @ q
            top = cand[np.argsort(-sims)[:k]]
            if gold_rows[i] in top:
                hits += 1
            rec += len(set(top.tolist()) & set(exact[i].tolist()))
        n = len(Qn)
        out.append({
            "index": str(index_dir), "beam": beam,
            "found@5": round(hits / n, 4),
            "recall@5": round(rec / (n * k), 4),
            "ms": round((time.perf_counter() - t0) / n * 1000, 3),
            "visited": round(visited / n, 1),
            **st,
            "build_s": meta["partitioning"].get("indexing_seconds"),
            "n": meta["chunk_count"],
            "encoder": meta["encoder"]["model"],
            "dim": meta["encoder"]["dim"],
            "variant": index_dir.name.replace("bench_", "")
                                     .replace("centroid_spill", "centroid+spill")
                                     .replace("spill02", "spill0.2"),
            "kb": index_dir.parent.name,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kbs", nargs="*", default=None)
    ap.add_argument("--beam", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--out", default="benchmarks/results.json")
    a = ap.parse_args()

    kbs = ([Path(p) for p in a.kbs] if a.kbs else
           sorted({p.parent.parent for p in
                   REPO.glob("reference-kb/**/bench_*/ann_tree.json")},
                  key=lambda d: d.name))
    rows: list[dict] = []
    for kb in kbs:
        qf = QDIR / f"{kb.name}.npz"
        if not qf.exists():
            print(f"⚠ {kb.name}: kein Query-Satz — übersprungen")
            continue
        d = np.load(qf, allow_pickle=True)
        Q, gold = d["queries"].astype(float), d["gold_rows"]
        idxs = sorted(kb.glob("bench_*"))
        E = np.load(idxs[0] / "embeddings.npy").astype(float)
        print(f"\n{kb.name}  ({len(Q)} Queries, {E.shape[0]} Chunks)")
        for ix in idxs:
            if not (ix / "ann_tree.json").exists():
                continue
            res = evaluate(ix, Q, gold, E, a.beam)
            rows.extend(res)
            r2 = [r for r in res if r["beam"] == 2][0]
            print(f"  {ix.name:22s} found {r2['found@5']:.3f} "
                  f"recall {r2['recall@5']:.3f}  {r2['ms']:.2f}ms  "
                  f"{r2['visited']:6.1f} besucht  "
                  f"{r2['leaves']:4d} Blätter  Tiefe {r2['depth_max']}")
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} Messungen → {a.out}")


if __name__ == "__main__":
    main()
