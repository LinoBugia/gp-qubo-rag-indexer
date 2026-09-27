"""
Die ganze Matrix auf einmal: Baum × Repräsentation × Suchstrategie.

    python3 benchmarks/evaluate_full.py [--kbs …] [--beams 1 2 4 8 16 32]

Drei Achsen, die sich unabhängig drehen lassen — und nur die erste kostet
einen Neubau:

  1. BAUM        Optimierer (Lloyd | Lloyd→DA | DA ohne Warmstart)
                 × Zielfunktion (gram = 2-Means-artig | knn = Modularity
                 des k-NN-Graphen).  Das sind die bench_*-Ordner.
  2. ANKER       k = 1 (ein Mittelwert) | 4 | 16.  Wird hier NACHTRÄGLICH
                 angehängt (add_anchors.py): Anker beschreiben die fertigen
                 Gruppen, sie verändern den Baum nicht.
  3. SUCHE       Level-Beam (b Knoten je Ebene) | Best-First (eine globale
                 Warteschlange, Budget = Expansionen).

Gemessen wird je Kombination: recall@5 gegen die exakte Cosine-Suche,
found@5, die Zahl der nachgerankten Kandidaten und die Score-Skalarprodukte
des Abstiegs — erst beide zusammen sind die Arbeit, die ein Verfahren kostet.
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                            # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import add_anchors as aa                                      # noqa: E402

K = 5


def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-30)


def load_tree(d: Path, k: int):
    """Baum laden und mit k Ankern beschriften (k=1 → der Mittelwert)."""
    tree = json.loads((d / "ann_tree.json").read_text())
    if k > 1:
        npz = d / f"anchors_k{k}.npz"
        if not npz.exists():
            aa.build(d, k)
        aa.attach(tree, npz)
    else:
        for nd in aa.preorder(tree):
            m = np.asarray(nd["mean"], dtype=float)
            nd["_S"] = (m / max(np.linalg.norm(m), 1e-30))[None, :]
    return tree


def level_beam(tree, q, beam):
    rows, sp = [], 0
    fr = [tree]
    while any(n.get("children") is not None for n in fr):
        cands, sc = [], []
        for nd in fr:
            kids = nd.get("children")
            for c in (kids if kids else [nd]):
                cands.append(c)
                sc.append(float((c["_S"] @ q).max()))
                sp += len(c["_S"])
        fr = [cands[int(i)] for i in np.argsort(-np.asarray(sc))[:beam]]
    for nd in fr:
        rows.extend(aa.leaf_rows(nd))
    return rows, sp


def best_first(tree, q, budget):
    tie = itertools.count()
    heap = [(-float((tree["_S"] @ q).max()), next(tie), tree)]
    rows, sp, exp = [], len(tree["_S"]), 0
    while heap and exp < budget:
        _, _, nd = heapq.heappop(heap)
        kids = nd.get("children")
        if kids is None:
            rows.extend(nd["rows"])
            continue
        exp += 1
        for c in kids:
            heapq.heappush(heap, (-float((c["_S"] @ q).max()), next(tie), c))
            sp += len(c["_S"])
    return rows, sp


def score_run(tree, Qn, En, exact, gold, fn, arg):
    tot = spt = hits = rec = 0
    for i, q in enumerate(Qn):
        rows, sp = fn(tree, q, arg)
        rows = list(dict.fromkeys(rows))
        tot += len(rows)
        spt += sp
        if not rows:
            continue
        c = np.asarray(rows)
        top = c[np.argsort(-(En[c] @ q))[:K]]
        hits += int(gold[i] in top)
        rec += len(set(top.tolist()) & set(exact[i].tolist()))
    n = len(Qn)
    return {"cands": round(tot / n, 1), "score_sp": round(spt / n, 1),
            "found@5": round(hits / n, 4), "recall@5": round(rec / (n * K), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kbs", nargs="*", default=None)
    ap.add_argument("--anchors", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--beams", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--budgets", nargs="+", type=int,
                    default=[8, 16, 32, 64, 128])
    ap.add_argument("--only", default="",
                    help="Komma-Liste von Varianten (leer = alle)")
    ap.add_argument("--out", default="benchmarks/results_full.json")
    a = ap.parse_args()

    kbs = ([Path(p) for p in a.kbs] if a.kbs else
           sorted({p.parent.parent for p in
                   REPO.glob("reference-kb/**/bench_*/ann_tree.json")},
                  key=lambda d: d.name))
    rows: list[dict] = []
    for kb in kbs:
        qf = HERE / "queries" / f"{kb.name}.npz"
        if not qf.exists():
            print(f"⚠ {kb.name}: kein Query-Satz — übersprungen")
            continue
        d = np.load(qf, allow_pickle=True)
        Qn = unit(d["queries"].astype(float))
        gold = d["gold_rows"]
        idxs = sorted(p.parent for p in kb.glob("bench_*/ann_tree.json"))
        if a.only:
            want = set(a.only.split(","))
            idxs = [d for d in idxs if d.name.replace("bench_", "") in want]
        if not idxs:
            continue
        En = unit(np.load(idxs[0] / "embeddings.npy").astype(float))
        exact = np.argsort(-(Qn @ En.T), axis=1)[:, :K]
        print(f"\n{kb.name}  ({len(Qn)} Queries, {len(En)} Chunks)")
        for ix in idxs:
            meta = json.loads((ix / "index_meta.json").read_text())
            base = {"kb": kb.name, "variant": ix.name.replace("bench_", ""),
                    "n": meta["chunk_count"],
                    "encoder": meta["encoder"]["model"],
                    "build_s": meta["partitioning"].get("indexing_seconds")}
            for k in a.anchors:
                tree = load_tree(ix, k)
                for b in a.beams:
                    rows.append({**base, "anchors": k, "search": "beam",
                                 "param": b,
                                 **score_run(tree, Qn, En, exact, gold,
                                             level_beam, b)})
                for bud in a.budgets:
                    rows.append({**base, "anchors": k, "search": "bestfirst",
                                 "param": bud,
                                 **score_run(tree, Qn, En, exact, gold,
                                             best_first, bud)})
                r = [x for x in rows if x["kb"] == base["kb"]
                     and x["variant"] == base["variant"]
                     and x["anchors"] == k and x["search"] == "beam"
                     and x["param"] == max(a.beams)][0]
                print(f"  {ix.name:20s} k={k:<3d} beam {max(a.beams):<3d} "
                      f"recall {r['recall@5']:.3f}  {r['cands']:5.0f} Kand")
    Path(REPO / a.out).write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} Messungen → {a.out}")


if __name__ == "__main__":
    main()
