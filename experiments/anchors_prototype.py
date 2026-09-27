"""Prototyp: k ANKER je Knoten statt EINEM Mittelwert.

Score(Knoten, q) = max_j cos(q, anker_j) statt cos(q, mittelwert).
Der Baum bleibt unveraendert — nur die Repraesentation der Knoten aendert
sich, also reine Query-Zeit-Massnahme ohne Neubau.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

def unit(V): return V/np.maximum(np.linalg.norm(V,axis=-1,keepdims=True),1e-30)
def leaf_rows(nd):
    return nd["rows"] if "rows" in nd else [r for c in nd.get("children",[]) for r in leaf_rows(c)]

kb = REPO/"reference-kb/RefKB_bge-m3"; ix = kb/"bench_plane"
q = np.load(REPO/"benchmarks/queries/RefKB_bge-m3.npz", allow_pickle=True)
Qn = unit(q["queries"].astype(float))
En = unit(np.load(ix/"embeddings.npy").astype(float))
tree = json.loads((ix/"ann_tree.json").read_text())
K = 5
exact = np.argsort(-(Qn @ En.T), axis=1)[:, :K]

def kmeans_anchors(X, k, iters=10, seed=0):
    """k Anker = k-Means-Zentren der Knotenpunkte (Einheitsnorm)."""
    if len(X) <= k: return unit(X)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = X[a==j]
            if len(m): C[j] = m.mean(axis=0)
        C = unit(C)
    return C

def attach(nd, k):
    rows = leaf_rows(nd)
    nd["_A"] = kmeans_anchors(En[rows], k) if len(rows) > 1 else unit(En[rows])
    for c in nd.get("children") or []: attach(c, k)

def descend_anchors(q, beam):
    """Beam-Abstieg, Score = max ueber die Anker des Knotens."""
    frontier = [tree]
    while any(n.get("children") is not None for n in frontier):
        cands, scores = [], []
        for nd in frontier:
            kids = nd.get("children")
            if kids is None:
                cands.append(nd); scores.append(float((nd["_A"] @ q).max()))
            else:
                for c in kids:
                    cands.append(c); scores.append(float((c["_A"] @ q).max()))
        pick = np.argsort(-np.asarray(scores))[:beam]
        frontier = [cands[int(i)] for i in pick]
    return frontier

print(f"{'k Anker':>8} {'beam 1':>8} {'beam 2':>8} {'beam 4':>8} {'ms/query':>9}")
base = None
for k in [1, 2, 4, 8, 16]:
    attach(tree, k)
    line, t_all = [], 0.0
    for beam in [1,2,4]:
        t0 = time.perf_counter(); rec = 0
        for i,qq in enumerate(Qn):
            rows=[]
            for nd in descend_anchors(qq, beam): rows.extend(leaf_rows(nd))
            rows=list(dict.fromkeys(rows)); c=np.asarray(rows)
            top=c[np.argsort(-(En[c]@qq))[:K]]
            rec += len(set(top.tolist()) & set(exact[i].tolist()))
        dt = time.perf_counter()-t0
        line.append(rec/(len(Qn)*K)*100)
        if beam==2: t_all = dt/len(Qn)*1000
    print(f"{k:8d} {line[0]:7.1f}% {line[1]:7.1f}% {line[2]:7.1f}% {t_all:8.3f}")
