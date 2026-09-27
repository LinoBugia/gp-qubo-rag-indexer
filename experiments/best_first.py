"""Best-First statt Level-Beam — gleicher Baum, andere Suchreihenfolge.

Der Level-Beam haelt pro EBENE b Knoten und verwirft alle anderen endgueltig.
Best-First haelt EINE globale Warteschlange ueber alle offenen Knoten und
expandiert immer den aussichtsreichsten, egal auf welcher Tiefe er liegt.
Ein zurueckgestellter Knoten aus dem anderen Wurzelzweig bleibt erreichbar —
das sind die "Astspruenge", ohne dass eine einzige Querkante noetig waere.

Budget = Anzahl der Knoten-Expansionen; das ist der Regler wie efSearch.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="1"
import heapq, itertools, json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
import rag_query as rq

def unit(V): return V/np.maximum(np.linalg.norm(V,axis=-1,keepdims=True),1e-30)
def leaf_rows(nd):
    return nd["rows"] if "rows" in nd else [r for c in nd.get("children",[]) for r in leaf_rows(c)]

kb = R/"reference-kb/RefKB_bge-m3"; ix = kb/"bench_plane"
q = np.load(R/"benchmarks/queries/RefKB_bge-m3.npz", allow_pickle=True)
Qn = unit(q["queries"].astype(float))
En = unit(np.load(ix/"embeddings.npy").astype(float))
tree = json.loads((ix/"ann_tree.json").read_text()); K=5
exact = np.argsort(-(Qn@En.T), axis=1)[:,:K]

# Knotenvektoren einmal cachen
def prep(nd):
    m = np.asarray(nd["mean"], float)
    nd["_v"] = m/max(np.linalg.norm(m), 1e-30)
    for c in nd.get("children") or []: prep(c)
prep(tree)

def best_first(qq, budget):
    """Expandiere `budget` mal den besten offenen Knoten.  Blaetter sammeln."""
    tie = itertools.count()
    heap = [(-float(tree["_v"]@qq), next(tie), tree)]
    rows, sp, expanded = [], 1, 0
    while heap and expanded < budget:
        _, _, nd = heapq.heappop(heap)
        kids = nd.get("children")
        if kids is None:
            rows.extend(nd["rows"])
            continue
        expanded += 1
        for c in kids:
            heapq.heappush(heap, (-float(c["_v"]@qq), next(tie), c))
            sp += 1
    # was noch in der Queue liegt und Blatt ist, zaehlt nicht mehr
    return rows, sp

def level_beam(qq, beam):
    rows, sp = [], 0
    fr = [tree]
    while any(n.get("children") is not None for n in fr):
        cands, sc = [], []
        for nd in fr:
            kids = nd.get("children")
            for c in (kids if kids else [nd]):
                cands.append(c); sc.append(float(c["_v"]@qq)); sp += 1
        fr = [cands[int(i)] for i in np.argsort(-np.asarray(sc))[:beam]]
    for nd in fr: rows.extend(leaf_rows(nd))
    return rows, sp

def score(fn, arg):
    tot=spt=rec=0
    for i,qq in enumerate(Qn):
        rows, sp = fn(qq, arg)
        rows = list(dict.fromkeys(rows)); tot += len(rows); spt += sp
        if not rows: continue
        c = np.asarray(rows)
        rec += len(set(c[np.argsort(-(En[c]@qq))[:K]].tolist()) & set(exact[i].tolist()))
    N = len(Qn)
    return tot/N, spt/N, rec/(N*K)*100

print(f"{'Verfahren':22s} {'Kandidaten':>11s} {'Score-SP':>9s} {'GESAMT':>8s} {'recall@5':>9s}")
for b in (2,4,8,16,32):
    t,s,r = score(level_beam, b)
    print(f"{'Level-Beam ' + str(b):22s} {t:11.0f} {s:9.0f} {t+s:8.0f} {r:8.1f}%")
print()
for bud in (8,16,32,64,128):
    t,s,r = score(best_first, bud)
    print(f"{'Best-First ' + str(bud):22s} {t:11.0f} {s:9.0f} {t+s:8.0f} {r:8.1f}%")
