"""Blatt-Nachbarschaftsnetz: lohnt es sich, von einem Blatt zu springen?

Zwei Konstruktionen:
  (a) ZENTROID-Nachbarn  — billig: die m aehnlichsten Blattzentroide.
  (b) ECHTE Nachbarn     — teuer, als Obergrenze: fuer jeden Punkt seine
      k naechsten Nachbarn nachschlagen und zaehlen, welche BLAETTER dadurch
      verbunden sind.  Das ist die Nachbarschaft, die der Baum zerschnitten
      hat, exakt gemessen.
Verglichen wird gegen den Beam bei GLEICHER Kandidatenzahl.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_query as rq

def unit(V): return V/np.maximum(np.linalg.norm(V,axis=-1,keepdims=True),1e-30)
def leaf_rows(nd):
    return nd["rows"] if "rows" in nd else [r for c in nd.get("children",[]) for r in leaf_rows(c)]

R = Path(__file__).resolve().parent.parent
kb = R/"reference-kb/RefKB_bge-m3"; ix = kb/"bench_plane"
q = np.load(R/"benchmarks/queries/RefKB_bge-m3.npz", allow_pickle=True)
Qn = unit(q["queries"].astype(float))
En = unit(np.load(ix/"embeddings.npy").astype(float))
tree = json.loads((ix/"ann_tree.json").read_text()); K=5
exact = np.argsort(-(Qn@En.T),axis=1)[:,:K]

# Blätter einsammeln
leaves = []
def collect(nd):
    if nd.get("children") is None: leaves.append(nd)
    else:
        for c in nd["children"]: collect(c)
collect(tree)
L = len(leaves)
rows_of = [np.asarray(leaf_rows(l)) for l in leaves]
leaf_of = np.empty(len(En), int)
for j,r in enumerate(rows_of): leaf_of[r] = j
C = unit(np.asarray([En[r].mean(axis=0) for r in rows_of]))
print(f"{L} Blätter, Median {int(np.median([len(r) for r in rows_of]))} Punkte\n")

# (a) Zentroid-Nachbarn
S = C @ C.T; np.fill_diagonal(S, -2)
nbr_cent = np.argsort(-S, axis=1)

# (b) echte Nachbarschaft: wer teilt sich Top-10-Nachbarn?
W = np.zeros((L, L))
NN = np.argsort(-(En @ En.T), axis=1)[:, 1:11]
for i in range(len(En)):
    a = leaf_of[i]
    for b in leaf_of[NN[i]]:
        if a != b: W[a, b] += 1
nbr_true = np.argsort(-W, axis=1)

def run(mode, m, beam=1):
    tot=rec=0
    for i,qq in enumerate(Qn):
        fr = rq.descend(tree, qq, beam)
        seen = {id(x) for x in fr}
        rows = []
        for nd in fr: rows.extend(leaf_rows(nd))
        if m:
            start = leaf_of[rows[0]] if rows else 0
            tab = nbr_cent if mode=="cent" else nbr_true
            for j in tab[start][:m]:
                rows.extend(rows_of[j].tolist())
        rows=list(dict.fromkeys(rows)); tot+=len(rows); c=np.asarray(rows)
        top=c[np.argsort(-(En[c]@qq))[:K]]
        rec+=len(set(top.tolist())&set(exact[i].tolist()))
    return rec/(len(Qn)*K)*100, tot/len(Qn)

print(f"{'Verfahren':34s} {'Kandidaten':>11s} {'recall@5':>9s}")
for beam in [1,2,4,8]:
    r,t = run("cent", 0, beam)
    print(f"{'Beam ' + str(beam) + ' (heute)':34s} {t:11.0f} {r:8.1f}%")
print()
for m in [1,2,4,8]:
    r,t = run("cent", m)
    print(f"{'Beam 1 + ' + str(m) + ' Zentroid-Nachbarn':34s} {t:11.0f} {r:8.1f}%")
print()
for m in [1,2,4,8]:
    r,t = run("true", m)
    print(f"{'Beam 1 + ' + str(m) + ' ECHTE Nachbarn':34s} {t:11.0f} {r:8.1f}%")
