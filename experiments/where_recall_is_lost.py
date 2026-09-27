"""Wo genau geht Recall verloren?  Ohne Spill gemessen (Variante `plane`)."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rag_query as rq

def unit(V): return V/np.maximum(np.linalg.norm(V,axis=-1,keepdims=True),1e-30)
def leaf_rows(nd):
    return nd["rows"] if "rows" in nd else [r for c in nd.get("children",[]) for r in leaf_rows(c)]

kb = REPO/"reference-kb/RefKB_bge-m3"
ix = kb/"bench_plane"                      # KEIN Spill — saubere Partition
q = np.load(REPO/"benchmarks/queries/RefKB_bge-m3.npz", allow_pickle=True)
Qn = unit(q["queries"].astype(float)); gold = q["gold_rows"]
En = unit(np.load(ix/"embeddings.npy").astype(float))
tree = json.loads((ix/"ann_tree.json").read_text())
K, n = 5, len(En)
exact = np.argsort(-(Qn @ En.T), axis=1)[:, :K]

# Blatt-Zugehörigkeit jedes Chunks
leaf_of = {}
def walk(nd, path=""):
    if nd.get("children") is None:
        for r in leaf_rows(nd): leaf_of[r] = path
    else:
        for i,c in enumerate(nd["children"]): walk(c, path+str(i))
walk(tree)
sizes = {}
for r,p in leaf_of.items(): sizes[p] = sizes.get(p,0)+1
print(f"{len(set(leaf_of.values()))} Blätter, Größe median "
      f"{int(np.median(list(sizes.values())))}, max {max(sizes.values())}\n")

# ── 1. OBERE SCHRANKE: wie viele Blätter enthalten die exakte Top-5? ──────
spread = np.array([len({leaf_of[int(r)] for r in exact[i]}) for i in range(len(Qn))])
print("Auf wie viele BLÄTTER verteilt sich die exakte Top-5?")
for k in range(1, 6):
    print(f"  {k} Blatt/Blätter: {(spread==k).mean()*100:5.1f} %")
print(f"  Mittel: {spread.mean():.2f} Blätter\n")

# obere Schranke für recall@5 bei beam b: Anteil der Top-5, der in den b
# häufigsten Blättern der Query liegt — bei PERFEKTEM Abstieg
print("Theoretische Obergrenze für recall@5 bei perfektem Abstieg:")
for beam in [1,2,4,8]:
    tot = 0
    for i in range(len(Qn)):
        cnt = {}
        for r in exact[i]:
            p = leaf_of[int(r)]; cnt[p] = cnt.get(p,0)+1
        # die beam grössten Blätter der Top-5 einsammeln
        tot += sum(sorted(cnt.values(), reverse=True)[:beam])
    print(f"  beam {beam}: {tot/(len(Qn)*K)*100:5.1f} %")

# ── 2. tatsächlicher Recall + wo der Abstieg abbiegt ─────────────────────
print("\nTatsächlich erreicht:")
for beam in [1,2,4,8]:
    rec = 0
    for i,qq in enumerate(Qn):
        rows=[]
        for nd in rq.descend(tree, qq, beam): rows.extend(leaf_rows(nd))
        rows=list(dict.fromkeys(rows)); c=np.asarray(rows)
        top=c[np.argsort(-(En[c]@qq))[:K]]
        rec += len(set(top.tolist()) & set(exact[i].tolist()))
    print(f"  beam {beam}: {rec/(len(Qn)*K)*100:5.1f} %")

# ── 3. Fehlertiefe: wo verliert der GREEDY-Pfad den wahren Nachbarn? ─────
print("\nWo biegt der Greedy-Pfad vom Blatt des NÄCHSTEN Nachbarn ab?")
depth_err = {}
for i,qq in enumerate(Qn):
    target = leaf_of[int(exact[i][0])]        # Blatt des Top-1
    nd, d = tree, 0
    while nd.get("children") is not None:
        nxt = rq.greedy_child(nd, qq)
        idx = nd["children"].index(nxt)
        if idx != int(target[d]):
            depth_err[d] = depth_err.get(d,0)+1
            break
        nd = nxt; d += 1
    else:
        depth_err[-1] = depth_err.get(-1,0)+1   # korrekt angekommen
ok = depth_err.pop(-1, 0)
print(f"  korrekt im Zielblatt: {ok/len(Qn)*100:.1f} %")
for d in sorted(depth_err):
    print(f"  falsch ab Tiefe {d}: {depth_err[d]/len(Qn)*100:5.1f} %")
