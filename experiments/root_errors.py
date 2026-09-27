"""Die 21 % Wurzelfehler: Split-Fehler oder Repraesentations-Fehler?"""
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

kb = REPO/"reference-kb/RefKB_bge-m3"; ix = kb/"bench_plane"
q = np.load(REPO/"benchmarks/queries/RefKB_bge-m3.npz", allow_pickle=True)
Qn = unit(q["queries"].astype(float))
En = unit(np.load(ix/"embeddings.npy").astype(float))
tree = json.loads((ix/"ann_tree.json").read_text())
exact = np.argsort(-(Qn @ En.T), axis=1)[:, :5]

A, B = tree["children"]
rowsA, rowsB = set(leaf_rows(A)), set(leaf_rows(B))
mA = unit(np.asarray(tree["children"][0]["mean"], float)[None])[0]
mB = unit(np.asarray(tree["children"][1]["mean"], float)[None])[0]
print(f"Wurzel: {len(rowsA)} | {len(rowsB)} Punkte, Zentroid-Cosine "
      f"m_A·m_B = {mA@mB:.4f}\n")

wrong = []
for i,qq in enumerate(Qn):
    side_true = 0 if int(exact[i][0]) in rowsA else 1
    side_go   = tree["children"].index(rq.greedy_child(tree, qq))
    cs = (qq@mA, qq@mB)
    wrong.append((side_go != side_true, abs(cs[0]-cs[1]), cs, side_true, i))

bad = [w for w in wrong if w[0]]
print(f"Wurzel falsch: {len(bad)/len(Qn)*100:.1f} % der Queries")
print(f"  Zentroid-Marge |cos_A - cos_B| bei RICHTIG: "
      f"{np.median([w[1] for w in wrong if not w[0]]):.4f}")
print(f"  Zentroid-Marge bei FALSCH            : "
      f"{np.median([w[1] for w in bad]):.4f}")

# Wie nah ist der wahre Nachbar dem Zentroid seiner Seite?
print("\nLiegt der wahre Top-1-Nachbar auf der Seite, die IHM naeher ist?")
cons = 0
for i in range(len(Qn)):
    r = int(exact[i][0]); v = En[r]
    side_is = 0 if r in rowsA else 1
    side_near = 0 if (v@mA) > (v@mB) else 1
    cons += (side_is == side_near)
print(f"  {cons/len(Qn)*100:.1f} % der Nachbarn liegen bei ihrem naeheren Zentroid")

# Kontrafaktisch: wuerde die Query mit MAX-Cosine-zu-Punkt richtig abbiegen?
print("\nWenn der Knoten statt des Mittelwerts seinen BESTEN Punkt zeigte:")
fixed = 0
for w in bad:
    i = w[4]; qq = Qn[i]
    best_A = max(En[list(rowsA)] @ qq); best_B = max(En[list(rowsB)] @ qq)
    if (0 if best_A > best_B else 1) == w[3]: fixed += 1
print(f"  {fixed}/{len(bad)} = {fixed/max(len(bad),1)*100:.1f} % der Wurzelfehler "
      f"waeren behoben")
