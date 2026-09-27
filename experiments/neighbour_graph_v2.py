"""Nachbarschaftsnetz, zweiter Anlauf — diesmal mit allem, was inzwischen da ist.

Drei Konstruktionen der Blatt-Nachbarschaft:
  (a) NAECHSTE   die m aehnlichsten Blattzentroide (erster Versuch)
  (b) DIVERS     HNSWs Heuristik: ein Kandidat wird nur Nachbar, wenn er dem
                 Blatt naeher ist als allen schon gewaehlten Nachbarn — so
                 zeigen die Kanten in VERSCHIEDENE Richtungen
  (c) ECHT       aus den tatsaechlichen k-NN-Kanten der Punkte: welche
                 Blaetter hat der Baum auseinandergerissen
und zwei Scores (k=1 Mittelwert / k=16 Anker).
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="1"
import json, sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(R/"benchmarks")); sys.path.insert(0,str(R))
import add_anchors as aa

def unit(V): return V/np.maximum(np.linalg.norm(V,axis=-1,keepdims=True),1e-30)
kb=R/"reference-kb/RefKB_bge-m3"; ix=kb/"bench_m-da"
q=np.load(R/"benchmarks/queries/RefKB_bge-m3.npz",allow_pickle=True)
Qn=unit(q["queries"].astype(float))
En=unit(np.load(ix/"embeddings.npy").astype(float))
exact=np.argsort(-(Qn@En.T),axis=1)[:,:5]; K=5

def load(k):
    t=json.loads((ix/"ann_tree.json").read_text())
    if k>1: aa.attach(t, ix/f"anchors_k{k}.npz")
    else:
        for nd in aa.preorder(t):
            m=np.asarray(nd["mean"],float); nd["_S"]=(m/max(np.linalg.norm(m),1e-30))[None,:]
    return t

t1=load(1)
leaves=[nd for nd in aa.preorder(t1) if nd.get("children") is None]
rows_of=[np.asarray(nd["rows"]) for nd in leaves]
leaf_of=np.empty(len(En),int)
for j,r in enumerate(rows_of): leaf_of[r]=j
C=unit(np.asarray([En[r].mean(axis=0) for r in rows_of]))
L=len(leaves)
S=C@C.T; np.fill_diagonal(S,-2)

# (a) naechste
nbr_near=np.argsort(-S,axis=1)

# (b) divers (HNSW-Heuristik)
def diverse(i,M=8):
    out=[]
    for c in np.argsort(-S[i]):
        if len(out)>=M: break
        if all(S[c,o] < S[i,c] for o in out):   # naeher an mir als an allen gewaehlten
            out.append(int(c))
    return out
nbr_div=np.asarray([diverse(i)+[0]*(8-len(diverse(i))) for i in range(L)])

# (c) echte k-NN-Kanten
NN=np.argsort(-(En@En.T),axis=1)[:,1:11]
W=np.zeros((L,L))
for i in range(len(En)):
    a=leaf_of[i]
    for b in leaf_of[NN[i]]:
        if a!=b: W[a,b]+=1
nbr_true=np.argsort(-W,axis=1)

def run(tree, beam, m, table):
    cand=sp=rec=0
    for i,qq in enumerate(Qn):
        fr=[tree]
        while any(n.get("children") is not None for n in fr):
            cs,sc=[],[]
            for nd in fr:
                kids=nd.get("children")
                for c in (kids if kids else [nd]):
                    cs.append(c); sc.append(float((c["_S"]@qq).max())); sp+=len(c["_S"])
            fr=[cs[int(x)] for x in np.argsort(-np.asarray(sc))[:beam]]
        rows=[]
        for nd in fr: rows.extend(aa.leaf_rows(nd))
        if m and table is not None:
            for s in {leaf_of[r] for r in rows}:
                for j in table[s][:m]: rows.extend(rows_of[j].tolist())
        rows=list(dict.fromkeys(rows)); cand+=len(rows); c=np.asarray(rows)
        rec+=len(set(c[np.argsort(-(En[c]@qq))[:K]].tolist())&set(exact[i].tolist()))
    N=len(Qn); return cand/N, sp/N, rec/(N*K)*100

for k in (1,16):
    tree=load(k)
    print(f"\n=== Score: k={k} {'(Mittelwert)' if k==1 else '(16 Anker)'} ===")
    print(f"{'Verfahren':28s} {'Arbeit':>8s} {'recall@5':>9s}")
    for beam in (2,4,8):
        c,s,r=run(tree,beam,0,None)
        print(f"{'Beam ' + str(beam):28s} {c+s:8.0f} {r:8.1f}%")
    for name,tab in (("+1 nächster","near"),("+1 divers","div"),("+1 echt","true")):
        tb={"near":nbr_near,"div":nbr_div,"true":nbr_true}[tab]
        for beam in (2,4):
            c,s,r=run(tree,beam,1,tb)
            print(f"{'Beam ' + str(beam) + ' ' + name:28s} {c+s:8.0f} {r:8.1f}%")
