"""Konvergenztest der WURZEL-Bisektion: E(steps) auf der echten 4178er-KB.

    python3 benchmarks/convergence.py


Die Frage, die dieser Test beantwortet: der Steps-Plan halbiert je Ebene
(`steps/2**depth`, Boden STEPS_FLOOR), und AUTO setzt Ebene 0 auf n/3 — bei
4178 Chunks also 1392 Steps für 4178 Variablen, 0.33 Steps je Variable.  Das
ist wenig, und ein unterkonvergierter DA würde jede Aussage über den Nutzen
des Annealings gegenüber k-Means wertlos machen.

Ergebnis (2026-09-23): 1392 → 44544 Steps, also 32× mehr, liefern dieselbe
Energie bis auf die 10. Stelle (−448388.6640981942 gegen −448388.6640981945).
Die Wurzel ist konvergiert; dass der DA dort fast nichts gegenüber Lloyd holt
(0.0007 %), ist die Landschaft, kein Budget-Problem.  Der Warmstart bringt
den Solver bereits in das Becken, aus dem er nicht mehr herausfindet.

Nur messen — am Steps-Schedule wird hier nichts geändert.
"""
import sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import graph_partitioning as gp

E = np.load(REPO / "reference-kb/RefKB_bge-m3/bench_default/embeddings.npy").astype(float)
X = gp.normalize_coords(E)                 # Produktionspfad: ZENTRIERT
idx = list(range(len(X)))
n = len(idx)
print(f"n = {n}, d = {X.shape[1]}   (Produktionslauf: 1392 Steps)\n")
print(f"{'steps':>7} {'/Var':>6} {'E_lloyd':>15} {'E_DA':>15} "
      f"{'DA-Gewinn':>10} {'vs 1392':>9} {'n_A':>6} {'sek':>7}")

base = None
for s in [1392, 2784, 5568, 11136, 22272, 44544]:
    st = {}
    t0 = time.perf_counter()
    A, B = gp.bisect_group(idx, X, total_n=n, depth=0, steps=s, num_mc=20,
                           iterations_lloyd=20, lloyd_mc=1, anneal=True,
                           stats=st, verbose=False)
    dt = time.perf_counter() - t0
    e, el = st.get("energy"), st.get("E_lloyd")
    if base is None:
        base = e
    print(f"{s:7d} {s/n:6.2f} {el:15.1f} {e:15.1f} "
          f"{(el-e)/abs(el)*100:9.4f}% {(base-e)/abs(base)*100:8.4f}% "
          f"{len(A):6d} {dt:7.1f}", flush=True)
