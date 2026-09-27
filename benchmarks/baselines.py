"""
Der eigene Index gegen den Stand der Technik — auf DIESEM Korpus.

    python3 benchmarks/baselines.py [--kb reference-kb/RefKB_bge-m3]

Warum nicht SIFT1M oder GloVe, die üblichen ANN-Datensätze: das sind Bild-
deskriptoren bzw. Wort-Vektoren, deren Nachbarschaftsstruktur mit der eines
Chunk-Korpus wenig zu tun hat.  Realistisch für einen RAG-Index ist die echte
Wissensbasis mit den echten Queries — dieselben 300 Wortfenster, dieselbe
exakte Cosine-Top-5 als Wahrheit, mit der auch `evaluate.py` misst.

Verglichen wird auf der PARETO-FRONT, nicht an einem Punkt: jeder Index hat
einen Qualitäts/Tempo-Regler (HNSW `efSearch`, IVF `nprobe`, hier der Beam),
und nur die ganze Kurve sagt, wer bei gleichem Recall günstiger ist.

ZWEI Kostenachsen, und die zweite ist die wichtigere:

  * **ms/query** — was der Nutzer spürt, aber der Vergleich misst hier auch
    die Sprache mit: `rag_query.descend` ist Python, HNSW und FAISS sind C++.
    Ein Faktor 20 in dieser Spalte kann reiner Interpreter-Overhead sein.
  * **Distanzberechnungen je Query** — implementierungsunabhängig, und damit
    das, was den ALGORITHMUS vergleicht.  Für diesen Index ist es die Zahl
    der Blatt-Kandidaten, die nachgerankt werden; FAISS zählt seine eigenen
    (`indexIVF_stats.ndis`, `hnsw_stats.ndis`).  Deshalb läuft HNSW hier in
    der FAISS-Variante: hnswlib gibt die Zahl nicht heraus.

Gemessen wird mit EINEM BLAS-Thread, warm, Median aus drei Runden — wie in
`time_query.py` und aus denselben Gründen.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rag_query as rq                                        # noqa: E402

K, ROUNDS, WARMUP = 5, 3, 40


def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-30)


def leaf_rows(node):
    if "rows" in node:
        return node["rows"]
    return [r for c in node.get("children", []) for r in leaf_rows(c)]


def timed(fn, warm_arg, full_arg):
    """Warmlaufen, dann Median aus ROUNDS über den vollen Satz."""
    fn(warm_arg)
    ts = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        out = fn(full_arg)
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def score(got, exact):
    """recall@5 gegen die exakte Suche + found@5 (Quellchunk gefunden)."""
    rec = sum(len(set(g[:K]) & set(e[:K])) for g, e in zip(got, exact))
    return rec / (len(got) * K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="reference-kb/RefKB_bge-m3")
    ap.add_argument("--variant", default="default",
                    help="welcher gebaute Index als eigene Kurve dient")
    ap.add_argument("--out", default="benchmarks/baselines.json")
    a = ap.parse_args()

    kb = REPO / a.kb
    q = np.load(REPO / f"benchmarks/queries/{kb.name}.npz", allow_pickle=True)
    Qn = unit(q["queries"].astype(np.float32))
    gold = q["gold_rows"]
    E = np.load(sorted(kb.glob("bench_*"))[0] / "embeddings.npy")
    En = unit(E.astype(np.float32))
    n, d = En.shape
    print(f"{kb.name}: {n} Chunks, {d}d, {len(Qn)} Queries\n")

    # Wahrheit: exakte Cosine-Top-5 (das ist auch die Brute-Force-Baseline)
    def brute(Q):
        return np.argsort(-(Q @ En.T), axis=1)[:, :K]
    exact, t_brute = timed(brute, Qn[:WARMUP], Qn)
    rows = [{"index": "exact (brute force)", "param": "—", "recall@5": 1.0,
             "found@5": float(np.mean([gold[i] in exact[i]
                                       for i in range(len(Qn))])),
             "ms": t_brute / len(Qn) * 1000, "build_s": 0.0,
             "bytes": int(En.nbytes)}]
    rows[0]["cands"] = float(n)
    print(f"{'index':22s} {'param':>10s} {'recall@5':>9s} {'found@5':>8s} "
          f"{'cands':>10s} {'ms/query':>9s} {'build':>8s}")
    print(f"{rows[0]['index']:22s} {'—':>10s} {1.0:9.3f} "
          f"{rows[0]['found@5']:8.3f} {float(n):10.0f} {rows[0]['ms']:9.3f} "
          f"{'—':>8s}")

    # ── HNSW (FAISS-Variante, weil sie ndis mitzählt) ────────────────────
    import faiss
    t0 = time.perf_counter()
    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 200
    h.add(En)
    build_h = time.perf_counter() - t0
    for ef in [5, 10, 20, 50, 100]:
        h.hnsw.efSearch = ef
        got, t = timed(lambda Q: h.search(Q, K)[1], Qn[:WARMUP], Qn)
        faiss.cvar.hnsw_stats.reset()
        h.search(Qn, K)
        nd = faiss.cvar.hnsw_stats.ndis / len(Qn)
        r = score(got, exact)
        f = float(np.mean([gold[i] in got[i] for i in range(len(Qn))]))
        rows.append({"index": "HNSW (M=16)", "param": f"ef={ef}",
                     "recall@5": r, "found@5": f, "cands": nd,
                     "ms": t / len(Qn) * 1000, "build_s": build_h})
        print(f"{'HNSW (M=16)':22s} {'ef=' + str(ef):>10s} {r:9.3f} {f:8.3f} "
              f"{nd:10.0f} {t / len(Qn) * 1000:9.3f} {build_h:7.1f}s")

    # ── FAISS IVF ────────────────────────────────────────────────────────
    nlist = max(4, int(np.sqrt(n)))
    t0 = time.perf_counter()
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(En)
    ivf.add(En)
    build_i = time.perf_counter() - t0
    for np_ in [1, 2, 4, 8, 16, 32]:
        ivf.nprobe = np_
        got, t = timed(lambda Q: ivf.search(Q, K)[1], Qn[:WARMUP], Qn)
        faiss.cvar.indexIVF_stats.reset()
        ivf.search(Qn, K)
        nd = faiss.cvar.indexIVF_stats.ndis / len(Qn)
        r = score(got, exact)
        f = float(np.mean([gold[i] in got[i] for i in range(len(Qn))]))
        rows.append({"index": f"FAISS IVF (nlist={nlist})",
                     "param": f"nprobe={np_}", "recall@5": r, "found@5": f,
                     "cands": nd, "ms": t / len(Qn) * 1000,
                     "build_s": build_i})
        print(f"{'FAISS IVF':22s} {'nprobe=' + str(np_):>10s} {r:9.3f} "
              f"{f:8.3f} {nd:10.0f} {t / len(Qn) * 1000:9.3f} {build_i:7.1f}s")

    # ── dieser Index ─────────────────────────────────────────────────────
    ix = kb / f"bench_{a.variant}"
    tree = json.loads((ix / "ann_tree.json").read_text())
    meta = json.loads((ix / "index_meta.json").read_text())
    build_o = meta["partitioning"].get("indexing_seconds", 0.0)
    En64 = En.astype(float)

    seen_cands = {}

    def descend(Q, beam):
        out, tot = [], 0
        for q_ in Q:
            r = []
            for nd in rq.descend(tree, q_, beam):
                r.extend(leaf_rows(nd))
            r = list(dict.fromkeys(r))
            tot += len(r)
            c = np.asarray(r)
            out.append(c[np.argsort(-(En64[c] @ q_))[:K]] if len(r) else
                       np.zeros(K, int))
        seen_cands[beam] = tot / len(Q)
        return out

    for beam in [1, 2, 4, 8, 16, 32]:
        got, t = timed(lambda Q: descend(Q, beam), Qn[:WARMUP], Qn)
        r = score(got, exact)
        f = float(np.mean([gold[i] in got[i] for i in range(len(Qn))]))
        rows.append({"index": f"gp-qubo ({a.variant})", "param": f"beam={beam}",
                     "recall@5": r, "found@5": f, "cands": seen_cands[beam],
                     "ms": t / len(Qn) * 1000, "build_s": build_o})
        print(f"{'gp-qubo ' + a.variant:22s} {'beam=' + str(beam):>10s} "
              f"{r:9.3f} {f:8.3f} {seen_cands[beam]:10.0f} "
              f"{t / len(Qn) * 1000:9.3f} {build_o:7.1f}s")

    Path(REPO / a.out).write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} Messpunkte → {a.out}")


if __name__ == "__main__":
    main()
