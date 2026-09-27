"""
Query-Zeit sauber messen — das ist eine eigene Messung, kein Nebenprodukt.

    python3 benchmarks/time_query.py

Die `ms` in results.json entstehen nebenbei, während numpy über alle Kerne
rechnet und andere Läufe die Maschine belasten; sie schwanken um Faktoren.
Hier wird dreierlei festgenagelt:

  * **ein BLAS-Thread** — die Zahl heißt dann, was sie soll: was EINE Query
    auf EINEM Kern kostet, statt was acht Kerne an einer Query tun.
  * **warmer Knoten-Cache** — `rag_query` legt die Kindvektoren beim ersten
    Abstieg am Knoten ab.  Wer kalt misst, misst den Cache-Aufbau mit und
    bekommt für Beam 1 eine höhere Zahl als für Beam 2, obwohl Beam 1
    weniger Arbeit macht.  Ein RAG-Dienst lädt den Index einmal und stellt
    danach viele Fragen — warm ist der ehrliche Fall.
  * **Median aus fünf Runden** über dieselben 300 Queries.

Gemessen wird exakt das, was eine Anfrage kostet: Abstieg durch den Baum
plus exaktes Nachranken der Blatt-Kandidaten.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import rag_query as rq                                        # noqa: E402

KBS = {"bge-m3": REPO / "reference-kb/RefKB_bge-m3",
       "qwen3-4b": REPO / "reference-kb/RefKB_qwen3-embedding-4b-fp16"}
# Variante → Ordnername (run_matrix schreibt bench_<name ohne Sonderzeichen>)
VARIANTS = ["lloyd-raw", "raw", "centroid", "plane", "lloyd", "centroid+spill",
            "default", "spill0.2", "eps", "mc4",
            # die ausgelieferten Defaults und ihr Lloyd-Gegenstueck
            "knn-da", "m-lloyd"]
DIRNAME = {"spill0.2": "spill02", "centroid+spill": "centroid_spill"}
ROUNDS, BEAMS, K, WARMUP = 5, [1, 2, 4], 5, 40


def unit(V: np.ndarray) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-30)


def leaf_rows(node: dict) -> list[int]:
    if "rows" in node:
        return node["rows"]
    return [r for c in node.get("children", []) for r in leaf_rows(c)]


def one_round(tree: dict, Qn: np.ndarray, En: np.ndarray, beam: int) -> float:
    """Sekunden für alle Queries: Abstieg + exaktes Nachranken."""
    t0 = time.perf_counter()
    for q in Qn:
        rows: list[int] = []
        for nd in rq.descend(tree, q, beam):
            rows.extend(leaf_rows(nd))
        rows = list(dict.fromkeys(rows))
        if rows:
            cand = np.asarray(rows)
            top = cand[np.argsort(-(En[cand] @ q))[:K]]
            top[0]          # Ergebnis anfassen, nichts wegwerfen
    return time.perf_counter() - t0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", default="", help="Komma-Liste statt aller")
    ap.add_argument("--beams", default="", help="Komma-Liste statt 1,2,4")
    ap.add_argument("--max-load", type=float, default=2.0,
                    help="oberhalb dieses load average nicht messen")
    a = ap.parse_args()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()] or VARIANTS
    beams = [int(b) for b in a.beams.split(",") if b.strip()] or BEAMS

    # Lastwaechter.  Ein BLAS-Thread schuetzt nur davor, dass WIR die Maschine
    # fluten — nicht davor, dass jemand anders es tut.  Unter Fremdlast kommen
    # Zahlen heraus, die um Faktoren daneben liegen (gemessen: 0.19 statt
    # 0.06 ms bei load 10.8), und die landen dann in der README.
    load = os.getloadavg()[0]
    if load > a.max_load:
        sys.exit(f"Abbruch: load average {load:.1f} > {a.max_load} — die "
                 f"Maschine ist belegt, die Zeiten waeren wertlos. Warten, "
                 f"oder bewusst mit --max-load ueberstimmen.")

    # Vorhandene Messungen behalten: ein Teillauf soll die Tabelle ergaenzen,
    # nicht die Zahlen ueberschreiben, auf die die README verweist.
    dest = REPO / "benchmarks/query_times.json"
    out = json.loads(dest.read_text()) if dest.exists() else []
    out = [r for r in out
           if (r["variant"], r["beam"]) not in {(v, b) for v in variants
                                                for b in beams}]
    for ename, kb in KBS.items():
        d = np.load(REPO / f"benchmarks/queries/{kb.name}.npz",
                    allow_pickle=True)
        Qn = unit(d["queries"].astype(float))
        En = unit(np.load(sorted(kb.glob("bench_*"))[0] / "embeddings.npy")
                  .astype(float))
        for v in variants:
            ix = kb / f"bench_{DIRNAME.get(v, v)}"
            if not (ix / "ann_tree.json").exists():
                continue
            tree = json.loads((ix / "ann_tree.json").read_text())
            for beam in beams:
                one_round(tree, Qn[:WARMUP], En, beam)      # Cache füllen
                ts = sorted(one_round(tree, Qn, En, beam) / len(Qn) * 1000
                            for _ in range(ROUNDS))
                out.append({"encoder": ename, "variant": v, "beam": beam,
                            "ms": round(statistics.median(ts), 3),
                            "ms_min": round(ts[0], 3)})
                print(f"{ename:9s} {v:15s} beam {beam}  "
                      f"{statistics.median(ts):6.3f} ms  (min {ts[0]:.3f})",
                      flush=True)
    out.sort(key=lambda r: (r["encoder"], r["variant"], r["beam"]))
    dest.write_text(json.dumps(out, indent=1))
    print(f"\n{len(out)} Zeitmessungen → benchmarks/query_times.json")


if __name__ == "__main__":
    main()
