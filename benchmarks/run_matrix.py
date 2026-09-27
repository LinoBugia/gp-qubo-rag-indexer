#!/usr/bin/env python3
"""
Benchmark-Matrix: jede Wissensbasis in jeder Pipeline-Variante indexieren.

    python3 benchmarks/run_matrix.py --kbs reference-kb/subsets/* reference-kb/RefKB_bge-m3
    python3 benchmarks/run_matrix.py --only plane,lloyd --jobs 4

Jeder Lauf ist ein eigener Subprozess (`rag_indexer.py`), mehrere laufen
parallel.  Ergebnis ist ein Index-Ordner je (KB × Variante); ausgewertet wird
er von benchmarks/evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Die Matrix.  Jede Zeile isoliert EINEN Mechanismus gegen den Default,
# damit man am Ergebnis ablesen kann, was der Mechanismus beiträgt.
VARIANTS: dict[str, list[str]] = {
    # ── Wer bisektiert? ──────────────────────────────────────────────────
    "lloyd":        ["--no-anneal", "--split-mode", "plane", "--spill", "0.1"],
    "da-blind":     ["--iterations-lloyd", "0",
                     "--split-mode", "plane", "--spill", "0.1"],
    # ── Was macht der Nachschritt? (immer Lloyd + DA) ────────────────────
    "raw":          ["--split-mode", "none"],
    "centroid":     ["--split-mode", "centroid", "--spill", "0"],
    "plane":        ["--split-mode", "plane", "--spill", "0"],
    # ── Überlappung ──────────────────────────────────────────────────────
    "centroid+spill": ["--split-mode", "centroid", "--spill", "0.1"],
    "default":      ["--split-mode", "plane", "--spill", "0.1"],
    "spill0.2":     ["--split-mode", "plane", "--spill", "0.2"],
    # ── ε: Ebene in die sauberste Lücke, nur Unsichere spillen ───────────
    #  0.0745 = q90 der Projektionsverschiebung, an bge-m3 gemessen
    #  (256 Chunks, 60-80-%-Wortfenster, neu eingebettet).
    "eps":          ["--split-mode", "plane", "--spill", "0.1",
                     "--eps", "0.0745"],
    # ── Kontrollen: was kommt von wem? ───────────────────────────────────
    #  lloyd-raw: k-Means ohne jeden Nachschritt — die nackte Untergrenze.
    #  mc4:       der DA bekommt 4 Lloyd-Startbecken statt einem (der
    #             Default lloyd_mc=1 lässt die Multistarts ungenutzt).
    "lloyd-raw":    ["--no-anneal", "--split-mode", "none"],
    "mc4":          ["--lloyd-mc", "4",
                     "--split-mode", "plane", "--spill", "0.1"],
    "lloyd-mc4":    ["--no-anneal", "--lloyd-mc", "4",
                     "--split-mode", "plane", "--spill", "0.1"],
    # ── Die Achsen vollstaendig kreuzen ──────────────────────────────────
    #  Ohne diese Zeilen steht Lloyd nur bei ZWEI Grenzziehungen (nearest
    #  centroid und plane+10 %), der DA bei sieben — dann ist "die
    #  Grenzziehung traegt mehr als der Optimierer" an zwei Punkten
    #  abgelesen statt an allen.  Lloyd ist billig, also gibt es jede
    #  DA-Grenzziehung auch ohne Annealing.
    "lloyd-centroid":   ["--no-anneal", "--split-mode", "centroid",
                         "--spill", "0"],
    "lloyd-plane":      ["--no-anneal", "--split-mode", "plane",
                         "--spill", "0"],
    "lloyd-centroid-spill": ["--no-anneal", "--split-mode", "centroid",
                             "--spill", "0.1"],
    "lloyd-spill02":    ["--no-anneal", "--split-mode", "plane",
                         "--spill", "0.2"],
    "lloyd-eps":        ["--no-anneal", "--split-mode", "plane",
                         "--spill", "0.1", "--eps", "0.0745"],
    # ══ Die Hauptmatrix: WER bisektiert x WIE wird bewertet ═════════════
    #  Spill durchgehend nur 5 %: Overlap kauft Recall mit Redundanz und
    #  verdeckt damit, was Optimierer und Repraesentation wirklich tragen.
    #
    #  Achse 1 — Optimierer:  Lloyd | Lloyd->DA | DA ohne Warmstart
    #  Achse 2 — Beam-Score:  k = 1 (ein Mittelwert) | 4 | 16 Anker
    "m-lloyd":        ["--no-anneal", "--split-mode", "plane", "--spill", "0.05"],
    "m-lloyd-k4":     ["--no-anneal", "--split-mode", "plane", "--spill", "0.05",
                       "--anchors", "4"],
    "m-lloyd-k16":    ["--no-anneal", "--split-mode", "plane", "--spill", "0.05",
                       "--anchors", "16"],
    "m-da":           ["--split-mode", "plane", "--spill", "0.05"],
    "m-da-k4":        ["--split-mode", "plane", "--spill", "0.05",
                       "--anchors", "4"],
    "m-da-k16":       ["--split-mode", "plane", "--spill", "0.05",
                       "--anchors", "16"],
    #  DA OHNE Lloyd-Warmstart — die Referenz, die zeigt, dass der Annealer
    #  allein nicht traegt (schlechteste Variante der ganzen Matrix).
    "m-dablind":      ["--iterations-lloyd", "0", "--split-mode", "plane",
                       "--spill", "0.05"],
    "m-dablind-k16":  ["--iterations-lloyd", "0", "--split-mode", "plane",
                       "--spill", "0.05", "--anchors", "16"],
    #  Untergrenzen ohne jede Grenzziehung und ohne Overlap
    "m-lloyd-bare":   ["--no-anneal", "--split-mode", "none"],
    "m-da-bare":      ["--split-mode", "none"],
    #  Anker ganz ohne Spill: was traegt die Repraesentation allein?
    #  Zielfunktion k-NN-Modularity — wirkt NUR mit dem Annealer (Lloyd
    #  bisektiert im Koordinatenraum und sieht die Graphmatrix nie).
    "knn-da":         ["--objective", "knn", "--split-mode", "plane",
                       "--spill", "0.05"],
    "knn-dablind":    ["--objective", "knn", "--iterations-lloyd", "0",
                       "--split-mode", "plane", "--spill", "0.05"],
}


COMMON = ["--no-visualize-first", "--stats", "--quiet", "--seed", "7"]
SKIP_EXISTING = False


def run_one(kb: Path, name: str, args: list[str], g: int) -> dict:
    """Einen Index bauen; fester Zielordner je Variante (keine Nummern-
    Kollision, wenn mehrere Laeufe parallel in dieselbe KB schreiben)."""
    dest = kb / f"bench_{name.replace('.', '').replace('+', '_')}"
    if dest.exists():
        if SKIP_EXISTING and (dest / "ann_tree.json").exists():
            print(f"  · {kb.name:26s} {name:16s} schon da — übersprungen")
            return {"kb": kb.name, "variant": name, "ok": True, "seconds": 0.0,
                    "index": str(dest), "args": args, "skipped": True}
        shutil.rmtree(dest)
    cmd = [sys.executable, "-u", str(REPO / "rag_indexer.py"), str(kb),
           "-o", str(dest), "-g", str(g), *args, *COMMON]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0
    out = {"kb": kb.name, "variant": name, "ok": ok,
           "seconds": round(time.time() - t0, 1),
           "index": str(dest) if ok else None, "args": args}
    if not ok:
        out["error"] = (proc.stderr or proc.stdout)[-400:]
    print(f"  {'✓' if ok else '✗'} {kb.name:26s} {name:16s} "
          f"{out['seconds']:7.1f}s  {out.get('index') or out.get('error','')}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kbs", nargs="+", required=True)
    ap.add_argument("--only", default="", help="Komma-Liste von Varianten")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("-g", type=int, default=20)
    ap.add_argument("--out", default="benchmarks/runs.json")
    ap.add_argument("--skip-existing", action="store_true",
                    help="fertige bench_<name>/ nicht neu bauen")
    args = ap.parse_args()

    names = ([n.strip() for n in args.only.split(",") if n.strip()]
             or list(VARIANTS))
    kbs = [Path(k) for k in args.kbs if Path(k).is_dir()]
    jobs = [(kb, n, VARIANTS[n]) for kb in kbs for n in names]
    print(f"{len(jobs)} Läufe ({len(kbs)} KBs × {len(names)} Varianten), "
          f"{args.jobs} parallel\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        res = list(ex.map(lambda j: run_one(j[0], j[1], j[2], args.g), jobs))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else []
    prev = [r for r in prev
            if (r["kb"], r["variant"]) not in {(x["kb"], x["variant"])
                                               for x in res}]
    out.write_text(json.dumps(prev + res, indent=1), encoding="utf-8")
    print(f"\n{sum(r['ok'] for r in res)}/{len(res)} ok in "
          f"{time.time() - t0:.0f}s → {out}")


if __name__ == "__main__":
    main()
