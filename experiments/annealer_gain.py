#!/usr/bin/env python3
"""
Wieviel Energie nimmt der Annealer Lloyd ab — und wo im Baum?

    python3 experiments/annealer_gain.py

`rag_indexer --stats` schreibt je Bisektion `E_lloyd` (die Energie des
Warmstarts) und `energy` (was der DA daraus gemacht hat).  Der Gewinn ist
`(E_lloyd - energy) / |E_lloyd|`.

Zwei Dinge, die man beim Lesen wissen muss:

  * **Median, nie Mittelwert.**  In kleinen Gruppen tief unten geht E_lloyd
    gegen null, und der relative Gewinn explodiert (ueber 16 000 % kommt vor).
    Ein Mittelwert misst dort nur, wie klein der Nenner war.
  * **Die Zielfunktion steht dabei.**  `gram` und `knn` sind verschiedene
    Energien; ihre Prozente sind NICHT gegeneinander lesbar, nur je Spalte.

Dass der Gewinn gross ist, heisst nicht, dass der Recall folgt — genau das
misst experiments/objective_vs_recall.py, und die Antwort dort ist nein.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
KBS = ["RefKB_bge-m3", "RefKB_qwen3-embedding-4b-fp16"]
VARIANTS = ["m-da", "knn-da", "m-dablind", "knn-dablind"]


def gains(stats_path: Path) -> tuple[list[float], str]:
    b = json.loads(stats_path.read_text())["bisections"]
    obj = b[0].get("objective", "gram")
    g = [(x["E_lloyd"] - x["energy"]) / abs(x["E_lloyd"]) * 100
         for x in b
         if x.get("E_lloyd") not in (None, 0) and x.get("energy") is not None]
    return g, obj


def main() -> None:
    any_row = False
    for kb in KBS:
        base = REPO / "reference-kb" / kb
        if not base.is_dir():
            continue
        print(f"\n{kb}")
        print(f"  {'variant':13s} {'objective':10s} {'n':>5s} {'root':>8s} "
              f"{'median':>8s} {'q90':>8s} {'= 0 %':>7s}")
        for v in VARIANTS:
            p = base / f"bench_{v}" / "indexing_stats.json"
            if not p.exists():
                continue
            g, obj = gains(p)
            if not g:
                continue
            any_row = True
            zero = sum(x <= 1e-9 for x in g) / len(g) * 100
            print(f"  {v:13s} {obj:10s} {len(g):5d} {g[0]:7.2f}% "
                  f"{np.median(g):7.2f}% {np.percentile(g, 90):7.2f}% "
                  f"{zero:6.1f}%")
    if not any_row:
        sys.exit("Keine indexing_stats.json gefunden — erst run_matrix.py.")
    print("\n'root' ist die Wurzelbisektion, 'q90' das 90-%-Quantil ueber alle\n"
          "Bisektionen, '= 0 %' der Anteil, in dem der DA Lloyds Loesung nicht\n"
          "verbessern konnte.")


if __name__ == "__main__":
    main()
