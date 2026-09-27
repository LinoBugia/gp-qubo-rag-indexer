#!/usr/bin/env python3
"""
Woran haengt der Recall — an der Partitionsguete oder an den Nachbarschaften?

    python3 experiments/objective_vs_recall.py

Die Doku behauptet an mehreren Stellen, dass nicht der Optimierer, sondern
die ZIELFUNKTION die bindende Nebenbedingung ist.  Dieses Skript ist der Beleg
dazu, und es ist der Grund, warum es `--objective knn` ueberhaupt gibt.

Ueber ALLE gebauten Baeume derselben Wissensbasis werden zwei Groessen gegen
recall@5 korreliert:

  * **E_root** — die QUBO-Energie der Wurzelbisektion, `pbf_energy_dense` auf
    derselben Gram-Matrix, die der Annealer minimiert.  Das ist Partitions-
    guete im Sinne der Zielfunktion: niedriger ist besser.
  * **cut@10** — der Anteil der symmetrischen 10-NN-Paare, die der Baum
    trennt (kein Blatt enthaelt beide).  Das ist das, was das Retrieval
    tatsaechlich ruiniert: ein zerschnittener echter Nachbar ist ein
    Kandidat, den der Abstieg nie zu sehen bekommt.

Beide Baum-Groessen sind auf DEMSELBEN Punktsatz gerechnet, also direkt
vergleichbar; der Recall kommt aus den bereits gemessenen Benchmarks bei
Beam 2 (results.json bzw. results_full.json), damit hier nichts neu
ausgewertet werden muss.

Erwartung laut docs/benchmarks.md: E_root korreliert schwach, cut@10 stark
negativ.
Das Skript sagt, ob das stimmt — es behauptet es nicht.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from clustering import clustering_gram_matrix                 # noqa: E402
from graph_partitioning import pbf_energy_dense               # noqa: E402

KB = REPO / "reference-kb/RefKB_bge-m3"
KNN_K = 10
BEAM = 2


def leaf_sets(node: dict, out: list[list[int]] | None = None) -> list[list[int]]:
    out = [] if out is None else out
    if node.get("children"):
        for c in node["children"]:
            leaf_sets(c, out)
    else:
        out.append(node["rows"])
    return out


def root_assignment(tree: dict, E: np.ndarray) -> np.ndarray:
    """x_i = 1 fuer Kind 0 (Seite t > tau).  Die Spill-Zone liegt in BEIDEN
    Kindern, deshalb entscheidet hier die Ebene und nicht die Mitgliedschaft —
    sonst waere die Zuordnung von der Reihenfolge abhaengig."""
    if "w" in tree and tree.get("tau") is not None:
        w = np.asarray(tree["w"], dtype=float)
        return (E @ w > float(tree["tau"])).astype(float)
    # Baeume ohne Ebene (--split-mode none/centroid): naeherer Kindmittelwert
    M = np.asarray([c["mean"] for c in tree["children"]], dtype=float)
    M = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-30)
    return (E @ M[0] > E @ M[1]).astype(float)


def knn_pairs(E: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Symmetrische k-NN-Paare (i<j) auf der Kugel, per Cosinus."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-30)
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    pairs = {(min(i, j), max(i, j))
             for i, row in enumerate(nb) for j in row}
    return np.asarray(sorted(pairs), dtype=np.int64)


def cut_fraction(leaves: list[list[int]], pairs: np.ndarray, n: int) -> float:
    """Anteil der k-NN-Paare, die KEIN gemeinsames Blatt mehr haben."""
    where: list[set[int]] = [set() for _ in range(n)]
    for lid, rows in enumerate(leaves):
        for r in rows:
            where[r].add(lid)
    cut = sum(1 for i, j in pairs if not (where[i] & where[j]))
    return cut / len(pairs)


def recall_table() -> dict[str, float]:
    """variant → recall@5 bei Beam 2, aus den vorhandenen Messungen."""
    out: dict[str, float] = {}
    for f, key in (("benchmarks/results.json", "beam"),
                   ("benchmarks/results_full.json", "param")):
        p = REPO / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if (r.get("kb") == KB.name and r.get(key) == BEAM
                    and r.get("anchors", 1) == 1
                    and r.get("search", "beam") == "beam"):
                out[r["variant"]] = r["recall@5"] * 100
    return out


def pearson(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a), np.asarray(b)
    x, y = x - x.mean(), y - y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / d) if d else float("nan")


def main() -> None:
    E = np.load(sorted(KB.glob("bench_*"))[0] / "embeddings.npy").astype(float)
    n = len(E)
    print(f"{KB.name}: {n} Chunks, {E.shape[1]}d")
    print("Gram-Matrix und k-NN-Graph einmal — beide Baumgroessen laufen "
          "darauf.")
    Q = clustering_gram_matrix(E)
    pairs = knn_pairs(E)
    print(f"{len(pairs)} symmetrische {KNN_K}-NN-Paare\n")

    rec = recall_table()
    rows = []
    for d in sorted(KB.glob("bench_*")):
        variant = d.name[len("bench_"):]
        tree_p = d / "ann_tree.json"
        if not tree_p.exists():
            continue
        # run_matrix ersetzt '.' und '+' im Ordnernamen — zurueckraten
        r = rec.get(variant) or rec.get(variant.replace("spill02", "spill0.2")) \
            or rec.get(variant.replace("_spill", "+spill"))
        if r is None:
            print(f"  · {variant:24s} kein Recall gemessen — uebersprungen")
            continue
        tree = json.loads(tree_p.read_text())
        leaves = leaf_sets(tree)
        e = pbf_energy_dense(Q, root_assignment(tree, E))
        c = cut_fraction(leaves, pairs, n)
        rows.append((variant, e, c * 100, r))
        print(f"  {variant:24s} E_root {e:12.1f}  cut@10 {c*100:5.2f} %  "
              f"recall {r:5.1f} %", flush=True)

    if len(rows) < 3:
        sys.exit("\nZu wenige Baeume fuer eine Korrelation.")
    # Zwei Populationen, und der Unterschied ist das eigentliche Ergebnis.
    # Die `*blind*`-Laeufe sind der DA OHNE Lloyd-Warmstart: dort ist die
    # Zielfunktion schlicht nicht geloest (Energie um Faktoren schlechter).
    # Nimmt man sie mit, korreliert die Energie scheinbar gut — man misst
    # dann nur "geloest gegen nicht geloest".  Interessant ist die Frage
    # DANACH: unter den Baeumen, die das Ziel erreichen, erklaert die Energie
    # noch irgendetwas?
    groups = [("alle Baeume", rows),
              ("nur warmgestartete", [r for r in rows if "blind" not in r[0]])]
    print(f"\n{len(rows)} Baeume auf demselben Punktsatz, Recall bei Beam 2.")
    stats = {}
    for name, sub in groups:
        _, es, cs, rs = zip(*sub)
        ce = pearson(list(es), list(rs))
        cc = pearson(list(cs), list(rs))
        spread = (max(es) - min(es)) / abs(float(np.mean(es))) * 100
        stats[name] = {"n": len(sub), "corr_energy_recall": ce,
                       "corr_cut_recall": cc, "energy_spread_pct": spread}
        print(f"\n  {name} (n = {len(sub)}), Energiespanne {spread:.2f} %")
        print(f"    corr(E_root , recall@5) = {ce:+.3f}"
              "   ← Partitionsguete im Sinne der Zielfunktion")
        print(f"    corr(cut@10 , recall@5) = {cc:+.3f}"
              "   ← zerschnittene echte Nachbarschaften")
    print("\nLesart: mit den blinden Laeufen misst die Energie nur, OB der\n"
          "Optimierer sein Ziel erreicht hat. Unter denen, die es erreichen,\n"
          "ist die Energie praktisch konstant — der Recall aber nicht. Was\n"
          "ihn erklaert, sind die zerschnittenen Nachbarschaften. Genau\n"
          "deshalb ist die Zielfunktion und nicht der Optimierer die Grenze.")
    out = REPO / "benchmarks/objective_vs_recall.json"
    out.write_text(json.dumps(
        {"kb": KB.name, "beam": BEAM, "knn_k": KNN_K,
         "trees": [{"variant": v, "E_root": e, "cut_pct": c, "recall_pct": r}
                   for v, e, c, r in rows],
         "correlations": stats}, indent=1))
    print(f"\n→ {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
