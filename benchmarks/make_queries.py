#!/usr/bin/env python3
"""
Query-Sätze für den Benchmark bauen — echte Embeddings, kein simuliertes
Rauschen.

    python3 benchmarks/make_queries.py reference-kb/RefKB_bge-m3 [--n 300]

Eine Benchmark-Query ist ein zufälliges zusammenhängendes 60–80-%-Wortfenster
eines Chunks, mit DEMSELBEN Encoder eingebettet, mit dem die Wissensbasis
gebaut wurde (RAG_INDEX_CONTRACT.md §7.1).  Das ist der RAG-Fall: die Frage
deckt einen Aspekt des Chunks ab, nicht den ganzen Chunk, und liegt deshalb
ein Stück neben ihm — genau die Sorte Abweichung, an der sich ein
Partitionsbaum bewähren muss.

Der Satz wird einmal je Wissensbasis erzeugt und gecacht, damit alle
Varianten desselben Index gegen EXAKT dieselben Queries antreten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_indexer as ri                                    # noqa: E402

OUT = Path(__file__).resolve().parent / "queries"


# Qwen3-Embedding ist ein Instruct-Modell: die Query-Seite gewinnt durch ein
# Aufgaben-Präfix, die Dokumente sind roh eingebettet.  Für den Vergleich der
# beiden Encoder wird OHNE Präfix gemessen (identische Methodik), der Effekt
# des Präfix separat.
QWEN_PREFIX = ("Instruct: Given a search query, retrieve relevant passages "
               "that answer the query\nQuery: ")


def build(run: Path, n: int, seed: int = 11, prefix: str = "",
          suffix: str = "") -> Path:
    specs, chunks, ids, emb = ri.load_run_folder(run)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(chunks), size=min(n, len(chunks)), replace=False)
    # Vertrag §2: der Vektor wurde aus embed_text berechnet, wenn vorhanden
    texts = [ri.perturb_text(chunks[int(r)].get("embed_text")
                             or chunks[int(r)].get("text", ""), rng)
             for r in rows]
    if prefix:
        texts = [prefix + t for t in texts]
    print(f"{run.name}: {len(rows)} Queries mit "
          f"{specs['embedding']['model']} einbetten "
          f"{'(mit Instruct-Präfix)' if prefix else ''}…")
    Q = ri.embed_texts(texts, specs["embedding"], verbose=True)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{run.name}{suffix}.npz"
    np.savez_compressed(path, queries=Q.astype(np.float32),
                        gold_rows=rows.astype(np.int32),
                        gold_ids=np.array([ids[int(r)] for r in rows]))
    meta = {"kb": run.name, "model": specs["embedding"]["model"],
            "dim": int(specs["embedding_dim"]), "n_queries": int(len(rows)),
            "window": "60-80 % contiguous words", "seed": seed}
    meta["prefix"] = prefix or None
    (OUT / f"{run.name}{suffix}.json").write_text(json.dumps(meta, indent=1))
    print(f"  → {path.name}  ({Q.shape[0]}×{Q.shape[1]})")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--qwen-prefix", action="store_true",
                    help="Instruct-Präfix voranstellen (Qwen3-Embedding); "
                         "landet in <kb>_instruct.npz")
    a = ap.parse_args()
    for r in a.runs:
        build(Path(r), a.n,
              prefix=QWEN_PREFIX if a.qwen_prefix else "",
              suffix="_instruct" if a.qwen_prefix else "")


if __name__ == "__main__":
    main()
