#!/usr/bin/env python3
"""
Teil-Wissensbasen aus einer RefKB schneiden — DOKUMENTWEISE, nicht zufällig.

    python3 benchmarks/make_subsets.py reference-kb/RefKB_bge-m3 500 1000 2000

Warum nicht zufällig: eine echte Knowledgebase ist thematisch geklumpt.
Zieht man Chunks quer über alle Dateien, zerstört das genau die Struktur,
die das Partitionieren ausnutzen soll — der Baum sähe künstlich schwer aus.
Also werden ganze Quelldokumente aufgesammelt, bis die Zielgröße erreicht
ist; `ids_by_file` aus specs.json liefert sie in Dokumentreihenfolge
(RAG_INDEX_CONTRACT.md §5).

Die Ausgabe ist wieder ein vertragstreuer Run-Ordner: chunks.jsonl,
embeddings.jsonl, specs.json mit korrigiertem chunk_count / ids /
ids_by_file / input_files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(run: Path) -> tuple[dict, list[dict], dict[str, list]]:
    specs = json.loads((run / "specs.json").read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in
              (run / specs["files"]["chunks"]).read_text(
                  encoding="utf-8").splitlines() if l.strip()]
    embs: dict[str, list] = {}
    if specs["files"].get("embeddings"):
        for line in (run / specs["files"]["embeddings"]).read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                embs[rec["id"]] = rec["embedding"]
    return specs, chunks, embs


def pick_files(specs: dict, target: int) -> list[str]:
    """Ganze Dateien aufsammeln, bis die Zielgröße am besten getroffen ist."""
    by_file = specs.get("ids_by_file") or {}
    if not by_file:                       # Fallback: aus den Chunks ableiten
        raise SystemExit("specs.json ohne ids_by_file — Contract §5 verletzt.")
    taken, n = [], 0
    for fname, ids in by_file.items():
        if n >= target:
            break
        # Datei nur nehmen, wenn sie das Ziel nicht massiv überschießt
        if n and n + len(ids) > target * 1.12:
            continue
        taken.append(fname)
        n += len(ids)
    return taken


def write_subset(run: Path, out: Path, target: int) -> dict:
    specs, chunks, embs = load(run)
    files = pick_files(specs, target)
    keep = {cid for f in files for cid in specs["ids_by_file"][f]}
    sub = [c for c in chunks if c["id"] in keep]

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as fh:
        for c in sub:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "embeddings.jsonl", "w", encoding="utf-8") as fh:
        for c in sub:
            fh.write(json.dumps({"id": c["id"], "embedding": embs[c["id"]]},
                                ensure_ascii=False) + "\n")

    s = dict(specs)
    s["input_files"] = files
    s["ids_by_file"] = {f: specs["ids_by_file"][f] for f in files}
    s["ids"] = [c["id"] for c in sub]
    s["chunk_count"] = len(sub)
    s["modes_by_file"] = {f: m for f, m in (specs.get("modes_by_file") or {}
                                            ).items() if f in files}
    s["subset_of"] = run.name
    s["subset_target"] = target
    (out / "specs.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"dir": out.name, "chunks": len(sub), "files": len(files),
            "target": target}


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    targets = [int(a) for a in sys.argv[2:]]
    base = run.parent / "subsets"
    for t in targets:
        info = write_subset(run, base / f"{run.name}_n{t}", t)
        print(f"{info['dir']:38s} {info['chunks']:5d} Chunks aus "
              f"{info['files']:3d} Dateien  (Ziel {info['target']})")


if __name__ == "__main__":
    main()
