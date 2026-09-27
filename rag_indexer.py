#!/usr/bin/env python3
"""
rag_indexer.py — text-embedder run folder → self-contained RAG query folder
============================================================================

Takes an output run folder of the text-embedder pipeline
(specs.json + chunks.jsonl + embeddings.jsonl), partitions the chunk
embeddings with the recursive DA graph partitioning (graph_partitioning.py)
and writes a self-contained, numbered index folder next to the data —
every run gets a fresh `rag_index_NNN/`, existing indexings are never
overwritten (a run folder can carry several, e.g. for parameter sweeps):

    <run_folder>/rag_index_001/
    ├── query.py               ← standalone mini terminal script (numpy only)
    ├── index_meta.json        ← encoder spec + partition params + counts
    ├── ann_tree.json          ← binary partition tree: means + leaf rows
    ├── embeddings.npy         ← (n, d) float32 chunk embeddings
    ├── chunks_manifest.json   ← chunk id → path/source/heading
    └── chunks/<id>.md         ← one file per chunk (the returned paths)

Indexing may take a while (one DA solve per bisection) — that is fine,
querying afterwards is a handful of scalar products.

Usage
-----
    python rag_indexer.py <run_folder> [-g 10] [-o OUT] [--seed S]
                          [--steps AUTO] [--num-mc 20]
                          [--cooling-a AUTO] [--cooling-b 0] [--cooling-d 2]
                          [--offset-increase 1000] [--gram-threshold 0]
                          [--iterations-lloyd 0] [--lloyd-mc 1]
                          [--solver fast|reference]
                          [--no-visualize-first] [--no-stats] [--quiet]

The DA parameters default to the calibrated values — leave them alone
unless you know why.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import graph_partitioning as gp
from graph_partitioning import partition_tree, tree_leaves

HERE = Path(__file__).resolve().parent
QUERY_TEMPLATE = HERE / "rag_query.py"


# ── Input loading ─────────────────────────────────────────────────────────────

def load_run_folder(run_dir: Path) -> tuple[dict, list[dict], list[str], np.ndarray]:
    """
    Read specs.json, chunks.jsonl and embeddings.jsonl from a text-embedder
    run folder.  Returns (specs, chunks, chunk_ids, embeddings) with row i of
    *embeddings* belonging to chunk_ids[i] / chunks[i].
    """
    specs_path = run_dir / "specs.json"
    if not specs_path.exists():
        sys.exit(f"Error: '{run_dir}' is not a text-embedder run folder "
                 f"(specs.json missing).")

    with open(specs_path, encoding="utf-8") as fh:
        specs = json.load(fh)

    chunks_by_id: dict[str, dict] = {}
    with open(run_dir / specs["files"]["chunks"], encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                chunks_by_id[rec["id"]] = rec

    chunk_ids: list[str] = []
    vectors: list[list[float]] = []
    emb_file = specs.get("files", {}).get("embeddings")
    if not emb_file:
        sys.exit(f"Error: '{run_dir}' is a chunk-only run (no embeddings) — "
                 f"nothing to index. See RAG_INDEX_CONTRACT.md §1.")
    with open(run_dir / emb_file, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                chunk_ids.append(rec["id"])
                vectors.append(rec["embedding"])

    missing = [cid for cid in chunk_ids if cid not in chunks_by_id]
    if missing:
        sys.exit(f"Error: {len(missing)} embedding ids have no chunk "
                 f"(first: {missing[0]}).")

    embeddings = np.asarray(vectors, dtype=float)
    dim = specs.get("embedding_dim")
    if dim is not None and embeddings.shape[1] != dim:
        sys.exit(f"Error: embeddings have {embeddings.shape[1]} dims, "
                 f"specs.json says {dim}.")

    chunks = [chunks_by_id[cid] for cid in chunk_ids]
    return specs, chunks, chunk_ids, embeddings


# ── ε-Kalibrierung: den Unsicherheitsradius am echten Encoder messen ──────────
#  Ein Embedding ist keine exakte Koordinate: derselbe Inhalt, anders
#  formuliert oder anders beschnitten, landet ein Stück daneben.  Wie weit,
#  hängt allein am Encoder — also messen wir es an ihm, statt es zu raten.
#
#  Störung: ein zufälliges zusammenhängendes FENSTER des Chunks (60–80 % der
#  Wörter).  Das ist der RAG-Fall — eine Query trifft selten den ganzen
#  Chunk, sondern einen Aspekt daraus.  Tippfehler oder Wort-Dropout wären
#  billiger zu erzeugen, treffen aber die falsche Fehlerquelle.
#
#  Gemessen wird nicht der Abstand im Einbettungsraum, sondern die Streuung
#  der PROJEKTION (siehe gp.estimate_eps) — nur die ist die Einheit, in der
#  die Trennebene rechnet.
EPS_SAMPLE = 256


def perturb_text(text: str, rng: np.random.Generator,
                 keep_lo: float = 0.6, keep_hi: float = 0.8) -> str:
    """Zufälliges zusammenhängendes Wortfenster (keep_lo..keep_hi des Chunks)."""
    words = text.split()
    if len(words) < 8:
        return text
    frac = float(rng.uniform(keep_lo, keep_hi))
    k = max(4, int(len(words) * frac))
    start = int(rng.integers(0, len(words) - k + 1))
    return " ".join(words[start:start + k])


def embed_texts(texts: list[str], emb_spec: dict,
                verbose: bool = True, timeout: float = 600.0) -> np.ndarray:
    """
    Texte mit demselben Encoder einbetten, mit dem der Index gebaut wurde.

    Unterstützt den ollama-Provider (POST /api/embed, lokal, kein Schlüssel).
    Der Timeout ist großzügig: ein großes Embedding-Modell (qwen3-4b-fp16 =
    8 GB) muss beim ersten Aufruf erst geladen werden, und wenn nebenher
    indexiert wird, dauert das leicht mehrere Minuten.
    Für andere Provider bricht die Messung ab, statt einen Ersatzencoder zu
    verwenden — ε eines fremden Modells wäre schlicht die falsche Zahl.
    """
    import urllib.request

    provider = emb_spec.get("provider", "ollama")
    if provider != "ollama":
        raise RuntimeError(
            f"ε-Messung unterstützt nur den ollama-Provider, der Index wurde "
            f"mit '{provider}' gebaut. Nutze --eps cos:<wert> (Schätzung aus "
            f"einer bekannten Paraphrasen-Cosine) oder --eps <zahl>.")

    base = emb_spec.get("base_url", "http://localhost:11434").rstrip("/")
    model = emb_spec["model"]
    batch = int(emb_spec.get("batch_size", 16))
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        payload = json.dumps({"model": model,
                              "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(
            f"{base}/api/embed", data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                f"ollama unter {base} nicht erreichbar ({exc}) — läuft "
                f"`ollama serve`, ist '{model}' gepullt? (Timeout "
                f"{timeout:.0f}s, Batch {batch})") from exc
        out.extend(data["embeddings"])
        if verbose:
            print(f"    ε-Sample: {min(i + batch, len(texts))}/{len(texts)} "
                  f"eingebettet", end="\r")
    if verbose:
        print()
    return np.asarray(out, dtype=float)


def measure_eps(chunks: list[dict], embeddings: np.ndarray, specs: dict, *,
                sample: int = EPS_SAMPLE, seed: int | None = None,
                quantile: float = gp.EPS_QUANTILE,
                verbose: bool = True) -> dict:
    """
    ε dynamisch am Encoder messen → dict mit eps + Diagnostik (für index_meta).
    """
    rng = np.random.default_rng(seed)
    n = len(chunks)
    take = min(int(sample), n)
    rows = rng.choice(n, size=take, replace=False)
    texts = [perturb_text(chunks[int(r)].get("embed_text")
                          or chunks[int(r)].get("text", ""), rng)
             for r in rows]

    if verbose:
        print(f"  ε-Kalibrierung: {take} Chunks, Fenster 60–80 %, Encoder "
              f"{specs['embedding']['model']} …")
    P_pert = embed_texts(texts, specs["embedding"], verbose=verbose)

    info = gp.estimate_eps(embeddings[rows], P_pert,
                           quantile=quantile,
                           directions_from=embeddings,
                           seed=int(rng.integers(0, 2 ** 31 - 1)))
    info["quantile"] = float(quantile)
    info["sample"] = take
    info["source"] = "measured"
    if verbose:
        print(f"  ε = {info['eps']:.5f}  (Median {info['median']:.5f}, "
              f"q99 {info['q99']:.5f}; {info['n_pairs']} Paare × "
              f"{info['n_dirs']} Richtungen)")
    return info


def resolve_eps(eps_arg, chunks, embeddings, specs, *,
                seed=None, verbose=True) -> tuple[float, dict]:
    """
    --eps in eine Zahl auflösen.  Erlaubt:
      <zahl>     direkt (Projektionseinheit)
      "auto"     am Encoder messen (measure_eps)
      "cos:0.9"  aus einer bekannten Paraphrasen-Cosine ableiten
    Rückgabe (eps, info-dict fürs Meta).
    """
    if isinstance(eps_arg, str):
        s = eps_arg.strip().lower()
        if s == "auto":
            info = measure_eps(chunks, embeddings, specs,
                               seed=None if seed is None else int(seed),
                               verbose=verbose)
            return float(info["eps"]), info
        if s.startswith("cos:"):
            cos = float(s[4:])
            val = gp.eps_from_cosine(cos, embeddings.shape[1])
            if verbose:
                print(f"  ε = {val:.5f}  (aus cos={cos:g}, d="
                      f"{embeddings.shape[1]}; Schätzung, nicht gemessen)")
            return val, {"eps": val, "source": "cosine", "cos": cos}
        eps_arg = float(s)
    val = float(eps_arg)
    return val, {"eps": val, "source": "fixed"}


# ── Tree export (rows instead of raw indices, no per-node index blowup) ───────

def export_tree(node: dict, anchor_sink: list | None = None) -> dict:
    """
    Serialize a partition_tree node for ann_tree.json: every node keeps its
    mean (for the query descent), leaves keep their embedding rows.

    Anker landen NICHT im JSON, sondern in *anchor_sink* → anchors.npy; der
    Knoten trägt nur [offset, count].  Als Text wären sie das Vielfache des
    ganzen Baums: 16 Anker × 1024 Dimensionen je Knoten sind bei 400 Chunks
    schon 20 MB JSON gegen 1.9 MB ohne.
    """
    out = {"depth": node["depth"], "n": node["n"],
           "mean": [round(v, 8) for v in node["mean"]]}
    if node.get("anchors") is not None and anchor_sink is not None:
        A = node["anchors"]
        out["anchors"] = [sum(len(x) for x in anchor_sink), len(A)]
        anchor_sink.append(A)
    if node.get("w") is not None:
        # Max-Margin-Trennebene: der Descent entscheidet über q·ŵ − τ.
        # σ normiert den Score, damit er über Knoten hinweg vergleichbar
        # ist (der Beam stellt Kandidaten verschiedener Eltern gegenüber).
        out["w"] = [round(v, 8) for v in node["w"]]
        out["tau"] = round(node["tau"], 10)
        out["sigma"] = round(node["sigma"], 10)
    if node["children"] is None:
        out["rows"] = node["indices"]
        out["children"] = None
    else:
        out["children"] = [export_tree(c, anchor_sink)
                           for c in node["children"]]
    return out


# ── Chunk files ───────────────────────────────────────────────────────────────

def write_chunk_files(chunks: list[dict], out_dir: Path) -> dict[str, dict]:
    """
    Write every chunk as chunks/<id>.md and return the manifest entries
    (paths relative to the index folder).
    """
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    for rec in chunks:
        rel = f"chunks/{rec['id']}.md"
        with open(out_dir / rel, "w", encoding="utf-8") as fh:
            fh.write(rec["text"])
        md = rec.get("metadata", {})
        entries[rec["id"]] = {
            "path": rel,
            "source": rec.get("source"),
            "heading_path": md.get("heading_path"),
            "token_count": md.get("token_count"),
        }
    return entries


# ── Mehrere Indexierungen pro Run-Ordner ─────────────────────────────────────

def list_index_dirs(run_dir: Path) -> list[Path]:
    """Alle fertigen Indexierungen eines Run-Ordners (rag_index,
    rag_index_001, …), sortiert nach Name — die letzte ist die neueste."""
    return sorted(p for p in run_dir.glob("rag_index*")
                  if (p / "index_meta.json").exists())


def next_index_dir(run_dir: Path) -> Path:
    """Nächster freier nummerierter Index-Ordner rag_index_NNN."""
    n = 1
    while (run_dir / f"rag_index_{n:03d}").exists():
        n += 1
    return run_dir / f"rag_index_{n:03d}"


# ── Core (importable — GUI benutzt das direkt) ────────────────────────────────

def build_index(
    run_dir: Path,
    out_dir: Path | None = None,
    *,
    g: int = 20,
    steps: int | None = None,
    num_mc: int = 20,
    offset_increase: float | str = "auto_gp",   # Zahl oder "auto_gp" (dynamisch im Solver)
    cooling_a: float | None = None,
    cooling_b: float = 0.0,
    cooling_d: float = 2.0,
    iterations_lloyd: int = 20,
    lloyd_mc: int = 1,
    cooling_schedule: str = "da_gp",
    cooling_per_group: bool = True,
    hold_steps: int | None = None,
    offset_k_escape: float = 5.0,
    split_mode: str = "plane",
    spill: float = 0.1,
    eps: float | str = 0.0,      # Zahl | "auto" | "cos:<wert>"
    anneal: bool = True,
    anchors: int = 1,            # Beam-Repräsentation: k Anker je Knoten
    objective: str | None = None,  # None → "knn" mit DA, "gram" ohne
    p_hot: float = 0.4,
    p_cold: float = 0.02,
    gamma: float = 0.05,
    seed: float | None = None,
    verbose: bool = True,
    progress_cb=None,
    visualize_first: bool = True,
    gram_threshold: float = 0.0,
    solver: str = "fast",
    save_stats: bool = True,
) -> tuple[Path, dict]:
    """
    Kompletter Indexing-Lauf: Run-Ordner lesen → DA-Partitionierung →
    rag_index/-Ordner schreiben.  progress_cb wird an partition_tree
    durchgereicht (Events: bisect_start / split / leaf — siehe dort).

    iterations_lloyd — > 0: k=2-Lloyd-Vorschritt vor jeder Bisection
    (zufälliger Start bis Konvergenz oder Limit, Lösung unverändert); Cooling
    wechselt auf ["da_gp", steps, None, lloyd_assign, E_lloyd, E_start, p_hot, p_cold, gamma].
    0 = aus → kalibriertes logarithmic-Cooling.

    lloyd_mc — > 1: PCA-orientierte Lloyd-Multi-Starts je Bisection
    (Sign-Splits entlang der Hauptkomponenten + λ-gewichtete Mischungen).
    Mit solver="fast" laufen num_mc DA-Trials PRO uniquer Lloyd-Lösung in
    einem Batch; "reference" startet nur ab der besten.

    solver — "fast" (annealing-qubo-optimizer: heuristischer batched
    numpy-DA, QUBO als Matrix, kein PBF-Dict) oder "reference"
    (annealing-cop-approximator: dict-basierter Originalpfad).  Gleiche
    Kalibrierung, gleiche Landschaft — beides Heuristiken ohne
    Optimalitätsgarantie.

    gram_threshold — Sparsity-Schwelle: |Q[i,j]| < thr wird schon bei der
    blockweisen Gram-Berechnung verworfen (sparse, nie eine dichte
    n×n-Matrix; 0 = aus).  save_stats — zusätzlich indexing_stats.json mit den KPIs
    jeder einzelnen Bisection schreiben (Energie, genutzte Steps, Zeiten,
    Transfers, …) — die Messwerte fürs Kalibrieren.

    Returns
    -------
    (out_dir, meta)  — Pfad des Index-Ordners und index_meta-Dict
    """
    run_dir = Path(run_dir).resolve()
    # Ohne explizites Ziel: nächste freie Nummer — bestehende Indexierungen
    # werden nie überschrieben, ein Run-Ordner kann mehrere tragen.
    out_dir = Path(out_dir).resolve() if out_dir else next_index_dir(run_dir)

    # ── 1. Load embedder run ─────────────────────────────────────────────────
    if verbose:
        print(f"Loading run folder: {run_dir.name}")
    specs, chunks, chunk_ids, embeddings = load_run_folder(run_dir)
    n, d = embeddings.shape
    if verbose:
        print(f"  {n} chunks × {d} dims  "
              f"(encoder: {specs['embedding']['model']} via {specs['embedding']['provider']})")

    # ── 2. Partition (this is the slow, DA-heavy part) ───────────────────────
    if verbose:
        print(f"\nPartitioning down to leaf size g={g} …")

    # KPI-Sammlung: split-Events tragen die Bisection-Statistiken; die
    # großen Index-Listen bleiben draußen.
    bisections: list[dict] = []

    def _cb(event: str, info: dict) -> None:
        if event == "split":
            rec = {k: v for k, v in info.items()
                   if k not in ("idx_A", "idx_B")}
            rec["n_A"] = len(info["idx_A"])
            rec["n_B"] = len(info["idx_B"])
            rec["seconds"] = round(info["seconds"], 3)
            bisections.append(rec)
        if progress_cb is not None:
            progress_cb(event, info)

    # Auto-Budget aus der GESAMTZAHL der Chunks (nicht pro Ebene): steps
    # sind der Wert für Ebene 0 und halbieren sich von dort mit der Tiefe.
    n_total = len(embeddings)
    if steps is None:
        steps = max(gp.STEPS_FLOOR, n_total // 3)
        if verbose:
            print(f"steps auto: n={n_total} → {steps} (n/3, Boden "
                  f"{gp.STEPS_FLOOR})")
    if hold_steps is None:
        hold_steps = n_total // 10
        if verbose:
            print(f"hold auto: n={n_total} → {hold_steps} (n/10)")

    # ε auflösen (Zahl | "auto" = am Encoder messen | "cos:<wert>") — vor
    # dem Baum, weil jede Bisection damit rechnet.
    # `knn` wirkt ausschliesslich ueber das QUBO — und das wird ohne DA gar
    # nicht erst gebaut.  Lloyd bisektiert im Koordinatenraum und kann eine
    # Graph-Modularitaet nicht optimieren; die Kombination waere also still
    # wirkungslos statt anders.
    # Default: k-NN-Modularity, wo ein Annealer sie optimieren kann.  Sie
    # baut Bäume, die weniger echte Nachbarschaften zerschneiden (gemessen
    # 61.2 % statt 64.0 %), und das bei 4× kürzerer Bauzeit, weil die
    # Graphmatrix dünn ist.  Ohne DA ist die Wahl gegenstandslos — Lloyd
    # bisektiert im Koordinatenraum und sieht keine Matrix.
    if objective is None:
        objective = "knn" if anneal else "gram"
    elif objective == "knn" and not anneal:
        sys.exit("Error: --objective knn braucht den Annealer — Lloyd "
                 "bisektiert im Koordinatenraum und sieht die Graphmatrix "
                 "nie. Entweder --anneal (default) oder --objective gram.")

    eps, eps_info = resolve_eps(eps, chunks, embeddings, specs,
                                seed=seed, verbose=verbose)

    t0 = time.time()
    root = partition_tree(
        embeddings, g,
        steps=steps, num_mc=num_mc,
        offset_increase=offset_increase,
        cooling_a=cooling_a, cooling_b=cooling_b, cooling_d=cooling_d,
        iterations_lloyd=iterations_lloyd, lloyd_mc=lloyd_mc,
        cooling_schedule=cooling_schedule,
        cooling_per_group=cooling_per_group,
        hold_steps=hold_steps, offset_k_escape=offset_k_escape,
        split_mode=split_mode, spill=spill, eps=eps,
        anneal=anneal,
        p_hot=p_hot, p_cold=p_cold, gamma=gamma,
        seed_gen=seed,
        verbose=verbose, progress_cb=_cb,
        visualize_first=visualize_first,
        gram_threshold=gram_threshold,
        solver=solver,
        objective=objective,
        anchors=anchors,
    )
    elapsed = time.time() - t0
    leaves = tree_leaves(root)
    sizes = [leaf["n"] for leaf in leaves]
    if verbose:
        print(f"\nPartitioning done in {elapsed:.1f}s — {len(leaves)} leaf groups, "
              f"sizes {min(sizes)}–{max(sizes)}")

    # ── 3. Write index folder ────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = write_chunk_files(chunks, out_dir)
    with open(out_dir / "chunks_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"row_to_id": chunk_ids, "chunks": manifest_entries},
                  fh, ensure_ascii=False, indent=2)

    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))

    anchor_sink: list = []
    tree_json = export_tree(root, anchor_sink)
    with open(out_dir / "ann_tree.json", "w", encoding="utf-8") as fh:
        json.dump(tree_json, fh, ensure_ascii=False)
    if anchor_sink:
        np.save(out_dir / "anchors.npy",
                np.concatenate([np.asarray(a, dtype=np.float32)
                                for a in anchor_sink]))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": run_dir.name,
        "encoder": {
            "provider": specs["embedding"]["provider"],
            "model":    specs["embedding"]["model"],
            "dim":      d,
        },
        "chunk_count": n,
        "partitioning": {
            "algorithm": "recursive_bisection_qubo_digital_annealing",
            "solver": solver,
            # Zielfunktion und Knoten-Repraesentation gehoeren hier hin: ein
            # knn-Index ist sonst von einem gram-Index nicht zu unterscheiden,
            # und ohne `anchors` weiss der Leser nicht, wonach der Beam scort.
            "objective": objective,
            "anchors": anchors,
            "g": g,
            "steps": steps,
            "num_mc": num_mc,
            "offset_increase": offset_increase,
            "cooling": ({"schedule": cooling_schedule, "p_hot": p_hot,
                         "p_cold": p_cold, "gamma": gamma,
                         "per_group": cooling_per_group,
                         "hold_steps": hold_steps,
                         "hold_max_frac": gp.HOLD_MAX_FRAC,
                         "offset_k_escape": offset_k_escape}
                        if iterations_lloyd > 0 else
                        {"schedule": "logarithmic",
                         "a": cooling_a if cooling_a is not None else "auto",
                         "b": cooling_b,
                         "d": cooling_d}),
            "split_mode": split_mode,
            "spill": spill,
            "eps": eps,
            "eps_info": eps_info,
            "anneal": anneal,
            "margin_quantile": gp.MARGIN_QUANTILE,
            "iterations_lloyd": iterations_lloyd,
            "lloyd_mc": lloyd_mc,
            "steps_schedule": {"type": "halving",
                               "per_depth": "steps/2**depth",
                               "min": gp.STEPS_FLOOR,
                               "mc_small_n": gp.MC_SMALL_N,
                               "mc_small_trials": gp.MC_SMALL_TRIALS},
            "gram_threshold": gram_threshold,
            "seed": seed,
            "num_leaves": len(leaves),
            "leaf_sizes": sizes,
            "tree_depth": max(leaf["depth"] for leaf in leaves),
            "indexing_seconds": round(elapsed, 1),
        },
    }
    with open(out_dir / "index_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    # ── Optionale KPI-Datei: eine Zeile pro Bisection + Summary ──────────────
    if save_stats:
        da_seconds = sum(b["seconds"] for b in bisections)
        by_depth: dict[int, list[dict]] = {}
        for b in bisections:
            by_depth.setdefault(b["depth"], []).append(b)
        stats = {
            "total_indexing_seconds": round(elapsed, 1),
            "da_seconds": round(da_seconds, 1),
            "overhead_seconds": round(elapsed - da_seconds, 1),
            "num_bisections": len(bisections),
            "transfers_total": sum(b.get("transfers", 0) for b in bisections),
            "per_depth": {
                str(dpt): {
                    "bisections": len(bs),
                    "seconds": round(sum(b["seconds"] for b in bs), 1),
                    "steps_used": bs[0].get("steps_used"),
                    "mean_energy": round(float(np.mean(
                        [b["energy"] for b in bs])), 2),
                    "min_energy": round(min(b["energy"] for b in bs), 2),
                } for dpt, bs in sorted(by_depth.items())
            },
            "bisections": bisections,
        }
        with open(out_dir / "indexing_stats.json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  KPIs → indexing_stats.json ({len(bisections)} Bisections, "
                  f"DA {da_seconds:.1f}s von {elapsed:.1f}s)")

    shutil.copy(QUERY_TEMPLATE, out_dir / "query.py")

    if verbose:
        print(f"\n✓ RAG index written → {out_dir}")
    return out_dir, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def _offset_arg(v: str) -> float | str:
    """--offset-increase: Zahl oder das Literal 'auto_gp' (dynamisch)."""
    return v if v == "auto_gp" else float(v)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a RAG query folder from a text-embedder run folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[1],
    )
    ap.add_argument("run_folder", help="text-embedder output run folder")
    ap.add_argument("-o", "--output", default=None,
                    help="index output folder (default: next free "
                         "<run_folder>/rag_index_NNN — never overwrites)")
    ap.add_argument("-g", "--group-size", type=int, default=20,
                    help="max leaf group size (default 20)")
    ap.add_argument("--steps", type=int, default=None,
                    help="DA-Steps auf Ebene 0; halbiert sich je Tiefe, Boden "
                         "%d. Ohne Angabe AUTO = n/3 mit n = Gesamtzahl der "
                         "Chunks" % gp.STEPS_FLOOR)
    ap.add_argument("--num-mc", type=int, default=20,
                    help="Monte Carlo runs per bisection (calibrated: 20)")
    ap.add_argument("--offset-increase", type=_offset_arg, default="auto_gp",
                    help="offset_increase_rate for the DA: Zahl oder "
                         "'auto_gp' = dynamisch im Solver aus der ΔE-Skala "
                         "(q50 der Bergauf-Flips / k_escape; default auto_gp)")
    ap.add_argument("--cooling-a", type=float, default=None,
                    help="a in cooling ['logarithmic', a, b] "
                         "(default: auto = d*N/(depth+1))")
    ap.add_argument("--cooling-b", type=float, default=0.0,
                    help="b in cooling ['logarithmic', a, b] (calibrated: 0)")
    ap.add_argument("--cooling-d", type=float, default=2.0,
                    help="d im Auto-a: a = d*N/(depth+1) (calibrated: 2)")
    ap.add_argument("--p-hot", type=float, default=0.4,
                    help="da_gp-Cooling: Akzeptanz eines medianen "
                         "Bergauf-Flips bei t=1 (default 0.4)")
    ap.add_argument("--p-cold", type=float, default=0.02,
                    help="da_gp-Cooling: Rest-Akzeptanz eines kleinen (q25) "
                         "Flips am Ende — die Einfriergrenze (default 0.02)")
    ap.add_argument("--gamma", type=float, default=0.05,
                    help="da_gp-Cooling: Deckel T(1) <= gamma*(E_start-E_lloyd) "
                         "— Warmstart nie einschmelzen (default 0.05)")
    ap.add_argument("--iterations-lloyd", type=int, default=20,
                    help="k=2-Lloyd-Vorschritt vor jeder Bisection: max. "
                         "Iterationen (zufälliger Start, stoppt bei "
                         "Konvergenz); Cooling wechselt auf 'da_gp' "
                         "(default 10; 0 = aus → logarithmic)")
    ap.add_argument("--lloyd-mc", type=int, default=1,
                    help="PCA-orientierte Lloyd-Multi-Starts je Bisection: "
                         "Sign-Splits entlang der Hauptkomponenten + "
                         "λ-gewichtete Mischungen. Mit --solver fast laufen "
                         "num_mc DA-Trials PRO uniquer Lloyd-Lösung "
                         "(default 1 = ein Start wie bisher)")
    ap.add_argument("--visualize-first", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="DA-Trajektorien-Plot für die ERSTE Bisection "
                         "jeder Tiefenebene (visual_inst=True; default an, "
                         "--no-visualize-first schaltet ab)")
    ap.add_argument("--gram-threshold", type=float, default=0.0,
                    help="Sparsity-Schwelle: |Q[i,j]| < thr wird schon bei "
                         "der Gram-Berechnung verworfen — sparse PBF, keine "
                         "dichte n×n-Matrix (default 0 = aus)")
    ap.add_argument("--cooling-schedule",
                    choices=["da_gp", "da_gp_floor"], default="da_gp",
                    help="Form der da_gp-Kurve: 'da_gp' faellt unter die "
                         "Einfriergrenze (kalibrierter Stand), 'da_gp_floor' "
                         "laeuft auf sie zu und friert nicht haerter ein "
                         "als p_cold verlangt (default da_gp)")
    ap.add_argument("--cooling-per-group",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Jede Lloyd-Startgruppe an IHREM Startvektor "
                         "kalibrieren (default an; --no-cooling-per-group "
                         "= ein gemeinsames Schedule aus der besten Loesung)")
    ap.add_argument("--hold-steps", type=int, default=None,
                    help="Plateau-Phase: Starttemperatur so viele Steps "
                         "konstant halten, bevor die Kurve faellt (Angabe "
                         "gilt fuer Ebene 0 und halbiert sich je Tiefe wie "
                         "die Steps, gedeckelt auf den Anteil "
                         f"{gp.HOLD_MAX_FRAC:g} der Steps). Gibt dem "
                         "Annealer eine echte Explorationsphase, damit er das "
                         "Lloyd-Becken verlassen kann. Ohne Angabe AUTO = "
                         "n/10 mit n = Gesamtzahl der Chunks; 0 = aus. Nur "
                         "fast-Optimizer")
    ap.add_argument("--offset-k-escape", type=float, default=5.0,
                    help="Skaliert den E_Offset bei offset_increase=auto_gp: "
                         "rate = q50(dE+)/k_escape. KLEINER = staerkerer "
                         "Offset = aggressivere Flucht aus lokalen Minima "
                         "(default 5)")
    ap.add_argument("--split-mode", choices=["none", "centroid", "plane"],
                    default="plane",
                    help="Nachschritt auf dem DA-Split — genau EINER. "
                         "'none': roher DA-Split. "
                         "'centroid': Punkte dem naeheren Zentroid zuordnen "
                         "(Hyperebene durch den Ursprung). 'plane': "
                         "Max-Margin-Ebene — gleiche Richtung, aber in die "
                         "groesste Luecke der Projektionen geschoben; der "
                         "Baum traegt dann (w, tau, sigma) und der Descent "
                         "entscheidet ueber q·w − tau (default plane)")
    ap.add_argument("--spill", type=float, default=0.1,
                    help="Spill-Zone: die spill·n Punkte, die der Ebene am "
                         "naechsten liegen, landen in BEIDEN Kindern "
                         "(Punktanteil, z.B. 0.05 = 5 Prozent). Faengt Queries nahe der "
                         "Trennebene dort, wo es keine Luecke gibt (oben im "
                         "Baum). Speicher waechst mit (1+spill)^Tiefe. "
                         "Mit --eps > 0 nur noch die OBERGRENZE. "
                         "Nur mit --split-mode centroid/plane "
                         "(default 0.1; 0 = aus)")
    ap.add_argument("--eps", default="0",
                    help="'auto' misst den Radius am ECHTEN Encoder (256 "
                         "Chunks, zufaelliges 60-80-Prozent-Wortfenster, neu "
                         "eingebettet, q90 der Projektionsstreuung — braucht "
                         "den ollama-Provider); 'cos:0.9' leitet ihn aus einer "
                         "bekannten Paraphrasen-Cosine ab; eine Zahl setzt ihn "
                         "direkt. "
                         "Unsicherheitsradius der Embeddings, gemessen in der "
                         "Einheit der Projektion t = p·w (NICHT als Abstand im "
                         "Einbettungsraum — siehe estimate_eps). > 0 macht den "
                         "Nachschritt fehlerbewusst: die Ebene sucht die "
                         "Luecke mit den WENIGSTEN Punkten in |t − tau| <= eps, "
                         "und gespillt wird genau diese Zone statt eines festen "
                         "Anteils (--spill deckelt sie nur noch). Wo die Ebene "
                         "sauber trennt, spillt dann gar nichts "
                         "(default 0 = aus, fester Anteil wie bisher)")
    ap.add_argument("--anneal", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Digital Annealing als Bisektor. --no-anneal baut den "
                         "Baum allein aus den Lloyd-Bisektionen: kein QUBO, "
                         "keine Gram-Matrix, O(n·d) statt O(n²) je Knoten — "
                         "der billige Referenzpfad, gegen den sich der DA "
                         "messen lassen muss. Braucht --iterations-lloyd > 0 "
                         "(default an)")
    ap.add_argument("--objective", choices=["gram", "knn"], default=None,
                    help="Was die Bisektion minimiert. `gram` = die "
                         "Ähnlichkeit ALLER Paare (bei balancierten Splits "
                         "2-Means) — misst Partitionsgüte. `knn` = "
                         "Modularity des k-NN-Graphen, also die Zahl der "
                         "zerschnittenen ECHTEN Nachbarschaften. Gemessen "
                         "(experiments/objective_vs_recall.py, 19 Baeume auf "
                         "demselben Punktsatz): die Gram-Energie erklaert den "
                         "recall@5 nicht mehr (+0.10), die zerschnittenen "
                         "10-NN-Paare zu -0.91. Ende zu Ende bringt der "
                         "Wechsel bisher nur ~+0.3 Punkte, also innerhalb des "
                         "Rauschens. "
                         "Nativ quadratisch (kein Grad-4-Problem wie exaktes "
                         "2-Means) und dünn besetzt. Default: knn mit "
                         "Annealer, gram ohne (dort ist die Wahl ohne "
                         "Wirkung).")
    ap.add_argument("--anchors", type=int, default=1, metavar="K",
                    help="Beam-Repräsentation: K Anker je Knoten statt EINEM "
                         "Mittelwert (k-Means-Zentren der Knotenpunkte). "
                         "Der Beam bewertet einen Teilbaum dann mit "
                         "max_j cos(q, a_j) statt cos(q, mean) — er erkennt "
                         "damit, dass ein Teilbaum passt, auch wenn dessen "
                         "Schwerpunkt woanders liegt. Ändert NICHTS an der "
                         "Trennung (die macht weiter Ebene bzw. Zentroid) und "
                         "nichts am Baum, nur an der Query-Bewertung. "
                         "Kostet K×d floats je Knoten. 1 = aus (default)")
    ap.add_argument("--solver", choices=["fast", "reference"], default="fast",
                    help="DA-Backend (Heuristik, keine Optimalitätsgarantie): "
                         "'fast' = annealing-qubo-optimizer "
                         "(batched numpy, QUBO als Matrix); 'reference' = "
                         "annealing-cop-approximator (dict-basierter Original"
                         "pfad, benötigt deren int-Key-Support). Default fast.")
    ap.add_argument("--stats", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="indexing_stats.json mit KPIs je Bisection schreiben "
                         "(default an, --no-stats schaltet ab)")
    ap.add_argument("--seed", type=float, default=None,
                    help="fixed RNG seed for reproducible partitions")
    ap.add_argument("--quiet", action="store_true", help="less progress output")
    ap.add_argument("--progress-json", action="store_true",
                    help="Progress-Events als '@@'-präfixierte JSON-Zeilen auf "
                         "stdout (Maschinen-Schnittstelle, nutzt die GUI für "
                         "das Subprozess-Indexing)")
    args = ap.parse_args()

    # Unvereinbare Flags sofort abfangen — der Waechter in build_index greift
    # sonst erst, nachdem die Wissensbasis geladen ist.
    if args.objective == "knn" and args.anneal is False:
        ap.error("--objective knn braucht den Annealer — Lloyd bisektiert im "
                 "Koordinatenraum und sieht die Graphmatrix nie. Entweder "
                 "--anneal (default) oder --objective gram.")

    progress_cb = None
    if args.progress_json:
        def progress_cb(event: str, info: dict) -> None:
            print("@@" + json.dumps({"event": event, **info}), flush=True)

    out_dir, _meta = build_index(
        Path(args.run_folder),
        Path(args.output) if args.output else None,
        g=args.group_size,
        steps=args.steps, num_mc=args.num_mc,
        offset_increase=args.offset_increase,
        cooling_a=args.cooling_a, cooling_b=args.cooling_b,
        cooling_d=args.cooling_d,
        iterations_lloyd=args.iterations_lloyd, lloyd_mc=args.lloyd_mc,
        cooling_schedule=args.cooling_schedule,
        cooling_per_group=args.cooling_per_group,
        hold_steps=args.hold_steps, offset_k_escape=args.offset_k_escape,
        split_mode=args.split_mode, spill=args.spill, eps=args.eps,
        anneal=args.anneal,
        objective=args.objective,
        anchors=args.anchors,
        p_hot=args.p_hot, p_cold=args.p_cold, gamma=args.gamma,
        seed=args.seed, visualize_first=args.visualize_first,
        gram_threshold=args.gram_threshold, solver=args.solver,
        save_stats=args.stats,
        verbose=not args.quiet,
        progress_cb=progress_cb,
    )
    if args.progress_json:
        print("@@" + json.dumps({"event": "done", "out_dir": str(out_dir)}),
              flush=True)
    print(f"  Query it with:")
    print(f"    python {out_dir / 'query.py'} --embedding <emb.json> -k 5")


if __name__ == "__main__":
    main()
