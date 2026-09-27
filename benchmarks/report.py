#!/usr/bin/env python3
"""
results.json → die Tabellen für die README.

    python3 benchmarks/report.py > benchmarks/REPORT.md

Der wichtigste Schnitt ist der erste: WER bisektiert (k-Means oder der
Digital Annealer) und WIE die Grenze danach gezogen wird, sind zwei
unabhängige Achsen.  Jede Variante wird deshalb als Paar (Optimierer,
Grenzziehung) ausgewiesen — „lloyd" ist nicht nacktes k-Means, sondern
k-Means MIT Ebene und Overlap; nackt ist `lloyd-raw`.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = [r for r in json.loads((HERE / "results.json").read_text())]
# Query-Zeiten kommen aus einem EIGENEN Lauf (time_query.py, ein BLAS-
# Thread, Median aus 5 Runden).  Die "ms" in results.json entstehen
# nebenbei, während numpy über alle Kerne rechnet, und schwanken um
# Faktoren mit der Systemlast — als veröffentlichte Zahl taugen sie nicht.
TIMES = {(t["encoder"], t["variant"], t["beam"]): t["ms"]
         for t in json.loads((HERE / "query_times.json").read_text())} \
    if (HERE / "query_times.json").exists() else {}

# (Optimierer, Grenzziehung, Flags) — die beiden Achsen getrennt benannt
LABEL = {
    # nackt: die Punkte gehen an das nächste Zentrum, keine Ebene, kein Overlap
    "lloyd-raw":      ("Lloyd", "nearest centroid",
                       "`--no-anneal --split-mode none`"),
    "raw":            ("DA", "nearest centroid", "`--split-mode none`"),
    "da-blind":       ("DA, no warm start", "plane + 10 %",
                       "`--iterations-lloyd 0`"),
    # Ebene, aber ohne Overlap
    "centroid":       ("DA", "centroid plane", "`centroid --spill 0`"),
    "plane":          ("DA", "max-margin plane", "`plane --spill 0`"),
    # Ebene + Overlap
    "lloyd":          ("Lloyd", "max-margin plane + 10 %", "`--no-anneal`"),
    "centroid+spill": ("DA", "centroid plane + 10 %",
                       "`centroid --spill 0.1`"),
    "default":        ("**DA**", "**max-margin plane + 10 %**",
                       "`plane --spill 0.1`"),
    "spill0.2":       ("DA", "max-margin plane + 20 %", "`plane --spill 0.2`"),
    "eps":            ("DA", "ε-aware plane + 10 %",
                       "`plane --spill 0.1 --eps 0.0745`"),
    # Multistart-Kontrollen
    "lloyd-mc4":      ("Lloyd ×4", "max-margin plane + 10 %",
                       "`--no-anneal --lloyd-mc 4`"),
    "mc4":            ("DA, 4 starts", "max-margin plane + 10 %",
                       "`--lloyd-mc 4`"),
}
# ── die neue Hauptmatrix (Spill 5 %, k Anker) ────────────────────────────
MATRIX = {
    "m-lloyd-bare":     ("Lloyd", "nearest centroid", 1),
    "m-da-bare":        ("Lloyd → DA", "nearest centroid", 1),
    "m-lloyd":          ("Lloyd", "plane + 5 %", 1),
    "m-lloyd-k4":       ("Lloyd", "plane + 5 %", 4),
    "m-lloyd-k16":      ("Lloyd", "plane + 5 %", 16),
    "m-da":             ("Lloyd → DA", "plane + 5 %", 1),
    "m-da-k4":          ("Lloyd → DA", "plane + 5 %", 4),
    "m-da-k16":         ("Lloyd → DA", "plane + 5 %", 16),
    "m-da-k16-nospill": ("Lloyd → DA", "plane, kein Overlap", 16),
    "m-dablind":        ("DA ohne Warmstart", "plane + 5 %", 1),
    "m-dablind-k16":    ("DA ohne Warmstart", "plane + 5 %", 16),
}

ORDER = ["lloyd-raw", "raw",                                   # keine Ebene
         "centroid", "plane",                                  # Ebene, kein Overlap
         "lloyd", "centroid+spill", "default", "spill0.2", "eps",
         "lloyd-mc4", "mc4", "da-blind"]                       # Kontrollen

SHORT = {"bge-m3": "bge-m3", "qwen3-embedding:4b-fp16": "qwen3-4b"}


def enc(r: dict) -> str:
    return SHORT.get(r.get("encoder", "?"), r.get("encoder", "?"))


def encoders() -> list[str]:
    seen = []
    for r in ROWS:
        if enc(r) not in seen:
            seen.append(enc(r))
    return seen


def sizes() -> list[int]:
    return sorted({r["n"] for r in ROWS})


def get(variant: str, n: int, e: str, beam: int = 2) -> dict | None:
    return next((r for r in ROWS if r.get("variant") == variant
                 and r["n"] == n and enc(r) == e and r["beam"] == beam), None)


def num(x, fmt="{:.1f}%", scale=100):
    return fmt.format(x * scale) if x is not None else "—"


def pair(r: dict | None) -> str:
    return (f"{num(r['found@5'])} / {num(r['recall@5'])}") if r else "— / —"


def delta(a: dict | None, b: dict | None) -> str:
    """b − a in Recall-Punkten."""
    if not a or not b:
        return "—"
    d = (b["recall@5"] - a["recall@5"]) * 100
    return f"{d:+.1f}"


# ── 0 — die Kreuztabelle: Optimierer vs. Grenzziehung ────────────────────
#  Jede Zeile hält die Grenzziehung fest und tauscht nur den Optimierer.
OPT_PAIRS = [("nearest centroid, no plane", "lloyd-raw", "raw"),
             ("max-margin plane + 10 % overlap", "lloyd", "default"),
             ("max-margin plane + 10 %, 4 restarts", "lloyd-mc4", "mc4")]
#  … und umgekehrt: Optimierer festhalten, Grenzziehung tauschen.
BND_PAIRS = [("Lloyd", "lloyd-raw", "lloyd"), ("DA", "raw", "default")]


def t_crosstab(beam: int = 2) -> str:
    n = max(sizes())
    out = ["**Which half of the pipeline does the work?** "
           f"found@5 / recall@5, {n} chunks, beam {beam}\n",
           "*Same boundary, different optimiser — the annealer against "
           "Lloyd alone:*\n",
           "| Boundary | Encoder | Lloyd | Digital Annealer | Δ recall |",
           "|---|---|---|---|---|"]
    for name, a, b in OPT_PAIRS:
        for e in encoders():
            ra, rb = get(a, n, e, beam), get(b, n, e, beam)
            if not (ra or rb):
                continue
            out.append(f"| {name} | {e} | {pair(ra)} | {pair(rb)} | "
                       f"**{delta(ra, rb)}** |")
    out += ["", "*Same optimiser, different boundary — the plane and the "
                "overlap against nearest-centroid assignment:*\n",
            "| Optimiser | Encoder | nearest centroid | "
            "max-margin plane + 10 % | Δ recall |",
            "|---|---|---|---|---|"]
    for name, a, b in BND_PAIRS:
        for e in encoders():
            ra, rb = get(a, n, e, beam), get(b, n, e, beam)
            if not (ra or rb):
                continue
            out.append(f"| {name} | {e} | {pair(ra)} | {pair(rb)} | "
                       f"**{delta(ra, rb)}** |")
    return "\n".join(out)


# ── 1 — alle Varianten × beide Encoder ───────────────────────────────────
def t_quality_full(beam: int = 2) -> str:
    es, n = encoders(), max(sizes())
    head = ["| Optimiser | Boundary | Flags |"]
    sub = ["|---|---|---|"]
    for e in es:
        head.append(f" {e} found | recall@5 | leaf entries | build |")
        sub.append("---|---|---|---|")
    out = ["".join(head).rstrip(), "".join(sub).rstrip()]
    for v in ORDER:
        opt, bnd, flags = LABEL[v]
        cells = [f"| {opt} | {bnd} | {flags} |"]
        for e in es:
            r = get(v, n, e, beam)
            if not r:
                cells.append(" — | — | — | — |")
                continue
            cells.append(f" {num(r['found@5'])} | {num(r['recall@5'])} | "
                         f"{r['leaf_entries']:,} | {r['build_s']:.0f} s |"
                         .replace(",", " "))
        out.append("".join(cells).rstrip())
    return (f"**All variants × both encoders — {n} chunks, beam {beam}, "
            f"300 queries each**\n\n" + "\n".join(out))


# ── 2 — die große Zeitmatrix ─────────────────────────────────────────────
def t_times() -> str:
    ns = sizes()
    out = ["| Optimiser | Boundary | Encoder | "
           + " | ".join(f"{n:,}".replace(",", " ") for n in ns)
           + " | total |",
           "|---|---|---|" + "---|" * (len(ns) + 1)]
    for v in ORDER:
        opt, bnd, _ = LABEL[v]
        for e in encoders():
            rs = [get(v, n, e) for n in ns]
            if not any(rs):
                continue
            tot = sum(r["build_s"] for r in rs if r)
            out.append(f"| {opt} | {bnd} | {e} | "
                       + " | ".join(f"{r['build_s']:.0f} s" if r else "—"
                                    for r in rs)
                       + f" | **{tot:.0f} s** |")
    return ("**Build time, every variant × size × encoder**\n\n"
            + "\n".join(out))


# ── 3 — Qualität über die Größen ─────────────────────────────────────────
def t_sizes(v: str = "default", beam: int = 2) -> str:
    es = encoders()
    out = ["| Chunks | " + " | ".join(f"{e} found | recall@5" for e in es)
           + " | Δ recall | leaf entries | overhang |",
           "|---|" + "---|" * (2 * len(es) + 3)]
    for n in sizes():
        cells = [f"| {n:,}".replace(",", " ")]
        ent = ovh = None
        rec = []
        for e in es:
            r = get(v, n, e, beam)
            cells.append(f" | {num(r['found@5']) if r else '—'} | "
                         f"{num(r['recall@5']) if r else '—'}")
            if r:
                ent, ovh = r["leaf_entries"], r["leaf_entries"] / r["n"]
                rec.append(r["recall@5"])
        cells.append(f" | {(rec[1] - rec[0]) * 100:+.1f}"
                     if len(rec) == 2 else " | —")
        cells.append(f" | {ent:,}".replace(",", " ") if ent else " | —")
        cells.append(f" | {ovh:.2f}× |" if ovh else " | — |")
        out.append("".join(cells))
    return (f"**Scaling — `{v}`, beam {beam}.**  Δ recall is the gain from "
            "the stronger encoder; it grows with the corpus.\n\n"
            + "\n".join(out))


# ── 4 — Beam-Breite ──────────────────────────────────────────────────────
def t_beam(v: str = "default") -> str:
    n = max(sizes())
    out = ["| Beam | Encoder | found | recall@5 | chunks visited | ms/query |",
           "|---|---|---|---|---|---|"]
    for beam in sorted({r["beam"] for r in ROWS}):
        for e in encoders():
            r = get(v, n, e, beam)
            if not r:
                continue
            ms = TIMES.get((e, v, beam), r["ms"])
            out.append(f"| {beam} | {e} | {num(r['found@5'])} | "
                       f"{num(r['recall@5'])} | {r['visited']:.0f} | "
                       f"{ms:.3f} |")
    return (f"**Beam width — `{v}` on {n} chunks.**  Costs nothing at build "
            "time and beats every indexing option above.  Times from "
            "`time_query.py`: one BLAS thread, median of five rounds.\n\n"
            + "\n".join(out))


def t_query_times(beam: int = 2) -> str:
    """5 — Query-Zeit je Variante, sauber gemessen."""
    if not TIMES:
        return ""
    es = encoders()
    vs = [v for v in ORDER if any((e, v, beam) in TIMES for e in es)]
    out = ["| Optimiser | Boundary | leaf entries | depth | chunks visited | "
           + " | ".join(f"{e} ms/query" for e in es) + " |",
           "|---|---|---|---|---|" + "---|" * len(es)]
    n = max(sizes())
    for v in vs:
        opt, bnd, _ = LABEL[v]
        r0 = get(v, n, es[0], beam)
        out.append(f"| {opt} | {bnd} | "
                   + (f"{r0['leaf_entries']:,}".replace(",", " ")
                      + f" | {r0['depth_max']} | {r0['visited']:.0f}"
                      if r0 else "— | — | —")
                   + " | "
                   + " | ".join(f"{TIMES[(e, v, beam)]:.3f}"
                                if (e, v, beam) in TIMES else "—"
                                for e in es) + " |")
    return (f"**Query time — {n} chunks, beam {beam}, one BLAS thread, "
            f"median of five rounds.**  Leaf entries, tree depth and visited "
            f"chunks are from the bge-m3 index.\n\n" + "\n".join(out))


def t_candidates() -> str:
    """Der faire Schnitt: recall gegen die Zahl der nachgerankten Chunks.

    Ein Index ist nur vergleichbar, wenn man ihn bei GLEICHER Arbeit misst.
    `visited` ist diese Arbeit — wie viele Chunks am Ende exakt gerankt
    werden.  Der Beam ist unser Regler dafür, `efSearch` bzw. `nprobe` der
    der Referenzindexe.
    """
    n = max(sizes())
    rows = []
    for v, (opt, bnd, k) in MATRIX.items():
        for b in sorted({r["beam"] for r in ROWS}):
            r = get(v, n, "bge-m3", b)
            if r:
                rows.append((r["visited"], f"{opt} · {bnd} · k={k}",
                             f"beam {b}", r["recall@5"], r["found@5"]))
    bl = HERE / "baselines.json"
    if bl.exists():
        for e in json.loads(bl.read_text()):
            if e.get("cands") and not e["index"].startswith("gp-qubo"):
                rows.append((e["cands"], e["index"], e["param"],
                             e["recall@5"], e["found@5"]))
    if not rows:
        return ""
    rows.sort()
    out = ["**Recall gegen Arbeit — alles auf einer Achse.**  "
           f"{n} Chunks, bge-m3, 300 Queries.  `Kandidaten` = exakt "
           "nachgerankte Chunks; das ist die Größe, die beide Seiten "
           "vergleichbar macht.\n",
           "| Kandidaten | Verfahren | Regler | recall@5 | found@5 |",
           "|---|---|---|---|---|"]
    for c, name, param, rec, f in rows:
        out.append(f"| {c:.0f} | {name} | {param} | {num(rec)} | {num(f)} |")
    return "\n".join(out)


if __name__ == "__main__":
    print("\n\n".join([t_crosstab(), t_quality_full(), t_times(),
                       t_sizes(), t_beam(), t_query_times(),
                       t_candidates()]))
