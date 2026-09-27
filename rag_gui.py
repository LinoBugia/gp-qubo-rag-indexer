#!/usr/bin/env python3
"""
rag_gui.py — CustomTkinter GUI zum Bauen & Testen einer RAG-Knowledgebase
==========================================================================

Workflow
--------
  1. Ordner wählen:
       · rag_index/ (oder Run-Ordner mit einem) → Index wird geladen
       · un-geindexter text-embedder Run-Ordner → Indexing startet
         AUTOMATISCH; die DA-Bisektionen sind live in der Punktwolke
         zu verfolgen — jede Gruppe eine Farbe, die sich mit jedem
         Split verfeinert
  2. Query eintippen → wird mit dem Encoder des Index embeddet (Provider,
     Modell + Dimension aus index_meta.json; sichtbar in der Karten-Leiste)
  3. Die Suche läuft über das echte query.py des Index-Ordners (Subprozess,
     Embedding via stdin) — die GUI testet also exakt das Terminal-Artefakt
  4. Ergebnis-Chunks lesen; Treffer + Query-Punkt werden in der Karte markiert

Karten (Dropdown, alle numpy-only, pro Index gecacht):
  PCA             linear, instant — Ladeansicht
  MDS (Cosine)    globale Cosine-Distanzen (Torgerson)
  t-SNE           beste semantische Cluster-Trennung (exakt, O(n²)/Iteration)
  Spectral (kNN)  Laplacian Eigenmaps auf dem Nachbarschaftsgraphen
  DA-Baum (Grid)  eine Rasterzelle pro Partitionsbaum-Leaf — zeigt den Index

Start:  python rag_gui.py
"""

from __future__ import annotations

import colorsys
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import numpy as np

import graph_partitioning as gp

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

DEFAULT_OLLAMA = "http://localhost:11434"
CANVAS_BG = "#181a1c"


# ── Backend helpers (kein UI-Code) ────────────────────────────────────────────

def list_index_dirs(run_dir: Path) -> list[Path]:
    """Alle fertigen Indexierungen eines Run-Ordners (rag_index,
    rag_index_001, …), sortiert nach Name — die letzte ist die neueste."""
    return sorted(p for p in run_dir.glob("rag_index*")
                  if (p / "index_meta.json").exists())


def resolve_folder(folder: Path) -> tuple[str, Path]:
    """
    Klassifiziert einen gewählten Ordner:
      ("index", <index_dir>)  — fertiger RAG-Index (bei mehreren
                                Indexierungen im Run-Ordner: die neueste)
      ("run",   <run_dir>)    — text-embedder Run-Ordner ohne Index
      ("none",  folder)       — nichts von beidem
    """
    if (folder / "index_meta.json").exists():
        return "index", folder
    idx = list_index_dirs(folder)
    if idx:
        return "index", idx[-1]
    if (folder / "specs.json").exists():
        return "run", folder
    return "none", folder


def ollama_models(base_url: str) -> list[str]:
    """Installierte Ollama-Modelle (wirft bei nicht erreichbarem Server)."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=5) as resp:
        data = json.loads(resp.read())
    return [m["name"] for m in data.get("models", [])]


def ollama_embed(base_url: str, model: str, text: str) -> list[float]:
    """Ein einzelnes Query-Embedding via POST /api/embed (wie text-embedder)."""
    payload = json.dumps({"model": model, "input": [text]}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["embeddings"][0]


# ── Google Gemini API (AI Studio) ────────────────────────────────────────────
# gemini-embedding-001, text-embedding-004, … — nur API-Key nötig (Free Tier).
# Key kommt aus dem GUI-Feld oder $GEMINI_API_KEY / $GOOGLE_API_KEY.

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_PROVIDERS = {"gemini", "google", "google-ai-studio", "googleai"}


def env_gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def gemini_models(api_key: str) -> list[str]:
    """Verfügbare Gemini-Modelle (Verbindungs- und Key-Check)."""
    req = urllib.request.Request(
        f"{GEMINI_BASE}/models", headers={"x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return [m["name"].split("/")[-1] for m in data.get("models", [])]


def gemini_embed(api_key: str, model: str, text: str,
                 output_dim: int | None = None) -> list[float]:
    """
    Query-Embedding via POST models/<model>:embedContent.
    output_dim (outputDimensionality) sorgt dafür, dass z. B.
    gemini-embedding-001 (nativ 3072) auf die Index-Dimension gekürzt wird —
    das Cosine-Ranking normalisiert ohnehin, Truncation ist damit safe.
    """
    model_path = model if model.startswith("models/") else f"models/{model}"
    body: dict = {"content": {"parts": [{"text": text}]}}
    if output_dim:
        body["outputDimensionality"] = int(output_dim)
    req = urllib.request.Request(
        f"{GEMINI_BASE}/{model_path}:embedContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["embedding"]["values"]


def embed_query(provider: str, conn: str, model: str, text: str,
                output_dim: int | None = None) -> list[float]:
    """
    Provider-Dispatch fürs Query-Embedding.
      ollama  → conn = Base-URL   (nomic, bge-m3, embeddinggemma, mxbai, …)
      gemini  → conn = API-Key    (gemini-embedding-001, text-embedding-004, …)
    """
    if provider.lower() in GEMINI_PROVIDERS:
        key = conn or env_gemini_key()
        if not key:
            raise RuntimeError(
                "Gemini-API-Key fehlt — ins Feld oben eintragen oder "
                "$GEMINI_API_KEY setzen (kostenloser Key: Google AI Studio).")
        return gemini_embed(key, model, text, output_dim=output_dim)
    return ollama_embed(conn or DEFAULT_OLLAMA, model, text)


def run_query_script(index_dir: Path, embedding: list[float],
                     k: int, beam: int, min_cand: int | None) -> dict:
    """query.py des Index-Ordners aufrufen; Embedding via stdin, JSON zurück."""
    cmd = [sys.executable, str(index_dir / "query.py"),
           "-k", str(k), "--beam", str(beam), "--json"]
    if min_cand is not None:
        cmd += ["--min-candidates", str(min_cand)]
    proc = subprocess.run(
        cmd, input=json.dumps(embedding),
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"query.py exit {proc.returncode}")
    return json.loads(proc.stdout)


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    2-D-PCA-Projektion (SVD, nur numpy).  Liefert normierte Koordinaten in
    [0,1]² plus die Basis, um später Query-Vektoren gleich zu projizieren.
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (Y - lo) / span, {"mean": mean, "Vt2": Vt[:2], "lo": lo, "span": span}


def project_query(q: np.ndarray, basis: dict) -> np.ndarray:
    """Query-Embedding mit derselben PCA-Basis in die Karte projizieren."""
    y = (q - basis["mean"]) @ basis["Vt2"].T
    return np.clip((y - basis["lo"]) / basis["span"], -0.05, 1.05)


# ── Weitere Karten-Projektionen (alle nur numpy) ──────────────────────────────

def _norm01(Y: np.ndarray) -> np.ndarray:
    """Koordinaten auf [0,1]² normieren."""
    lo, hi = Y.min(axis=0), Y.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (Y - lo) / span


def _cosine_dist(E: np.ndarray) -> np.ndarray:
    """Paarweise Cosine-Distanzen 1 − cos(vi, vj)."""
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return np.clip(1.0 - En @ En.T, 0.0, 2.0)


def mds_2d(E: np.ndarray) -> np.ndarray:
    """
    Klassisches MDS (Torgerson) auf Cosine-Distanzen: erhält globale
    Distanzverhältnisse im Embedding-Raum — oft ehrlicher als PCA, die bei
    768-dim Embeddings nur die zwei varianzstärksten Achsen zeigt.
    """
    D2 = _cosine_dist(E) ** 2
    n = len(D2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    top = np.argsort(w)[::-1][:2]
    return _norm01(V[:, top] * np.sqrt(np.maximum(w[top], 0.0)))


def spectral_2d(E: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Laplacian Eigenmaps auf dem kNN-Graphen (Cosine): betont lokale
    Nachbarschaften — Cluster liegen kompakt beieinander.
    """
    n = len(E)
    k = min(k, n - 1)
    S = 1.0 - _cosine_dist(E)
    np.fill_diagonal(S, -np.inf)
    W = np.zeros((n, n))
    nn = np.argsort(-S, axis=1)[:, :k]
    for i in range(n):
        W[i, nn[i]] = np.maximum(S[i, nn[i]], 0.0)
    W = np.maximum(W, W.T)
    d = W.sum(axis=1)
    d[d == 0] = 1.0
    Dinv = np.diag(1.0 / np.sqrt(d))
    L = np.eye(n) - Dinv @ W @ Dinv
    _, V = np.linalg.eigh(L)
    return _norm01(V[:, 1:3])            # 2. + 3. kleinster Eigenvektor


def tsne_2d(E: np.ndarray, *, iters: int = 400, seed: int = 0) -> np.ndarray:
    """
    Kompaktes exaktes t-SNE (O(n²) pro Iteration) auf Cosine-Distanzen —
    beste Trennung semantischer Cluster, dafür die teuerste Karte.
    Für Korpora bis ein paar tausend Chunks völlig ausreichend.
    """
    n = len(E)
    D2 = _cosine_dist(E) ** 2
    target = np.log(max(5, min(30, (n - 1) // 3)))

    # Perplexitäts-Kalibrierung: binäre Suche über beta je Punkt
    P = np.zeros((n, n))
    for i in range(n):
        di = D2[i].copy()
        di[i] = np.inf
        beta, blo, bhi = 1.0, 0.0, np.inf
        for _ in range(50):
            p = np.exp(-di * beta)
            s = p.sum()
            p = p / s if s > 0 else p
            Hs = -(p[p > 0] * np.log(p[p > 0])).sum()
            if abs(Hs - target) < 1e-3:
                break
            if Hs > target:
                blo, beta = beta, beta * 2 if bhi == np.inf else (beta + bhi) / 2
            else:
                bhi, beta = beta, beta / 2 if blo == 0 else (beta + blo) / 2
        p = np.exp(-D2[i] * beta)
        p[i] = 0.0
        P[i] = p / max(p.sum(), 1e-12)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n, 2)) * 1e-2
    dY = np.zeros_like(Y)
    for it in range(iters):
        Pm = P * (4.0 if it < 100 else 1.0)          # early exaggeration
        sq = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        num = 1.0 / (1.0 + sq)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        Lm = (Pm - Q) * num
        grad = 4.0 * ((np.diag(Lm.sum(axis=1)) - Lm) @ Y)
        dY = (0.5 if it < 250 else 0.8) * dY - 50.0 * grad
        Y += dY
        Y -= Y.mean(axis=0)
    return _norm01(Y)


def tree_grid_2d(tree: dict, n: int) -> np.ndarray:
    """
    DA-Baum-Ansicht: jedes Leaf des Partitionsbaums bekommt eine Rasterzelle
    (links oben → rechts unten = Baumreihenfolge), die Chunks liegen als
    Mini-Grid darin — zeigt den Index selbst statt des Embedding-Raums.
    """
    import math

    def leaves(node):
        if node["children"] is None:
            return [node]
        return leaves(node["children"][0]) + leaves(node["children"][1])

    lv = leaves(tree)
    cols = math.ceil(math.sqrt(len(lv)))
    Y = np.zeros((n, 2))
    for li, leaf in enumerate(lv):
        cx, cy = li % cols, li // cols
        m = len(leaf["rows"])
        sub = math.ceil(math.sqrt(m))
        for k, r in enumerate(leaf["rows"]):
            sx, sy = k % sub, k // sub
            Y[r] = [cx + 0.14 + 0.72 * (sx + 0.5) / sub,
                    cy + 0.14 + 0.72 * (sy + 0.5) / sub]
    Y[:, 1] = Y[:, 1].max() - Y[:, 1]                # oben beginnen
    return _norm01(Y)


def interp_query_xy(q: np.ndarray, E: np.ndarray, xy: np.ndarray,
                    k: int = 10) -> np.ndarray:
    """
    Query-Position für Karten ohne lineare Basis (MDS/t-SNE/Spectral/Baum):
    gewichtetes Mittel der Positionen der k cosine-ähnlichsten Chunks.
    """
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    qn = q / max(np.linalg.norm(q), 1e-12)
    sims = En @ qn
    top = np.argsort(-sims)[:min(k, len(E))]
    w = np.maximum(sims[top], 0.0) + 1e-9
    w = w / w.sum()
    return (xy[top] * w[:, None]).sum(axis=0)


# Ab dieser Punktzahl rechnen die teuren Karten auf einer Landmark-Teilmenge
# exakt und platzieren die übrigen Punkte per Cosine-kNN dazwischen — sonst
# skaliert MDS/Spectral O(n³) bzw. t-SNE O(n²·Iterationen) nicht mehr.
LANDMARK_LIMITS = {"MDS (Cosine)": 1500, "Spectral (kNN)": 1500, "t-SNE": 1000}


def landmark_map(E: np.ndarray, exact_fn, m: int, *,
                 k: int = 10, seed: int = 0) -> np.ndarray:
    """
    Karte für große Korpora: exact_fn läuft exakt auf m zufälligen
    Landmark-Punkten, jeder übrige Punkt wird als gewichtetes Mittel der
    Positionen seiner k cosine-ähnlichsten Landmarks platziert.
    """
    n = len(E)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    Y_lm = exact_fn(E[idx])
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    S = En @ En[idx].T                     # (n, m) Cosine zu den Landmarks
    kk = min(k, m)
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    w = np.maximum(np.take_along_axis(S, top, axis=1), 0.0) + 1e-9
    w /= w.sum(axis=1, keepdims=True)
    Y = (Y_lm[top] * w[..., None]).sum(axis=1)
    Y[idx] = Y_lm
    return _norm01(Y)


def tree_phases(tree: dict) -> list[list[tuple[str, list[int]]]]:
    """
    Splitting-Phasen aus dem gespeicherten Partitionsbaum rekonstruieren.

    phases[p] = Schnitt durch den Baum auf Tiefe p: Liste von (pfad, rows).
      Phase 0 → 1 Gruppe (alles), Phase 1 → 2 Gruppen, Phase 2 → ≤4, …
      bis zur maximalen Tiefe (= fertige Leaves).  Leaves, die früher
    fertig wurden, bleiben in tieferen Phasen als eigene Gruppe stehen.
    """
    def rows_of(node):
        if node["children"] is None:
            return list(node["rows"])
        return rows_of(node["children"][0]) + rows_of(node["children"][1])

    maxd = 0

    def probe(node):
        nonlocal maxd
        maxd = max(maxd, node["depth"])
        if node["children"]:
            probe(node["children"][0])
            probe(node["children"][1])

    probe(tree)

    phases: list[list[tuple[str, list[int]]]] = []
    for p in range(maxd + 1):
        out: list[tuple[str, list[int]]] = []

        def cut(node, path):
            if node["children"] is None or node["depth"] == p:
                out.append((path, rows_of(node)))
            else:
                cut(node["children"][0], path + "0")
                cut(node["children"][1], path + "1")

        cut(tree, "")
        phases.append(out)
    return phases


MAP_CHOICES = ["PCA", "MDS (Cosine)", "t-SNE", "Spectral (kNN)", "DA-Baum (Grid)"]

MAP_CAPTIONS = {
    "PCA":            "PCA — lineare Projektion (maximale Varianz)",
    "MDS (Cosine)":   "MDS — globale Cosine-Distanzen erhalten",
    "t-SNE":          "t-SNE — lokale semantische Cluster getrennt",
    "Spectral (kNN)": "Spectral — kNN-Nachbarschaftsgraph (Laplacian Eigenmaps)",
    "DA-Baum (Grid)": "DA-Baum — jede Zelle = ein Leaf des Partitionsbaums",
}


def path_color(path: str) -> str:
    """
    Farbe für einen Baumpfad ("0"/"1"-String): der Hue-Kreis wird wie ein
    binäres Intervall geteilt — Kinder erben die Umgebung ihrer Eltern,
    benachbarte Subtrees bekommen verwandte Farbtöne.
    """
    lo, hi = 0.0, 1.0
    for bit in path:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if bit == "0" else (mid, hi)
    h = (lo + hi) / 2
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.7, 0.95)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def doc_color(i: int) -> str:
    """Gut unterscheidbare Farbe für Dokument Nr. i (Goldener-Schnitt-Hues)."""
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.92)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


# ── Punktwolken-Canvas ────────────────────────────────────────────────────────

class PointCloud(tk.Canvas):
    """
    2-D-Scatter der Chunk-Embeddings.

    · Basisfärbung: gleiche Farbe = gleiches Quelldokument (Legende oben
      links); während des Indexings zeigen die Farben die DA-Gruppen.
    · Query-Ansicht (set_query_view): die vom Beam durchsuchten Kandidaten
      behalten ihre Farbe, alles andere wird Richtung Hintergrund gedimmt;
      getroffene Chunks werden GRAU ausgefüllt (weißer Ring + Rangnummer),
      der gelesene Chunk bekommt einen blauen Auswahlring, die Query ein ×.
    · Klickbar: on_click(row) wird mit dem nächstgelegenen Punkt aufgerufen
      — auch für getroffene (graue) Punkte.  Klick ins Leere (kein Punkt in
      Reichweite) ruft on_click(None) auf → Auswahl/Query-Ansicht aufheben.
    """

    R = 4          # Punktradius
    PAD = 24       # Rand
    SELECT = "#4da3ff"
    HIT_FILL = "#8f8f8f"   # "ausgegraut" — Treffer-Füllung nach der Query
    CLICK_PX = 12          # max. Klick-Distanz zum nächsten Punkt

    def __init__(self, master, **kw):
        super().__init__(master, bg=CANVAS_BG, highlightthickness=0,
                         cursor="hand2", **kw)
        self.xy: np.ndarray | None = None      # (n,2) in [0,1]
        self.colors: list[str] = []            # Basisfarben (Dokumente)
        self.hits: list[tuple[int, int]] = []  # [(row, rank), ...]
        self.candidates: set[int] | None = None  # None = keine Query-Ansicht
        self.selected: int | None = None
        self.query_xy: np.ndarray | None = None
        self.caption = ""
        self.legend: list[tuple[str, str]] = []  # [(farbe, label), ...]
        self.on_click = None                     # callable(row) | None
        self._dim_cache: dict[str, str] = {}
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Button-1>", self._handle_click)

    def _handle_click(self, event) -> None:
        if self.xy is None or self.on_click is None:
            return
        to_px = self._scale()
        best, best_d2 = None, float(self.CLICK_PX) ** 2
        for i, p in enumerate(self.xy):
            x, y = to_px(p)
            d2 = (x - event.x) ** 2 + (y - event.y) ** 2
            if d2 < best_d2:
                best, best_d2 = i, d2
        self.on_click(best)          # None = Klick ins Leere (deselecten)

    # ── Zustand ───────────────────────────────────────────────────────────────
    def set_points(self, xy_norm: np.ndarray, color: str = "gray50") -> None:
        self.xy = xy_norm
        self.colors = [color] * len(xy_norm)
        self.hits, self.candidates, self.selected, self.query_xy = [], None, None, None
        self.redraw()

    def color_rows(self, rows: list[int], color: str) -> None:
        for r in rows:
            self.colors[r] = color
        self.redraw()

    def set_query_view(self, candidates: list[int] | None,
                       hits: list[tuple[int, int]],
                       query_xy: np.ndarray | None) -> None:
        self.candidates = set(candidates) if candidates is not None else None
        self.hits, self.query_xy, self.selected = hits, query_xy, None
        self.redraw()

    def set_selected(self, row: int | None) -> None:
        self.selected = row
        self.redraw()

    def clear(self) -> None:
        self.xy = None
        self.delete("all")

    # ── Zeichnen ──────────────────────────────────────────────────────────────
    def _dim(self, color: str) -> str:
        """Farbe Richtung Hintergrund abdunkeln (für nicht durchsuchte Punkte)."""
        if color not in self._dim_cache:
            try:
                cr, cg, cb = (int(color[i:i + 2], 16) for i in (1, 3, 5))
            except ValueError:            # benannte Farbe wie "gray50"
                cr = cg = cb = 127
            br, bg_, bb = (int(CANVAS_BG[i:i + 2], 16) for i in (1, 3, 5))
            f = 0.22                      # 22 % Farbe, 78 % Hintergrund
            self._dim_cache[color] = (
                f"#{int(f*cr + (1-f)*br):02x}"
                f"{int(f*cg + (1-f)*bg_):02x}"
                f"{int(f*cb + (1-f)*bb):02x}")
        return self._dim_cache[color]

    def _scale(self):
        w = max(self.winfo_width(), 50)
        h = max(self.winfo_height(), 50)
        return (lambda p: (self.PAD + p[0] * (w - 2 * self.PAD),
                           h - self.PAD - p[1] * (h - 2 * self.PAD)))

    def redraw(self) -> None:
        self.delete("all")
        if self.xy is None:
            self.create_text(
                self.winfo_width() // 2, self.winfo_height() // 2,
                text="Keine Knowledgebase geladen", fill="gray50",
                font=("", 13))
            return
        to_px = self._scale()
        r = self.R
        hit_rows = {row for row, _rank in self.hits}
        query_view = self.candidates is not None

        # Basispunkte (in der Query-Ansicht: Nicht-Kandidaten gedimmt)
        for i, p in enumerate(self.xy):
            if i in hit_rows:
                continue                  # Treffer werden oben drauf gezeichnet
            x, y = to_px(p)
            color = self.colors[i]
            if query_view and i not in self.candidates:
                color = self._dim(color)
            self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="")

        # Treffer: vergrößert, GRAU ausgefüllt ("verbraucht"), weißer Ring + Rang
        for row, rank in self.hits:
            x, y = to_px(self.xy[row])
            rr = r + 3
            self.create_oval(x - rr, y - rr, x + rr, y + rr,
                             fill=self.HIT_FILL, outline="white", width=2)
            self.create_text(x + rr + 5, y - rr - 2, text=str(rank),
                             fill="white", font=("", 11, "bold"), anchor="w")

        # Auswahlring für den gerade gelesenen Chunk
        if self.selected is not None:
            x, y = to_px(self.xy[self.selected])
            rs = r + 8
            self.create_oval(x - rs, y - rs, x + rs, y + rs,
                             outline=self.SELECT, width=2)

        # Query-Punkt als Kreuz
        if self.query_xy is not None:
            x, y = to_px(self.query_xy)
            s = 7
            self.create_line(x - s, y - s, x + s, y + s, fill="white", width=2)
            self.create_line(x - s, y + s, x + s, y - s, fill="white", width=2)
            self.create_text(x + s + 4, y, text="Query", fill="white",
                             font=("", 10), anchor="w")
        # Legende (Dokument → Farbe) oben links
        ly = 14
        for color, label in self.legend:
            self.create_rectangle(10, ly - 4, 18, ly + 4, fill=color, outline="")
            self.create_text(24, ly, text=label, fill="gray75",
                             font=("", 10), anchor="w")
            ly += 16

        if self.caption:
            self.create_text(10, self.winfo_height() - 10, text=self.caption,
                             fill="gray55", font=("", 10), anchor="sw")


# ── Indexing-Parameter: Modell + Dialog ───────────────────────────────────────
#  Die Werte leben in einem schlichten dict (dem "Modell"), nicht in Widgets.
#  Der Dialog liest es beim Öffnen und schreibt es beim OK zurück — so kann
#  das Indexing laufen, ohne dass irgendein Fenster offen sein muss, und die
#  Hauptleiste bleibt frei für das, was man wirklich ständig sieht.

PARAM_DEFAULTS: dict[str, str | bool] = {
    "g": "20",
    "eps": "0",
    "gram_threshold": "0",
    "solver": "fast",
    "steps": "", "num_mc": "20", "hold_steps": "",
    "iterations_lloyd": "20", "lloyd_mc": "1",
    "cooling_schedule": "da_gp", "cooling_per_group": True,
    "p_hot": "0.4", "p_cold": "0.02", "gamma": "0.05",
    "anneal": True, "offset_increase": "auto_gp", "offset_k_escape": "5",
    "split_mode": "plane", "spill": "0.1",
    "seed": "", "visualize_first": True, "save_stats": True,
}

# Farben der Pipeline-Grafik (docs/img/pipeline.svg), auf den dunklen GUI-Grund
# umgesetzt: Pflicht / abschaltbar / eine Wahl / gehört zum Annealer.
C_FIX, C_OPT, C_ALT, C_DA = "#94a3b8", "#22c55e", "#f59e0b", "#818cf8"
C_CARD, C_CARD_DA = "#23262e", "#212536"


class PipelineDialog(ctk.CTkToplevel):
    """
    Die Indexing-Pipeline als Formular — ein Schritt pro Karte, in derselben
    Reihenfolge und mit denselben Farben wie docs/img/pipeline.svg.
    """

    def __init__(self, master, params: dict, n_chunks, on_start) -> None:
        super().__init__(master)
        self.title("Indexing-Pipeline konfigurieren")
        self.geometry("1020x820")
        self.minsize(820, 560)
        self.on_start = on_start
        self.vals = dict(params)
        self.n_chunks = n_chunks
        self.w: dict[str, ctk.CTkBaseClass] = {}
        self._da_widgets: list[tuple] = []      # (frame, widgets) für Ausgrauen

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_head()
        self._build_body()
        self._build_foot()
        self._sync_enabled()

        self.transient(master)
        self.after(120, self._grab)

    def _grab(self) -> None:
        try:
            self.grab_set()
            self.focus_force()
        except Exception:                       # noqa: BLE001
            pass

    # ── Bausteine ────────────────────────────────────────────────────────────
    def _entry(self, parent, key, width=70, placeholder="") -> ctk.CTkEntry:
        e = ctk.CTkEntry(parent, width=width, placeholder_text=placeholder)
        val = self.vals.get(key, "")
        if val not in ("", None):
            e.insert(0, str(val))
        self.w[key] = e
        return e

    def _field(self, parent, col, label, key, width=70, placeholder="",
               hint="") -> int:
        """Ein beschriftetes Eingabefeld; gibt die nächste freie Spalte."""
        box = ctk.CTkFrame(parent, fg_color="transparent", height=1)
        box.grid(row=0, column=col, padx=(0, 14), sticky="w")
        ctk.CTkLabel(box, text=label, text_color="gray60",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0,
                                                     sticky="w")
        self._entry(box, key, width, placeholder).grid(row=1, column=0,
                                                       sticky="w")
        if hint:
            ctk.CTkLabel(box, text=hint, text_color="gray45",
                         font=ctk.CTkFont(size=10)).grid(row=2, column=0,
                                                         sticky="w")
        return col + 1

    def _seg(self, parent, col, label, key, values, width=150) -> int:
        box = ctk.CTkFrame(parent, fg_color="transparent", height=1)
        box.grid(row=0, column=col, padx=(0, 14), sticky="w")
        ctk.CTkLabel(box, text=label, text_color="gray60",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0,
                                                     sticky="w")
        sw = ctk.CTkSegmentedButton(box, values=values, width=width, height=26)
        sw.set(str(self.vals.get(key, values[0])))
        sw.grid(row=1, column=0, sticky="w")
        self.w[key] = sw
        return col + 1

    def _check(self, parent, col, text, key) -> int:
        c = ctk.CTkCheckBox(parent, text=text, checkbox_width=18,
                            checkbox_height=18,
                            font=ctk.CTkFont(size=11))
        if self.vals.get(key):
            c.select()
        c.grid(row=0, column=col, padx=(0, 16), sticky="w", pady=(14, 0))
        self.w[key] = c
        return col + 1

    def _card(self, parent, nr, title, sub, *, kind="fix", da=False,
              indent=0, fill=None):
        """
        Eine Schritt-Karte wie eine Zeile der Pipeline-Grafik: links der
        farbige Balken und die Beschreibung, rechts die Parameter.
        """
        col = {"fix": C_FIX, "opt": C_OPT, "alt": C_ALT}[kind]
        outer = ctk.CTkFrame(parent, fg_color=fill or
                             (C_CARD_DA if da else C_CARD), corner_radius=8,
                             height=1)
        outer.pack(fill="x", padx=(6 + indent, 6), pady=3)
        outer.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(outer, fg_color=col, width=4, height=1,
                     corner_radius=2).grid(
            row=0, column=0, sticky="ns", padx=(6, 10), pady=7)

        head = ctk.CTkFrame(outer, fg_color="transparent", height=1)
        head.grid(row=0, column=1, sticky="w", pady=(7, 7))
        row0 = ctk.CTkFrame(head, fg_color="transparent", height=1)
        row0.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row0, text=nr, text_color="gray50", width=34,
                     anchor="w", font=ctk.CTkFont(size=11, family="monospace")
                     ).pack(side="left")
        ctk.CTkLabel(row0, text=title, anchor="w",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        if da:
            ctk.CTkLabel(row0, text=" DA ", text_color=C_DA,
                         fg_color="#2b2f45", corner_radius=6,
                         font=ctk.CTkFont(size=10, weight="bold")).pack(
                side="left", padx=8)
        if sub:
            ctk.CTkLabel(head, text=sub, text_color="gray50", anchor="w",
                         justify="left", wraplength=430,
                         font=ctk.CTkFont(size=11)).grid(
                row=1, column=0, sticky="w", padx=(34, 0), pady=(1, 0))

        body = ctk.CTkFrame(outer, fg_color="transparent", height=1)
        body.grid(row=0, column=2, sticky="e", padx=(16, 14), pady=7)
        return outer, body

    # ── Aufbau ───────────────────────────────────────────────────────────────
    def _build_head(self) -> None:
        head = ctk.CTkFrame(self, fg_color="transparent", height=1)
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(head, text="Die Pipeline einer Bisektion",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, sticky="w")
        n_txt = (f"{self.n_chunks} Chunks → {self.n_chunks} Annealing-Variablen "
                 f"an der Wurzel" if isinstance(self.n_chunks, int)
                 else "Chunkzahl unbekannt")
        ctk.CTkLabel(head, text=n_txt, text_color="gray55",
                     font=ctk.CTkFont(size=11)).grid(row=1, column=0,
                                                     sticky="w")
        leg = ctk.CTkFrame(head, fg_color="transparent", height=1)
        leg.grid(row=0, column=1, rowspan=2, sticky="e")
        for i, (c, t) in enumerate(((C_FIX, "läuft immer"),
                                    (C_OPT, "abschaltbar"),
                                    (C_ALT, "eine Wahl"),
                                    (C_DA, "nur für den DA"))):
            ctk.CTkFrame(leg, fg_color=c, width=10, height=10,
                         corner_radius=3).grid(row=0, column=i * 2,
                                               padx=(12, 4))
            ctk.CTkLabel(leg, text=t, text_color="gray55",
                         font=ctk.CTkFont(size=11)).grid(row=0,
                                                         column=i * 2 + 1)

    def _build_body(self) -> None:
        sc = ctk.CTkScrollableFrame(self, fg_color="transparent")
        sc.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)

        # 0 — ε
        _, b = self._card(sc, "0", "ε kalibrieren",
                          "Unsicherheitsradius des Encoders — auto misst ihn "
                          "an 256 Chunks (nur ollama)", kind="opt")
        c = self._field(b, 0, "ε", "eps", 110,
                        hint="auto · cos:0.9 · Zahl · 0 = aus")

        # 1 — Normalisierung (kein Parameter)
        self._card(sc, "1", "Normalisieren",
                   "Gruppe zentrieren (Σx̃=0), auf mittlere Länge 1 skalieren "
                   "— immer an, deshalb braucht es keinen Balance-Term")

        # 2/3/4 — nur für den DA
        _, b = self._card(sc, "2", "Gram-Matrix",
                          "Q = x̃x̃ᵀ, jede paarweise Ähnlichkeit — O(n²)",
                          kind="alt", da=True, indent=22)
        self._field(b, 0, "Threshold", "gram_threshold", 90,
                    hint="0 = dicht · >0 = sparse")
        self._da_widgets.append(b)

        _, b = self._card(sc, "3", "QUBO",
                          "E(x) = −½ Σ Qᵢⱼ sᵢsⱼ — eine Binärvariable je Chunk",
                          kind="alt", da=True, indent=22)
        self._seg(b, 0, "Solver", "solver", ["fast", "referenz"], 160)
        self._da_widgets.append(b)

        _, b = self._card(sc, "4", "Budget",
                          "Steps halbieren sich je Ebene, Trials passen sich "
                          "der Gruppengröße an", da=True, indent=22)
        c = self._field(b, 0, "steps", "steps", 90, "auto",
                        hint=f"auto = n/3 (Boden {gp.STEPS_FLOOR})")
        c = self._field(b, c, "MC-Trials", "num_mc", 70)
        c = self._field(b, c, "hold", "hold_steps", 90, "auto",
                        hint="Plateau, auto = n/10")
        self._da_widgets.append(b)

        # 5–7 — die Bisektion
        outer, _ = self._card(sc, "5–7", "Bisektion",
                              "der Schnitt selbst — mindestens einer der "
                              "beiden muss laufen, jeder allein genügt")
        inner = ctk.CTkFrame(outer, fg_color="transparent", height=1)
        inner.grid(row=1, column=0, columnspan=3, sticky="ew",
                   padx=(20, 12), pady=(0, 10))
        inner.grid_columnconfigure(0, weight=1)

        self.f_lloyd = ctk.CTkFrame(inner, fg_color="#1d2a22", height=1,
                                    corner_radius=8, border_width=1,
                                    border_color=C_OPT)
        self.f_lloyd.pack(fill="x", pady=(0, 8))
        self._sub_head(self.f_lloyd, "5", "Lloyd-Warmstart",
                       "k=2 von PCA-orientierten Starts, dedupliziert",
                       "lloyd_on", self.vals.get("iterations_lloyd") not in
                       ("0", 0, ""))
        lb = ctk.CTkFrame(self.f_lloyd, fg_color="transparent")
        lb.pack(fill="x", padx=(90, 12), pady=(0, 10))
        c = self._field(lb, 0, "Iterationen", "iterations_lloyd", 70)
        self._field(lb, c, "Starts (lloydMC)", "lloyd_mc", 70,
                    hint="PCA-orientierte Multistarts")
        self.f_lloyd_body = lb

        self.f_da = ctk.CTkFrame(inner, fg_color="#1d2436", corner_radius=8,
                                 height=1, border_width=1,
                                 border_color=C_OPT)
        self.f_da.pack(fill="x")
        self._sub_head(self.f_da, "6+7", "Digital Annealing",
                       "erst die Temperatur kalibrieren, dann lösen",
                       "anneal", bool(self.vals.get("anneal", True)), da=True)
        # 6 — Kalibrierung
        r6 = ctk.CTkFrame(self.f_da, fg_color="transparent")
        r6.pack(fill="x", padx=(58, 12), pady=(0, 2))
        ctk.CTkFrame(r6, fg_color=C_ALT, width=3, height=1,
                     corner_radius=2).pack(
            side="left", fill="y", pady=2)
        r6b = ctk.CTkFrame(r6, fg_color="transparent", height=1)
        r6b.pack(side="left", fill="x", expand=True, padx=(10, 0))
        ctk.CTkLabel(r6b, text="6  Cooling-Kalibrierung", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(r6b, text="Temperatur aus der ΔE-Skala der "
                               "Lloyd-Lösungen", text_color="gray50",
                     font=ctk.CTkFont(size=11)).pack(anchor="w")
        f6 = ctk.CTkFrame(r6b, fg_color="transparent", height=1)
        f6.pack(anchor="w", pady=(6, 8))
        c = self._seg(f6, 0, "Schedule", "cooling_schedule",
                      ["da_gp", "floor"], 130)
        c = self._field(f6, c, "p_hot", "p_hot", 60)
        c = self._field(f6, c, "p_cold", "p_cold", 60)
        c = self._field(f6, c, "γ", "gamma", 60)
        self._check(f6, c, "T je Startgruppe", "cooling_per_group")
        # 7 — Solve
        r7 = ctk.CTkFrame(self.f_da, fg_color="transparent")
        r7.pack(fill="x", padx=(58, 12), pady=(0, 10))
        ctk.CTkFrame(r7, fg_color="gray40", width=3, height=1,
                     corner_radius=2).pack(
            side="left", fill="y", pady=2)
        r7b = ctk.CTkFrame(r7, fg_color="transparent", height=1)
        r7b.pack(side="left", fill="x", expand=True, padx=(10, 0))
        ctk.CTkLabel(r7b, text="7  Der Solve", anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(r7b, text="alle Trials einer Startgruppe als ein Batch",
                     text_color="gray50",
                     font=ctk.CTkFont(size=11)).pack(anchor="w")
        f7 = ctk.CTkFrame(r7b, fg_color="transparent", height=1)
        f7.pack(anchor="w", pady=(6, 4))
        c = self._field(f7, 0, "E-Offset", "offset_increase", 90,
                        hint="auto_gp oder Zahl")
        self._field(f7, c, "kEsc", "offset_k_escape", 60,
                    hint="kleiner = stärkere Flucht")

        # 8 — Nachschritt
        _, b = self._card(sc, "8", "Nachschritt",
                          "wo die Grenze am Ende liegt", kind="alt")
        self._seg(b, 0, "split-mode", "split_mode",
                  ["none", "centroid", "plane"], 220)

        # 9 — Overlap
        _, b = self._card(sc, "9", "Überlappung",
                          "Sicherheitsnetz obendrauf: wer in BEIDE Kinder "
                          "darf — braucht eine Grenze aus Schritt 8",
                          kind="opt")
        self._field(b, 0, "spill", "spill", 80,
                    hint="Punktanteil · mit ε nur die Obergrenze")

        # Rekursion + Lauf-Optionen
        _, b = self._card(sc, "→", "Rekursion & Lauf",
                          "beide Hälften erneut, bis eine Gruppe ≤ g Punkte hat")
        c = self._field(b, 0, "Leaf-Größe g", "g", 70)
        c = self._field(b, c, "Seed", "seed", 90, "random")
        c = self._check(b, c, "Visualize first", "visualize_first")
        self._check(b, c, "KPIs schreiben", "save_stats")

    def _sub_head(self, parent, nr, title, sub, key, on, da=False) -> None:
        h = ctk.CTkFrame(parent, fg_color="transparent", height=1)
        h.pack(fill="x", padx=12, pady=(10, 4))
        sw = ctk.CTkSwitch(h, text="", width=44, command=self._sync_enabled,
                           progress_color=C_OPT)
        sw.select() if on else sw.deselect()
        sw.pack(side="left")
        self.w[key] = sw
        ctk.CTkLabel(h, text=nr, text_color="gray50", width=34, anchor="w",
                     font=ctk.CTkFont(size=11, family="monospace")).pack(
            side="left")
        tt = ctk.CTkFrame(h, fg_color="transparent", height=1)
        tt.pack(side="left", fill="x", expand=True)
        row = ctk.CTkFrame(tt, fg_color="transparent", height=1)
        row.pack(anchor="w")
        ctk.CTkLabel(row, text=title,
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        if da:
            ctk.CTkLabel(row, text=" DA ", text_color=C_DA,
                         fg_color="#2b2f45", corner_radius=6,
                         font=ctk.CTkFont(size=10, weight="bold")).pack(
                side="left", padx=8)
        ctk.CTkLabel(tt, text=sub, text_color="gray50", anchor="w",
                     font=ctk.CTkFont(size=11)).pack(anchor="w")

    def _build_foot(self) -> None:
        foot = ctk.CTkFrame(self, fg_color="transparent", height=1)
        foot.grid(row=2, column=0, sticky="ew", padx=16, pady=(4, 14))
        foot.grid_columnconfigure(0, weight=1)
        self.lbl_warn = ctk.CTkLabel(foot, text="", text_color="#f08a8a",
                                     anchor="w", font=ctk.CTkFont(size=11))
        self.lbl_warn.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(foot, text="Abbrechen", width=110, fg_color="gray30",
                      hover_color="gray25", command=self.destroy).grid(
            row=0, column=1, padx=6)
        self.btn_go = ctk.CTkButton(foot, text="▶ Indexieren starten",
                                    width=190, command=self._start)
        self.btn_go.grid(row=0, column=2)

    # ── Verhalten ────────────────────────────────────────────────────────────
    def _set_state(self, widget, state: str) -> None:
        for child in widget.winfo_children():
            if isinstance(child, (ctk.CTkEntry, ctk.CTkSegmentedButton,
                                  ctk.CTkCheckBox, ctk.CTkOptionMenu)):
                try:
                    child.configure(state=state)
                except Exception:               # noqa: BLE001
                    pass
            else:
                self._set_state(child, state)

    def _sync_enabled(self) -> None:
        """DA aus → alles ausgrauen, was nur der Annealer braucht."""
        da_on = bool(self.w["anneal"].get())
        lloyd_on = bool(self.w["lloyd_on"].get())
        for frame in self._da_widgets:
            self._set_state(frame, "normal" if da_on else "disabled")
        for f in (self.f_da,):
            f.configure(border_color=C_OPT if da_on else "gray30")
        self.f_lloyd.configure(border_color=C_OPT if lloyd_on else "gray30")
        self._set_state(self.f_lloyd_body, "normal" if lloyd_on else "disabled")
        for r in self.f_da.winfo_children()[1:]:
            self._set_state(r, "normal" if da_on else "disabled")
        ok = da_on or lloyd_on
        self.lbl_warn.configure(
            text="" if ok else "Beide Bisektoren aus — es gibt dann keinen "
                               "Split. Mindestens Lloyd oder DA muss laufen.")
        self.btn_go.configure(state="normal" if ok else "disabled")

    def _collect(self) -> dict:
        out = dict(self.vals)
        for key, widget in self.w.items():
            if isinstance(widget, ctk.CTkEntry):
                out[key] = widget.get().strip()
            elif isinstance(widget, ctk.CTkSegmentedButton):
                out[key] = widget.get()
            elif isinstance(widget, (ctk.CTkCheckBox, ctk.CTkSwitch)):
                out[key] = bool(widget.get())
        if not out.pop("lloyd_on", True):
            out["iterations_lloyd"] = "0"
        elif str(out.get("iterations_lloyd", "")).strip() in ("", "0"):
            out["iterations_lloyd"] = "20"
        return out

    def _start(self) -> None:
        vals = self._collect()
        self.destroy()
        self.on_start(vals)


# ── GUI ───────────────────────────────────────────────────────────────────────

class RagTesterApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RAG Query Tester — DA Partition Tree")
        self.geometry("1280x800")
        self.minsize(980, 600)

        self.params: dict[str, str | bool] = dict(PARAM_DEFAULTS)
        self._budget_prefill: dict[str, str] = {}
        self._specs: dict | None = None          # specs.json der Quelle
        self._model_installed = None             # True | False | "offline"
        self._n_chunks: int | None = None
        self._root_terms: str | None = None

        self.index_dir: Path | None = None
        self.meta: dict | None = None
        self.manifest: dict | None = None
        self.pca_basis: dict | None = None
        self._row_of: dict[str, int] = {}
        self._embeddings: np.ndarray | None = None
        self._last_emb: np.ndarray | None = None
        self._provider = "ollama"
        self._ollama_url = DEFAULT_OLLAMA
        self._gemini_key = env_gemini_key()
        self._tree: dict | None = None
        self._maps: dict[str, np.ndarray] = {}     # Karten-Cache je Index
        self._map_notes: dict[str, str] = {}       # z.B. "Landmark-Näherung …"
        self._map_name = "PCA"
        self._map_target = "PCA"                   # vom Nutzer gewünschte Karte
        self._map_busy: str | None = None          # Name der Karte in Arbeit
        self._doc_coloring: list[tuple[list[int], str]] = []
        self._doc_legend: list[tuple[str, str]] = []
        self._qview: tuple | None = None           # (candidate_rows, hits)
        self._color_mode = "doc"                   # "doc" | "gp"
        self._pending_run: Path | None = None      # vorgemerkt, wartet auf ▶
        self._last_run_dir: Path | None = None
        self._index_dirs: dict[str, Path] = {}     # Name → Indexierung (Dropdown)
        self._phases: list[list[tuple[str, list[int]]]] = []
        self._phase = 0
        self._maxd = 0
        self._replaying = False
        self.results: list[dict] = []
        self._result_buttons: list[ctk.CTkButton] = []
        self._indexing = False
        self._active_thr = 0.0
        self._index_proc: subprocess.Popen | None = None
        self._points_done = 0

        # Thread-sichere UI-Updates: Worker posten Callables in die Queue,
        # der Main-Thread arbeitet sie im after()-Takt ab.
        self._uiq: queue.Queue = queue.Queue()

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(40, self._poll_ui_queue)
        self.after(200, self.check_ollama)

    def _on_close(self) -> None:
        """Fenster zu → laufenden Indexing-Subprozess mitbeenden."""
        p = self._index_proc
        if p is not None and p.poll() is None:
            p.terminate()
        self.destroy()

    def _post(self, fn) -> None:
        """Aus Worker-Threads: UI-Update in den Main-Thread übergeben."""
        self._uiq.put(fn)

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                fn = self._uiq.get_nowait()
                try:
                    fn()
                except Exception as exc:        # noqa: BLE001
                    self.set_status(f"✗ UI-Update: {exc}", error=True)
        except queue.Empty:
            pass
        self.after(40, self._poll_ui_queue)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # — Zeile 0: Ordner + Indexing-Parameter + Ollama-Status —
        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        top.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(top, text="Ordner wählen…", width=150,
                      command=self.pick_folder).grid(row=0, column=0, padx=8, pady=8)
        # Dropdown: zwischen mehreren Indexierungen desselben Run-Ordners wechseln
        self.index_menu = ctk.CTkOptionMenu(
            top, values=["—"], width=150,
            command=self._switch_index, state="disabled")
        self.index_menu.grid(row=0, column=1, padx=(0, 6))
        self.lbl_index = ctk.CTkLabel(top, text="Kein Index geladen", anchor="w",
                                      justify="left", text_color="gray70")
        self.lbl_index.grid(row=0, column=2, sticky="ew", padx=6)


        # — Zeile 1: Metadaten der Knowledgebase ————————————————————————————
        #  Der Platz, den früher die drei DA-Parameterreihen belegt haben.
        #  Die Indexing-Parameter stehen jetzt im Pipeline-Dialog (öffnet
        #  sich beim Klick auf „Indexieren"), hier steht dafür das, was man
        #  ständig sehen will: womit diese Knowledgebase gebaut wurde.
        self.meta_panel = ctk.CTkFrame(top, fg_color="transparent")
        self.meta_panel.grid(row=1, column=0, columnspan=9, sticky="ew",
                             padx=8, pady=(2, 8))
        self.meta_panel.grid_columnconfigure(0, weight=1)
        self._meta_cards: list[ctk.CTkFrame] = []
        self._render_meta([])

        self.btn_reindex = ctk.CTkButton(
            top, text="⟳ Neu indexieren", width=150,
            command=self.reindex_current, state="disabled")
        self.btn_reindex.grid(row=0, column=9, padx=(10, 8))

        # Wrapping der beiden langen Labels: an die FRAME-Breite gekoppelt
        # (nie ans Label selbst binden — wraplength ändert die Labelgröße
        # und würde sofort das nächste <Configure> auslösen: Endlosschleife).
        top.bind("<Configure>", lambda e:
                 self.lbl_index.configure(wraplength=max(e.width - 900, 250)))

        # Verbindungsfeld: Ollama-Base-URL oder Gemini-API-Key — je nachdem,
        # mit welchem Provider die geladene Knowledgebase embeddet wurde.
        self.lbl_conn = ctk.CTkLabel(top, text="Ollama:")
        self.lbl_conn.grid(row=0, column=5, padx=(10, 2))
        self.entry_url = ctk.CTkEntry(top, width=200)
        self.entry_url.insert(0, DEFAULT_OLLAMA)
        self.entry_url.grid(row=0, column=6, padx=(0, 4))
        self.lbl_ollama = ctk.CTkLabel(top, text="● Backend?", text_color="gray60",
                                       width=110, anchor="w")
        self.lbl_ollama.grid(row=0, column=7, padx=(2, 4))
        ctk.CTkButton(top, text="Neu prüfen", width=90,
                      command=self.check_ollama).grid(row=0, column=8, padx=(0, 8))

        # — Zeile 1: Query + Parameter —
        qrow = ctk.CTkFrame(self)
        qrow.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        qrow.grid_columnconfigure(0, weight=1)

        self.entry_query = ctk.CTkEntry(
            qrow, placeholder_text="RAG-Query eingeben … (Enter = Suchen)")
        self.entry_query.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        self.entry_query.bind("<Return>", lambda _e: self.start_query())

        ctk.CTkLabel(qrow, text="Chunks k:").grid(row=0, column=1, padx=(6, 2))
        self.entry_k = ctk.CTkEntry(qrow, width=48)
        self.entry_k.insert(0, "5")
        self.entry_k.grid(row=0, column=2)

        ctk.CTkLabel(qrow, text="Beam:").grid(row=0, column=3, padx=(10, 2))
        self.entry_beam = ctk.CTkEntry(qrow, width=48)
        self.entry_beam.insert(0, "2")
        self.entry_beam.grid(row=0, column=4)

        ctk.CTkLabel(qrow, text="min. Kand.:").grid(row=0, column=5, padx=(10, 2))
        self.entry_mincand = ctk.CTkEntry(qrow, width=56, placeholder_text="auto")
        self.entry_mincand.grid(row=0, column=6)

        self.btn_query = ctk.CTkButton(qrow, text="Suchen", width=110,
                                       command=self.start_query, state="disabled")
        self.btn_query.grid(row=0, column=7, padx=8)

        # — Zeile 2: Ergebnisliste links, Tabs (Chunk / Punktwolke) rechts —
        mid = ctk.CTkFrame(self, fg_color="transparent")
        mid.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        mid.grid_columnconfigure(0, weight=2, uniform="mid")
        mid.grid_columnconfigure(1, weight=3, uniform="mid")
        mid.grid_rowconfigure(0, weight=1)

        self.frame_results = ctk.CTkScrollableFrame(mid, label_text="Ergebnisse")
        self.frame_results.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.frame_results.grid_columnconfigure(0, weight=1)

        # DA-Terminal-Log: belegt während des Indexings die Ergebnis-Spalte
        # (zeigt den stdout des rag_indexer-Subprozesses, v.a. die
        # MC-Trial-Zeilen des DigitalAnnealing-Solvers).
        self.txt_dalog = ctk.CTkTextbox(mid, wrap="none",
                                        font=ctk.CTkFont(size=11, family="Menlo"),
                                        text_color="#9fd49f")
        self.txt_dalog.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.txt_dalog.configure(state="disabled")
        self.txt_dalog.grid_remove()                 # erst bei Indexing sichtbar

        # Rechts: Karten-Leiste + Punktwolke (immer sichtbar), Chunk darunter
        right = ctk.CTkFrame(mid)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=11)   # Karte
        right.grid_rowconfigure(3, weight=9)    # Chunk-Text

        mapbar = ctk.CTkFrame(right, fg_color="transparent")
        mapbar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        mapbar.grid_columnconfigure(4, weight=1)

        ctk.CTkLabel(mapbar, text="Karte:").grid(row=0, column=0, padx=(2, 4))
        self.map_menu = ctk.CTkOptionMenu(
            mapbar, values=MAP_CHOICES, width=140,
            command=self._switch_map, state="disabled")
        self.map_menu.grid(row=0, column=1)
        ctk.CTkLabel(mapbar, text="Farben:").grid(row=0, column=2, padx=(12, 4))
        self.color_menu = ctk.CTkOptionMenu(
            mapbar, values=["Dokumente", "GP-Gruppen"], width=130,
            command=self._on_color_mode, state="disabled")
        self.color_menu.grid(row=0, column=3)
        self.lbl_encoder = ctk.CTkLabel(
            mapbar, text="Encoder: —", anchor="e", text_color="gray70",
            font=ctk.CTkFont(size=12))
        self.lbl_encoder.grid(row=0, column=4, sticky="ew", padx=(10, 2))

        # Zweite Zeile: Splitting-Phasen (GP-Modus) — Slider + Replay
        ctk.CTkLabel(mapbar, text="Phase:").grid(row=1, column=0,
                                                 padx=(2, 4), pady=(4, 2))
        self.phase_slider = ctk.CTkSlider(
            mapbar, from_=0, to=1, number_of_steps=1, width=220,
            command=self._on_phase_slider, state="disabled")
        self.phase_slider.grid(row=1, column=1, columnspan=2,
                               sticky="w", pady=(4, 2))
        self.lbl_phase = ctk.CTkLabel(mapbar, text="–", width=110, anchor="w",
                                      text_color="gray70")
        self.lbl_phase.grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(4, 2))
        self.btn_replay = ctk.CTkButton(
            mapbar, text="▶ Splitting abspielen", width=150,
            command=self._replay, state="disabled")
        self.btn_replay.grid(row=1, column=4, sticky="e", padx=(10, 2),
                             pady=(4, 2))

        self.cloud = PointCloud(right)
        self.cloud.grid(row=1, column=0, sticky="nsew", padx=6, pady=(4, 2))

        self.lbl_chunk_head = ctk.CTkLabel(right, text="Chunk-Ansicht",
                                           anchor="w",
                                           font=ctk.CTkFont(weight="bold"))
        self.lbl_chunk_head.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 2))
        self.txt_chunk = ctk.CTkTextbox(right, wrap="word",
                                        font=ctk.CTkFont(size=13))
        self.txt_chunk.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))
        self.txt_chunk.configure(state="disabled")

        # — Zeile 3: Progressbar + Statusleiste (eigener Frame, Text bricht
        #   um statt abgeschnitten zu werden) —
        bottom = ctk.CTkFrame(self)
        bottom.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        bottom.grid_columnconfigure(1, weight=1)
        self.progress = ctk.CTkProgressBar(bottom, width=220)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, padx=(10, 12), pady=6)
        self.lbl_status = ctk.CTkLabel(bottom, text="Bereit.", anchor="w",
                                       justify="left", text_color="gray70")
        self.lbl_status.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=4)
        # wraplength dynamisch an die Fensterbreite koppeln
        bottom.bind("<Configure>", lambda e: self.lbl_status.configure(
            wraplength=max(e.width - 280, 300)))

        # Esc: Auswahl + Query-Ansicht aufheben → komplette Wolke
        self.bind("<Escape>", lambda _e: self._reset_query_view())

    # ── Ordnerwahl: laden oder automatisch indexen ────────────────────────────
    def pick_folder(self) -> None:
        if self._indexing:
            return
        folder = filedialog.askdirectory(
            title="rag_index-, Run- oder un-geindexten text-embedder-Ordner wählen")
        if not folder:
            return
        kind, target = resolve_folder(Path(folder))

        if kind == "index":
            self.load_index(target)
        elif kind == "run":
            self._prepare_run(target)      # KEIN Auto-Start — nur vorbereiten
        else:
            self.lbl_index.configure(
                text=f"✗ '{Path(folder).name}' ist weder RAG-Index noch "
                     f"text-embedder-Run (index_meta.json/specs.json fehlen)",
                text_color="#e06060")

    # ── Mehrere Indexierungen pro Run-Ordner ──────────────────────────────────
    def _populate_index_menu(self, index_dir: Path | None) -> None:
        """Dropdown mit allen Indexierungen des zugehörigen Run-Ordners
        füllen (rag_index, rag_index_001, …) und die aktive auswählen."""
        self._index_dirs = {}
        run_dir = None
        if index_dir is not None and (index_dir.parent / "specs.json").exists():
            run_dir = index_dir.parent
        if run_dir is not None:
            for p in list_index_dirs(run_dir):
                self._index_dirs[p.name] = p
        elif index_dir is not None:          # z.B. exportierte KB ohne Run
            self._index_dirs[index_dir.name] = index_dir

        if self._index_dirs:
            names = list(self._index_dirs)
            self.index_menu.configure(values=names,
                                      state="normal" if len(names) > 1
                                      else "disabled")
            self.index_menu.set(index_dir.name if index_dir is not None
                                and index_dir.name in self._index_dirs
                                else names[-1])
        else:
            self.index_menu.configure(values=["—"], state="disabled")
            self.index_menu.set("—")

    def _switch_index(self, name: str) -> None:
        target = self._index_dirs.get(name)
        if target is None or self._indexing:
            if self.index_dir is not None:
                self.index_menu.set(self.index_dir.name)
            return
        if self.index_dir is not None and target == self.index_dir:
            return
        self.set_status(f"Wechsle zu Indexierung „{name}“ …")
        self.load_index(target)

    # ── Index laden (inkl. Karte) ─────────────────────────────────────────────
    def load_index(self, index_dir: Path) -> None:
        def worker() -> None:
            try:
                with open(index_dir / "index_meta.json", encoding="utf-8") as fh:
                    meta = json.load(fh)
                with open(index_dir / "chunks_manifest.json", encoding="utf-8") as fh:
                    manifest = json.load(fh)
                with open(index_dir / "ann_tree.json", encoding="utf-8") as fh:
                    tree = json.load(fh)
                E = np.load(index_dir / "embeddings.npy").astype(float)
                xy, basis = pca_2d(E)
                specs = None
                spath = index_dir.parent / "specs.json"
                if spath.exists():
                    try:
                        with open(spath, encoding="utf-8") as fh:
                            specs = json.load(fh)
                    except Exception:           # noqa: BLE001
                        specs = None
                self._post(lambda: self._index_loaded(
                    index_dir, meta, manifest, xy, basis, E, tree, specs))
            except Exception as exc:            # noqa: BLE001
                msg = str(exc)
                self._post(lambda: self.set_status(f"✗ Index laden: {msg}",
                                                      error=True))
        threading.Thread(target=worker, daemon=True).start()

    def _index_loaded(self, index_dir, meta, manifest, xy, basis, E, tree,
                      specs=None) -> None:
        self.index_dir, self.meta, self.manifest = index_dir, meta, manifest
        self._specs = specs
        self.pca_basis = basis
        self._embeddings = E
        self._tree = tree
        self._last_emb = None
        self._qview = None
        self._maps = {"PCA": xy}                    # Cache frisch je Index
        self._map_notes = {}
        self._map_name = "PCA"
        self._row_of = {cid: i for i, cid in enumerate(manifest["row_to_id"])}
        enc, part = meta["encoder"], meta["partitioning"]
        self.lbl_index.configure(
            text=(f"✓ {index_dir.parent.name} / {index_dir.name}  ·  "
                  f"{meta['chunk_count']} Chunks"
                  f"  ·  {part['num_leaves']} Leaves, Tiefe {part['tree_depth']}, "
                  f"g={part['g']}"),
            text_color="#7ec97e")
        self._populate_index_menu(index_dir)
        self.lbl_encoder.configure(
            text=f"Encoder: {enc['model']}  ·  {enc['provider']}  ·  {enc['dim']}d")
        self.btn_query.configure(state="normal")
        self._pending_run = None
        self.btn_reindex.configure(state="normal", text="⟳ Neu indexieren")
        self.map_menu.configure(state="normal")
        self.map_menu.set("PCA")
        self.progress.set(1.0)
        self._apply_provider(enc["provider"])

        # Dokumentfarben einmal berechnen (gelten für alle Karten)
        sources: list[str] = []
        rows_by_source: dict[str, list[int]] = {}
        for row, cid in enumerate(manifest["row_to_id"]):
            src = manifest["chunks"][cid]["source"] or "?"
            if src not in rows_by_source:
                sources.append(src)
                rows_by_source[src] = []
            rows_by_source[src].append(row)
        self._doc_coloring = []
        self._doc_legend = []
        for i, src in enumerate(sources):
            color = doc_color(i)
            self._doc_coloring.append((rows_by_source[src], color))
            self._doc_legend.append(
                (color, src if len(src) <= 34 else src[:31] + "…"))

        # Splitting-Phasen aus dem Baum rekonstruieren (GP-Ansicht + Replay)
        self._phases = tree_phases(tree)
        self._maxd = len(self._phases) - 1
        self._phase = self._maxd
        self.phase_slider.configure(state="normal", to=self._maxd,
                                    number_of_steps=max(self._maxd, 1))
        self.phase_slider.set(self._maxd)
        self.color_menu.configure(state="normal")
        self.color_menu.set("Dokumente")
        self._color_mode = "doc"
        self.btn_replay.configure(state="normal")

        # Die Parameter des geladenen Index werden ins Formular übernommen,
        # damit „⟳ Neu indexieren“ dort weitermacht, wo dieser Index aufhörte.
        self._params_from_meta(meta)
        n_c = meta["chunk_count"]
        self._prefill_budget(n_c)

        # Root-QUBO-Größe: echte Termzahl aus indexing_stats.json (sparse bei
        # thr>0), sonst die dichte Formel n²/2+n.  Nur wenn der DA lief.
        terms_txt = None
        if part.get("anneal", True):
            terms_txt = f"{n_c * (n_c + 1) // 2 + n_c:,} Terme (dicht)"
            try:
                with open(index_dir / "indexing_stats.json",
                          encoding="utf-8") as fh:
                    b0 = json.load(fh)["bisections"][0]
                terms_txt = f"{b0['pbf_terms']:,} Terme"
                if b0.get("gram_zeroed_frac"):
                    terms_txt += f", {b0['gram_zeroed_frac']:.0%} verworfen"
            except Exception:               # noqa: BLE001
                pass
        self._root_terms = terms_txt
        self._refresh_meta()
        self._check_model_installed(
            (specs or {}).get("embedding", {}) or
            {"provider": enc.get("provider"), "model": enc.get("model")})

        self._apply_map("PCA")
        self.set_status(f"Index geladen: {index_dir}")
        self.check_ollama()

        # Frisch geindext → Splitting-Prozess einmal langsam abspielen
        if getattr(self, "_replay_after_load", False):
            self._replay_after_load = False
            self.after(400, self._replay)

    # ── Farbmodus + Splitting-Phasen ──────────────────────────────────────────
    def _recolor(self) -> None:
        """Basisfarben gemäß Modus (Dokumente / GP-Phase) neu setzen."""
        if self.cloud.xy is None or self._embeddings is None:
            return
        if self._color_mode == "doc":
            for rows, color in self._doc_coloring:
                for r in rows:
                    self.cloud.colors[r] = color
            self.cloud.legend = self._doc_legend
            self.lbl_phase.configure(text="–")
            suffix = "Farbe = Quelldokument"
        else:
            groups = self._phases[self._phase]
            for path, rows in groups:
                color = path_color(path)
                for r in rows:
                    self.cloud.colors[r] = color
            self.cloud.legend = []
            sizes = sorted((len(rw) for _p, rw in groups), reverse=True)
            self.lbl_phase.configure(
                text=f"{self._phase}/{self._maxd} · {len(groups)} Gruppen")
            suffix = (f"GP-Splitting Phase {self._phase}/{self._maxd} · "
                      f"{len(groups)} Gruppen (Größen {sizes[-1]}–{sizes[0]} "
                      f"= DA-Variablen der nächsten Bisection)")
        self.cloud.caption = (self._map_caption() + " · " + suffix
                              + " · Klick öffnet Chunk · Leer-Klick/Esc = zurück")
        self.cloud.redraw()

    def _map_caption(self) -> str:
        note = self._map_notes.get(self._map_name, "")
        return MAP_CAPTIONS[self._map_name] + (f" · {note}" if note else "")

    def _on_color_mode(self, choice: str) -> None:
        self._color_mode = "gp" if choice.startswith("GP") else "doc"
        self._recolor()

    def _on_phase_slider(self, value: float) -> None:
        p = int(round(value))
        if p != self._phase:
            self._phase = p
            if self._color_mode != "gp":
                self._color_mode = "gp"
                self.color_menu.set("GP-Gruppen")
            self._recolor()

    def _set_phase(self, p: int) -> None:
        self._phase = max(0, min(p, self._maxd))
        self.phase_slider.set(self._phase)
        self._recolor()

    def _replay(self) -> None:
        """Splitting-Prozess abspielen: Phase 0 → … → Leaves."""
        if self._embeddings is None:
            return
        if self._replaying:                        # laufendes Replay stoppen
            self._replaying = False
            self.btn_replay.configure(text="▶ Splitting abspielen")
            return
        self._replaying = True
        self._color_mode = "gp"
        self.color_menu.set("GP-Gruppen")
        self.btn_replay.configure(text="■ Stop")

        def step(p: int) -> None:
            if not self._replaying or p > self._maxd:
                self._replaying = False
                self.btn_replay.configure(text="▶ Splitting abspielen")
                return
            self._set_phase(p)
            n_groups = len(self._phases[p])
            self.set_status(f"Splitting-Replay: Phase {p}/{self._maxd} — "
                            f"{n_groups} Gruppe{'n' if n_groups != 1 else ''}")
            self.after(900, lambda: step(p + 1))

        step(0)

    # ── Karten-Auswahl ────────────────────────────────────────────────────────
    def _compute_map(self, name: str) -> tuple[np.ndarray, str]:
        """Karte berechnen (läuft im Worker-Thread) → (xy, Hinweis-Text)."""
        E = self._embeddings
        exact = {"MDS (Cosine)": mds_2d, "t-SNE": tsne_2d,
                 "Spectral (kNN)": spectral_2d}
        if name in exact:
            limit = LANDMARK_LIMITS[name]
            if len(E) > limit:
                xy = landmark_map(E, exact[name], limit)
                return xy, f"Landmark-Näherung ({limit}/{len(E)} exakt)"
            return exact[name](E), ""
        if name == "DA-Baum (Grid)":
            return tree_grid_2d(self._tree, len(E)), ""
        return pca_2d(E)[0], ""

    def _switch_map(self, name: str) -> None:
        if self._embeddings is None:
            self.map_menu.set(self._map_name)
            return
        if name in self._maps:                # Cache-Wechsel geht immer sofort
            self._apply_map(name)
            return
        if self._map_busy is not None:
            self.map_menu.set(self._map_name)
            self.set_status(f"Karte „{self._map_busy}“ wird noch berechnet — "
                            f"danach nochmal wählen.")
            return
        self._map_busy = name
        self._map_target = name
        self.map_menu.set(name)
        n = len(self._embeddings)
        limit = LANDMARK_LIMITS.get(name)
        approx = f" — Landmark-Näherung, {limit} exakt" \
            if limit is not None and n > limit else ""
        self.set_status(f"Berechne Karte „{name}“ ({n} Punkte{approx}) …")

        def worker() -> None:
            try:
                xy, note = self._compute_map(name)
                def done():
                    self._maps[name] = xy
                    self._map_notes[name] = note
                    self._map_busy = None
                    # nur übernehmen, wenn der Nutzer nicht längst auf eine
                    # andere (gecachte) Karte gewechselt hat
                    if self._map_target == name:
                        self._apply_map(name)
                    else:
                        self.set_status(f"✓ Karte „{name}“ fertig (im Cache).")
                self._post(done)
            except Exception as exc:            # noqa: BLE001
                msg = str(exc)
                def fail():
                    self._map_busy = None
                    if self._map_target == name:
                        self._map_target = self._map_name
                        self.map_menu.set(self._map_name)
                    self.set_status(f"✗ Karte „{name}“: {msg}", error=True)
                self._post(fail)

        threading.Thread(target=worker, daemon=True).start()

    def _query_pos(self, emb: np.ndarray, map_name: str,
                   xy: np.ndarray) -> np.ndarray:
        """Query-Marker-Position: PCA exakt (lineare Basis), sonst
        kNN-Interpolation über die ähnlichsten Chunk-Positionen."""
        if map_name == "PCA" and self.pca_basis is not None:
            return project_query(emb, self.pca_basis)
        return interp_query_xy(emb, self._embeddings, xy)

    def _apply_map(self, name: str) -> None:
        """Karte anzeigen; Färbung (Modus) + laufende Query-Ansicht übernehmen."""
        xy = self._maps[name]
        self._map_name = name
        self._map_target = name
        self.map_menu.set(name)
        self.cloud.set_points(xy)
        self.cloud.on_click = self._map_click
        self._recolor()
        if self._qview is not None and self._last_emb is not None:
            candidates, hits = self._qview
            self.cloud.set_query_view(
                candidates, hits, self._query_pos(self._last_emb, name, xy))

    def _params_from_meta(self, meta: dict) -> None:
        """Die im Index gespeicherten Einstellungen ins Formular spiegeln."""
        part = meta.get("partitioning", {})
        cool = part.get("cooling", {})
        P = self.params

        def put(key, value, fmt=str):
            if value is not None:
                P[key] = fmt(value)

        put("g", part.get("g"))
        put("steps", part.get("steps"))
        put("num_mc", part.get("num_mc"))
        put("gram_threshold", part.get("gram_threshold"), lambda v: f"{v:g}")
        put("iterations_lloyd", part.get("iterations_lloyd"))
        put("lloyd_mc", part.get("lloyd_mc"))
        put("split_mode", part.get("split_mode"))
        put("spill", part.get("spill"), lambda v: f"{v:g}")
        put("eps", part.get("eps"), lambda v: f"{v:g}")
        put("offset_increase", part.get("offset_increase"),
            lambda v: v if isinstance(v, str) else f"{v:g}")
        P["anneal"] = bool(part.get("anneal", True))
        P["solver"] = ("referenz" if part.get("solver") == "reference"
                       else "fast")
        if str(cool.get("schedule", "")).startswith("da_gp"):
            P["cooling_schedule"] = ("floor"
                                     if cool.get("schedule") == "da_gp_floor"
                                     else "da_gp")
            P["cooling_per_group"] = bool(cool.get("per_group", True))
            put("p_hot", cool.get("p_hot"), lambda v: f"{v:g}")
            put("p_cold", cool.get("p_cold"), lambda v: f"{v:g}")
            put("gamma", cool.get("gamma"), lambda v: f"{v:g}")
            put("hold_steps", cool.get("hold_steps"))
            put("offset_k_escape", cool.get("offset_k_escape"),
                lambda v: f"{v:g}")
        seed = part.get("seed")
        P["seed"] = "" if seed is None else f"{seed:g}"

    # ── Metadaten-Panel ──────────────────────────────────────────────────────
    def _render_meta(self, groups: list[tuple[str, list[tuple]]]) -> None:
        """
        Karten mit den Metadaten der Knowledgebase.  *groups* ist
        [(Kartentitel, [(Label, Wert, Farbe|None), …]), …] — alles, was
        gezeigt wird, kommt aus specs.json / index_meta.json, nichts ist
        hier fest verdrahtet.
        """
        for card in self._meta_cards:
            card.destroy()
        self._meta_cards = []
        if not groups:
            lbl = ctk.CTkLabel(
                self.meta_panel, anchor="w", text_color="gray45",
                font=ctk.CTkFont(size=11),
                text="Noch keine Knowledgebase geladen — „Ordner wählen…“, "
                     "dann zeigt diese Leiste Encoder, Chunking und Korpus.")
            lbl.grid(row=0, column=0, sticky="w")
            self._meta_cards.append(lbl)
            return
        for col, (title, rows) in enumerate(groups):
            card = ctk.CTkFrame(self.meta_panel, fg_color="#23262e",
                                corner_radius=8, height=1)
            card.grid(row=0, column=col, sticky="nsew", padx=(0, 8))
            self.meta_panel.grid_columnconfigure(col, weight=1, uniform="meta")
            ctk.CTkLabel(card, text=title, anchor="w", text_color="gray55",
                         font=ctk.CTkFont(size=10, weight="bold")).grid(
                row=0, column=0, columnspan=2, sticky="w", padx=10,
                pady=(7, 2))
            for r, item in enumerate(rows, start=1):
                label, value = item[0], item[1]
                color = item[2] if len(item) > 2 else None
                ctk.CTkLabel(card, text=label, anchor="w", text_color="gray45",
                             font=ctk.CTkFont(size=11)).grid(
                    row=r, column=0, sticky="w", padx=(10, 8))
                ctk.CTkLabel(card, text=str(value), anchor="w",
                             text_color=color or "gray75",
                             font=ctk.CTkFont(size=11)).grid(
                    row=r, column=1, sticky="w", padx=(0, 10))
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkFrame(card, fg_color="transparent", height=6).grid(
                row=len(rows) + 1, column=0)
            self._meta_cards.append(card)

    def _meta_groups(self, specs: dict | None, meta: dict | None
                     ) -> list[tuple[str, list[tuple]]]:
        """specs.json (+ index_meta.json) → Karteninhalt."""
        groups: list[tuple[str, list[tuple]]] = []
        emb = dict((specs or {}).get("embedding", {}))
        if meta and not emb:
            enc = meta.get("encoder", {})
            emb = {"model": enc.get("model"), "provider": enc.get("provider")}
        if emb:
            rows = [("Modell", emb.get("model", "?"))]
            # Dynamisch: was immer sonst im embedding-Block steht
            for k, v in emb.items():
                if k in ("model",) or v in (None, ""):
                    continue
                rows.append((k.replace("_", " "), v))
            dim = ((specs or {}).get("embedding_dim")
                   or (meta or {}).get("encoder", {}).get("dim"))
            if dim:
                rows.append(("Dimension", f"{dim}d"))
            rows.append(("lokal", *self._model_state(emb)))
            groups.append(("ENCODER", rows))

        chunk = (specs or {}).get("chunking", {})
        if chunk:
            groups.append(("CHUNKING", [(k.replace("_", " "), v)
                                        for k, v in chunk.items()
                                        if v not in (None, "")][:6]))

        corp = []
        n = (specs or {}).get("chunk_count") or (meta or {}).get("chunk_count")
        if n:
            corp.append(("Chunks", f"{n:,}".replace(",", " ")))
        files = (specs or {}).get("input_files") or []
        if files:
            corp.append(("Dateien", len(files)))
            corp.append(("zuerst", files[0][:26] + ("…" if len(files[0]) > 26
                                                    else "")))
        created = (specs or {}).get("created_utc")
        if created:
            corp.append(("erstellt", str(created)[:10]))
        if corp:
            groups.append(("KORPUS", corp))

        if meta:
            part = meta.get("partitioning", {})
            rows = [("Leaves", part.get("num_leaves", "?")),
                    ("Tiefe", part.get("tree_depth", "?")),
                    ("g", part.get("g", "?")),
                    ("Methode", ("Lloyd" if part.get("anneal") is False
                                 else f"DA/{part.get('solver', '?')}")),
                    ("Split", f"{part.get('split_mode', '?')}"
                              + (f" +spill {part['spill']:g}"
                                 if part.get("spill") else "")),
                    ("Bauzeit", f"{part.get('indexing_seconds', '?')} s")]
            if self._root_terms:
                rows.insert(3, ("Root-QUBO", self._root_terms))
            groups.append(("INDEX", rows))
        return groups

    def _model_state(self, emb: dict) -> tuple[str, str]:
        """Ist das Encoder-Modell der KB auf diesem Rechner nutzbar?"""
        state = self._model_installed
        if emb.get("provider") != "ollama":
            return "API-Modell (kein Download nötig)", "gray60"
        if state is None:
            return "wird geprüft …", "gray50"
        if state is False:
            return f"✗ fehlt — ollama pull {emb.get('model', '')}", "#e0a840"
        if state == "offline":
            return "Backend nicht erreichbar", "gray55"
        return "✓ installiert", "#7ec97e"

    def _check_model_installed(self, emb: dict) -> None:
        """Modellliste im Hintergrund holen (Netzwerk) und Panel auffrischen."""
        if emb.get("provider") != "ollama":
            self._model_installed = None
            return
        self._model_installed = None
        base = (emb.get("base_url") or self.entry_url.get().strip()
                or DEFAULT_OLLAMA)
        want = str(emb.get("model", ""))

        def worker() -> None:
            try:
                have = ollama_models(base)
                # ollama hängt gern ein ":latest" an
                ok = any(m == want or m.split(":")[0] == want.split(":")[0]
                         for m in have)
            except Exception:                   # noqa: BLE001
                ok = "offline"
            self._post(lambda: self._model_state_done(ok))

        threading.Thread(target=worker, daemon=True).start()

    def _model_state_done(self, state) -> None:
        self._model_installed = state
        self._refresh_meta()

    def _refresh_meta(self) -> None:
        self._render_meta(self._meta_groups(self._specs, self.meta))

    def _prefill_budget(self, n_chunks) -> None:
        """steps/hold aus der GESAMTZAHL der Chunks vorbelegen (n/3 bzw.
        n/10) — die Werte gelten für Ebene 0 und halbieren sich von dort
        mit der Tiefe.  Nur füllen, was der Nutzer nicht selbst gesetzt
        hat: ein leerer Wert bedeutet 'auto', ein von uns vorbelegter
        wird bei einer anderen KB neu berechnet."""
        try:
            n = int(n_chunks)
        except (TypeError, ValueError):
            return
        if n <= 0:
            return
        self._n_chunks = n
        for key, val in (("steps", max(gp.STEPS_FLOOR, n // 3)),
                         ("hold_steps", n // 10)):
            cur = str(self.params.get(key, "")).strip()
            if cur and cur != str(self._budget_prefill.get(key, "")):
                continue                      # vom Nutzer überschrieben
            self.params[key] = str(val)
            self._budget_prefill[key] = str(val)

    def _prepare_run(self, run_dir: Path) -> None:
        """Un-geindexten Run-Ordner vormerken — Indexing startet erst auf
        expliziten Klick (▶ Indexieren), damit vorher in Ruhe die
        DA-Parameter eingestellt werden können."""
        try:
            with open(run_dir / "specs.json", encoding="utf-8") as fh:
                specs = json.load(fh)
            n = specs.get("chunk_count", "?")
            model = specs.get("embedding", {}).get("model", "?")
        except Exception:                   # noqa: BLE001
            specs, n, model = None, "?", "?"
        self._pending_run = run_dir
        self._specs = specs
        self._prefill_budget(n)
        self._populate_index_menu(None)
        self.lbl_index.configure(
            text=(f"◔ Run-Ordner erkannt: {run_dir.name}  ·  {n} Chunks "
                  f"({model})  ·  noch kein Index"),
            text_color="#e0a840")
        self.btn_reindex.configure(text="▶ Indexieren", state="normal")
        self._refresh_meta()
        if specs:
            self._check_model_installed(specs.get("embedding", {}))
        self.set_status(
            f"Bereit zum Indexieren: {n} Chunks = {n} Annealing-Variablen an "
            f"der Wurzel. „▶ Indexieren“ öffnet die Pipeline-Einstellungen.")

    def reindex_current(self) -> None:
        """Öffnet die Pipeline-Einstellungen; gestartet wird erst aus dem
        Dialog heraus.  Ziel ist entweder der vorgemerkte Run-Ordner
        (▶ Indexieren) oder die Quelle des geladenen Index (⟳ Neu)."""
        if self._indexing:
            return
        if self._pending_run is not None:
            run_dir = self._pending_run
        elif self.index_dir is not None:
            run_dir = self.index_dir.parent
            if not (run_dir / "specs.json").exists():
                self.set_status("✗ Quell-Run-Ordner nicht gefunden "
                                "(specs.json fehlt).")
                return
        else:
            return

        n = getattr(self, "_n_chunks", None)
        if n is None and self.meta:
            n = self.meta.get("chunk_count")

        def go(vals: dict) -> None:
            self.params.update(vals)
            self._pending_run = None
            self.start_indexing(run_dir)

        dlg = PipelineDialog(self, self.params, n, go)
        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)

    # ── Automatisches Indexing mit Live-Karte ─────────────────────────────────
    def start_indexing(self, run_dir: Path) -> None:
        """Startet den Indexing-Subprozess mit den Werten aus self.params."""
        P = self.params

        def num(key, default, cast=float):
            raw = str(P.get(key, "")).strip()
            try:
                return cast(raw) if raw != "" else default
            except ValueError:
                raise ValueError(f"{key}: „{raw}“ ist keine Zahl") from None

        try:
            g = max(2, num("g", 20, int))
            steps_raw = str(P.get("steps", "")).strip()
            steps = int(steps_raw) if steps_raw else None
            num_mc = num("num_mc", 20, int)
            off_raw = (str(P.get("offset_increase", "")).strip() or "auto_gp")
            # "auto_gp": Offset wird im Solver dynamisch aus der ΔE-Skala
            # bestimmt (q50 der Bergauf-Flips / k_escape) statt absolut
            offset = off_raw if off_raw == "auto_gp" else float(off_raw)
            p_hot = num("p_hot", 0.4)
            p_cold = num("p_cold", 0.02)
            gamma = num("gamma", 0.05)
            seed_raw = str(P.get("seed", "")).strip()
            seed = float(seed_raw) if seed_raw else None
            gram_threshold = num("gram_threshold", 0.0)
            iterations_lloyd = num("iterations_lloyd", 20, int)
            lloyd_mc = max(1, num("lloyd_mc", 1, int))
            solver = "reference" if P.get("solver") == "referenz" else "fast"
            cooling_schedule = ("da_gp_floor"
                                if P.get("cooling_schedule") == "floor"
                                else "da_gp")
            cooling_per_group = bool(P.get("cooling_per_group", True))
            split_mode = str(P.get("split_mode", "plane"))
            spill = max(0.0, num("spill", 0.1))
            # ε darf auch "auto" (am Encoder messen) oder "cos:0.9" sein —
            # als Text durchreichen, der Indexer löst es auf.
            eps = (str(P.get("eps", "0")).strip() or "0")
            if eps.lower() != "auto" and not eps.lower().startswith("cos:"):
                eps = str(max(0.0, float(eps)))
            anneal = bool(P.get("anneal", True))
            hold_raw = str(P.get("hold_steps", "")).strip()
            hold_steps = max(0, int(hold_raw)) if hold_raw else None
            offset_k_escape = num("offset_k_escape", 5.0)
            visualize_first = bool(P.get("visualize_first", True))
            save_stats = bool(P.get("save_stats", True))
        except ValueError as exc:
            self.set_status(f"✗ {exc}")
            return

        if not anneal and iterations_lloyd <= 0:
            self.set_status("✗ Lloyd und DA beide aus — dann gibt es keinen "
                            "Split. Mindestens einer muss laufen.")
            return

        self._indexing = True
        self._active_thr = gram_threshold
        self._last_run_dir = run_dir
        self.btn_query.configure(state="disabled")
        self.progress.set(0)
        self.lbl_index.configure(
            text=f"⏳ Indexing läuft: {run_dir.name} (g={g}) …",
            text_color="#e0a840")
        self.cloud.caption = "DA-Partitionierung läuft — jede Farbe = eine Gruppe"
        self.cloud.legend = []
        self.cloud.on_click = None
        self._show_dalog(clear=True)
        self._append_dalog(f"▶ Indexing {run_dir.name}  (g={g}, "
                           f"steps={steps if steps is not None else 'auto'}, "
                           f"MC={num_mc}, thr={gram_threshold:g}, "
                           f"lloyd={iterations_lloyd}×{lloyd_mc}, "
                           f"cooling={cooling_schedule}"
                           f"{'/Start' if cooling_per_group else ''}, "
                           f"hold={hold_steps if hold_steps is not None else 'auto'}, "
                           f"kEsc={offset_k_escape:g}, "
                           f"split={split_mode}"
                           f"{f'+spill{spill:g}' if split_mode != 'none' and spill else ''}"
                           f"{f'+ε{eps}' if eps not in ('0', '0.0') else ''}, "
                           f"solver={solver if anneal else 'lloyd-only'})")
        self._replaying = False
        for w in (self.map_menu, self.color_menu, self.phase_slider,
                  self.btn_replay, self.btn_reindex,
                  self.index_menu):
            w.configure(state="disabled")
        self.map_menu.set("PCA")

        def worker() -> None:
            # Das Indexing läuft als SUBPROZESS: DA-Solver + PBF-Bau sind
            # reines Python und würden im Thread den GIL monopolisieren
            # (UI friert ein); Matplotlib (Visualize first) braucht auf
            # macOS zudem einen Main-Thread — im Subprozess hat es einen.
            try:
                from rag_indexer import load_run_folder
                _specs, _chunks, _ids, E = load_run_folder(run_dir)
                xy, _basis = pca_2d(E)
                self._points_done = 0
                self._total_points = len(E)
                self._post(lambda: self.cloud.set_points(xy, color="gray40"))

                here = Path(__file__).resolve().parent
                # -u: unbuffered — sonst hängen die Solver-Prints (MC-Trials)
                # im Pipe-Puffer und das DA-Log bleibt minutenlang leer
                cmd = [sys.executable, "-u", str(here / "rag_indexer.py"),
                       str(run_dir), "-g", str(g),
                       *(["--steps", str(steps)] if steps is not None else []),
                       "--num-mc", str(num_mc),
                       "--offset-increase", str(offset),
                       "--p-hot", str(p_hot),
                       "--p-cold", str(p_cold),
                       "--gamma", str(gamma),
                       "--gram-threshold", str(gram_threshold),
                       "--iterations-lloyd", str(iterations_lloyd),
                       "--lloyd-mc", str(lloyd_mc),
                       "--cooling-schedule", cooling_schedule,
                       "--cooling-per-group" if cooling_per_group
                       else "--no-cooling-per-group",
                       "--split-mode", split_mode,
                       "--spill", str(spill),
                       "--eps", str(eps),
                       "--anneal" if anneal else "--no-anneal",
                       *(["--hold-steps", str(hold_steps)]
                         if hold_steps is not None else []),
                       "--offset-k-escape", str(offset_k_escape),
                       "--solver", solver,
                       "--visualize-first" if visualize_first
                       else "--no-visualize-first",
                       "--stats" if save_stats else "--no-stats",
                       "--progress-json", "--quiet"]
                if seed is not None:
                    cmd += ["--seed", str(seed)]

                proc = subprocess.Popen(
                    cmd, cwd=str(here), text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self._index_proc = proc
                tail: list[str] = []
                out_dir: Path | None = None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if line.startswith("@@"):
                        evt = json.loads(line[2:])
                        ev = evt.pop("event")
                        if ev == "done":
                            out_dir = Path(evt["out_dir"])
                        else:
                            self._post(lambda e=ev, i=evt:
                                       self._progress_event(e, i))
                    elif line.strip():
                        tail.append(line)
                        del tail[:-30]
                        self._post(lambda l=line: self._append_dalog(l))
                rc = proc.wait()
                self._index_proc = None
                if rc == 0 and out_dir is not None:
                    final = out_dir
                    self._post(lambda: self._indexing_done(final))
                else:
                    msg = " · ".join(tail[-4:]) or f"Indexer-Exit {rc}"
                    self._post(lambda: self._indexing_failed(msg))
            except Exception as exc:            # noqa: BLE001
                msg = str(exc)
                self._index_proc = None
                self._post(lambda: self._indexing_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    # ── DA-Terminal-Log (linke Spalte während des Indexings) ─────────────────
    def _show_dalog(self, clear: bool = False) -> None:
        if clear:
            self.txt_dalog.configure(state="normal")
            self.txt_dalog.delete("1.0", "end")
            self.txt_dalog.configure(state="disabled")
        self.frame_results.grid_remove()
        self.txt_dalog.grid()

    def _show_results_panel(self) -> None:
        self.txt_dalog.grid_remove()
        self.frame_results.grid()

    def _log_at_bottom(self) -> bool:
        """Klebt der Blick am Ende des Logs? (Terminal-Verhalten)"""
        try:
            return self.txt_dalog.yview()[1] >= 0.999
        except Exception:                            # noqa: BLE001
            return True

    def _append_dalog(self, line: str) -> None:
        # Wie ein Terminal: nur mitscrollen, wenn der Blick ohnehin unten
        # klebt.  Wer hochgescrollt hat, um etwas zu lesen, bleibt stehen —
        # sonst reißt es einem den Text während des Indexings weg.
        follow = self._log_at_bottom()
        self.txt_dalog.configure(state="normal")
        self.txt_dalog.insert("end", line + "\n")
        # begrenzen, damit stundenlange Läufe die UI nicht vollschreiben;
        # aber NICHT während jemand oben liest (das Kürzen oben würde die
        # Sicht verschieben) — nachgeholt, sobald er wieder unten ist
        if follow:
            n_lines = int(self.txt_dalog.index("end-1c").split(".")[0])
            if n_lines > 3000:
                self.txt_dalog.delete("1.0", f"{n_lines - 2000}.0")
            self.txt_dalog.see("end")
        self.txt_dalog.configure(state="disabled")

    def _progress_event(self, event: str, info: dict) -> None:
        if event == "bisect_start":
            n = info["n"]
            est = n * (n + 1) // 2 + n      # symmetrisch zusammengelegtes PBF
            if self._active_thr > 0:
                qubo = f"QUBO sparse ≤ {est:,} Terme (thr={self._active_thr:g})"
            else:
                qubo = f"QUBO {est:,} Terme"
            self.set_status(
                f"DA-Bisection Tiefe {info['depth']} "
                f"(Pfad '{info['path'] or 'root'}'): {n} Chunks = "
                f"{n} Annealing-Variablen, {qubo} …")
        elif event == "pbf_built":
            zf = info.get("gram_zeroed_frac", 0)
            self.set_status(
                f"PBF Tiefe {info['depth']} fertig: {info['pbf_terms']:,} Terme"
                + (f" ({zf:.0%} verworfen)" if zf else "")
                + " — DA-Solve läuft …")
        elif event == "lloyd_done":
            # lloyd_mc > 1: PCA-orientierte Multi-Starts — wie viele davon
            # in verschiedenen Becken landeten, sagt der unique-Zähler
            starts = ""
            if info.get("lloyd_mc", 1) > 1:
                starts = (f", Starts {info['lloyd_mc']}→"
                          f"{info.get('lloyd_unique', '?')} unique")
            self.set_status(
                f"Lloyd Tiefe {info['depth']}: {info['lloyd_iters']} it "
                f"({'konvergiert' if info['lloyd_converged'] else 'Limit'}), "
                f"Split {info.get('lloyd_n_A', '?')}/{info.get('lloyd_n_B', '?')}"
                f"{starts}, "
                f"E {info['E_start']:.1f} → {info['E_lloyd']:.1f} "
                f"— DA-Solve läuft …")
        elif event == "split":
            self.cloud.color_rows(info["idx_A"], path_color(info["path"] + "0"))
            self.cloud.color_rows(info["idx_B"], path_color(info["path"] + "1"))
            extra = ""
            if "pbf_terms" in info:
                extra = f", PBF {info['pbf_terms']:,} Terme"
                if info.get("gram_zeroed_frac"):
                    extra += f" ({info['gram_zeroed_frac']:.0%} verworfen)"
            self.set_status(
                f"Split Tiefe {info['depth']}: {len(info['idx_A'])} / "
                f"{len(info['idx_B'])} Punkte  ({info['seconds']:.2f}s DA{extra})")
        elif event == "leaf":
            self.cloud.color_rows(info["indices"], path_color(info["path"]))
            self._points_done += len(info["indices"])
            self.progress.set(self._points_done / max(self._total_points, 1))

    def _indexing_done(self, out_dir: Path) -> None:
        self._indexing = False
        self._append_dalog(f"✓ Indexing fertig → {out_dir}")
        self.set_status(f"✓ Indexing fertig → {out_dir}")
        self._replay_after_load = True     # Splitting einmal in Ruhe zeigen
        self.load_index(out_dir)

    def _indexing_failed(self, msg: str) -> None:
        self._indexing = False
        self._append_dalog(f"✗ Abbruch: {msg}")
        self.progress.set(0)
        self.lbl_index.configure(text=f"✗ Indexing fehlgeschlagen: {msg}",
                                 text_color="#e06060")
        # Run-Ordner wieder vormerken → Parameter anpassen und erneut ▶
        if self._last_run_dir is not None:
            self._pending_run = self._last_run_dir
            self.btn_reindex.configure(text="▶ Indexieren", state="normal")
        self._populate_index_menu(self.index_dir)
        self.set_status(f"✗ {msg}", error=True)

    # ── Provider-Umschaltung (Ollama-URL ↔ Gemini-Key im selben Feld) ─────────
    def _apply_provider(self, provider: str) -> None:
        new = "gemini" if provider.lower() in GEMINI_PROVIDERS else "ollama"
        if new == self._provider:
            return
        # Feldinhalt der bisherigen Rolle sichern
        if self._provider == "gemini":
            self._gemini_key = self.entry_url.get().strip()
        else:
            self._ollama_url = self.entry_url.get().strip() or DEFAULT_OLLAMA
        self._provider = new
        self.entry_url.delete(0, "end")
        if new == "gemini":
            self.lbl_conn.configure(text="Gemini-Key:")
            self.entry_url.configure(show="•",
                                     placeholder_text="API-Key ($GEMINI_API_KEY)")
            if self._gemini_key:
                self.entry_url.insert(0, self._gemini_key)
        else:
            self.lbl_conn.configure(text="Ollama:")
            self.entry_url.configure(show="", placeholder_text="")
            self.entry_url.insert(0, self._ollama_url)

    # ── Backend-Status (Ollama oder Gemini API) ───────────────────────────────
    def check_ollama(self) -> None:
        conn = self.entry_url.get().strip()
        provider = self._provider

        def worker() -> None:
            try:
                if provider == "gemini":
                    key = conn or env_gemini_key()
                    if not key:
                        raise RuntimeError("Gemini-API-Key fehlt")
                    models = gemini_models(key)
                else:
                    models = ollama_models(conn or DEFAULT_OLLAMA)
                ok, detail = True, models
            except Exception as exc:            # noqa: BLE001
                ok, detail = False, str(exc)
            self._post(lambda: self._ollama_result(ok, detail))

        threading.Thread(target=worker, daemon=True).start()

    def _ollama_result(self, ok: bool, detail) -> None:
        if not ok:
            self.lbl_ollama.configure(text="● offline", text_color="#e06060")
            return
        if self.meta:
            want = self.meta["encoder"]["model"]
            have = any(m.split(":")[0] == want.split(":")[0] for m in detail)
            if have:
                self.lbl_ollama.configure(text="● bereit", text_color="#7ec97e")
            else:
                self.lbl_ollama.configure(text=f"● {want} fehlt!",
                                          text_color="#e0a840")
        else:
            self.lbl_ollama.configure(text="● läuft", text_color="#7ec97e")

    # ── Query ─────────────────────────────────────────────────────────────────
    def start_query(self) -> None:
        if self.index_dir is None or self.meta is None or self._indexing:
            return
        query = self.entry_query.get().strip()
        if not query:
            self.set_status("Leere Query.")
            return
        try:
            k = max(1, int(self.entry_k.get()))
            beam = max(1, int(self.entry_beam.get()))
            mc_raw = self.entry_mincand.get().strip()
            min_cand = int(mc_raw) if mc_raw else None
        except ValueError:
            self.set_status("k / Beam / min. Kandidaten müssen Zahlen sein.")
            return

        conn = self.entry_url.get().strip()
        provider = self.meta["encoder"]["provider"]
        model = self.meta["encoder"]["model"]
        dim = self.meta["encoder"]["dim"]
        is_gemini = provider.lower() in GEMINI_PROVIDERS
        index_dir = self.index_dir

        self.btn_query.configure(state="disabled", text="Suche …")
        self.set_status(f"Embedde Query mit {model} ({provider}) …")

        def worker() -> None:
            try:
                t0 = time.time()
                emb = embed_query(provider, conn, model, query,
                                  output_dim=dim if is_gemini else None)
                t_embed = (time.time() - t0) * 1000

                t0 = time.time()
                data = run_query_script(index_dir, emb, k, beam, min_cand)
                t_query = (time.time() - t0) * 1000

                self._post(lambda: self._show_results(
                    data, np.asarray(emb), t_embed, t_query))
            except Exception as exc:            # noqa: BLE001
                msg = str(exc)
                self._post(lambda: self._query_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _query_failed(self, msg: str) -> None:
        self.btn_query.configure(state="normal", text="Suchen")
        self.set_status(f"✗ {msg}", error=True)

    def _show_results(self, data: dict, emb: np.ndarray,
                      t_embed: float, t_query: float) -> None:
        self.btn_query.configure(state="normal", text="Suchen")
        self._show_results_panel()          # DA-Log ggf. wieder ausblenden
        self.results = data["results"]

        for btn in self._result_buttons:
            btn.destroy()
        self._result_buttons.clear()

        for i, r in enumerate(self.results):
            head = " › ".join(r["heading_path"] or []) or Path(r["path"]).stem
            label = (f"#{r['rank']}   {r['score']:.4f}   {r['source']}\n"
                     f"      {head}")
            btn = ctk.CTkButton(
                self.frame_results, text=label, anchor="w",
                fg_color="transparent", border_width=1, border_color="gray40",
                font=ctk.CTkFont(size=12),
                command=lambda idx=i: self.show_chunk(idx),
            )
            btn.grid(row=i, column=0, sticky="ew", padx=4, pady=3)
            self._result_buttons.append(btn)

        # Karte: durchsuchte Kandidaten hell, Rest gedimmt, Treffer grau + Query
        self._last_emb = emb
        if self.manifest is not None and self._embeddings is not None:
            hits = [(self._row_of[r["id"]], r["rank"]) for r in self.results
                    if r["id"] in self._row_of]
            self._qview = (data.get("candidate_rows"), hits)
            xy = self._maps[self._map_name]
            self.cloud.caption = (self._map_caption()
                                  + " · Hell = durchsucht · Grau+Ring = Treffer"
                                  + " · Leer-Klick/Esc = zurück")
            self.cloud.set_query_view(
                data.get("candidate_rows"), hits,
                self._query_pos(emb, self._map_name, xy))

        self.set_status(
            f"✓ {len(self.results)} Chunks  ·  Embedding {t_embed:.0f} ms  ·  "
            f"Query {t_query:.0f} ms (Subprozess)  ·  "
            f"Beam {data['beam']}, {data['candidates_ranked']} Kandidaten exakt gerankt")
        if self.results:
            self.show_chunk(0)

    def _fill_chunk_view(self, header: str, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            text = f"(Chunk-Datei nicht lesbar: {exc})"
        self.lbl_chunk_head.configure(text=header)
        self.txt_chunk.configure(state="normal")
        self.txt_chunk.delete("1.0", "end")
        self.txt_chunk.insert("1.0", f"{path}\n{'─' * 80}\n\n{text}")
        self.txt_chunk.configure(state="disabled")

    def show_chunk(self, idx: int) -> None:
        r = self.results[idx]
        head = " › ".join(r["heading_path"] or [])
        self._fill_chunk_view(
            f"#{r['rank']}  ·  Score {r['score']:.4f}  ·  {r['source']}"
            + (f"  ·  {head}" if head else ""),
            Path(r["path"]))

        # Auswahl in der Karte spiegeln
        if self._row_of and r["id"] in self._row_of:
            self.cloud.set_selected(self._row_of[r["id"]])

        for i, btn in enumerate(self._result_buttons):
            btn.configure(border_color="#3a7ebf" if i == idx else "gray40")

    def _map_click(self, row: int | None) -> None:
        """Klick auf einen Punkt der Wolke — öffnet den Chunk dahinter.

        Getroffene (graue) Punkte sind Teil der Ergebnisliste → normale
        Auswahl inkl. Listen-Highlight; alle anderen werden direkt aus dem
        Manifest geöffnet, mit Cosine-Score zur letzten Query falls vorhanden.
        Klick ins Leere (row=None): erst Auswahlring weg, beim zweiten Mal
        die Query-Ansicht zurücksetzen → wieder die komplette Wolke.
        """
        if self.manifest is None or self.index_dir is None:
            return
        if row is None:
            if self.cloud.selected is not None:
                self.cloud.set_selected(None)
                for btn in self._result_buttons:
                    btn.configure(border_color="gray40")
            elif self._qview is not None:
                self._reset_query_view()
            return
        for i, r in enumerate(self.results):
            if self._row_of.get(r["id"]) == row:
                self.show_chunk(i)
                return

        cid = self.manifest["row_to_id"][row]
        info = self.manifest["chunks"][cid]
        head = " › ".join(info.get("heading_path") or [])
        score_txt = ""
        if self._last_emb is not None and self._embeddings is not None:
            e, q = self._embeddings[row], self._last_emb
            denom = float(np.linalg.norm(q) * np.linalg.norm(e)) or 1.0
            score_txt = f"Score {float(q @ e) / denom:.4f}  ·  "
        self._fill_chunk_view(
            f"○ kein Treffer  ·  {score_txt}{info['source']}"
            + (f"  ·  {head}" if head else ""),
            self.index_dir / info["path"])

        self.cloud.set_selected(row)
        for btn in self._result_buttons:
            btn.configure(border_color="gray40")

    def _reset_query_view(self) -> None:
        """Query-Markierungen (Dimmen, Treffer, ×) + Auswahl entfernen —
        die Wolke zeigt wieder alle Punkte in voller Farbe."""
        if self._embeddings is None:
            return
        self._qview = None
        self.cloud.set_query_view(None, [], None)
        for btn in self._result_buttons:
            btn.configure(border_color="gray40")
        self._recolor()
        self.set_status("Ansicht zurückgesetzt — komplette Wolke.")

    # ── Status ────────────────────────────────────────────────────────────────
    def set_status(self, msg: str, error: bool = False) -> None:
        self.lbl_status.configure(text=msg,
                                  text_color="#e06060" if error else "gray70")


if __name__ == "__main__":
    app = RagTesterApp()
    app.mainloop()
