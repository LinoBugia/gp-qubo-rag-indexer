#!/usr/bin/env python3
"""
Erzeugt die Grafiken fuer README und docs/ als SVG.

    python3 docs/img/make_figures.py

Die Punktwolken und Trennlinien sind GERECHNET, nicht gemalt: dieselbe Logik
wie in graph_partitioning.py (Richtung = Differenz der beiden Zentroide, τ in
der Mitte einer Lücke der Projektionen, ε-Zone um die Ebene), nur in 2-D und
ohne die Kugel-Normierung — die würde eine Gerade im Bild zu einem Kegel
machen und nichts erklären.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent        # docs/img — hier liegen die SVGs
REPO = OUT.parent.parent

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#ffffff"
INK = "#0f172a"
MUTED = "#64748b"
FAINT = "#cbd5e1"
BLUE = "#2563eb"
RED = "#e11d48"
GREEN = "#059669"
AMBER = "#d97706"
VIOLET = "#7c3aed"
DA_INK = "#4f46e5"          # gehört zum Annealer
DA_BG = "#eef2ff"
FONT = "ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def head(w: int, h: int) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">',
            f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>']


def txt(x, y, s, size=13, fill=INK, anchor="start", weight="400", font=FONT,
        opacity=1.0) -> str:
    s = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{font}" opacity="{opacity}">{s}</text>')


def dot(x, y, r=3.1, fill=MUTED, opacity=1.0) -> str:
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" '
            f'opacity="{opacity}"/>')


def split_dot(x, y, r=4.0, left=None, right=None, opacity=1.0) -> str:
    """Halb links / halb rechts gefärbt — ein Punkt, der in BEIDE Kinder geht."""
    left, right = left or BLUE, right or RED
    return (f'<path d="M {x:.1f} {y - r:.1f} A {r} {r} 0 0 0 {x:.1f} '
            f'{y + r:.1f} Z" fill="{left}" opacity="{opacity}"/>'
            f'<path d="M {x:.1f} {y - r:.1f} A {r} {r} 0 0 1 {x:.1f} '
            f'{y + r:.1f} Z" fill="{right}" opacity="{opacity}"/>')


def line(x1, y1, x2, y2, stroke=INK, w=1.4, dash=None, opacity=1.0) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d} opacity="{opacity}" '
            f'stroke-linecap="round"/>')


def box(x, y, w, h, fill="none", stroke=FAINT, sw=1.2, rx=8, dash=None,
        opacity=1.0) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
            f'{d} opacity="{opacity}"/>')


def badge(x, y, label="DA", col=DA_INK, size=9.5) -> str:
    """Kleines Etikett: dieser Schritt existiert nur für den Annealer."""
    w = len(label) * size * 0.72 + 12
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="15" '
            f'rx="7.5" fill="{col}" opacity="0.13"/>'
            + txt(x + w / 2, y + 11, label, size, col, anchor="middle",
                  weight="700", font=MONO))


# ── Geometrie: dieselben Regeln wie im Indexer, in 2-D ───────────────────────

def two_means(P: np.ndarray, iters: int = 40, seed: int = 0) -> np.ndarray:
    """Lloyd k=2, deterministisch gestartet an den PC1-Vorzeichen."""
    C = P - P.mean(axis=0)
    u = np.linalg.svd(C, full_matrices=False)[2][0]
    a = (C @ u > 0).astype(int)
    for _ in range(iters):
        if a.min() == a.max():
            a[int(np.argmax(C @ u))] = 1 - a[0]
        m0, m1 = P[a == 0].mean(axis=0), P[a == 1].mean(axis=0)
        new = (((P - m1) ** 2).sum(1) < ((P - m0) ** 2).sum(1)).astype(int)
        if np.array_equal(new, a):
            break
        a = new
    return a


def plane_of(P: np.ndarray, a: np.ndarray, *, shift: bool, eps: float = 0.0,
             q: float = 0.2):
    """(w, tau) nach der Regel aus plane_from_split: Richtung aus den
    Zentroiden, τ in der Lücke mit den wenigsten unsicheren Punkten."""
    mA, mB = P[a == 0].mean(axis=0), P[a == 1].mean(axis=0)
    w = mA - mB
    w = w / np.linalg.norm(w)
    t = P @ w
    tau0 = 0.5 * (mA + mB) @ w              # "centroid": Mitte zwischen beiden
    if not shift:
        return w, float(tau0)
    ts = np.sort(t)
    n = len(ts)
    lo, hi = max(1, int(q * n)), min(n - 1, int((1 - q) * n))
    if hi <= lo:
        return w, float(tau0)
    gaps = ts[lo + 1:hi + 1] - ts[lo:hi]
    mids = 0.5 * (ts[lo:hi] + ts[lo + 1:hi + 1])
    if eps > 0:
        occ = (np.searchsorted(ts, mids + eps, side="right")
               - np.searchsorted(ts, mids - eps, side="left"))
        k = int(np.lexsort((-gaps, occ))[0])
    else:
        k = int(np.argmax(gaps))
    return w, float(mids[k])


def cloud(n: int = 150, seed: int = 11) -> np.ndarray:
    """Fünf Gruppen unterschiedlicher Dichte — nicht kugelrund, nicht gleich groß."""
    rng = np.random.default_rng(seed)
    specs = [((0.26, 0.70), 0.085, 34), ((0.34, 0.36), 0.070, 30),
             ((0.66, 0.72), 0.075, 32), ((0.78, 0.33), 0.090, 30),
             ((0.52, 0.53), 0.055, 24)]
    pts = []
    for (cx, cy), s, k in specs:
        p = rng.normal([cx, cy], [s, s * 0.78], size=(k, 2))
        th = rng.uniform(0, math.pi)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        pts.append((p - [cx, cy]) @ R.T + [cx, cy])
    P = np.vstack(pts)[:n]
    return np.clip(P, 0.04, 0.96)


def on_line(w, tau, n=500):
    """Punkte der Geraden {p·w = tau} innerhalb des Panels [0,1]²."""
    pts = []
    for t in np.linspace(-0.8, 1.8, n):
        if abs(w[1]) > abs(w[0]):
            p = np.array([t, (tau - w[0] * t) / w[1]])
        else:
            p = np.array([(tau - w[1] * t) / w[0], t])
        if -0.01 <= p[0] <= 1.01 and -0.01 <= p[1] <= 1.01:
            pts.append(p)
    return pts


def clip_line(w, tau, x0, y0, side):
    """Ganze Gerade als ein Segment (Pixelkoordinaten) oder None."""
    pts = on_line(w, tau)
    if len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    return (x0 + a[0] * side, y0 + (1 - a[1]) * side,
            x0 + b[0] * side, y0 + (1 - b[1]) * side)


def seg_in_cell(wv, tau, cons, x0, y0, side, **kw) -> list[str]:
    """
    Die Gerade nur dort zeichnen, wo sie wirklich trennt: innerhalb der
    Zelle, die ihre Elternebenen aufspannen (cons = [(w, tau, sign), …]).
    Das ist die Bisektion — eine Ebene der Tiefe d gilt nur in ihrem Teil
    des Raums, nicht über das ganze Bild.
    """
    pts = on_line(wv, tau)
    runs, cur = [], []
    for p in pts:
        ok = all((float(p @ wc) - tc) * sg > 0 for wc, tc, sg in cons)
        if ok:
            cur.append(p)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < 2:
            continue
        a, b = r[0], r[-1]
        out.append(line(x0 + a[0] * side, y0 + (1 - a[1]) * side,
                        x0 + b[0] * side, y0 + (1 - b[1]) * side, **kw))
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  1 — Das Konzept: rekursive Bisektion
# ═════════════════════════════════════════════════════════════════════════════

def fig_concept() -> None:
    P = cloud()
    W, side, pad, gap = 980, 196, 26, 22
    H = 552
    s = head(W, H)
    s.append(txt(pad, 30, "Recursive bisection", 16, INK, weight="650"))
    s.append(txt(pad, 50, "Each level splits every group in two — the tree of "
                          "split planes IS the index.", 12.5, MUTED))

    # Rekursiv splitten; jede Ebene merkt sich die Zelle, in der sie gilt
    cells = [(np.ones(len(P), bool), [])]       # (Maske, Constraints)
    levels = [np.zeros(len(P), int)]
    planes: list[list] = [[]]
    for _ in range(3):
        nxt, pl, lab = [], list(planes[-1]), np.zeros(len(P), int)
        for gid, (m, cons) in enumerate(cells):
            idx = np.where(m)[0]
            if m.sum() < 8:
                lab[idx] = gid * 2
                nxt.append((m, cons))
                nxt.append((np.zeros(len(P), bool), cons))
                continue
            a = two_means(P[m])
            w, tau = plane_of(P[m], a, shift=True)
            pl.append((w, tau, cons))
            side0 = P @ w > tau
            # Farbe MUSS von der Ebene kommen, nicht von der
            # k-Means-Zuordnung: die Ebene ist ja gegenüber den Zentroiden
            # verschoben, sonst lägen Punkte sichtbar auf der falschen Seite
            # ihrer eigenen Trennlinie.
            lab[idx] = gid * 2 + (~side0[idx]).astype(int)
            nxt.append((m & side0, cons + [(w, tau, +1)]))
            nxt.append((m & ~side0, cons + [(w, tau, -1)]))
        cells, _ = nxt, None
        levels.append(lab)
        planes.append(pl)

    pal = [BLUE, RED, GREEN, AMBER, VIOLET, "#0891b2", "#be185d", "#4d7c0f"]
    y0 = 68
    for i, (lab, pls) in enumerate(zip(levels, planes)):
        x0 = pad + i * (side + gap)
        s.append(box(x0, y0, side, side, fill="#fbfcfe"))
        for (w, tau, cons) in pls:
            s.extend(seg_in_cell(w, tau, cons, x0, y0, side,
                                 stroke=INK, w=1.5, opacity=0.8))
        for (px, py), g in zip(P, lab):
            s.append(dot(x0 + px * side, y0 + (1 - py) * side, 2.9,
                         MUTED if i == 0 else pal[g % len(pal)],
                         0.55 if i == 0 else 0.92))
        n_groups = 1 if i == 0 else len(np.unique(lab))
        s.append(txt(x0, y0 + side + 19,
                     f"depth {i}" if i else "the point cloud", 12.5, INK,
                     weight="600"))
        s.append(txt(x0 + side, y0 + side + 19,
                     f"{n_groups} group" + ("s" if n_groups > 1 else ""),
                     12, MUTED, anchor="end"))

    # ── Der Abstieg: links der Baum, rechts die Bedingung je Ebene ──────────
    #  Der Baum zeigt die Struktur, die Kette rechts zeigt, was pro Ebene
    #  tatsächlich gerechnet wird — eine einzige Ja/Nein-Frage.
    ty = y0 + side + 50
    s.append(txt(pad, ty, "The query walks the same structure — one if per "
                          "level, and the branch not taken is never touched:",
                 12.5, MUTED))

    bx, by, bw, lvl_h = pad + 6, ty + 36, 520, 46
    path = (0, 1, 3, 6)                     # der Weg, den eine Query nimmt
    for d in range(4):
        k = 2 ** d
        for j in range(k):
            cx = bx + bw * (j + 0.5) / k
            cy = by + d * lvl_h
            if d < 3:
                for c in (2 * j, 2 * j + 1):
                    hot = j == path[d] and c == path[d + 1]
                    s.append(line(cx, cy + 5, bx + bw * (c + 0.5) / (2 * k),
                                  by + (d + 1) * lvl_h - 5,
                                  VIOLET if hot else FAINT,
                                  2.2 if hot else 1.2))
            on = j == path[d]
            s.append(dot(cx, cy, 5.6 if on else 4,
                         VIOLET if on else FAINT, 1.0))

    # Rechte Spalte: je Ebene die Bedingung, die dort ausgewertet wird
    cx0, cw, ch = bx + bw + 74, 286, 34
    conds = [("q·ŵ₀ ≥ τ₀", "yes"), ("q·ŵ₁ ≥ τ₁", "no"),
             ("q·ŵ₂ ≥ τ₂", "yes")]
    for d, (cond, answer) in enumerate(conds):
        cy = by + d * lvl_h
        hot_x = bx + bw * (path[d] + 0.5) / (2 ** d)
        s.append(line(hot_x + 10, cy, cx0 - 8, cy, FAINT, 1.2, "3 3"))
        s.append(box(cx0, cy - ch / 2, cw, ch, fill=BG, stroke=VIOLET, sw=1.5))
        s.append(txt(cx0 + 14, cy + 5, cond, 13, INK, weight="650", font=MONO))
        s.append(txt(cx0 + cw - 14, cy + 5, answer, 11.5, VIOLET, anchor="end",
                     weight="700", font=MONO))
        s.append(txt(cx0 + cw / 2 + 24, cy + 5, "→", 12, MUTED,
                     anchor="middle"))
    cy = by + 3 * lvl_h
    hot_x = bx + bw * (path[3] + 0.5) / 8
    s.append(line(hot_x + 10, cy, cx0 - 8, cy, FAINT, 1.2, "3 3"))
    s.append(box(cx0, cy - ch / 2, cw, ch, fill="#f5f3ff", stroke=VIOLET,
                 sw=1.8))
    s.append(txt(cx0 + 14, cy + 5, "leaf — ≤ g chunks", 12.5, VIOLET,
                 weight="700"))
    s.append(txt(cx0 + cw - 14, cy + 5, "rank exactly", 11, MUTED,
                 anchor="end"))
    s.append(txt(cx0, by + 3 * lvl_h + 38,
                 "one dot product per level · the other half of the tree is "
                 "never opened", 11.5, MUTED))
    s.append("</svg>")
    (OUT / "concept-bisection.svg").write_text("\n".join(s), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  2 — Die Trennebene: none / centroid / plane / ε
# ═════════════════════════════════════════════════════════════════════════════

def fig_split() -> None:
    # Zwei ungleiche Gruppen (eine dicht, eine locker) plus ein paar Punkte
    # dazwischen — so liegt die breiteste Lücke NICHT in der Mitte, und der
    # Unterschied zwischen "centroid" und "plane" ist überhaupt sichtbar.
    rng = np.random.default_rng(12)
    A = rng.normal([0.26, 0.54], [0.062, 0.150], size=(48, 2))
    B = rng.normal([0.77, 0.47], [0.070, 0.165], size=(34, 2))
    mid = np.c_[rng.uniform(0.46, 0.56, 4), rng.uniform(0.15, 0.88, 4)]
    P = np.clip(np.vstack([A, B, mid]), 0.05, 0.95)
    a = np.r_[np.zeros(48, int), np.ones(34, int),
              (mid[:, 0] > 0.50).astype(int)]
    # Was der rohe DA-Split stehen lässt: Punkte, die in der QUBO-korrekten
    # Gruppe liegen, aber näher am anderen Zentroid sitzen.
    stray = [np.argmax(A[:, 0]), 48 + int(np.argmin(B[:, 0])),
             int(np.argmax(A[:, 0] - 0.3 * A[:, 1]))]
    for i in stray:
        a[i] = 1 - a[i]

    W, side, pad, gap = 980, 205, 22, 26
    H = 400
    s = head(W, H)
    s.append(txt(pad, 28, "The post-step on the split", 16, INK, weight="650"))
    s.append(txt(pad, 47, "The direction is the same in all four — what changes "
                          "is where the boundary sits, and how much overlap "
                          "it gets.",
                 12.5, MUTED))

    eps = 0.10
    panels = [
        ("none", "raw split — strays stay lost", None, False, 0.0),
        ("centroid", "every point joins the nearer centroid", "c", False, 0.0),
        ("plane", "shifted into the widest gap", "p", True, 0.0),
        ("plane + spill", "works on centroid just as well", "e", True, eps),
    ]
    y0 = 92
    # Die drei --split-mode-Werte gehören zusammen; spill steht daneben,
    # weil es KEIN vierter Modus ist, sondern ein Sicherheitsnetz obendrauf.
    grp_w = 3 * side + 2 * gap
    s.append(txt(pad, y0 - 30, "--split-mode  —  pick exactly one", 12.5, INK,
                 weight="650", font=MONO))
    s.append(line(pad, y0 - 20, pad + grp_w, y0 - 20, FAINT, 1.2))
    sx = pad + grp_w + gap
    s.append(txt(sx, y0 - 30, "--spill  —  safety net on top", 12.5, VIOLET,
                 weight="650", font=MONO))
    s.append(line(sx, y0 - 20, sx + side, y0 - 20, VIOLET, 1.2, opacity=0.5))
    s.append(line(sx - gap / 2, y0 - 36, sx - gap / 2, y0 + side + 44,
                  FAINT, 1.2, dash="4 4"))

    for i, (name, sub, kind, shift, e) in enumerate(panels):
        x0 = pad + i * (side + gap)
        s.append(box(x0, y0, side, side, fill="#fbfcfe"))
        # alles Weitere innerhalb des Panels halten (Bänder sind breit)
        s.append(f'<clipPath id="c{i}"><rect x="{x0}" y="{y0}" '
                 f'width="{side}" height="{side}" rx="8"/></clipPath>')
        s.append(f'<g clip-path="url(#c{i})">')

        if kind is None:
            lab = a
        else:
            w, tau = plane_of(P, a, shift=shift, eps=e)
            lab = (P @ w <= tau).astype(int)
            c = clip_line(w, tau, x0, y0, side)
            # Marge und ε als BAND zeichnen (Linienbreite = echte Breite in
            # Pixeln) — gestrichelte Ränder verschwinden bei kleiner Marge.
            if c and e > 0:
                s.append(line(*c, stroke=VIOLET, w=2 * e * side, opacity=0.18))
            if c and kind == "p":
                m = float(np.min(np.abs(P @ w - tau)))
                s.append(line(*c, stroke=GREEN, w=max(2 * m * side, 2.5),
                              opacity=0.3))
            if kind == "c":
                # Was "centroid" tut, sichtbar gemacht: jeder Punkt hängt an
                # dem Zentroid, der ihm näher ist; der Kreis um jedes Zentroid
                # geht bis zu dessen entferntestem Punkt.
                for g in (0, 1):
                    m = lab == g
                    ctr = P[m].mean(axis=0)
                    ccx, ccy = x0 + ctr[0] * side, y0 + (1 - ctr[1]) * side
                    col = BLUE if g == 0 else RED
                    for px, py in P[m]:
                        s.append(line(ccx, ccy, x0 + px * side,
                                      y0 + (1 - py) * side, col, 0.6,
                                      "1 2", 0.35))
                    rad = float(np.max(np.linalg.norm(P[m] - ctr, axis=1)))
                    s.append(f'<circle cx="{ccx:.1f}" cy="{ccy:.1f}" '
                             f'r="{rad * side:.1f}" fill="none" stroke="{col}" '
                             f'stroke-width="1.1" stroke-dasharray="4 3" '
                             f'opacity="0.55"/>')
                    s.append(f'<circle cx="{ccx:.1f}" cy="{ccy:.1f}" r="5.2" '
                             f'fill="{BG}" stroke="{col}" stroke-width="2.2"/>')
            if c:
                s.append(line(*c, stroke=INK, w=1.8))

        for j, ((px, py), g) in enumerate(zip(P, lab)):
            cx, cy = x0 + px * side, y0 + (1 - py) * side
            spilled = e > 0 and abs(float(P[j] @ w) - tau) <= e
            if spilled:
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.4" '
                         f'fill="none" stroke="{VIOLET}" stroke-width="1.6" '
                         f'opacity="0.95"/>')
            if kind is None and j in stray:      # die verirrten Punkte
                s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.8" '
                         f'fill="none" stroke="{AMBER}" stroke-width="1.6"/>')
            if spilled:                          # in BEIDEN Kindern
                s.append(split_dot(cx, cy, 4.0))
            else:
                s.append(dot(cx, cy, 3.1, BLUE if g == 0 else RED, 0.92))

        s.append("</g>")
        s.append(txt(x0, y0 + side + 20, name, 13, INK, weight="650",
                     font=MONO))
        s.append(txt(x0, y0 + side + 38, sub, 11.5, MUTED))

    ly = y0 + side + 68
    for i, (col, lb, wd, op) in enumerate([
            (INK, "split plane", 1.8, 1.0),
            (GREEN, "margin — the empty corridor the shift buys", 9, 0.3),
            (AMBER, "closer to the other centroid: lost to the descent", 1.8, 1.0),
            (VIOLET, "spill zone — --spill (share) or --eps (measured)", 9, 0.25)]):
        lx = pad + [0, 118, 0, 400][i]
        ry = ly + (26 if i > 1 else 0)
        s.append(line(lx, ry - 4, lx + 22, ry - 4, col, wd,
                      "3 3" if i == 2 else None, op))
        s.append(txt(lx + 30, ry, lb, 11.5, MUTED))
    # halb/halb: der gespillte Punkt
    hx = pad + 752
    s.append(split_dot(hx + 10, ly + 22, 5.0))
    s.append(f'<circle cx="{hx + 10}" cy="{ly + 22}" r="7.4" fill="none" '
             f'stroke="{VIOLET}" stroke-width="1.5"/>')
    s.append(txt(hx + 26, ly + 26, "in BOTH children", 11.5,
                 MUTED))
    s.append("</svg>")
    (OUT / "split-modes.svg").write_text("\n".join(s), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  3 — Die adaptive Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def fig_pipeline() -> None:
    """
    Die Pipeline einer Bisection.  Zwei Achsen stecken im Layout:
    EINRÜCKUNG = Abhängigkeit (was nur für den DA gebaut wird), FARBE =
    Wahlfreiheit (grau Pflicht, grün abschaltbar, amber eine Wahl).
    """
    W = 980
    # (nr, titel, untertitel, parameter, art, eingerückt)
    rows = [
        ("0", "Calibrate ε", "measure the encoder's uncertainty on a sample",
         "--eps auto | cos:0.9 | 0", "opt", False),
        ("1", "Normalise", "centre the group (Σx̃=0), scale to mean length 1",
         "always on — this is why no balance term is needed", "fix", False),
        ("2", "Gram matrix", "Q = x̃x̃ᵀ — every pairwise similarity, O(n²)",
         "--gram-threshold  (0 = dense · >0 = sparse)", "alt", True),
        ("3", "QUBO", "E(x) = −½ Σ Qᵢⱼ sᵢsⱼ — one binary variable per chunk",
         "--solver fast (matrix) | reference (PBF dict)", "alt", True),
        ("4", "Budget", "steps halve per level, trials adapt to group size",
         "--steps · --num-mc · --hold-steps", "fix", True),
        ("5-7", "Bisection", "the cut itself — at least one of the two runs, "
         "either one alone is a complete method", None, "grp", False),
        ("8", "Post-step", "where the boundary finally sits",
         "--split-mode none | centroid | plane", "alt", False),
        ("9", "Overlap", "a safety net on top: who may belong to both sides",
         "--spill (share) · --eps (measured)", "opt", False),
        ("10", "Node representation",
         "not the cut — how the BEAM judges a subtree on the way down",
         "--anchors 1 (one mean) | K (max over K anchors)", "opt", False),
    ]
    # Die beiden Bisektoren als Unterboxen EINER Pflichtbox: der Schritt
    # selbst ist nicht optional, nur die Wahl, wer ihn ausführt.
    subs = [
        ("5", "Lloyd warm start", "k=2 from PCA-oriented starts, deduplicated",
         "--iterations-lloyd 0  turns this off", ()),
        ("6+7", "Digital annealing", None, "--no-anneal  turns this off",
         # in Laufreihenfolge: erst kalibrieren, dann lösen.  Die
         # Kalibrierung ist selbst eine Wahl (amber) und liest das
         # Lloyd-Ergebnis, deshalb steht sie hier drin.
         (("6", "Cooling calibration",
           "temperature from the ΔE scale of the Lloyd solutions",
           "--cooling-schedule da_gp | floor · --p-hot · --p-cold · --gamma",
           AMBER),
          ("7", "The solve",
           "all trials of a start group as one batch",
           "--offset-increase · --offset-k-escape", MUTED))),
    ]
    row_h, gap, top = 52, 10, 132
    sub_h, det_h, grp_pad = 30, 36, 46
    IND = 34                                   # Einrückung der DA-Vorstufen
    sub_hs = [(sub_h + (16 if ss else 0)) + det_h * len(d)
              for _, _, ss, _, d in subs]
    grp_h = grp_pad + sum(h + 10 for h in sub_hs)
    H = top + (len(rows) - 1) * (row_h + gap) + grp_h + gap + 96

    s = head(W, H)
    s.append(txt(26, 32, "The adaptive pipeline", 16, INK, weight="650"))
    s.append(txt(26, 52, "One bisection, start to finish. Colour says how free "
                         "the choice is, indentation says what it depends on.",
                 12.5, MUTED))
    leg = [(INK, "always runs"), (GREEN, "can be switched off"),
           (AMBER, "pick one of several")]
    for i, (col, lb) in enumerate(leg):
        lx = 26 + i * 168
        s.append(f'<rect x="{lx}" y="{70}" width="12" height="12" rx="3" '
                 f'fill="{col}" opacity="0.85"/>')
        s.append(txt(lx + 19, 80, lb, 11.5, MUTED))
    s.append(badge(26 + 3 * 168, 69))
    s.append(txt(26 + 3 * 168 + 38, 80, "exists only for the annealer — "
                                        "--no-anneal drops all of it",
                 11.5, DA_INK))

    colour = {"fix": INK, "opt": GREEN, "alt": AMBER, "grp": INK}
    y = top
    for i, (nr, title, sub, param, kind, ind) in enumerate(rows):
        col = colour[kind]
        h = grp_h if kind == "grp" else row_h
        x = 26 + (IND if ind else 0)
        w_box = W - 26 - x
        s.append(box(x, y, w_box, h, fill=DA_BG if ind else "#fbfcfe",
                     stroke=DA_INK if ind else FAINT,
                     sw=1.2 if not ind else 1.0,
                     dash="5 4" if kind in ("opt", "alt") else None,
                     opacity=1.0 if not ind else 0.85))
        s.append(f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2.5" '
                 f'fill="{col}" opacity="0.9"/>')
        s.append(txt(x + 22, y + 22, nr, 12, MUTED, font=MONO, weight="600"))
        tx = x + (58 if kind == "grp" else 44)
        s.append(txt(tx, y + 22, title, 13.5, INK, weight="650"))
        if ind:
            s.append(badge(tx + len(title) * 7.3 + 10, y + 11))
        s.append(txt(tx, y + 39, sub, 11.5, MUTED))
        if param:
            s.append(txt(W - 42, y + 32, param, 11.5, col, anchor="end",
                         font=MONO))

        if kind == "grp":
            sy = y + grp_pad
            for j, (snr, stitle, ssub, sparam, det) in enumerate(subs):
                h_sub = sub_hs[j]
                is_da = bool(det)
                s.append(box(x + 32, sy, W - 58 - x - 32, h_sub,
                             fill=DA_BG if is_da else BG,
                             stroke=GREEN, sw=1.3, dash="5 4", opacity=0.95))
                s.append(f'<rect x="{x + 32}" y="{sy}" width="4" '
                         f'height="{h_sub}" rx="2" fill="{GREEN}" '
                         f'opacity="0.9"/>')
                s.append(txt(x + 50, sy + 20, snr, 11.5, MUTED, font=MONO,
                             weight="600"))
                s.append(txt(x + 86, sy + 20, stitle, 12.5, INK, weight="650"))
                if is_da:
                    s.append(badge(x + 86 + len(stitle) * 6.8 + 10, sy + 9))
                if ssub:
                    s.append(txt(x + 86, sy + 35, ssub, 11, MUTED))
                s.append(txt(W - 74, sy + 20, sparam, 11, GREEN,
                             anchor="end", font=MONO))
                dy = sy + sub_h
                for dnr, dtitle, dsub, dparam, dcol in det:
                    s.append(f'<rect x="{x + 54}" y="{dy + 2}" width="3" '
                             f'height="{det_h - 10}" rx="1.5" fill="{dcol}" '
                             f'opacity="0.9"/>')
                    s.append(txt(x + 68, dy + 14, dnr, 11, MUTED, font=MONO,
                                 weight="600"))
                    s.append(txt(x + 92, dy + 14, dtitle, 11.5, INK,
                                 weight="650"))
                    s.append(txt(x + 92, dy + 28, dsub, 10.5, MUTED))
                    s.append(txt(W - 74, dy + 21, dparam, 10.5, dcol,
                                 anchor="end", font=MONO))
                    dy += det_h
                if j < len(subs) - 1:
                    s.append(line(W / 2, sy + h_sub + 1, W / 2,
                                  sy + h_sub + 9, FAINT, 1.4))
                sy += h_sub + 10

        if i < len(rows) - 1:
            s.append(line(W / 2, y + h + 1, W / 2, y + h + gap - 1,
                          FAINT, 1.4))
        y += h + gap

    s.append(txt(26, y + 17, "recursion — repeat on both halves until a group "
                             "holds ≤ g points", 12.5, MUTED))
    s.append(txt(26, y + 42, "Every knob above is a measurement, not a taste: "
                             "--stats writes what each one did.", 12,
                 MUTED, opacity=0.9))
    s.append("</svg>")
    (OUT / "pipeline.svg").write_text("\n".join(s), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  4 — Das ε-Modell: jedes Embedding ist eine Kugel
# ═════════════════════════════════════════════════════════════════════════════

def fig_eps_model() -> None:
    # Zwei Wolken, die linke mit innerer Struktur, dazwischen ein dünn
    # besetzter Korridor.  Genau so entsteht der Fall, um den es geht: die
    # BREITESTE Lücke liegt in der dichten linken Wolke, die SAUBERSTE
    # Stelle im Korridor.  (Empirisch gesucht, siehe README.)
    rng = np.random.default_rng(4)
    P = np.vstack([
        np.c_[rng.normal(0.19, 0.032, 20), rng.uniform(0.12, 0.90, 20)],
        np.c_[rng.normal(0.36, 0.028, 14), rng.uniform(0.12, 0.90, 14)],
        np.c_[rng.uniform(0.46, 0.62, 11), rng.uniform(0.12, 0.90, 11)],
        np.c_[rng.normal(0.79, 0.050, 24), rng.uniform(0.12, 0.90, 24)],
    ])
    P = np.clip(P, 0.06, 0.94)
    a = (P[:, 0] > 0.52).astype(int)
    eps = 0.06

    w, tau_gap = plane_of(P, a, shift=True, eps=0.0)
    _, tau_eps = plane_of(P, a, shift=True, eps=eps)
    t = P @ w
    cut_gap = int((np.abs(t - tau_gap) <= eps).sum())
    cut_eps = int((np.abs(t - tau_eps) <= eps).sum())

    W, side, pad, gap = 980, 235, 26, 34
    H = 412
    s = head(W, H)
    s.append(txt(pad, 30, "ε — every embedding is a sphere, not a point",
                 16, INK, weight="650"))
    s.append(txt(pad, 50, "The encoder places the same content in slightly "
                          "different places depending on how it is phrased. "
                          "A boundary that cuts through that uncertainty is "
                          "guessing.", 12.5, MUTED))

    panels = [
        ("the assumption", f"radius ε — measured, not assumed", None),
        ("--eps 0  ·  widest gap", f"cuts {cut_gap} spheres", tau_gap),
        (f"--eps {eps:g}  ·  fewest cut", f"cuts {cut_eps} spheres", tau_eps),
    ]
    y0 = 78
    for i, (name, sub, tau) in enumerate(panels):
        x0 = pad + i * (side + gap)
        s.append(box(x0, y0, side, side, fill="#fbfcfe"))
        s.append(f'<clipPath id="e{i}"><rect x="{x0}" y="{y0}" '
                 f'width="{side}" height="{side}" rx="8"/></clipPath>')
        s.append(f'<g clip-path="url(#e{i})">')
        if tau is not None:
            c = clip_line(w, tau, x0, y0, side)
            if c:
                s.append(line(*c, stroke=INK, w=1.8))
        for j, (px, py) in enumerate(P):
            cx, cy = x0 + px * side, y0 + (1 - py) * side
            hit = tau is not None and abs(float(t[j] - tau)) <= eps
            s.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" '
                     f'r="{eps * side:.1f}" fill="{VIOLET}" '
                     f'opacity="{0.22 if hit else 0.07}" stroke="{VIOLET}" '
                     f'stroke-width="{1.4 if hit else 0.7}" '
                     f'stroke-opacity="{0.9 if hit else 0.3}"/>')
        for j, (px, py) in enumerate(P):
            cx, cy = x0 + px * side, y0 + (1 - py) * side
            hit = tau is not None and abs(float(t[j] - tau)) <= eps
            if hit:
                s.append(split_dot(cx, cy, 4.0))
            else:
                s.append(dot(cx, cy, 3.0,
                             MUTED if tau is None else
                             (BLUE if a[j] == 0 else RED), 0.9))
        s.append("</g>")
        s.append(txt(x0, y0 + side + 21, name, 12.5, INK, weight="650",
                     font=MONO))
        s.append(txt(x0, y0 + side + 39, sub, 11.5,
                     VIOLET if tau is not None else MUTED))

    ly = y0 + side + 64
    s.append(txt(pad, ly, "The widest gap is not the safest place: a gap can "
                          "be wide and still sit inside a dense region. With "
                          "ε the plane is placed where the fewest spheres are "
                          "cut —", 11.5, MUTED))
    s.append(txt(pad, ly + 16, "and exactly those that still are get spilled "
                               "into both children. This shift is only "
                               "available under --split-mode plane; centroid "
                               "has no freedom to move.", 11.5, MUTED))
    s.append("</svg>")
    (OUT / "epsilon-model.svg").write_text("\n".join(s), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  5 — Wie ε gemessen wird
# ═════════════════════════════════════════════════════════════════════════════

MEASURED = {"median": 0.02087, "q90": 0.06950, "q99": 0.15915,
            "sigma_root": 0.09098, "encoder": "nomic-embed-text",
            "pairs": 135, "dirs": 254}


def fig_eps_measure() -> None:
    W, H, pad = 980, 340, 26
    m = MEASURED
    s = head(W, H)
    s.append(txt(pad, 30, "How ε is measured", 16, INK, weight="650"))
    s.append(txt(pad, 50, "Not a guess and not a constant: ε is a property of "
                          "the encoder, so we perturb real chunks and watch "
                          "where they land.", 12.5, MUTED))

    # ── 1. Der Chunk und sein Fenster ───────────────────────────────────────
    bx, by, bw2, bh = pad, 84, 214, 132
    s.append(box(bx, by, bw2, bh, fill="#fbfcfe"))
    rng = np.random.default_rng(2)
    for k in range(9):
        ly = by + 16 + k * 13
        lw = bw2 - 28 - float(rng.uniform(0, 42))
        inside = 2 <= k <= 6
        s.append(line(bx + 14, ly, bx + 14 + lw, ly,
                      VIOLET if inside else FAINT, 4.5, None,
                      0.5 if inside else 0.9))
    s.append(f'<rect x="{bx + 9}" y="{by + 40}" width="{bw2 - 18}" '
             f'height="{5 * 13}" rx="5" fill="none" stroke="{VIOLET}" '
             f'stroke-width="1.4" stroke-dasharray="4 3"/>')
    s.append(txt(bx, by + bh + 19, "1 · a random 60–80 % word window",
                 12, INK, weight="650"))
    s.append(txt(bx, by + bh + 36, "256 chunks, the same encoder as the index",
                 11.5, MUTED))

    # ── 2. Beide Fassungen einbetten ────────────────────────────────────────
    ax = bx + bw2 + 42
    s.append(txt(ax - 26, by + 66, "→", 20, MUTED))
    px, py, pw = ax, by, 214
    s.append(box(px, py, pw, bh, fill="#fbfcfe"))
    o = np.array([0.36, 0.56])
    d = np.array([0.30, -0.14])
    for p, col, lab in ((o, INK, "original"), (o + d, VIOLET, "perturbed")):
        cx, cy = px + p[0] * pw, py + (1 - p[1]) * bh
        s.append(dot(cx, cy, 5.0, col))
        s.append(txt(cx + 9, cy - 6, lab, 10.5, col))
    s.append(line(px + o[0] * pw, py + (1 - o[1]) * bh,
                  px + (o + d)[0] * pw, py + (1 - (o + d)[1]) * bh,
                  VIOLET, 1.4, "4 3"))
    # Projektionsachse ŵ
    s.append(line(px + 14, py + bh - 22, px + pw - 14, py + bh - 22,
                  MUTED, 1.4))
    s.append(txt(px + pw - 14, py + bh - 30, "ŵ", 11.5, MUTED, anchor="end"))
    for p, col in ((o, INK), (o + d, VIOLET)):
        cx = px + p[0] * pw
        s.append(line(cx, py + (1 - p[1]) * bh, cx, py + bh - 22, col, 0.9,
                      "2 2", 0.6))
        s.append(dot(cx, py + bh - 22, 3.4, col))
    s.append(line(px + o[0] * pw, py + bh - 12,
                  px + (o + d)[0] * pw, py + bh - 12, VIOLET, 1.6))
    s.append(txt(px + (o[0] + d[0] / 2) * pw, py + bh - 2, "|Δt|", 10.5,
                 VIOLET, anchor="middle", weight="600"))
    s.append(txt(px, py + bh + 19, "2 · the shift along a split direction",
                 12, INK, weight="650"))
    s.append(txt(px, py + bh + 36, "projected, not the raw distance",
                 11.5, MUTED))

    # ── 3. Die Verteilung ───────────────────────────────────────────────────
    hx = px + pw + 42
    hw = W - hx - pad
    s.append(txt(hx - 26, by + 66, "→", 20, MUTED))
    s.append(box(hx, by, hw, bh, fill="#fbfcfe"))
    # Lognormal an Median und q90 angepasst (die Kurve illustriert, die
    # Marker sind gemessen)
    mu = math.log(m["median"])
    sg = math.log(m["q90"] / m["median"]) / 1.2816
    xmax = m["q99"] * 1.45
    xs = np.linspace(1e-4, xmax, 220)
    ys = np.exp(-((np.log(xs) - mu) ** 2) / (2 * sg ** 2)) / xs
    ys = ys / ys.max()
    pts = " ".join(f"{hx + 14 + v / xmax * (hw - 28):.1f},"
                   f"{by + bh - 30 - y * (bh - 58):.1f}"
                   for v, y in zip(xs, ys))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{VIOLET}" '
             f'stroke-width="1.8"/>')
    s.append(f'<polygon points="{hx + 14},{by + bh - 30} {pts} '
             f'{hx + 14 + (hw - 28)},{by + bh - 30}" fill="{VIOLET}" '
             f'opacity="0.12"/>')
    s.append(line(hx + 14, by + bh - 30, hx + hw - 14, by + bh - 30,
                  MUTED, 1.2))
    marks = ((m["median"], "median", MUTED, "end", -5, 26),
             (m["q90"], "ε = q90", VIOLET, "start", 5, 26),
             (m["sigma_root"], "σ(t) at the root", INK, "start", 5, 62))
    for val, lab, col, anc, dx, ty in marks:
        vx = hx + 14 + val / xmax * (hw - 28)
        hot = lab.startswith("ε")
        s.append(line(vx, by + 16, vx, by + bh - 30, col, 1.4 if hot else 1.1,
                      None if hot else "3 3", 1.0 if hot else 0.5))
        s.append(txt(vx + dx, ty + by, lab, 10.5, col, anchor=anc,
                     weight="650" if hot else "400"))
        s.append(txt(vx + dx, ty + by + 13, f"{val:.3f}", 10, col,
                     anchor=anc, opacity=0.8))
    s.append(txt(hx, by + bh + 19, "3 · ε = the 90th percentile of |Δt|",
                 12, INK, weight="650"))
    s.append(txt(hx, by + bh + 36, f"{m['encoder']}: ε = {m['q90']:.3f} against "
                                   f"a root spread of {m['sigma_root']:.3f}",
                 11.5, MUTED))

    s.append(txt(pad, H - 34, f"Markers are measured ({m['pairs']} pairs × "
                              f"{m['dirs']} directions); the curve is a "
                              "lognormal fitted to the median and q90.",
                 11, MUTED, opacity=0.85))
    s.append(txt(pad, H - 18, "ε = 0.76 σ at the root: up there almost "
                              "nothing is safely separable, which is why "
                              "--spill caps the zone. Deeper down the zone "
                              "empties by itself.", 11, MUTED, opacity=0.85))
    s.append("</svg>")
    (OUT / "epsilon-measure.svg").write_text("\n".join(s), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
#  6 — Heatmap der Benchmark-Matrix
# ═════════════════════════════════════════════════════════════════════════════

def _heat(v: float, lo: float, hi: float) -> str:
    """Wert → Farbe (blass bei niedrig, kräftig bei hoch)."""
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))
    # von hellem Grau-Blau nach kräftigem Indigo/Grün
    r0, g0, b0 = 0xF1, 0xF5, 0xF9
    r1, g1, b1 = 0x15, 0x6F, 0x4A
    return (f"#{int(r0 + (r1 - r0) * t):02x}"
            f"{int(g0 + (g1 - g0) * t):02x}"
            f"{int(b0 + (b1 - b0) * t):02x}")


def fig_heatmap(results="benchmarks/results.json", metric="recall@5",
                beam=2, out="benchmark-heatmap.svg",
                scale="global") -> None:
    """
    Die ganze Matrix als Bild: Zeilen = Pipeline-Variante, Spalten =
    Wissensbasis-Größe je Encoder, Farbe = Qualität.  Die Zahlen stehen
    trotzdem drin — eine Heatmap, die man nur anschauen und nicht ablesen
    kann, ist Dekoration.
    """
    import json as _json
    rows_raw = _json.loads((REPO / results).read_text())
    rows_raw = [r for r in rows_raw if r.get("beam") == beam]
    if not rows_raw:
        print("heatmap: keine Daten")
        return

    SHORT = {"bge-m3": "bge-m3", "qwen3-embedding:4b-fp16": "qwen3-4b"}
    order = ["lloyd-raw", "raw", "centroid", "plane", "lloyd", "lloyd-mc4",
             "da-blind", "centroid+spill", "default", "mc4", "spill0.2", "eps"]
    # Optimierer · Grenzziehung — die beiden Achsen immer getrennt benennen,
    # sonst liest sich "lloyd" wie nacktes k-Means, obwohl Ebene und Overlap
    # drinstecken.
    names = {"lloyd-raw": "Lloyd · nearest centroid",
             "raw": "DA · nearest centroid",
             "centroid": "DA · centroid plane",
             "plane": "DA · max-margin plane",
             "lloyd": "Lloyd · plane + 10 %",
             "lloyd-mc4": "Lloyd ×4 · plane + 10 %",
             "da-blind": "DA, no warm start · plane + 10 %",
             "centroid+spill": "DA · centroid plane + 10 %",
             "default": "DA · plane + 10 %  (default)",
             "mc4": "DA, 4 starts · plane + 10 %",
             "spill0.2": "DA · plane + 20 %",
             "eps": "DA · ε-plane + 10 %"}
    encs, sizes = [], sorted({r["n"] for r in rows_raw})
    for r in rows_raw:
        e = SHORT.get(r.get("encoder", "?"), "?")
        if e not in encs:
            encs.append(e)
    cells = {(r.get("variant"), SHORT.get(r.get("encoder"), "?"), r["n"]):
             r.get(metric) for r in rows_raw}
    used = [v for v in order if any((v, e, n) in cells
                                    for e in encs for n in sizes)]
    vals = [v for v in cells.values() if v is not None]
    # Farbskala SPALTENWEISE: sonst übertönt der Größeneffekt (83 % bei 529
    # Chunks, 60 % bei 4178) jeden Unterschied zwischen den Varianten, und
    # die Heatmap zeigt nur noch, dass große Wissensbasen schwerer sind.
    #  ... und zwar als RÜCKSTAND auf den Spaltenbesten, in Prozentpunkten,
    #  mit EINER Skala für die ganze Matrix.  Spreizt man stattdessen jede
    #  Spalte auf ihr eigenes Min/Max, sieht ein Rückstand von 3 Punkten
    #  genauso dramatisch aus wie einer von 15 — die Farbe behauptet dann
    #  Unterschiede, die die Zahlen nicht hergeben.
    best: dict = {}
    for e in encs:
        for n in sizes:
            col = [cells[(v, e, n)] for v in used if (v, e, n) in cells]
            best[(e, n)] = max(col) if col else 1.0
    deltas = [best[(e, n)] - cells[(v, e, n)]
              for v in used for e in encs for n in sizes
              if (v, e, n) in cells]
    dmax = max(max(deltas), 0.02)          # Spannweite der ganzen Matrix
    lo, hi = min(vals), max(vals)

    cw, ch, lw = 62, 26, 288
    W = lw + len(encs) * len(sizes) * cw + 40
    H = 110 + len(used) * ch + 56
    s = head(W, H)
    s.append(txt(20, 28, f"Benchmark matrix — {metric}, beam {beam}", 15, INK,
                 weight="650"))
    s.append(txt(20, 45, "300 questions per column, identical for every row; "
                         "columns differ in encoder and corpus size.",
                 11.5, MUTED))
    # Der Hinweis auf gestrichelte Zellen nur, wenn es welche GIBT — sonst
    # erklaert die Legende etwas, das der Betrachter nirgends sieht.
    missing = any((v, e, n) not in cells
                  for v in used for e in encs for n in sizes)
    s.append(txt(20, 60, "Each row names optimiser · boundary — the "
                         "two independent axes."
                         + ("  Dashed = not measured." if missing else ""),
                 11.5, MUTED))

    # Kopfzeilen: Encoder-Gruppen, darunter die Größen
    for ei, e in enumerate(encs):
        x0 = lw + ei * len(sizes) * cw
        s.append(f'<rect x="{x0}" y="76" width="{len(sizes) * cw - 6}" '
                 f'height="18" rx="5" fill="{DA_INK}" opacity="0.13"/>')
        s.append(txt(x0 + (len(sizes) * cw - 6) / 2, 89, e, 11.5, DA_INK,
                     anchor="middle", weight="700"))
        for si, n in enumerate(sizes):
            s.append(txt(x0 + si * cw + cw / 2 - 3, 108,
                         f"{n:,}".replace(",", " "), 10.5, MUTED,
                         anchor="middle"))

    for ri, v in enumerate(used):
        y = 116 + ri * ch
        s.append(txt(lw - 12, y + 17, names.get(v, v), 11.5, INK,
                     anchor="end"))
        for ei, e in enumerate(encs):
            for si, n in enumerate(sizes):
                x = lw + (ei * len(sizes) + si) * cw
                val = cells.get((v, e, n))
                if val is None:
                    s.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" '
                             f'height="{ch - 4}" rx="4" fill="none" '
                             f'stroke="{FAINT}" stroke-dasharray="3 3"/>')
                    continue
                if scale == "column":   # Rückstand je Spalte
                    d = best[(e, n)] - val          # 0 = bester der Spalte
                    col_fill = _heat(dmax - d, 0.0, dmax)
                    strong = (dmax - d) / dmax > 0.55
                    is_best = d <= 1e-9
                else:
                    col_fill = _heat(val, lo, hi)
                    strong = (val - lo) / max(hi - lo, 1e-9) > 0.55
                    is_best = val >= hi
                s.append(f'<rect x="{x}" y="{y}" width="{cw - 4}" '
                         f'height="{ch - 4}" rx="4" fill="{col_fill}"/>')
                s.append(txt(x + (cw - 4) / 2, y + 15, f"{val * 100:.0f}",
                             10.5, "#ffffff" if strong else INK,
                             anchor="middle",
                             weight="700" if is_best else "400"))
    y = 116 + len(used) * ch + 16
    legend = (f"numbers in percent · colour = gap to the best of THIS "
              f"column, one scale for the whole matrix: strong = leading, "
              f"pale = {dmax * 100:.0f} points behind."
              if scale == "column" else
              f"numbers in percent · one colour scale for the whole table: "
              f"pale = {lo:.0%}, strong = {hi:.0%} · bold = best in its "
              f"column")
    s.append(txt(20, y + 8, legend, 11, MUTED))
    s.append("</svg>")
    (OUT / out).write_text("\n".join(s), encoding="utf-8")
    print(f"{out}: {len(used)} Varianten × {len(encs) * len(sizes)} Spalten")


# ═════════════════════════════════════════════════════════════════════════════
#  7 — Pareto: Qualität gegen Arbeit, wir und der Stand der Technik
# ═════════════════════════════════════════════════════════════════════════════

WORK_NOTE = """\
"Arbeit" ist die Zahl der Skalarprodukte je Query: der Abstieg (Score gegen
Knoten bzw. Graph-Navigation) PLUS das exakte Nachranken der Kandidaten.  Nur
Kandidaten zu zaehlen schmeichelt jedem Verfahren, das seine Arbeit in die
Navigation verlegt — uns mit Ankern zum Beispiel."""


def _load(results, baselines):
    import json as _json
    return (_json.loads((REPO / results).read_text()),
            _json.loads((REPO / baselines).read_text()))


def _hull(pts):
    """Monoton steigende obere Huelle: mehr Arbeit darf nie weniger Recall
    ausweisen, sonst ist der Punkt vom billigeren dominiert."""
    out, best = [], 0.0
    for w, r in sorted(pts):
        if r > best:
            best = r
            out.append((w, best))
    return out


def _curve(rows, kb, variant, anchors, *, x="total"):
    pts = [(r["cands"] + (r["score_sp"] if x == "total" else 0.0),
            r["recall@5"] * 100)
           for r in rows
           if r["kb"] == kb and r["variant"] == variant
           and r["anchors"] == anchors and r["search"] == "beam"]
    return _hull(pts)


def work_at(pts, target):
    """Arbeit der GUENSTIGSTEN GEMESSENEN Einstellung, die `target` % Recall
    erreicht — None, wenn keine sie erreicht.

    Bewusst nicht interpoliert: beam, ef und nprobe sind diskrete Knoepfe, und
    ein interpolierter Punkt ist eine Konfiguration, die man nicht einstellen
    kann.  Die Knopf-Koernung ist bei allen vier Verfahren aehnlich (jeweils
    ungefaehr Verdopplung), also vergleicht das faire Punkte."""
    for w, r in pts:
        if r >= target:
            return w
    return None


def _axes(s, L, Rm, T, Bm, W, H, x0, x1, y0, y1, xticks, xlabel):
    sx = lambda w: L + (math.log10(max(w, x0)) - math.log10(x0)) / (
        math.log10(x1) - math.log10(x0)) * (W - L - Rm)
    sy = lambda r: H - Bm - (min(max(r, y0), y1) - y0) / (y1 - y0) * (H - T - Bm)
    for r in range(int(y0 // 10 * 10 + 10), int(y1) + 1, 10):
        y = sy(r)
        s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-Rm}" y2="{y:.1f}" '
                 f'stroke="{FAINT}" stroke-width="1"'
                 + ('' if r % 20 == 0 else ' opacity="0.45"') + '/>')
        s.append(txt(L - 10, y + 4, f"{r}%", 11, MUTED, anchor="end"))
    for w in xticks:
        x = sx(w)
        s.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-Bm}" '
                 f'stroke="{FAINT}" stroke-width="1" opacity="0.45"/>')
        s.append(txt(x, H - Bm + 18, f"{w:,}".replace(",", " "), 11, MUTED,
                     anchor="middle"))
    s.append(txt((L + W - Rm) / 2, H - Bm + 40, xlabel, 11.5, MUTED,
                 anchor="middle"))
    return sx, sy


def _plot(s, pts, sx, sy, col, dash, wd):
    d = " ".join(f"{'M' if i == 0 else 'L'} {sx(w):.1f} {sy(r):.1f}"
                 for i, (w, r) in enumerate(pts))
    s.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wd}" '
             f'stroke-linejoin="round"'
             + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
    for w, r in pts:
        s.append(f'<circle cx="{sx(w):.1f}" cy="{sy(r):.1f}" r="3" '
                 f'fill="{col}"/>')


def _legend(s, x, y, entries, note=()):
    for col, dash, wd, label, sub in entries:
        s.append(f'<line x1="{x}" y1="{y}" x2="{x+26}" y2="{y}" '
                 f'stroke="{col}" stroke-width="{wd}"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
        s.append(txt(x + 32, y + 4, label, 11.5, INK, weight="600"))
        y += 15
        if sub:
            s.append(txt(x + 32, y + 4, sub, 10.5, MUTED))
            y += 15
        y += 7
    for i, n in enumerate(note):
        s.append(txt(x, y + 12 + i * 15, n, 10.5, MUTED))


LLOYD_INK = "#0891b2"       # gehört zu Lloyd — billig, aber eine Stufe drunter
HNSW_INK = "#15803d"
IVF_INK = "#b45309"


def fig_pareto(results="benchmarks/results_full.json",
               baselines="benchmarks/baselines.json",
               out="pareto.svg", encoder="bge",
               kb="RefKB_bge-m3") -> None:
    """
    Recall gegen Arbeit — die einzige Achse, auf der sich Indexe ehrlich
    vergleichen lassen.  Drei Protagonisten in der Reihenfolge, in der sie
    liegen: HNSW klar vorn, dann Lloyd->DA, dann Lloyd allein.  FAISS IVF
    laeuft als zweite Referenz mit, weil wir sie schlagen.
    """
    R, B = _load(results, baselines)

    ours = _curve(R, kb, "knn-da", 1)
    lloyd = _curve(R, kb, "m-lloyd", 1)
    hnsw = _hull([(e["cands"], e["recall@5"] * 100) for e in B
                  if e["index"].startswith("HNSW") and e.get("cands")])
    ivf = _hull([(e["cands"], e["recall@5"] * 100) for e in B
                 if e["index"].startswith("FAISS") and e.get("cands")])

    W, H = 940, 762
    L, Rm, T, Bm = 74, 246, 84, 212
    s = head(W, H)
    s.append(txt(20, 30, "Quality against work", 16, INK, weight="650"))
    s.append(txt(20, 50, "4178 chunks, bge-m3, 300 questions · work = scalar "
                         "products per query (descent + re-ranking), "
                         "logarithmic", 11.5, MUTED))
    s.append(txt(20, 68, "Up and to the left is better. Every curve is one "
                         "index swept over its own accuracy knob — beam width "
                         "for the tree, ef / nprobe for the baselines.",
                 11.5, MUTED))

    sx, sy = _axes(s, L, Rm, T, Bm, W, H, 25.0, 2600.0, 35.0, 100.0,
                   (30, 100, 300, 1000, 2500), "scalar products per query")

    _plot(s, ivf, sx, sy, IVF_INK, "6 3", 1.7)
    _plot(s, lloyd, sx, sy, LLOYD_INK, "5 3", 2.2)
    _plot(s, ours, sx, sy, DA_INK, "", 2.6)
    _plot(s, hnsw, sx, sy, HNSW_INK, "", 2.6)

    _legend(s, W - Rm + 14, T + 4, [
        (HNSW_INK, "", 2.6, "HNSW (M=16)", "the winner — hnswlib, ef 5…100"),
        (DA_INK, "", 2.6, "this index · Lloyd → DA",
         "defaults: --objective knn, beam 1…32"),
        (LLOYD_INK, "5 3", 2.2, "this index · Lloyd only",
         "--no-anneal, same tree shape"),
        (IVF_INK, "6 3", 1.7, "FAISS IVF (nlist=64)", "nprobe 1…32"),
    ])

    rows = [("recall@5", "HNSW", "Lloyd→DA", "Lloyd", "FAISS IVF")]
    for tgt in (90.0, 95.0, 97.0):
        rows.append((f"{tgt:.0f} %",) + tuple(
            ("not reached" if v is None else f"{v:,.0f}".replace(",", " "))
            for v in (work_at(hnsw, tgt), work_at(ours, tgt),
                      work_at(lloyd, tgt), work_at(ivf, tgt))))
    ty = H - 158
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            s.append(txt(20 + j * 100, ty + i * 17, cell, 11,
                         MUTED if i == 0 else INK,
                         weight="600" if i == 0 else "400"))
    s.append(txt(20 + 5 * 100 + 14, ty,
                 "Scalar products of the cheapest measured setting that",
                 10.5, MUTED))
    s.append(txt(20 + 5 * 100 + 14, ty + 15,
                 "reaches that recall — no interpolation, because beam, ef",
                 10.5, MUTED))
    s.append(txt(20 + 5 * 100 + 14, ty + 30,
                 "and nprobe are discrete knobs.", 10.5, MUTED))
    for i, ln in enumerate((
            "Lloyd is the cheap floor: at equal work it trails Lloyd→DA by "
            "0.5–3.2 points here, and builds the tree in 14 s, not 223 s.",
            "Above 96 % it stops reaching the target at all with one mean per "
            "node. So the annealer does buy something — a couple of points,",
            "and the last stretch of the curve — for about ten times the build "
            "time. That is the whole trade, stated as a number.",
            "Neither comes near HNSW: 2.8× the work at 90 %, 3.8× at 95 %, "
            "4.0× at 97 %. What the tree holds is the left edge — below ~100",
            "scalar products a graph has not finished navigating yet.")):
        s.append(txt(20, H - 84 + i * 18, ln, 11.5, MUTED))
    s.append("</svg>")
    (OUT / out).write_text("\n".join(s), encoding="utf-8")
    print(f"{out}: 4 Kurven · "
          f"95 % @ HNSW {work_at(hnsw, 95):.0f} / DA {work_at(ours, 95):.0f} / "
          f"Lloyd {work_at(lloyd, 95):.0f} / IVF {work_at(ivf, 95):.0f} SP")


# ═════════════════════════════════════════════════════════════════════════════
#  8 — Anker: dieselbe Messung zweimal gezaehlt
# ═════════════════════════════════════════════════════════════════════════════

def fig_anchors(results="benchmarks/results_full.json",
                out="anchors.svg", kb="RefKB_bge-m3",
                variant="knn-da") -> None:
    """
    Die Anker-Achse, links ehrlich und rechts schmeichelnd gezaehlt.

    k Anker ersetzen den Gruppen-Mittelwert im Beam-Score durch
    max_j cos(q, a_j) — der Baum aendert sich nicht, nur die Bewertung der
    Knoten.  Rechts (nur Kandidaten) sieht das nach einem klaren Gewinn aus,
    links (Kandidaten + Abstieg) bezahlt man ihn mit Navigation.
    """
    R, _ = _load(results, "benchmarks/baselines.json")

    STYLE = {1: (DA_INK, "", 2.6, "k = 1 · one mean (default)"),
             4: ("#0ea5e9", "5 3", 2.0, "k = 4 anchors"),
             16: ("#38bdf8", "2 3", 2.0, "k = 16 anchors")}

    W, H = 990, 700
    PW = 400                                    # Panel-Breite
    s = head(W, H)
    s.append(txt(20, 30, "Anchors: the same measurement, counted two ways",
                 16, INK, weight="650"))
    s.append(txt(20, 50, "4178 chunks, bge-m3, 300 questions · tree built with "
                         "the defaults (Lloyd → DA, --objective knn), beam "
                         "1…32 · both axes logarithmic", 11.5, MUTED))

    panels = [
        (20, "total", "scalar products per query",
         (30, 100, 300, 1000, 3000), 25.0, 7500.0,
         "Counted honestly", "descent + re-ranking"),
        (20 + PW + 96, "cands", "re-rank candidates per query",
         (10, 30, 100, 300), 10.0, 450.0,
         "Counted the flattering way", "re-rank candidates only"),
    ]
    for px, mode, xlabel, xticks, x0, x1, title, sub in panels:
        s.append(txt(px, 86, title, 13, INK, weight="650"))
        s.append(txt(px, 103, sub, 11, MUTED))
        L, Rm, T, Bm = px + 40, W - (px + PW), 118, 176
        sx, sy = _axes(s, L, Rm, T, Bm, W, H, x0, x1, 35.0, 100.0,
                       xticks, xlabel)
        for k in (16, 4, 1):
            col, dash, wd, _ = STYLE[k]
            _plot(s, _curve(R, kb, variant, k, x=mode), sx, sy, col, dash, wd)

    lx = 20
    for k in (1, 4, 16):
        col, dash, wd, label = STYLE[k]
        s.append(f'<line x1="{lx}" y1="{H-110}" x2="{lx+24}" y2="{H-110}" '
                 f'stroke="{col}" stroke-width="{wd}"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
        s.append(txt(lx + 30, H - 106, label, 11.5, INK))
        lx += 190

    rows = [r for r in R if r["kb"] == kb and r["variant"] == variant
            and r["search"] == "beam" and r["param"] == 8]
    by_k = {r["anchors"]: r for r in rows}
    lines = []
    if {1, 4, 16} <= set(by_k):
        lines.append("At beam 8 all three look at the same ~104 candidates — "
                     + " · ".join(
                         f"k={k}: {by_k[k]['recall@5']*100:.1f} % for "
                         f"{by_k[k]['cands']+by_k[k]['score_sp']:.0f} SP"
                         for k in (1, 4, 16)) + ".")
    lines += [
        "So anchors do what they promise: at an equal candidate count they "
        "raise recall by 3.0 and 6.6 points — the right panel is real, not a "
        "trick.",
        "The bill arrives on the left. Scoring every node against k anchors "
        "costs k times the descent, and widening the beam buys the same points "
        "cheaper:",
        "k=1 at beam 32 reaches 96.0 % for 787 SP, where k=16 needs 2108 SP "
        "for 93.3 %. Hence the default --anchors 1 — the option stays, the "
        "recommendation does not."]
    for i, ln in enumerate(lines):
        s.append(txt(20, H - 80 + i * 18, ln, 11.5, MUTED))
    s.append("</svg>")
    (OUT / out).write_text("\n".join(s), encoding="utf-8")
    print(f"{out}: 3 Anker-Stufen × 2 Zaehlweisen")


if __name__ == "__main__":
    fig_concept()
    fig_split()
    fig_pipeline()
    fig_eps_model()
    fig_eps_measure()
    fig_heatmap()
    fig_heatmap(metric="found@5", out="benchmark-heatmap-found.svg")
    fig_pareto()
    fig_anchors()
    for f in sorted(OUT.glob("*.svg")):
        print(f"{f.name:28s} {f.stat().st_size / 1024:6.1f} KB")
