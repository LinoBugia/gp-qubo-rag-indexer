"""
graph_partitioning.py — Recursive Binary Graph Partitioning via QUBO/DA
========================================================================

Recursively bisects an n-dimensional point cloud using a QUBO formulation
solved by Digital Annealing until all leaf groups contain ≤ g points.

Pipeline per bisection level
-----------------------------
  1. normalize_coords: centre the group (Σ x̃ = 0) and scale to mean length 1
  2. Gram matrix Q = x̃ x̃ᵀ, dense or sparse (gram_threshold)
  3. QUBO (A, b, c) for solver="fast", or PBF dict for solver="reference";
     both encode E(x) = −½·Σ Q_ij s_i s_j  =  −2‖S_A‖²  (centred ⇒ no
     linear term) — see create_pbf
  4. Budget: steps = max(STEPS_FLOOR, steps/2**depth), auto-MC below
     MC_SMALL_N, plateau hold_steps/2**depth
  5. Lloyd warm start with PCA-oriented multi-starts (lloyd_mc), deduped;
     NO balance repair — that would destroy the start values
  6. Cooling calibration per start group (da_gp / da_gp_floor)
  7. Digital annealing, all trials of a group as one batch
  8. reassign_to_split_centroids: freeze both centroids, re-assign every
     point to the nearer one — that is the split for the next level
     (there is NO rebalancing step any more)
  9. Recurse on each sub-group until |group| ≤ g

Full prose version incl. the query side: docs/pipeline.md and docs/query.md.

The recursion works on *global indices* into one shared coordinate array —
no point-object identity tricks, and the result is an explicit binary tree
(parent → children) that a query can descend.

Imports
-------
  Funcs_Annealers  ← from annealing-cop-approximator/Code  (pbf_min_solver, createPolyDict)
  clustering.clustering_gram_matrix
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

# ── External: annealing-cop-approximator/Code  (solver="reference") ────────────
_COP_CODE = (
    Path(__file__).resolve().parent.parent / "annealing-cop-approximator" / "Code"
)
if str(_COP_CODE) not in sys.path:
    sys.path.insert(0, str(_COP_CODE))

# Der Referenz-Import darf fehlschlagen (z.B. während Umbauten an der alten
# Lib) — dann steht nur der "fast"-Solver zur Verfügung.
try:
    from Funcs_Annealers import *              # noqa: F401, F403  (createPolyDict, …)
    from Funcs_Optimizers import pbf_min_solver  # noqa: E402
    _REFERENCE_IMPORT_ERROR: Exception | None = None
except Exception as _exc:                      # noqa: BLE001
    _REFERENCE_IMPORT_ERROR = _exc

# ── External: annealing-qubo-optimizer/Code  (solver="fast", batched numpy) ─────
_AQS_CODE = (
    Path(__file__).resolve().parent.parent / "annealing-qubo-optimizer" / "Code"
)


def _import_fast_solver():
    """Lazy-Import der neuen QUBO-Lib (erst beim ersten fast-Solve nötig)."""
    if str(_AQS_CODE) not in sys.path:
        sys.path.insert(0, str(_AQS_CODE))
    from Funcs_Qubo_Optimizers import qubo_min_solver
    from Funcs_Qubo_Annealing3 import eval_qubo
    return qubo_min_solver, eval_qubo

# ── Local ─────────────────────────────────────────────────────────────────────
from clustering import clustering_gram_matrix, normalize_coords, sparse_gram_entries


# ═══════════════════════════════════════════════════════════════════════════════
#  PBF construction — INTEGER-KEYS (gepackt)
#
#  Key-Schema (Basis n, Bänder pro Grad):
#      quadratisch (i,j), i <= j :  key = i*n + j          ∈ [0, n²)
#      linear      (i,)          :  key = n*n + i          ∈ [n², n²+n)
#  Decode: key >= n² → linear, i = key − n²
#          sonst     → i, j = divmod(key, n)
#  Ein int-Key kostet 28 B statt 112 B (Tupel + 2 geboxte ints) → das
#  PBF-Dict schrumpft um Faktor ~2.7, int-Hashing ist zudem schneller.
# ═══════════════════════════════════════════════════════════════════════════════

def unpack_key(key: int, n: int) -> tuple:
    """Gepackten PBF-Key zurück in das Monom-Tupel: (i,) linear, (i,j) quad."""
    if key >= n * n:
        return (key - n * n,)
    return divmod(key, n)


# Key-Format für den REFERENZ-Solver (annealing-cop-approximator):
#   False → Tupel-Keys (i,) / (i,j) — das versteht die alte Lib HEUTE;
#           mehr Speicher (~2.7×), aber lauffähig.
#   True  → gepackte int-Keys — erst umstellen, wenn Linos int-Key-Umbau
#           in der alten Lib (createPolyDict/Eval_Delta_Energy/evalPBF)
#           fertig ist.
# Der fast-Optimizer ist davon unabhängig (Matrixform, keine PBF-Dicts).
REFERENCE_INT_KEYS = False

# ── Steps- und MC-Schedule je Bisection (Linos Kalibrierung) ─────────────────
#   step_annealing = max(STEPS_FLOOR, steps / 2**depth)   — Halbieren je Ebene
#   n < MC_SMALL_N  ->  mindestens MC_SMALL_TRIALS Monte-Carlo-Trials
STEPS_FLOOR = 200
MC_SMALL_N = 100
MC_SMALL_TRIALS = 100

# Batch-Deckel gegen die Speicherbandbreite.  Der Optimizer haelt pro Step
# rund acht (mc,n)-Arrays; ab mc*n ~ 5e5 Elementen faellt das Working Set
# aus dem L2 und der Step wird DRAM-bandbreitenlimitiert (Faktor ~3).  Das
# Effizienzplateau liegt bei mc*n zwischen 1e5 und 2.5e5 (~12 ns pro
# geprueftem Flip) — siehe die Performance-Section im Optimizer-README.
# Ohne Deckel laeuft der Indexer bei grossen KBs genau dort hinein: eine
# 100k-KB liegt auf den Ebenen 0-2 bei mc*n = 5e5 .. 2e6.
# MC_TARGET_MCN = 0 schaltet den Deckel ab (Verhalten vor 2026-09-07).
MC_TARGET_MCN = 2.0e5     # Ziel fuer num_mc * n
MC_CAP_MIN = 8            # so weit darf der Deckel num_mc hoechstens druecken

# Plateau-Phase am Anfang des Annealings (hold_steps): die Starttemperatur
# wird so viele Steps konstant gehalten, bevor die Kurve abfällt — eine
# echte Explorationsphase, damit der DA das Lloyd-Becken verlassen kann.
# Halbiert sich je Ebene wie die Steps und wird auf HOLD_MAX_FRAC der
# tatsächlich gefahrenen Steps gedeckelt (sonst frisst das Plateau bei
# tiefen Ebenen, wo STEPS_FLOOR greift, den ganzen Lauf).
HOLD_MAX_FRAC = 0.5


def create_pbf(Q: np.ndarray, int_keys: bool = True) -> dict:
    """
    Build the PBF (Pseudo-Boolean Function) for binary graph clustering
    directly from the dense normalized Gram matrix.

    SYMMETRISCH ZUSAMMENGELEGT — nur noch n(n+1)/2 statt n² quadratische
    Terme:

        linear    : pbf[n²+i]  = Q_row[i] + Q_col[i]  = 2·d_i  (zuerst!)
                    with Q_row = np.sum(Q, axis=0),  Q_col = np.sum(Q, axis=1)
        quadratic : pbf[i·n+j] = −2·(Q[i,j] + Q[j,i]) = −4·Q[i,j]   für i < j
                    pbf[i·n+i] = −2·Q[i,i]                          (Diagonale)

    LINEARTERM (2026-08-08 auf 2·d_i gebracht, vorher Q_row·Q_col = d_i²):
    die Referenzformulierung ist die SPIN-Form min Σ Q_ij s_i s_j mit
    s ∈ {−1,+1}.  Die Umrechnung s = 2x−1 auf x ∈ {0,1} erzeugt

        Σ_ij Q_ij s_i s_j = 4·Σ_ij Q_ij x_i x_j − 4·Σ_i d_i x_i + const

    der Linearterm ist also proportional zu d_i, nicht zu d_i².  In der
    Skalierung hier (A = −2Q) ist das b_i = 2·d_i, und damit gilt exakt
    E(x) = −½ · Σ Q_ij s_i s_j (verifiziert, Spannweite 1.6e-14).

    PRAKTISCH IRRELEVANT, aber formal richtig: normalize_coords ZENTRIERT
    die Punkte, also ist Σ x̃_i = 0 und damit d_i = x̃_i·S = 0 (gemessen
    |d_i| ≤ 4e-15 gegen eine Quadratterm-Skala von 0.15).  Beide Varianten
    des Linearterms sind im echten Pfad Maschinenrauschen; die Zielfunktion
    ist effektiv E(x) = −2‖S_A‖² mit S_A = Σ_{i∈A} x̃_i.

    Bezug zum 2-Means-Ziel 2·n1·n2·‖μ1−μ2‖²: bei BALANCIERTEN Splits
    exakt äquivalent (Korrelation −1.0000000000), über alle Splits nur
    −0.867 — der Faktor n1·n2 fehlt in der QUBO-Form.  Die Zentrierung
    reguliert die Größe aber von selbst (A = alles ⇒ S_A = 0 ⇒ E = 0):
    die Enumeration bei n=16 findet das globale Minimum bei |A| = 8,
    identisch mit dem besten balancierten Split.  Ein Balance-Term
    λ(Σx − n/2)² ist daher nicht nötig.

    Keys sind GEPACKTE INTS (Schema siehe oben / unpack_key) — 2.7×
    weniger Speicher als Tupel-Keys, schnelleres Hashing.

    Da x_i·x_j = x_j·x_i, ist E(x) exakt gleich der Vollform mit beiden
    Richtungen — gleiche Landschaft, gleiche ΔE, Kalibrierung unberührt.
    Halber Speicher, und der DA iteriert pro Flip über halb so viele
    Monome.  (Sparse-Pfad mit Threshold: siehe create_pbf_sparse.)
    """
    n = Q.shape[0]
    Q_row = np.sum(Q, axis=0)
    Q_col = np.sum(Q, axis=1)

    pbf: dict = {}
    n2 = n * n
    for i in range(n):                             # linear zuerst (Reihenfolge!)
        # 2·d_i (Q symmetrisch → Q_row + Q_col = 2·Zeilensumme)
        pbf[(n2 + i) if int_keys else (i,)] = float(Q_row[i] + Q_col[i])

    ii, jj = np.triu_indices(n)
    coef = -2.0 * (Q[ii, jj] + Q[jj, ii])          # i<j: −4·Q_ij (exakt beide Seiten)
    diag = ii == jj
    coef[diag] = -2.0 * Q[ii[diag], jj[diag]]      # Diagonale nur einfach
    if int_keys:
        keys = (ii.astype(np.int64) * n + jj).tolist()   # gepackte int-Keys
    else:
        keys = list(zip(ii.tolist(), jj.tolist()))       # Tupel (alte Lib)
    pbf.update(zip(keys, coef.tolist()))

    return pbf


def create_pbf_sparse(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                      Q_row: np.ndarray, Q_col: np.ndarray,
                      int_keys: bool = True) -> dict:
    """
    PBF direkt aus der sparse Gram-Darstellung (sparse_gram_entries).

    Identische Formel wie create_pbf (Linearterm 2·d_i, symmetrisch
    zusammengelegt: obere Dreiecksmatrix, i<j mit −4·Q_ij, Diagonale
    −2·Q_ii) — und Einträge unter der Schwelle kommen NIRGENDS vor: nicht
    als quadratische Terme und nicht in den Zeilen-/Spaltensummen
    Q_row/Q_col der Linearterme.
    """
    n = len(Q_row)
    pbf: dict = {}
    n2 = n * n
    for i in range(n):                      # linear zuerst (Reihenfolge!)
        pbf[(n2 + i) if int_keys else (i,)] = float(Q_row[i] + Q_col[i])
    keep = rows <= cols                     # obere Dreiecksmatrix inkl. Diagonale
    r, c_, v = rows[keep], cols[keep], vals[keep]
    coef = np.where(r == c_, -2.0 * v, -4.0 * v)   # Symmetriepartner zusammengelegt
    if int_keys:
        keys = (r.astype(np.int64) * n + c_).tolist()    # gepackte int-Keys
    else:
        keys = list(zip(r.tolist(), c_.tolist()))        # Tupel (alte Lib)
    pbf.update(zip(keys, coef.tolist()))
    return pbf


# ═══════════════════════════════════════════════════════════════════════════════
#  QUBO-Matrixform (solver="fast") — energie-identisch zu create_pbf(_sparse)
#
#  Kanonische Form der neuen Lib:  E(x) = xᵀAx + bᵀx + c
#      A = -(Q + Qᵀ), Diagonale 0        (Gram symmetrisch → A_ij = -2·Q_ij)
#      b = Q_row + Q_col − 2·diag(Q)     (= 2·d_i − 2·Q_ii, da x_i² = x_i)
#  ⇒ identisch zu  Σ 2·d_i·x_i − 2·xᵀQx  (pbf_energy_dense) —
#  gleiche Landschaft wie der PBF-Pfad, gleiche ΔE.  Es entsteht KEIN
#  PBF-Dict mehr (Zeile k von A ist die Monomliste von x_k).
#  Zur Korrektur des Linearterms (d_i² → 2·d_i) siehe create_pbf.
# ═══════════════════════════════════════════════════════════════════════════════

def qubo_from_gram_dense(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Dichte Gram-Matrix → (A, b, c) für qubo_min_solver."""
    Q_row = np.sum(Q, axis=0)
    Q_col = np.sum(Q, axis=1)
    b = Q_row + Q_col - 2.0 * np.diag(Q)        # 2·d_i − 2·Q_ii
    A = -(Q + Q.T)
    np.fill_diagonal(A, 0.0)
    return A, b, 0.0


def qubo_from_gram_coo(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                       Q_row: np.ndarray, Q_col: np.ndarray):
    """Sparse Gram (COO, beide Ordnungen) → (A_csr, b, c) — Einträge unter
    der Schwelle existieren nirgends (wie create_pbf_sparse)."""
    from scipy import sparse
    n = len(Q_row)
    b = (Q_row + Q_col).astype(np.float64)      # 2·d_i
    diag = rows == cols
    b[rows[diag]] -= 2.0 * vals[diag]
    off = ~diag
    A = sparse.csr_matrix((-2.0 * vals[off], (rows[off], cols[off])),
                          shape=(n, n))
    return A, b, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Descent-konsistente Neuzuordnung (Linos Idee, 2026-08-08)
# ═══════════════════════════════════════════════════════════════════════════════
#  Das QUBO ordnet nach PAARWEISER Ähnlichkeit unter Balance-Zwang zu, der
#  Descent in query.py fragt aber nach ZENTROID-Nähe.  Ein Punkt kann also
#  im Split korrekt in A liegen und trotzdem näher an m_B sitzen — genau
#  den findet der Descent nie wieder.
#
#  Abhilfe: nach dem Split die beiden Zentroide FESTHALTEN und alle Punkte
#  des Knotens neu dem näheren zuordnen.  Danach ist die Ebene per
#  Konstruktion descent-konsistent — vorausgesetzt, der Baum speichert
#  eben diese Split-Zentroide (nicht die Mittelwerte der neuen Gruppen,
#  sonst wandert der Bezugspunkt wieder weg).
#
#  Danach gilt eine GARANTIE: ein Punkt liegt auf jeder Ebene im Kind mit
#  dem näheren Zentroid, ein gieriger Abstieg (beam=1) findet ihn also
#  immer wieder — gemessen exakt 100 % Selbst-Retrieval.
#
#  Gemessen an einem fertigen 4082er-Index: 3.6 % aller Zuordnungen sind
#  inkonsistent, mit den Spitzen auf Tiefe 2/4/5 (je ~5 %) — also genau
#  dort, wo die Descent-Fehler sitzen.  Die Balance leidet nur mäßig
#  (schiefster Split 33/67), die Baumtiefe bleibt unverändert.
#  Gemessen: der Guard greift bei echten Embeddings GAR NICHT — der
#  schiefste Split nach Neuzuordnung war 33/67 (n=57), weit weg von einer
#  Entartung, und Selbst-Retrieval mit beam=1 ist mit wie ohne Guard exakt
#  100 %.  Er steht auf 1.0 (= aus), damit die Descent-Konsistenz
#  lückenlos gilt; der Mechanismus bleibt für pathologische Daten drin.
REASSIGN_MAX_IMB = 1.0      # |n_A − n_B| / n darf höchstens so groß werden
                            # (1.0 = kein Limit)


def reassign_to_split_centroids(
    idx_A: list[int], idx_B: list[int], X: np.ndarray,
    max_imbalance: float = REASSIGN_MAX_IMB,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray, int]:
    """
    Alle Punkte beider Gruppen dem näheren der beiden (festgehaltenen)
    Split-Zentroide zuordnen — Cosine, wie der Descent.

    Rückgabe: (neu_A, neu_B, mean_A, mean_B, umgehängte Punkte).
    Die zurückgegebenen Zentroide sind die Split-Zentroide VOR der
    Neuzuordnung — sie gehören in den Baum, damit die Konsistenz hält.
    """
    a = np.asarray(idx_A, dtype=int)
    b = np.asarray(idx_B, dtype=int)
    m_A = X[a].mean(axis=0)
    m_B = X[b].mean(axis=0)

    allidx = np.concatenate([a, b])
    side = np.concatenate([np.zeros(len(a), dtype=int),
                           np.ones(len(b), dtype=int)])
    P = X[allidx]
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
    nA = m_A / max(float(np.linalg.norm(m_A)), 1e-30)
    nB = m_B / max(float(np.linalg.norm(m_B)), 1e-30)
    sim = np.stack([Pn @ nA, Pn @ nB], axis=1)

    new = np.argmax(sim, axis=1)
    margin = np.abs(sim[:, 0] - sim[:, 1])
    n = len(allidx)

    # Guard: Wechsel mit der KLEINSTEN Marge zurücknehmen, bis das
    # Ungleichgewicht wieder im Rahmen ist (kleine Marge = wackelige
    # Entscheidung, die kostet am wenigsten).
    limit = int(max_imbalance * n)
    diff = int((new == 0).sum() - (new == 1).sum())
    if abs(diff) > limit:
        over = 0 if diff > 0 else 1              # überfüllte Seite
        cand = np.where((new == over) & (side != over))[0]   # Zuwanderer
        cand = cand[np.argsort(margin[cand])]                # schwächste zuerst
        need = (abs(diff) - limit + 1) // 2
        for k in cand[:need]:
            new[k] = 1 - over

    moved = int((new != side).sum())
    return (allidx[new == 0].tolist(), allidx[new == 1].tolist(),
            m_A, m_B, moved)



# ═══════════════════════════════════════════════════════════════════════════════
#  PBF energy evaluation  (E_start / E_Lloyd für das da_gp-Cooling)
# ═══════════════════════════════════════════════════════════════════════════════

def pbf_energy_dense(Q: np.ndarray, x: np.ndarray) -> float:
    """
    E(x) des kalibrierten PBF, direkt aus der dichten Gram-Matrix:

        E(x) = Σ_i Q_row[i]·Q_col[i]·x_i  −  2·xᵀQx

    (identisch zur Auswertung des create_pbf-Dicts, nur als BLAS-Ausdruck).
    """
    x = np.asarray(x, dtype=float)
    lin = float(np.dot(np.sum(Q, axis=0) * np.sum(Q, axis=1), x))
    return lin - 2.0 * float(x @ Q @ x)


def pbf_energy_sparse(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
                      Q_row: np.ndarray, Q_col: np.ndarray,
                      x: np.ndarray) -> float:
    """E(x) des sparse PBF (create_pbf_sparse) aus der COO-Darstellung."""
    x = np.asarray(x, dtype=float)
    lin = float(np.dot(Q_row * Q_col, x))
    quad = -2.0 * float(np.sum(vals * x[rows] * x[cols]))
    return lin + quad


# ═══════════════════════════════════════════════════════════════════════════════
#  Lloyd (k=2) — optionaler Vorschritt vor dem DA
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  Max-Margin-Trennebene (2026-09-21)
# ═══════════════════════════════════════════════════════════════════════════════
#  Die Zentroid-Zuordnung ist geometrisch eine Hyperebene DURCH DEN URSPRUNG
#  mit Normale w = m̂_A − m̂_B:  "näher an A"  ⟺  p·w > 0.  Punkte dicht an
#  dieser Ebene sind die fragilen — eine Query, die nur leicht neben ihrem
#  Chunk liegt, kippt auf die falsche Seite.
#
#  Hält man die Richtung w fest, ist die Lage der Ebene ein EINDIMENSIONALES
#  Problem: projiziere t_i = p_i·ŵ, dann ist jede Parallelverschiebung nur
#  ein Schwellwert τ, und die Marge ist min_i |t_i − τ|.  Die beste Schwelle
#  liegt in der Mitte der GRÖSSTEN LÜCKE der sortierten t_i — sortieren,
#  Differenzen, Maximum.  (Derselbe Gedanke wie ein CART-Split, nur entlang
#  einer schrägen Richtung statt einer Koordinatenachse.)
#
#  Garantie: hat ein Knoten Marge δ, dann biegt der gierige Abstieg für jede
#  Query q mit ‖q − p‖ < δ genauso ab wie für den Chunk p — denn
#  |q·ŵ − p·ŵ| ≤ ‖q − p‖ < δ.
#
#  Gemessen am 4082er-Baum: die beste Lücke ist im Median 15.5× so breit wie
#  der typische Nachbarabstand der Projektionen, die Marge verdreifacht sich.
#  ABER das gilt vor allem TIEF im Baum (ab Tiefe 5 leert sich die Zone
#  |t−τ| < 0.02 von 2.0 % auf 0.8 %, bei Tiefe 6 auf 0.0 %).  Oben ist der
#  Raum im relevanten Maßstab ein Kontinuum: an der Wurzel bleibt die beste
#  Marge bei 0.001 gegen eine Streuung von 0.18.
#
#  Deshalb zusätzlich SPILL: die spill·n Punkte, die der Ebene am nächsten
#  liegen, gehen in BEIDE Kinder.  Das braucht keine Lücke und fängt genau
#  die Fälle, für die Verschieben oben nichts ausrichten kann.  Preis ist
#  Speicher: (1 + spill)^depth.  Als PUNKTANTEIL definiert, nicht als
#  σ-Vielfaches — 1σ wären rund 68 % der Punkte und der Baum explodiert.
MARGIN_QUANTILE = 0.2       # Lücke nur im mittleren Korridor suchen
                            # (sonst spaltet die größte Lücke einen einzelnen
                            #  Ausreißer ab und der Baum entartet zur Liste)


def plane_from_split(
    idx_A: list[int], idx_B: list[int], X: np.ndarray, *,
    margin_quantile: float = MARGIN_QUANTILE, spill: float = 0.0,
    shift: bool = True, eps: float = 0.0,
):
    """
    Aus einem fertigen Split die Max-Margin-Trennebene bestimmen.

    Richtung ŵ = normiert(m̂_A − m̂_B) aus den Zentroiden des Splits, Lage τ
    in der Mitte einer Lücke der Projektionen (im mittleren Korridor).
    Punkte werden danach neu zugeordnet: t > τ → A, sonst B.

    shift=False: die Ebene bleibt bei τ = 0, also GENAU die Zuordnung zum
    näheren Zentroid — das ist der split_mode "centroid".  Die Richtung,
    der Spill und das gespeicherte (w, τ, σ) sind dieselben; verschoben
    wird nur nicht.  So sind "wo liegt die Ebene" und "wie viel
    Überlappung" zwei unabhängige Regler.

    eps — Unsicherheitsradius der Embeddings, gemessen IN DER EINHEIT DER
    PROJEKTION t (siehe estimate_eps).  Jeder Punkt trägt damit keine
    Koordinate mehr, sondern ein Intervall [t_i − eps, t_i + eps]; wird es
    von der Ebene geschnitten, ist die Zuordnung dieses Punktes nicht
    belastbar.  eps ändert beides:
      τ-Wahl  — unter allen Lücken gewinnt die mit den WENIGSTEN Punkten in
                der Zone |t − τ| ≤ eps, bei Gleichstand die breiteste.
                eps = 0 macht jede Zone leer ⇒ exakt die alte
                Größte-Lücke-Regel.
      Spill   — es spillen genau die Punkte der Zone statt eines festen
                Anteils.  Wo die Ebene sauber durch eine breite Lücke geht
                (tief im Baum die Regel), spillt gar nichts.

    spill > 0: diese Punkte landen in BEIDEN Gruppen.  Als Punktanteil
    definiert (nicht als σ-Vielfaches), damit der Überhang selbst-
    begrenzend und vorhersagbar ist: die Gesamtpunktzahl wächst je Ebene
    um (1 + spill), über die Tiefe also (1 + spill)^depth.  Mit eps > 0
    ist spill nur noch die OBERGRENZE — sind mehr Punkte in der Zone,
    spillen die spill·n ebenennächsten; sind es weniger, spillen weniger.
    So bleibt die Kostenschranke aus docs/pipeline.md erhalten, während die
    Auswahl datengetrieben wird.

    Rückgabe None, wenn die Ebene entartet, sonst dict mit
    w, tau, sigma, margin, idx_A, idx_B, moved, spilled, in_eps.
    """
    a = np.asarray(idx_A, dtype=int)
    b = np.asarray(idx_B, dtype=int)
    allidx = np.concatenate([a, b])
    P = X[allidx]
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)

    mA = Pn[:len(a)].mean(axis=0)
    mB = Pn[len(a):].mean(axis=0)
    mA = mA / max(float(np.linalg.norm(mA)), 1e-30)
    mB = mB / max(float(np.linalg.norm(mB)), 1e-30)
    w = mA - mB
    nw = float(np.linalg.norm(w))
    if nw < 1e-12:                      # Zentroide fallen zusammen
        return None
    w = w / nw

    t = Pn @ w
    n = len(t)
    ts = np.sort(t)
    sigma = float(np.std(t))

    lo = max(1, int(margin_quantile * n))
    hi = min(n - 1, int((1.0 - margin_quantile) * n))
    if not shift or hi <= lo:           # centroid-Modus / zu klein
        tau, margin = 0.0, float(np.min(np.abs(t)))
    else:
        # Kandidaten sind die Mitten aller Lücken im mittleren Korridor.
        # Bewertet wird lexikographisch (Zonenbelegung, −Lückenbreite):
        # zuerst so wenige unsichere Punkte wie möglich, dann so viel
        # Marge wie möglich.  Mit eps = 0 ist die Belegung überall 0 und
        # es bleibt die reine Größte-Lücke-Regel.
        gaps = ts[lo + 1:hi + 1] - ts[lo:hi]
        mids = 0.5 * (ts[lo:hi] + ts[lo + 1:hi + 1])
        if eps > 0.0:
            # #{ |t − τ| ≤ eps } für alle Kandidaten auf einen Schlag
            occ = (np.searchsorted(ts, mids + eps, side="right")
                   - np.searchsorted(ts, mids - eps, side="left"))
            k = int(np.lexsort((-gaps, occ))[0])
        else:
            k = int(np.argmax(gaps))
        tau = float(mids[k])
        margin = 0.5 * float(gaps[k])

    side = t > tau                      # True = Seite A
    if not side.any() or side.all():
        return None

    old = np.concatenate([np.ones(len(a), bool), np.zeros(len(b), bool)])
    moved = int((side != old).sum())

    new_A = allidx[side]
    new_B = allidx[~side]
    dist = np.abs(t - tau)
    in_eps = int((dist <= eps).sum()) if eps > 0.0 else 0

    # Wer spillt?  eps = 0 → die k_spill ebenennächsten Punkte (fester
    # Anteil).  eps > 0 → die Punkte der Unsicherheitszone, aber nie mehr
    # als k_spill; liegen mehr in der Zone, gewinnen die nächsten.
    k_spill = int(spill * n)
    if eps > 0.0:
        k_spill = min(k_spill, in_eps)

    spilled = 0
    if k_spill > 0:
        near = np.argpartition(dist, k_spill - 1)[:k_spill]
        zone = np.zeros(n, dtype=bool)
        zone[near] = True
        extra_A = allidx[zone & ~side]              # gehören zu B, auch nach A
        extra_B = allidx[zone & side]
        new_A = np.concatenate([new_A, extra_A])
        new_B = np.concatenate([new_B, extra_B])
        spilled = int(zone.sum())

    return {"w": w, "tau": tau, "sigma": sigma, "margin": margin,
            "idx_A": new_A.tolist(), "idx_B": new_B.tolist(),
            "moved": moved, "spilled": spilled, "in_eps": in_eps}


# ═══════════════════════════════════════════════════════════════════════════════
#  ε — Unsicherheitsradius der Embeddings (2026-09-23)
# ═══════════════════════════════════════════════════════════════════════════════
#  Ein Embedding ist keine exakte Koordinate.  Derselbe Inhalt, anders
#  formuliert oder anders beschnitten, landet ein Stück daneben; eine Query
#  zu einem Chunk erst recht.  Nennt man diesen Radius ε, dann ist jeder
#  Punkt eine KUGEL, und eine Trennebene, die eine Kugel schneidet, trifft
#  für diesen Punkt eine Entscheidung, die das Embedding gar nicht hergibt.
#
#  Genau diese Punkte — und nur die — gehören in beide Kinder.  Das ersetzt
#  den festen Spill-Anteil durch ein Kriterium: nicht "die 10 % nächsten",
#  sondern "alle, deren Unsicherheit über die Ebene reicht".
#
#  EINHEIT.  Der Descent vergleicht t = p̂·ŵ, also die Projektion des
#  NORMIERTEN Punktes auf eine Einheitsrichtung.  ε muss in derselben
#  Einheit vorliegen, nicht als Abstand im Einbettungsraum: eine Störung
#  der Länge ‖Δp̂‖ verschiebt t um höchstens ‖Δp̂‖ (Cauchy-Schwarz), im
#  Mittel aber nur um ~‖Δp̂‖/√d — bei d = 1024 ein Faktor 32.  Wer den
#  vollen Abstand einsetzt, spillt den halben Baum.
#
#  Deshalb wird ε direkt als Streuung der PROJEKTION gemessen: gestörte
#  Paare (p̂, p̂′) auf Richtungen projizieren, wie sie im Baum wirklich
#  vorkommen (Differenzen zufälliger Punktpaare — das ist die Statistik
#  einer Zentroid-Differenz), und davon ein hohes Quantil nehmen.
EPS_QUANTILE = 0.9          # so viel der Störungen soll ε abdecken
EPS_DIRS = 256              # Richtungen für die Projektionsstatistik


def estimate_eps(
    P: np.ndarray,
    P_pert: np.ndarray,
    *,
    quantile: float = EPS_QUANTILE,
    n_dirs: int = EPS_DIRS,
    directions_from: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """
    ε aus gestörten Embedding-Paaren schätzen — in Projektionseinheiten.

    P, P_pert — (S, d): S Embeddings und dieselben S Inhalte nach einer
    Störung (Paraphrase, Ausschnitt, Tippfehler …).  Beide werden normiert,
    gepaart müssen sie zeilenweise zusammengehören.

    directions_from — (M, d) Punktwolke, aus deren zufälligen Paar-
    differenzen die Testrichtungen gezogen werden.  Das ist die richtige
    Referenz, weil die Splitrichtung ŵ = m̂_A − m̂_B im Baum genau so
    entsteht.  None → isotrope Zufallsrichtungen (konservativer Ersatz).

    Rückgabe: {"eps", "median", "q90", "q99", "n_pairs", "n_dirs"} — eps ist
    das gewählte Quantil von |(p̂ − p̂′)·ŵ|.
    """
    rng = np.random.default_rng(seed)

    def _unit(V):
        V = np.asarray(V, dtype=float)
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-30)

    A = _unit(P)
    B = _unit(P_pert)
    if A.shape != B.shape:
        raise ValueError(f"P {A.shape} und P_pert {B.shape} müssen "
                         f"zeilenweise gepaart sein.")
    D = A - B                                   # Störvektoren

    d = A.shape[1]
    if directions_from is not None:
        R = _unit(directions_from)
        m = len(R)
        i = rng.integers(0, m, size=n_dirs)
        j = rng.integers(0, m, size=n_dirs)
        W = R[i] - R[j]
        keep = np.linalg.norm(W, axis=1) > 1e-12
        W = W[keep] if keep.any() else rng.standard_normal((n_dirs, d))
    else:
        W = rng.standard_normal((n_dirs, d))
    W = _unit(W)

    shifts = np.abs(D @ W.T).ravel()            # |Δt| über alle Paar×Richtung
    return {
        "eps": float(np.quantile(shifts, quantile)),
        "median": float(np.median(shifts)),
        "q90": float(np.quantile(shifts, 0.9)),
        "q99": float(np.quantile(shifts, 0.99)),
        "n_pairs": int(len(D)),
        "n_dirs": int(len(W)),
    }


def eps_from_cosine(cos_sim: float, dim: int) -> float:
    """
    ε ohne Messung, aus einer bekannten Paraphrasen-Ähnlichkeit.

    cos_sim ist die typische Cosine zwischen zwei Formulierungen desselben
    Inhalts (bge-m3: ~0.9).  Der Abstand der normierten Vektoren ist dann
    ‖Δp̂‖ = √(2 − 2·cos); die Projektion auf eine Richtung, die mit der
    Störung nichts zu tun hat, sieht davon im Mittel den Anteil 1/√d.

    Nur eine Hausnummer für den Fall, dass keine gestörten Embeddings
    vorliegen — estimate_eps misst dasselbe am echten Encoder.
    """
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * cos_sim)) / np.sqrt(max(dim, 1)))


def lloyd_bisect(
    xg: np.ndarray,
    iterations: int,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    """
    k-Means-Bisektion (Lloyd, k=2) auf den normalisierten Gruppenkoordinaten.

    Start ist eine ZUFÄLLIGE Binärzuweisung (oder *start*, falls gegeben);
    iteriert bis sich die Zuweisung nicht mehr ändert oder *iterations*
    erreicht ist.  Leere Seiten werden repariert (der am weitesten vom
    verbleibenden Zentroid entfernte Punkt wechselt die Seite).

    Returns
    -------
    (assign, iters_used, converged)
      assign     : (n,) int-Array 0/1
      iters_used : tatsächlich gelaufene Lloyd-Iterationen
      converged  : True, wenn vor Erreichen von *iterations* stabil
    """
    n = len(xg)
    if start is None:
        assign = np.random.randint(0, 2, size=n)
    else:
        assign = np.asarray(start, dtype=int).copy()

    iters_used = 0
    converged = False
    for it in range(int(iterations)):
        # Leere Seite reparieren, sonst ist kein Zentroid definiert
        for side in (0, 1):
            if not np.any(assign == side):
                other_c = xg[assign == 1 - side].mean(axis=0)
                far = int(np.argmax(np.linalg.norm(xg - other_c, axis=1)))
                assign[far] = side
        c0 = xg[assign == 0].mean(axis=0)
        c1 = xg[assign == 1].mean(axis=0)
        d0 = ((xg - c0) ** 2).sum(axis=1)
        d1 = ((xg - c1) ** 2).sum(axis=1)
        new = (d1 < d0).astype(int)
        iters_used = it + 1
        if np.array_equal(new, assign):
            converged = True
            break
        assign = new
    return assign, iters_used, converged


# ═══════════════════════════════════════════════════════════════════════════════
#  Knoten-Repräsentation: k Anker statt einem Mittelwert
# ═══════════════════════════════════════════════════════════════════════════════
#  Der Beam braucht je Knoten eine Antwort auf "wie gut passt dieser Teilbaum
#  zur Query" — und die richtige Antwort wäre max_p cos(q, p) über alle Punkte
#  p des Teilbaums.  EIN Mittelwert beantwortet das schlecht: gemessen am
#  4178er-Baum haben die beiden Wurzelkinder eine Zentroid-Cosine von 0.905,
#  zeigen also fast in dieselbe Richtung, und 21 % aller Queries biegen schon
#  an der Wurzel falsch ab.  100 % dieser Fehler verschwinden, wenn der Knoten
#  statt des Mittelwerts seinen besten Punkt zeigen darf.
#
#  k Anker (k-Means-Zentren der Knotenpunkte) sind die praktikable Fassung
#  davon: sie beschreiben die Gruppe als Vereinigung von k Kugeln statt als
#  eine Kugel um den Schwerpunkt, und max_j cos(q, a_j) approximiert das
#  Maximum über die Punkte.  Das ist dieselbe Idee wie die 2026-08 verworfene
#  Radius-Schranke, nur feiner: eine Kugel um den Mittelwert war zu grob, um
#  je einen Ast abzuschneiden.
#
#  WICHTIG: Anker SEPARIEREN nichts.  Die Trennung macht weiter die Ebene
#  (w, tau) bzw. der nähere Zentroid; die Anker ändern nur den Beam-Score.
#  Sie leben deshalb auch im ROHEN Raum, wie mean, w und tau.

# ═══════════════════════════════════════════════════════════════════════════════
#  Zielfunktion: Ähnlichkeit (Gram) oder Nachbarschaft (k-NN-Modularity)
# ═══════════════════════════════════════════════════════════════════════════════
#  Die Gram-Matrix bewertet JEDES Paar nach seiner Ähnlichkeit.  Das ist (bei
#  balancierten Splits) 2-Means — und misst damit Partitionsgüte, nicht das,
#  was ein RAG-Index braucht.  Gemessen am 4178er-Korpus:
#
#      Korrelation mit recall@5 …   QUBO-Energie (Gram)     −0.31
#                                   zerschnittene k-NN      −0.93
#
#  Was zählt, ist also: wie viele ECHTE Nachbarschaften zerschneidet der
#  Schnitt.  Das ist Graph-Bisection auf dem k-NN-Graphen — nativ quadratisch,
#  kein Grad-4-Problem wie das exakte 2-Means mit seinen |A|-Nennern, und
#  dünn besetzt (0.3 % statt 100 %).
#
#  Balance kommt über das MODULARITY-Nullmodell statt über einen Penalty:
#      B_ij = A_ij − d_i d_j / (2m)
#  Ein Schnitt, der alles auf eine Seite legt, bringt damit nichts mehr — die
#  erwarteten Kanten heben die tatsächlichen exakt auf.  Das ist dieselbe
#  Rolle, die im Gram-Pfad die Zentrierung spielt (Σx̃ = 0 ⇒ A=alles ⇒ E=0),
#  nur für einen Graphen statt für eine Punktwolke.

KNN_K = 10          # Nachbarn je Punkt, aus denen der Graph gebaut wird


def knn_modularity_matrix(coords: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """
    Modularity-Matrix des symmetrischen k-NN-Graphen der Punkte.

    Nutzt dieselbe O(n²)-Ähnlichkeitsrechnung wie die Gram-Matrix — der
    k-NN-Graph fällt dabei als Top-k je Zeile praktisch gratis ab.
    """
    P = np.asarray(coords, dtype=float)
    n = len(P)
    if n < 3:
        return np.zeros((n, n))
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
    S = Pn @ Pn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), kk), idx.ravel()] = 1.0
    A = np.maximum(A, A.T)              # symmetrisieren
    d = A.sum(axis=1)
    m2 = d.sum()                        # = 2·|E|
    if m2 <= 0:
        return A
    return A - np.outer(d, d) / m2


def node_anchors(P: np.ndarray, k: int, *, iters: int = 10,
                 seed: int = 0) -> np.ndarray:
    """
    k Anker einer Punktmenge: k-Means auf der Kugel (Cosine), Zentren normiert.

    Weniger Punkte als k → die Punkte selbst.  k <= 1 → der Mittelwert, also
    exakt das bisherige Verhalten.
    """
    P = np.asarray(P, dtype=float)
    if k <= 1 or len(P) <= 1:
        return _unit_rows(P.mean(axis=0)[None, :])
    if len(P) <= k:
        return _unit_rows(P)
    rng = np.random.default_rng(seed)
    C = P[rng.choice(len(P), k, replace=False)].copy()
    for _ in range(int(iters)):
        a = np.argmax(P @ C.T, axis=1)
        for j in range(k):
            m = P[a == j]
            if len(m):
                C[j] = m.mean(axis=0)
        C = _unit_rows(C)
    return C


def _unit_rows(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-30)


def lloyd_cost(xg: np.ndarray, assign: np.ndarray) -> float:
    """
    2-Means-Zielwert Σ_c Σ_{i∈c} ‖x_i − μ_c‖² einer Zuweisung — O(n·d).

    Nur für den Pfad OHNE Annealing gedacht: dort existiert kein QUBO, also
    auch keine QUBO-Energie, mit der die Lloyd-Multistarts sonst gegen-
    einander bewertet werden.  Bei balancierten Splits ist das dieselbe
    Rangfolge (2-Means ⇔ QUBO, siehe Modulkopf), kostet aber O(n·d) statt
    O(n²).
    """
    a = np.asarray(assign, dtype=int)
    total = 0.0
    for side in (0, 1):
        m = a == side
        if not m.any():
            continue
        pts = xg[m]
        total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


# ═══════════════════════════════════════════════════════════════════════════════
#  PCA-Scores für orientierte Lloyd-Multi-Starts
# ═══════════════════════════════════════════════════════════════════════════════

def pca_scores(xg: np.ndarray, k: int, iters: int = 3,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-k Hauptkomponenten-SCORES via randomized SVD (Halko) auf den
    zentrierten Gruppenkoordinaten — O(n·d·k), ~ms selbst am Root
    (gemessen: 18 ms @ n=4082, d=1024; exakte Gram-eigsh: 3.6 s).

    Gram-Seite der PCA: scores[:, i] ist die Projektion aller n Punkte auf
    die i-te Hauptkomponente (= Gram-Eigenvektor · σ_i), eigvals[i] = σ_i²
    die zugehörige Varianz.  Die d-Raum-Richtungen werden nie gebraucht —
    ein Sign-Split `scores[:, i] > 0` trennt die beiden Halbräume senkrecht
    zur Komponente (Schwerpunkte in entgegengesetzten Richtungen).

    Returns
    -------
    (eigvals (k,), scores (n, k))
    """
    n, d = xg.shape
    k = max(1, min(k, min(n, d) - 1))
    p = max(2, min(4, min(n, d) - k))            # Oversampling
    rng = np.random.default_rng(seed)
    Y = xg @ rng.standard_normal((d, k + p))
    for _ in range(iters):                        # Power-Iterationen + Re-Orth.
        Y, _ = np.linalg.qr(xg @ (xg.T @ Y))
    Qb, _ = np.linalg.qr(Y)
    Ub, s, _ = np.linalg.svd(Qb.T @ xg, full_matrices=False)
    U = Qb @ Ub
    return s[:k] ** 2, U[:, :k] * s[:k]


# ═══════════════════════════════════════════════════════════════════════════════
#  Single bisection step
# ═══════════════════════════════════════════════════════════════════════════════

def bisect_group(
    idx: list[int],
    X: np.ndarray,
    *,
    total_n: int,
    depth: int = 0,
    steps: int = 200,
    num_mc: int = 20,
    offset_increase: float | str = "auto_gp",   # Zahl oder "auto_gp" (dynamisch im Solver)
    cooling_a: float | None = None,
    cooling_b: float = 0.0,
    cooling_d: float = 2.0,
    iterations_lloyd: int = 10,
    lloyd_mc: int = 1,
    p_hot: float = 0.3,
    p_cold: float = 0.01,
    gamma: float = 0.05,
    cooling_schedule: str = "da_gp",
    cooling_per_group: bool = True,
    hold_steps: int = 0,
    offset_k_escape: float = 25.0,
    split_mode: str = "plane",     # "none" | "centroid" | "plane"
    spill: float = 0.1,
    seed_gen: float | None = None,
    verbose: bool = True,
    visual_inst: bool = False,
    gram_threshold: float = 0.0,
    solver: str = "fast",
    anneal: bool = True,
    objective: str = "gram",   # "gram" | "knn"
    stats: dict | None = None,
    progress_cb=None,
) -> tuple[list[int], list[int]]:
    """
    Split the group given by global indices *idx* into two sub-groups via
    one DA bisection.  The pbf_min_solver call and all its parameters are
    identical to the calibrated original.

    solver — "fast": annealing-qubo-optimizer (heuristischer batched
    numpy-DA, keine Optimalitätsgarantie; QUBO direkt als Matrix,
    KEIN PBF-Dict); "reference": alte
    annealing-cop-approximator-Lib (dict-basiert, kalibrierter Originalpfad;
    benötigt deren int-Key-Support).  Gleiche Kalibrierung (da_gp/auto_gp),
    gleiche Energie-Landschaft — nur der Rechenweg unterscheidet sich.

    anneal — False schaltet das Digital Annealing AUS: der Split ist dann
    die beste Lloyd-Lösung.  Damit entfällt der gesamte QUBO-Zweig
    (Gram-Matrix, PBF/Matrixform, Cooling, Solve), die Bisection kostet nur
    noch O(n·d·iterations) statt O(n²) — der billige Referenzpfad, gegen
    den sich der DA messen lassen muss.  Braucht iterations_lloyd > 0;
    beides aus ergibt keinen Split und ist ein Fehler.

    iterations_lloyd — > 0 schaltet den Lloyd-Vorschritt ein: k=2-Lloyd von
    zufälligem Start, bis konvergiert oder iterations_lloyd erreicht — die
    Lösung bleibt UNVERÄNDERT (auch unbalanciert).  Das Cooling
    wird dann als ["da_gp", steps, None, lloyd_assign, E_lloyd, E_start, p_hot, p_cold, gamma]
    übergeben — den ΔE-Vektor (Slot 2) berechnet pbf_min_solver selbst.
    0 (Default) = aus → kalibriertes ["logarithmic", a, b] wie bisher.

    lloyd_mc — > 1: PCA-orientierte Lloyd-Multi-Starts. Trial m < k nutzt den
    Sign-Split entlang der m-ten Hauptkomponente (Schwerpunkte in entgegen-
    gesetzten Richtungen, k = min(4, lloyd_mc)); weitere Trials streuen
    ENTGEGEN den Hauptkomponenten — Zufallsrichtungen, aus denen der
    Top-PC-Anteil herausprojiziert ist.  Grund: entlang PC1 konvergiert
    Lloyd immer ins selbe Becken, neue Becken liegen im Komplement.
    Jeder Start wird per Lloyd verfeinert, Duplikate
    (gleiches Becken) fliegen raus, die BESTE Lösung kalibriert das Cooling.
    solver="fast": ALLE uniquen Lösungen gehen als Startgruppen in den DA
    (num_mc Trials pro Lösung, ein Batch); solver="reference": nur die beste
    (die alte Lib kann keine Startlisten).  1 = bisheriges Verhalten.

    visual_inst — an pbf_min_solver durchgereicht (DA-Trajektorien-Plot).

    gram_threshold — Sparsity-Schwelle: Einträge |Q[i,j]| < threshold werden
    schon WÄHREND der blockweisen Gram-Berechnung verworfen — die dichte
    n×n-Matrix entsteht nie, und ein verworfener Eintrag existiert nirgends
    (weder als quadratischer Term noch in den Zeilen-/Spaltensummen der
    Linearterme).  0 = aus → kalibrierter dichter Originalpfad.

    stats — optionales dict; wird in-place mit KPIs der Bisection gefüllt
    (n, steps_used, energy, valid_trials, reassigned, …).

    Parameters
    ----------
    idx            : global indices of the current group's points in X
    X              : (N, d) full coordinate array
    total_n        : size of the *original* full dataset (for auto-a)
    depth          : current recursion depth (0-indexed)
    steps          : number of annealing steps
    num_mc         : Monte Carlo repetitions (best result is kept)
    offset_increase: offset_increase_rate for pbf_min_solver
    cooling_a      : a in ["logarithmic", a, b];  None → auto:
                     a = d·total_n/(depth+1)  (kalibrierter Default, d=2)
    cooling_b      : b in ["logarithmic", a, b]  (Default 0)
    cooling_d      : Faktor d im Auto-a  (nur wirksam bei cooling_a=None)
    seed_gen       : global RNG seed (fixed for reproducibility)
    verbose        : print progress

    Returns
    -------
    (idx_A, idx_B) — global index lists
    """
    if seed_gen is None:
        seed_gen = random.uniform(1, 1000)

    if not anneal and iterations_lloyd <= 0:
        raise ValueError(
            "anneal=False und iterations_lloyd=0 zusammen ergeben keinen "
            "Split — mindestens einer der beiden Bisektoren muss laufen.")

    use_fast = solver == "fast"
    if anneal:
        if use_fast:
            _qubo_min_solver, _eval_qubo = _import_fast_solver()
        elif _REFERENCE_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"solver='reference' nicht verfügbar — Import der "
                f"annealing-cop-approximator-Lib schlug fehl: "
                f"{_REFERENCE_IMPORT_ERROR}")

    n = len(idx)
    group = X[idx]

    # 1+2. Gram matrix → QUBO.
    #   thr = 0 : kalibrierter dichter Originalpfad (unverändert).
    #   thr > 0: sparse Pfad — |Q[i,j]| < thr wird schon WÄHREND der
    #            blockweisen Gram-Berechnung verworfen (nie eine dichte
    #            n×n-Matrix im Speicher); Linearterme aus den Überlebenden.
    #   fast    : Matrixform (A, b, c) statt PBF-Dict — Gram-Daten werden
    #             sofort nach dem Umbau freigegeben.
    #   anneal=False: der ganze Block entfällt — ohne DA gibt es kein QUBO
    #            zu lösen, und O(n²) wäre die teuerste Zeile der Bisection.
    n_terms = 0
    zeroed_frac = 0.0
    if not anneal:
        pass
    elif gram_threshold > 0.0:
        rows_, cols_, vals_, Q_row_, Q_col_ = sparse_gram_entries(
            group, gram_threshold)
        zeroed_frac = 1.0 - len(vals_) / (n * n)
        if use_fast:
            A_, b_, c_ = qubo_from_gram_coo(rows_, cols_, vals_,
                                            Q_row_, Q_col_)
            n_terms = int(np.count_nonzero(rows_ <= cols_)) + n
            del rows_, cols_, vals_, Q_row_, Q_col_
        else:
            pbf = create_pbf_sparse(rows_, cols_, vals_, Q_row_, Q_col_,
                                    int_keys=REFERENCE_INT_KEYS)
            n_terms = len(pbf)
    else:
        Q = (knn_modularity_matrix(group) if objective == "knn"
             else clustering_gram_matrix(group))
        zeroed_frac = float((Q == 0.0).mean())
        if use_fast:
            A_, b_, c_ = qubo_from_gram_dense(Q)
            n_terms = n * (n + 1) // 2 + n
            del Q
        else:
            pbf = create_pbf(Q, int_keys=REFERENCE_INT_KEYS)
            n_terms = len(pbf)
    if anneal:
        if verbose:
            print(f"    {'QUBO' if use_fast else 'PBF'}: {n_terms} terms")
        if progress_cb:
            # Sofort melden, sobald das QUBO steht — der DA-Solve danach kann
            # lange dauern, die echte (sparse) Termzahl ist ab hier bekannt.
            progress_cb("pbf_built", {"depth": depth, "n": n,
                                      "pbf_terms": n_terms,
                                      "gram_zeroed_frac": round(zeroed_frac, 4)})

    # Steps-Schedule je Tiefe: HALBIEREN pro Ebene (steps/2^depth), Boden
    # STEPS_FLOOR.  Kleine Gruppen brauchen deutlich weniger Steps als das
    # frühere harmonische steps/(depth+1) — die gesparte Zeit geht in mehr
    # MC-Trials (siehe MC_SMALL_N/MC_SMALL_TRIALS).
    step_annealing = max(STEPS_FLOOR, int(steps / (2 ** depth)))

    # Kleine Gruppen: viele billige MC-Trials statt weniger langer Läufe
    # (der batched Solver rechnet alle Trials in EINEM Durchgang, bei n<100
    # kostet das kaum mehr als ein einzelner Trial).
    if n < MC_SMALL_N:
        num_mc = max(num_mc, MC_SMALL_TRIALS)

    # Deckel: mc*n aus dem DRAM-Regime heraushalten.  Wirkt nur nach unten
    # und nur bei grossen Gruppen — genau dort, wo laut Kalibrierung ohnehin
    # wenige Trials genuegen (die obersten Ebenen konvergieren klar, alle
    # Lloyd-Starts landen im selben Becken).
    if MC_TARGET_MCN > 0:
        cap = max(MC_CAP_MIN, int(MC_TARGET_MCN / max(n, 1)))
        if num_mc > cap:
            if verbose:
                print(f"    num_mc {num_mc} → {cap} (Batch-Deckel: "
                      f"mc·n bliebe sonst bei {num_mc * n:.2e})")
            num_mc = cap

    # Plateau-Phase: halbiert sich je Ebene wie die Steps, gedeckelt auf
    # HOLD_MAX_FRAC der tatsächlich gefahrenen Steps.
    hold_used = 0
    if hold_steps and hold_steps > 1:
        hold_used = int(hold_steps / (2 ** depth))
        hold_used = min(hold_used, int(HOLD_MAX_FRAC * step_annealing))
        hold_used = max(hold_used, 0)
        if hold_used > 1 and not use_fast and verbose:
            print("   Hinweis: Plateau (hold_steps) kennt nur der fast-Optimizer "
                  "— im Referenz-Pfad wird es ignoriert.")

    # 3a. Optionaler Lloyd-Vorschritt (iterations_lloyd > 0):
    #     k=2-Lloyd von zufälligem Start bis Konvergenz oder Limit.
    #     E_start (zufälliger Startvektor) und E_lloyd
    #     (reparierte Lloyd-Lösung) vermessen die Energielandschaft — beide
    #     wandern zusammen mit dem Lloyd-Assignment in den Cooling-Vektor.
    lloyd_info: dict | None = None
    if iterations_lloyd > 0:
        xg = normalize_coords(group)
        if not anneal:
            # Kein QUBO vorhanden → die Multistarts nach ihrem eigenen
            # 2-Means-Zielwert ranken (O(n·d) statt O(n²)).
            _energy = lambda v: lloyd_cost(xg, v)    # noqa: E731
        elif use_fast:
            _energy = lambda v: float(_eval_qubo(   # noqa: E731
                A_, b_, c_, np.asarray(v, dtype=float)))
        elif gram_threshold > 0.0:
            _energy = lambda v: pbf_energy_sparse(   # noqa: E731
                rows_, cols_, vals_, Q_row_, Q_col_, v)
        else:
            _energy = lambda v: pbf_energy_dense(Q, v)   # noqa: E731

        assign0 = np.random.randint(0, 2, size=n)
        E_start = _energy(assign0)

        if lloyd_mc > 1:
            # PCA-orientierte Multi-Starts: erst reine PC-Sign-Splits
            # (deterministisch, orthogonal), dann λ-gewichtete Zufalls-
            # mischungen der Top-PCs.  Lloyd verfeinert jeden Start;
            # Lösungen, die ins gleiche Becken kollabieren, fliegen raus
            # (ein Split und sein Komplement sind identisch).
            k_pc = max(1, min(4, int(lloyd_mc), n - 2))
            eigvals, scores = pca_scores(xg, k_pc)
            # Orthonormalbasis der Top-PCs (Scores sind bereits orthogonal)
            U_pc = scores / np.maximum(
                np.linalg.norm(scores, axis=0, keepdims=True), 1e-30)
            solutions: list[tuple[float, np.ndarray, int, bool]] = []
            seen: set[bytes] = set()
            for m in range(int(lloyd_mc)):
                if m < k_pc:
                    s = scores[:, m]            # reiner PC-Sign-Split
                else:
                    # ENTGEGEN den Hauptkomponenten streuen: Zufalls-
                    # richtung, aus der der Top-PC-Anteil herausprojiziert
                    # wird.  Entlang PC1 konvergiert Lloyd ohnehin immer ins
                    # selbe Becken — neue Becken liegen im Komplement.
                    # (gemessen: mehr unique Loesungen und bei Tiefe 2 ein
                    #  um 1.4 % besserer Split als Mischungen der Top-PCs)
                    v = np.random.standard_normal(n)
                    v -= U_pc @ (U_pc.T @ v)
                    s = v
                st = (s > 0).astype(int)
                if not 0 < st.sum() < n:          # degenerierte Richtung
                    st = assign0
                a, it_u, conv = lloyd_bisect(xg, iterations_lloyd, start=st)
                canon = a if a[0] == 0 else 1 - a
                key = canon.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                solutions.append((float(_energy(a)), a, int(it_u), bool(conv)))
            solutions.sort(key=lambda t: t[0])
            E_lloyd, assign, lloyd_iters, lloyd_conv = solutions[0]
            lloyd_starts = [sol[1] for sol in solutions]
        else:
            assign, lloyd_iters, lloyd_conv = lloyd_bisect(
                xg, iterations_lloyd, start=assign0)
            E_lloyd = float(_energy(assign))
            lloyd_starts = [assign]

        # KEINE Balance-Reparatur: die Lloyd-Lösung bleibt unverändert
        # (auch unbalanciert) — Hin- und Herschieben würde die Startwerte
        # zerstören. Rebalancing passiert nur am ENDE auf dem DA-Ergebnis.
        n_B = int(assign.sum())

        lloyd_info = {
            "iterations_lloyd": int(iterations_lloyd),
            "lloyd_iters": int(lloyd_iters),
            "lloyd_converged": bool(lloyd_conv),
            "lloyd_n_A": n - n_B,
            "lloyd_n_B": n_B,
            "lloyd_mc": int(lloyd_mc),
            "lloyd_unique": len(lloyd_starts),
            "E_start": float(E_start),
            "E_lloyd": float(E_lloyd),
        }
        if verbose:
            extra = (f", Starts {int(lloyd_mc)}→{len(lloyd_starts)} unique"
                     if lloyd_mc > 1 else "")
            print(f"    Lloyd: {lloyd_iters} it "
                  f"({'konvergiert' if lloyd_conv else 'Limit'}), "
                  f"Split {n - n_B}/{n_B}{extra}, "
                  f"E_start={E_start:.2f} → E_lloyd={E_lloyd:.2f}")
        if progress_cb:
            progress_cb("lloyd_done", {"depth": depth, "n": n, **lloyd_info})

    # 3a'. Ohne Annealing ist die Lloyd-Lösung bereits das Ergebnis —
    #      Cooling, Solve und die Auswahl unter den MC-Trials entfallen.
    if not anneal:
        assignment = assign.tolist()
        idx_A = [idx[i] for i in range(n) if assignment[i] == 0]
        idx_B = [idx[i] for i in range(n) if assignment[i] == 1]
        if stats is not None:
            stats.update({
                "n": n, "solver": "lloyd", "anneal": False,
                "steps_used": 0, "cooling": "none",
                "energy": lloyd_info["E_lloyd"], "valid_trials": 0,
                "num_mc": 0, "pbf_terms": 0,
                **lloyd_info,
            })
        return idx_A, idx_B

    # 3b. Cooling schedule
    #     Lloyd an  → ["da_gp", steps, delta_E_vec, initial_varAssignement,
    #                  E_lloyd, E_start]; Slot 2 (ΔE-Vektor) bleibt None —
    #                  den berechnet pbf_min_solver selbst im Solve.
    #     Lloyd aus → kalibriertes ["logarithmic", a, b];
    #                  a = cooling_a (fest) oder auto = d*total_n/(depth+1)
    a = cooling_a if cooling_a is not None else cooling_d * total_n / (depth + 1)
    if lloyd_info is not None:
        # cooling_schedule: "da_gp" (kalibriert, faellt UNTER die
        # Einfriergrenze) oder "da_gp_floor" (gleiche Kalibrierung, aber
        # die Kurve laeuft AUF die Einfriergrenze zu statt darunter)
        cooling_param = [cooling_schedule, int(step_annealing), None,
                         assign.tolist(),
                         lloyd_info["E_lloyd"], lloyd_info["E_start"],
                         float(p_hot), float(p_cold), float(gamma)]
    else:
        cooling_param = ["logarithmic", a, cooling_b]

    # 3c. Gram-Daten freigeben, BEVOR der lange DA-Solve läuft — ab hier
    #     wird nur noch das PBF-Dict bzw. (A, b, c) gebraucht.  Sonst hängen
    #     Q (dicht: n²·8 B, Root ~133 MB) bzw. die COO-Arrays den ganzen
    #     Solve über zusätzlich im Speicher.  (fast: schon in 1+2 passiert.)
    if not use_fast:
        if gram_threshold > 0.0:
            del rows_, cols_, vals_, Q_row_, Q_col_
        else:
            del Q
    del group
    if iterations_lloyd > 0:
        del xg, assign0, _energy

    # 4+5. Solve — gleiche Draw-Reihenfolge der Seeds in beiden Pfaden
    seed_rand = [random.uniform(0, 1000) for _ in range(num_mc)]

    if use_fast:
        Min_VarAss, Min, Trajectories, _infos = _qubo_min_solver(
            A_, b_, c_,
            steps=int(step_annealing),
            num_MC=num_mc,
            cooling_param=cooling_param,
            seed_rand=[int(s) for s in seed_rand],
            seed_gen_initial_varAssignment=int(seed_gen),
            visual_inst=visual_inst,
            offset_increase_rate=offset_increase,
            save_csv=False,
            save_addinfo=True,
            # Lloyd an → jede unique Lloyd-Lösung wird eine Startgruppe
            # mit num_mc Trials (lloyd_mc=1: genau eine, wie bisher);
            # Lloyd aus → geseedeter Zufallsstart für alle Trials
            initial_varAssignements_pre=(np.stack(lloyd_starts)
                                         if lloyd_info is not None else None),
            random_start=False,
            # jede Startgruppe an IHREM Startvektor kalibrieren (die beste
            # Lloyd-Loesung sitzt tiefer -> groessere dE-Skala -> ihr c ist
            # fuer die flacheren Startpunkte zu heiss)
            cooling_per_group=cooling_per_group,
            # Explorationsphase: Starttemperatur hold_used Steps halten
            hold_steps=hold_used,
            offset_k_escape=offset_k_escape,
        )
    else:
        pbf_var_dict = createPolyDict(pbf, n)

        Min_VarAss, Min, Trajectories, result_List = pbf_min_solver(
            pbf,
            pbf_var_dict,
            "digitalAnnealing",
            int(step_annealing),
            num_mc,
            cooling_param,
            seed_rand,
            seed_gen,
            visual_inst=visual_inst,
            offset_increase_rate=offset_increase,
            save_csv=False,
            save_addinfo=True,
            # Lloyd an → alle MC-Trials starten VON der Lloyd-Lösung
            # (nicht nur Kalibrierung); Lloyd aus → geseedeter Start wie bisher
            initial_varAssignement_pre=(assign.tolist()
                                        if lloyd_info is not None else []),
            random_start=False,
        )

    # 6. Best assignment → global index lists
    #    Achtung: DigitalAnnealing liefert für einen MC-Trial eine LEERE
    #    VarAssignment, wenn kein Step E <= E_init erreicht hat (das
    #    Min-Tracking im Solver beginnt erst NACH dem ersten Schritt).
    #    Solche Trials sind unbrauchbar → bestes Ergebnis unter den Trials
    #    MIT vollständigem Assignment wählen.
    valid = [i for i, va in enumerate(Min_VarAss) if len(va) == n]
    if not valid:
        if lloyd_info is not None:
            # Mit Lloyd-Warmstart ist "kein Trial besser als der Start" kein
            # Fehler, sondern heißt: die Lloyd-Lösung IST das beste bekannte
            # Ergebnis → direkt verwenden.
            if verbose:
                print("    ⚠ Kein MC-Trial besser als Lloyd-Start — "
                      "nutze Lloyd-Lösung direkt.")
            best = None
            best_energy = lloyd_info["E_lloyd"]
            assignment = assign.tolist()
        else:
            raise RuntimeError(
                f"DA hat in keinem der {num_mc} MC-Trials den Startzustand "
                f"verbessert (n={n}, steps={int(step_annealing)}) — kein "
                f"VarAssignment verfügbar. steps/num_mc erhöhen oder "
                f"Cooling (a/b/d) anpassen.")
    else:
        best = valid[int(np.argmin([Min[i] for i in valid]))]
        best_energy = float(Min[best])
        assignment = Min_VarAss[best]

    idx_A = [idx[i] for i in range(n) if assignment[i] == 0]
    idx_B = [idx[i] for i in range(n) if assignment[i] == 1]

    # KEIN Rebalancing mehr: das Verschieben war eine reine Größenkorrektur,
    # die den vom DA gefundenen Split gegen dessen eigene Zielfunktion
    # zurechtbog.  Die Neuzuordnung an den Split-Zentroiden bestimmt die
    # Gruppengröße ohnehin neu — gemessen war Rebalancing nie besser
    # (Self-Retrieval beam=2: 98.9 % mit, 99.4 % ohne) und der Baum bleibt
    # gesund (mittlere Tiefe 6.8 statt 6.9).

    if stats is not None:
        stats.update({
            "n": n,
            "solver": solver,
            "anneal": True,
            "steps_used": int(step_annealing),
            "cooling": (cooling_schedule if lloyd_info is not None
                        else "logarithmic"),
            "cooling_per_group": bool(cooling_per_group),
            "hold_steps": int(hold_used),
            "offset_k_escape": float(offset_k_escape),
            "cooling_a": float(a),
            "cooling_b": float(cooling_b),
            "energy": best_energy,
            "valid_trials": len(valid),
            "num_mc": num_mc,
            "pbf_terms": n_terms,
            "objective": objective,
            "gram_threshold": gram_threshold,
            "gram_zeroed_frac": round(zeroed_frac, 4),
        })
        if lloyd_info is not None:
            stats.update(lloyd_info)

    return idx_A, idx_B


# ═══════════════════════════════════════════════════════════════════════════════
#  Recursive driver → explicit binary tree
# ═══════════════════════════════════════════════════════════════════════════════

def partition_tree(
    coords,
    g: int,
    *,
    steps: int = 200,
    num_mc: int = 20,
    offset_increase: float | str = "auto_gp",   # Zahl oder "auto_gp" (dynamisch im Solver)
    cooling_a: float | None = None,
    cooling_b: float = 0.0,
    cooling_d: float = 2.0,
    iterations_lloyd: int = 10,
    lloyd_mc: int = 1,
    p_hot: float = 0.3,
    p_cold: float = 0.01,
    gamma: float = 0.05,
    cooling_schedule: str = "da_gp",
    cooling_per_group: bool = True,
    hold_steps: int = 0,
    offset_k_escape: float = 25.0,
    split_mode: str = "plane",     # "none" | "centroid" | "plane"
    spill: float = 0.1,
    eps: float = 0.0,
    seed_gen: float | None = None,
    verbose: bool = True,
    progress_cb=None,
    visualize_first: bool = False,
    gram_threshold: float = 0.0,
    solver: str = "fast",
    anneal: bool = True,
    objective: str = "gram",   # "gram" | "knn"
    anchors: int = 1,
) -> dict:
    """
    Recursively bisect *coords* until every leaf group has ≤ g points and
    return an explicit binary tree.

    solver — "fast" (annealing-qubo-optimizer, heuristischer batched
    numpy-DA) oder
    "reference" (annealing-cop-approximator, dict-basierter Originalpfad).

    anneal — False baut den Baum ohne Digital Annealing, allein aus den
    Lloyd-Bisektionen (siehe bisect_group).  Der Referenzpfad, um zu
    messen, was der DA gegenüber reinem k-Means beiträgt.

    eps — Unsicherheitsradius der Embeddings in Projektionseinheiten
    (siehe estimate_eps).  > 0 macht den Nachschritt fehlerbewusst: die
    Ebene sucht die Lücke mit den wenigsten unsicheren Punkten, und nur
    die Punkte der ε-Zone spillen (spill wird zur Obergrenze).

    iterations_lloyd — > 0: vor JEDER Bisection ein k=2-Lloyd-Vorschritt
    (zufälliger Start, bis Konvergenz oder Limit, Lösung bleibt unverändert);
    das Cooling wechselt auf ["da_gp", steps, None, lloyd_assign, E_lloyd,
    E_start].  0 = aus → kalibriertes logarithmic-Cooling wie bisher.

    lloyd_mc — > 1: PCA-orientierte Lloyd-Multi-Starts pro Bisection
    (PC-Sign-Splits + λ-gewichtete Mischungen, randomized SVD ~ms);
    solver="fast" fährt num_mc DA-Trials PRO uniquer Lloyd-Lösung
    (ein Batch), "reference" nur ab der besten.  Siehe bisect_group.

    gram_threshold — Sparsity-Schwelle für die Gram-Matrix jeder Bisection:
    |Q[i,j]| < threshold wird schon bei der blockweisen Berechnung
    verworfen — sparse Speicherung, nie eine dichte n×n-Matrix
    (0 = aus → kalibrierter dichter Originalpfad).

    cooling_a / cooling_b / cooling_d — logarithmic-Schedule
    ["logarithmic", a, b]; cooling_a=None nutzt den dynamischen Auto-Wert
    a = d·N/(depth+1) mit d = cooling_d (kalibrierter Default 2).

    visualize_first — True: die ERSTE Bisection jeder Tiefenebene wird mit
    visual_inst=True gelöst (DA-Trajektorien-Plot), alle weiteren derselben
    Ebene ohne — mehr wäre unübersichtlich.

    Every node carries the centroid ("mean") of its group, so a query can
    descend the tree by comparing an incoming embedding against the two
    child means at each level.

    progress_cb — optional callable(event: str, info: dict) for live
    observers (e.g. a GUI).  Events:
      "bisect_start" {depth, path, n}
      "pbf_built"    {depth, n, pbf_terms, gram_zeroed_frac}  — sobald das
                     PBF fertig ist (vor dem DA-Solve)
      "lloyd_done"   {depth, n, iterations_lloyd, lloyd_iters,
                      lloyd_converged, lloyd_n_A, lloyd_n_B, E_start, E_lloyd}
                     — nur bei iterations_lloyd > 0, vor dem DA-Solve
      "split"        {depth, path, idx_A, idx_B, seconds, + KPIs der
                      Bisection: n, steps_used, cooling_a/b, energy,
                      valid_trials, num_mc, pbf_terms,
                      gram_threshold, gram_zeroed_frac}
      "leaf"         {depth, path, indices}
    *path* is the binary tree path from the root ("" = root, "0" = child A,
    "01" = child A → child B, …).  The callback runs in the indexing thread.

    Node format
    -----------
    {
      "depth":    int,
      "n":        int,            # group size
      "mean":     [float, ...],   # centroid of the group (raw space)
      "indices":  [int, ...],     # global indices of the group's points
      "children": [nodeA, nodeB]  # or None for leaves
    }

    Returns
    -------
    root node dict
    """
    X = np.asarray(coords, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array (n points × d dims), got {X.shape}.")

    total_n = len(X)
    k_anchors = int(anchors)
    if seed_gen is None:
        seed_gen = random.uniform(1, 1000)

    seen_depths: set[int] = set()   # für visualize_first: erste Bisection je Ebene

    def _build(idx: list[int], depth: int, path: str = "",
               mean_override: np.ndarray | None = None) -> dict:
        indent = "  " * depth
        # mean_override: bei aktiver Neuzuordnung trägt der Knoten den
        # SPLIT-Zentroid seines Elternteils, nicht den Mittelwert seiner
        # (nachträglich veränderten) Punktmenge — nur so bleibt die
        # Descent-Konsistenz erhalten, die die Neuzuordnung herstellt.
        node = {
            "depth": depth,
            "n": len(idx),
            "mean": (X[idx].mean(axis=0) if mean_override is None
                     else mean_override).tolist(),
            "indices": list(idx),
            "children": None,
        }
        # Beam-Repräsentation: k Anker statt des einen Mittelwerts.  Ändert
        # NICHTS an der Trennung — nur daran, wie gut der Beam erkennt, dass
        # dieser Teilbaum zur Query passt.  k <= 1 lässt den Baum unverändert.
        if k_anchors > 1 and len(idx) > 1:
            node["anchors"] = node_anchors(
                X[idx], k_anchors, seed=(hash(path) & 0xFFFF)).tolist()

        if len(idx) <= g:
            if verbose:
                print(f"{indent}[leaf, depth={depth}]  {len(idx)} points ≤ g={g} → done")
            if progress_cb:
                progress_cb("leaf", {"depth": depth, "path": path,
                                     "indices": list(idx)})
            return node

        if verbose:
            a_show = (cooling_a if cooling_a is not None
                      else cooling_d * total_n / (depth + 1))
            print(f"{indent}[depth={depth}]  Bisecting {len(idx)} points  "
                  f"(cooling a={a_show:.1f}, b={cooling_b:g}) …")
        if progress_cb:
            progress_cb("bisect_start", {"depth": depth, "path": path,
                                         "n": len(idx)})
        t0 = time.time()

        first_of_level = depth not in seen_depths
        seen_depths.add(depth)

        bstats: dict = {}
        idx_A, idx_B = bisect_group(
            idx, X,
            total_n=total_n, depth=depth,
            steps=steps, num_mc=num_mc,
            offset_increase=offset_increase,
            cooling_a=cooling_a, cooling_b=cooling_b, cooling_d=cooling_d,
            iterations_lloyd=iterations_lloyd,
            lloyd_mc=lloyd_mc,
            p_hot=p_hot, p_cold=p_cold, gamma=gamma,
            cooling_schedule=cooling_schedule,
            cooling_per_group=cooling_per_group,
            hold_steps=hold_steps,
            offset_k_escape=offset_k_escape,
            seed_gen=seed_gen,
            verbose=verbose,
            visual_inst=visualize_first and first_of_level,
            gram_threshold=gram_threshold,
            solver=solver,
            anneal=anneal,
            objective=objective,
            stats=bstats,
            progress_cb=progress_cb,
        )

        if verbose:
            print(f"{indent}  → A: {len(idx_A)}, B: {len(idx_B)}"
                  f"  ({time.time() - t0:.2f}s)")

        # Degenerate split → treat as leaf
        if len(idx_A) == 0 or len(idx_B) == 0:
            if verbose:
                print(f"{indent}  ⚠ Degenerate split — treating as leaf")
            if progress_cb:
                progress_cb("leaf", {"depth": depth, "path": path,
                                     "indices": list(idx)})
            return node

        # Descent-konsistente Neuzuordnung an den festgehaltenen
        # Split-Zentroiden — DAS ist dann der Split für die nächste Ebene.
        # Nachschritt auf dem DA-Split — GENAU EINER, split_mode wählt:
        #   "none"     → der rohe DA-Split bleibt stehen
        #   "centroid" → Punkte dem näheren der beiden festgehaltenen
        #                Zentroide zuordnen (Hyperebene durch den Ursprung)
        #   "plane"    → dieselbe Richtung, aber als Max-Margin-Ebene in die
        #                größte Lücke geschoben, optional mit Spill-Zone
        # "plane" ersetzt "centroid" vollständig; beides zugleich gibt es
        # nicht, die Ebene IST die verschobene Zentroid-Trennung.
        mean_A = mean_B = None
        if split_mode in ("plane", "centroid"):
            # Max-Margin-Ebene: Richtung aus den Zentroiden, Lage in die
            # größte Lücke.  Der Knoten trägt danach (w, τ, σ); der Descent
            # entscheidet über q·ŵ − τ statt über Zentroid-Cosine.
            shift = split_mode == "plane"
            pl = plane_from_split(idx_A, idx_B, X, spill=spill, shift=shift,
                                  eps=eps)
            # Entartungs-Guard: mit Spill darf ein Kind NIE so groß werden
            # wie der Elternknoten, sonst terminiert die Rekursion nicht.
            if pl is not None and max(len(pl["idx_A"]),
                                      len(pl["idx_B"])) >= len(idx):
                pl2 = plane_from_split(idx_A, idx_B, X, spill=0.0,
                                       shift=shift, eps=eps)
                if verbose:
                    print(f"{indent}  ⚠ Spill hätte ein Kind so groß wie den "
                          f"Knoten gemacht — ohne Spill gesplittet")
                pl = pl2
            if pl is not None:
                idx_A, idx_B = pl["idx_A"], pl["idx_B"]
                node["w"] = pl["w"].tolist()
                node["tau"] = float(pl["tau"])
                node["sigma"] = float(pl["sigma"])
                bstats.update({"reassigned": pl["moved"],
                               "margin": pl["margin"],
                               "spilled": pl["spilled"],
                               "in_eps": pl["in_eps"]})
                if verbose:
                    eps_txt = (f", {pl['in_eps']} in ε-Zone" if eps > 0
                               else "")
                    print(f"{indent}  ⟂ {split_mode}: Marge {pl['margin']:.5f} "
                          f"(σ={pl['sigma']:.4f}), {pl['moved']} umgehängt, "
                          f"{pl['spilled']} gespillt{eps_txt} → A: {len(idx_A)}, "
                          f"B: {len(idx_B)}")

        if progress_cb:
            progress_cb("split", {"depth": depth, "path": path,
                                  "idx_A": list(idx_A), "idx_B": list(idx_B),
                                  "seconds": time.time() - t0, **bstats})

        node["children"] = [
            _build(idx_A, depth + 1, path + "0", mean_A),
            _build(idx_B, depth + 1, path + "1", mean_B),
        ]
        return node

    return _build(list(range(total_n)), 0)


def tree_leaves(node: dict) -> list[dict]:
    """Return all leaf nodes of a partition tree (left-to-right)."""
    if node["children"] is None:
        return [node]
    return tree_leaves(node["children"][0]) + tree_leaves(node["children"][1])


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience wrapper (flat interface matching original GraphPartitioning)
# ═══════════════════════════════════════════════════════════════════════════════

def graph_partitioning(
    coords,
    g: int,
    *,
    num_mc_gp: int = 20,
    steps: int = 200,
    seed_gen: float | None = None,
) -> list[list]:
    """
    Flat entry-point: partition until every group has ≤ g points and return
    the leaf groups as lists of points.
    """
    print(f"GraphPartitioning: {len(coords)} points → groups of ≤ {g}")
    root = partition_tree(
        coords, g,
        steps=steps, num_mc=num_mc_gp,
        seed_gen=seed_gen,
    )
    X = np.asarray(coords, dtype=float)
    groups = [X[leaf["indices"]].tolist() for leaf in tree_leaves(root)]
    print(f"\nResult: {len(groups)} groups")
    for i, g_i in enumerate(groups):
        print(f"  Group {i}: {len(g_i)} points")
    return groups


# ═══════════════════════════════════════════════════════════════════════════════
#  ANN map generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_ann_map(
    coords,
    g: int,
    *,
    steps: int = 200,
    num_mc: int = 20,
    offset_increase: float | str = "auto_gp",   # Zahl oder "auto_gp" (dynamisch im Solver)
    cooling_a: float | None = None,
    cooling_b: float = 0.0,
    cooling_d: float = 2.0,
    iterations_lloyd: int = 10,
    lloyd_mc: int = 1,
    p_hot: float = 0.3,
    p_cold: float = 0.01,
    gamma: float = 0.05,
    cooling_schedule: str = "da_gp",
    cooling_per_group: bool = True,
    hold_steps: int = 0,
    offset_k_escape: float = 25.0,
    split_mode: str = "plane",     # "none" | "centroid" | "plane"
    spill: float = 0.1,
    eps: float = 0.0,
    seed_gen: float | None = None,
    verbose: bool = True,
    save_path: str | None = None,
    compact: bool = False,
    gram_threshold: float = 0.0,
    solver: str = "fast",
    anneal: bool = True,
    objective: str = "gram",   # "gram" | "knn"
    anchors: int = 1,
) -> dict:
    """
    Run the recursive partitioning and return (+ optionally save) the ANN map.

    Output format
    -------------
    {
      "coordinates": { "0": [...], "1": [...], ... },
      "ANN_MAP": {                      # flat per-depth view (as before)
        "d_1": {
          "d": 1, "num_groups": 2,
          "Group_0": { "indices": [...], "mean": [...] },
          "Group_1": { ... }
        },
        ...
      },
      "TREE": { ... }                   # explicit binary tree with child
    }                                   # pointers — use this for querying

    Returns
    -------
    dict  (the ANN map, also saved to save_path if provided)
    """
    root = partition_tree(
        coords, g,
        steps=steps, num_mc=num_mc,
        offset_increase=offset_increase,
        cooling_a=cooling_a, cooling_b=cooling_b, cooling_d=cooling_d,
        iterations_lloyd=iterations_lloyd,
        lloyd_mc=lloyd_mc,
        p_hot=p_hot, p_cold=p_cold, gamma=gamma,
        cooling_schedule=cooling_schedule,
        cooling_per_group=cooling_per_group,
        hold_steps=hold_steps,
        offset_k_escape=offset_k_escape,
        split_mode=split_mode, spill=spill, eps=eps,
        seed_gen=seed_gen,
        verbose=verbose,
        gram_threshold=gram_threshold,
        solver=solver,
        anneal=anneal,
        objective=objective,
        anchors=anchors,
    )

    X = np.asarray(coords, dtype=float)

    ann_map: dict = {}
    ann_map["coordinates"] = {str(i): X[i].tolist() for i in range(len(X))}

    # Flat per-depth view: every bisection's two children, grouped by depth
    by_depth: dict[int, list[dict]] = {}

    def _collect(node: dict) -> None:
        if node["children"] is None:
            return
        for child in node["children"]:
            by_depth.setdefault(child["depth"], []).append(child)
        for child in node["children"]:
            _collect(child)

    _collect(root)

    ann_map["ANN_MAP"] = {}
    for d in sorted(by_depth):
        groups = by_depth[d]
        depth_entry: dict = {"d": d, "num_groups": len(groups)}
        for g_idx, grp in enumerate(groups):
            depth_entry[f"Group_{g_idx}"] = {
                "indices": grp["indices"],
                "mean":    [round(v, 8) for v in grp["mean"]],
            }
        ann_map["ANN_MAP"][f"d_{d}"] = depth_entry

    # Explicit tree (means rounded, same as flat view)
    def _export(node: dict) -> dict:
        out = {
            "depth": node["depth"],
            "n":     node["n"],
            "mean":  [round(v, 8) for v in node["mean"]],
        }
        if node["children"] is None:
            out["indices"] = node["indices"]
            out["children"] = None
        else:
            out["children"] = [_export(c) for c in node["children"]]
        return out

    ann_map["TREE"] = _export(root)

    if save_path is not None:
        import json
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(ann_map, fh, indent=None if compact else 2, ensure_ascii=False)
        size_kb = out.stat().st_size / 1024
        print(f"✓ ANN map saved → {out}  ({size_kb:.1f} KB)")

    return ann_map


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo (structure test with mock solver — no DA required)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _mock_bisect(idx, X, **kw):
        """Deterministic 50/50 split (no DA) — for structure testing only."""
        mid = len(idx) // 2
        return idx[:mid], idx[mid:]

    _real_bisect = bisect_group
    bisect_group = _mock_bisect  # noqa: F811

    rng = np.random.default_rng(0)
    points = rng.standard_normal((32, 4)).tolist()

    print("── Recursive partitioning demo (mock solver) ─────────────────")
    root = partition_tree(points, g=4, verbose=True)
    leaves = tree_leaves(root)
    sizes = [leaf["n"] for leaf in leaves]
    print(f"\nLeaf groups: {len(leaves)}  (expected ~8 for g=4, n=32)")
    print(f"Sizes: {sizes}  max={max(sizes)}  min={min(sizes)}")
    assert all(s <= 4 for s in sizes), "Some groups exceed g!"
    covered = sorted(i for leaf in leaves for i in leaf["indices"])
    assert covered == list(range(32)), "Index coverage broken!"
    print("✓ All groups ≤ g, all indices covered exactly once")

    bisect_group = _real_bisect  # restore
