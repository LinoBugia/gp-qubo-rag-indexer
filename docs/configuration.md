# Configuration

Every setting of the indexing pipeline in one table, and the measurement
behind each chosen default.

---

## Every parameter, in one table

The GUI shows these in the top bar (rows 1–3); the CLI names are
`rag_indexer.py`'s flags. Grouped by the pipeline step they control.

| Step | GUI | CLI | Default | What it does |
|---|---|---|---|---|
| 0·8·9 | `ε` | `--eps` | 0 (off) | The encoder's uncertainty radius, in the unit of the projection `t = p·ŵ` — not a distance in embedding space (see [planning for the encoder's own error](pipeline.md#planning-for-the-encoders-own-error)). `auto` measures it on the real encoder (256 chunks, random 60–80 % word window, re-embedded, q90 of the projected shift; needs the `ollama` provider). `cos:0.9` derives it from a known paraphrase cosine. A number sets it directly. Effect: the plane picks the gap with the **fewest** points inside `\|t−τ\| ≤ ε` instead of the widest one, and exactly those points spill. `0` reproduces the old behaviour bit for bit. |
| 2 | `thr` | `--gram-threshold` | 0 | Gram sparsity threshold: entries `\|Q[i,j]\| < thr` are dropped **during** the blockwise Gram computation — the dense n×n matrix never materialises, and dropped entries exist nowhere, neither as quadratic terms nor inside the row/column sums. 0 = off. |
| 2 | — | `--objective` | **auto: `knn` with the annealer, `gram` without** | **What the bisection minimises.** `gram` = the similarity of *all* pairs (2-means for balanced splits) — partition quality. `knn` = modularity of the symmetric 10-NN graph, i.e. the number of severed *true* neighbourhoods; natively quadratic (no degree-4 problem like exact 2-means) and sparse. Lloyd bisects in coordinate space and never sees a graph matrix, so `--no-anneal --objective knn` exits with an error instead of quietly ignoring the flag. Measured: [what the objective can and cannot explain](benchmarks.md#what-the-objective-can-and-cannot-explain). |
| 3 | `Solver` | `--solver` | `fast` | DA backend: `fast` = [annealing-qubo-optimizer](../../annealing-qubo-optimizer) (batched numpy, Gram→(A,b,c) directly, no PBF dict); `reference` = [annealing-cop-approximator](../../annealing-cop-approximator) (dict-based original path). Energy-identical to ≤1e-13, and ~58× faster end to end on the GP path (measured at n=300 with num_MC=8, whole solve including setup). The optimizer's own per-step numbers are x41 per trial at n=2000 and ~x76 with 8 trials batched — same picture, different slice. |
| 4 | `steps` | `--steps` | **auto = n/3** | Annealing steps at depth 0, where `n` is the **total chunk count**. Per level: `max(STEPS_FLOOR, steps / 2**depth)` — halving, floor `STEPS_FLOOR = 200`. Empty field = auto; the GUI pre-fills the computed value once it knows the chunk count. |
| 4 | `MC` | `--num-mc` | 20 | Monte-Carlo trials per start group. Groups below `MC_SMALL_N = 100` points get at least `MC_SMALL_TRIALS = 100` — the batched optimizer runs them in one pass. Capped from above by `MC_TARGET_MCN = 2e5`: past `mc·n ≈ 5e5` the working set leaves L2 and the step goes DRAM-bandwidth bound (~3× more expensive). The cap only lowers `num_mc`, never below `MC_CAP_MIN = 8`, and only on the top levels of large bases; below ~12 k chunks it never triggers. `MC_TARGET_MCN = 0` disables it. |
| 4 | `hold` | `--hold-steps` | **auto = n/10** | **Plateau**: hold the start temperature for that many steps before the curve falls. Added on top — the cooling curve keeps all `steps` support points and the same end temperature, the run just gets `hold` steps longer. Applies to depth 0 and halves per level like the steps, capped at `HOLD_MAX_FRAC = 0.5` of the steps used. `fast` optimizer only. |
| 5 | `lloyd` | `--iterations-lloyd` | 20 | Lloyd iterations of the warm start before the DA; stops early on convergence. 0 = no warm start → the schedule falls back to `["logarithmic", a, b]`. |
| 5 | `lloydMC` | `--lloyd-mc` | 1 | PCA-oriented Lloyd multi-starts per bisection: trials 1..4 are sign splits of the leading principal components, further ones scatter **against** them (random direction minus its top-PC projection). Every unique Lloyd solution becomes its own start group. Measured: `mc=4` captures the whole gain, `mc=16` was never better. |
| 6 | `Cooling` | `--cooling-schedule` | `da_gp` | `da_gp` = calibrated curve `c/ln(1+t^d)`, which ends **below** the freezing limit; `floor` = `da_gp_floor`, which approaches `T_freeze` instead and never undercuts it. |
| 6 | `T je Start` | `--cooling-per-group` / `--no-…` | on | Calibrate each Lloyd start group on **its own** start vector. Without it, the deepest solution's ΔE scale sets one shared `c` that is 1.6–2.6× too hot for the flatter start points and melts them. |
| 6 | `p_hot` | `--p-hot` | 0.4 | Target acceptance of a median uphill flip at `t=1` → `T_hot = q50/ln(n/p_hot)`. Larger = hotter start. |
| 6 | `p_cold` | `--p-cold` | 0.02 | Residual acceptance of a small (q25) flip at the end → `T_freeze = q25/ln(n/p_cold)`. Larger = warmer end. |
| 6 | `γ` | `--gamma` | 0.05 | Cap against melting the warm start: `T(1) ≤ γ · (E_random − E_lloyd)`. Affects **temperature only**, never the E_Offset. |
| 6 | — | `--cooling-a/-b/-d` | auto / 0 / 2 | Only used when `--iterations-lloyd 0` turns the warm start off; the schedule then falls back to `["logarithmic", a, b]` with `a = d · N / (depth+1)` when `a` is unset. With the warm start on (the default) these are ignored. |
| 7 | `DA` | `--anneal` / `--no-anneal` | on | Whether digital annealing runs at all. `--no-anneal` builds the tree from the Lloyd bisections alone and skips the Gram matrix, the QUBO and the cooling with it — O(n·d) instead of O(n²) per node (0.03 s vs 3.6 s on a 600-point group). Requires `--iterations-lloyd > 0`; both off is an error. The baseline the DA has to beat. |
| 7 | `offset` | `--offset-increase` | `auto_gp` | E_Offset increment per step in which **no** flip was accepted. A number = absolute; `auto_gp` = derived from the measured ΔE scale. |
| 7 | `kEsc` | `--offset-k-escape` | 5 | Scales the `auto_gp` offset: `rate = q50(ΔE⁺) / kEsc`. After `kEsc` rejected steps the offset has grown to a median uphill barrier. **Smaller = stronger offset = more aggressive escape.** Unrelated to `γ`. |
| 8 | `Split` | `--split-mode` | **`plane`** | The post-step on the DA split — **exactly one of three**, they replace each other. `none`: keep the raw DA split. `centroid`: assign every point to the nearer of the two frozen centroids, a hyperplane through the origin; this makes the level descent-consistent (self-retrieval 93.2 % → 98.9 %). `plane`: the same direction, shifted along it into the widest gap of the projections, so every point gets a margin to the boundary; the node then carries `(ŵ, τ, σ)`. `plane` is the shifted version of `centroid` — there is no combining the two. |
| 9 | `spill` | `--spill` | **0.1** | Overlap as a **fraction of points**: the `f·n` points closest to the boundary go into **both** children, so a query near the boundary reaches its chunk either way. Works with `centroid` **and** `plane` — it is a safety net on top of the boundary, not a way of choosing it, and needs no gap in the data, which is why it covers the top levels. With `--eps > 0` it becomes the **ceiling** instead of the target. Costs `(N/g)^log₂(1+f)` leaf entries — see [What spill costs](pipeline.md#what-spill-costs). `0` = off. |
| 10 | `anchors` | `--anchors` | 1 (off) | **How the beam judges a subtree.** With `1` a node is represented by a single mean, and the beam scores it `cos(q, mean)`. With `K > 1` the node carries K anchors (k-means centres of its own points) and the score becomes `max_j cos(q, aⱼ)` — an approximation of `max_p cos(q, p)` over the subtree, which is what the beam actually wants to know. Changes **nothing** about the partition: the split is still decided by the plane `q·ŵ ≷ τ` or the nearer centroid. Anchors are stored in `anchors.npy` (the node keeps only `[offset, count]`); as JSON text they would dwarf the tree. See [Why one mean is not enough](pipeline.md#why-one-mean-is-not-enough). |
| — | `g` | `-g` / `--group-size` | 20 | Maximum leaf size — the recursion stops when a group holds ≤ g points. |
| — | `seed` | `--seed` | random | Seed of the run. Note that tree builds are **not** reproducible today — parts of the recursion draw from ungeseeded global RNGs. |
| — | `Visualize first` | `--visualize-first` / `--no-…` | on | Solve the FIRST bisection of every tree level with `visual_inst=True`, so the trajectory plot opens once per level instead of once per bisection. |
| — | `KPIs` | `--stats` / `--no-stats` | on | Write `indexing_stats.json`: total/DA/overhead seconds, and per bisection group size, steps used, plateau, cooling parameters, margin, best DA energy, valid MC trials, reassigned and spilled points, term count and zeroed Gram fraction — plus per-depth aggregates. This file is the measurement record for calibration work. |

Exact semantics of the schedule slots, the `da_gp` calibration and the three
exploration knobs (`hold_steps`, `offset_k_escape`, `p_hot`/`γ`) are documented
in the optimizer's [graph-partitioning section](../../annealing-qubo-optimizer/README.md#graph-partitioning-specifics-gp) — see [the `da_gp` contract](../../annealing-qubo-optimizer/README.md#the-da_gp-contract-slots), [calibration](../../annealing-qubo-optimizer/README.md#calibration) and [the three exploration controls](../../annealing-qubo-optimizer/README.md#the-three-exploration-controls).

## Why these defaults

Four settings are *chosen*, not derived, so they are named here with the
measurement that chose them:

| Default | Why | Measured in |
|---|---|---|
| `--beam 8` (was 2) | The cheapest quality knob there is: costs nothing at build time, needs no rebuild, and beats every indexing option in the matrix. Beam 2 → 8 goes from 59.6 % to 86.7 % recall@5 on the default tree. | [The cheapest quality knob is the beam](benchmarks.md#the-cheapest-quality-knob-is-the-beam) |
| `--objective knn` with the annealer, `gram` without | Among trees that solve the Gram objective the energy no longer explains recall at all (+0.10), while the cut 10-NN pairs explain it at −0.91. The k-NN modularity optimises the quantity that matters. End to end it is worth about +0.3 points — inside the noise — so this default is a bet on the diagnosis, not on a measured win. | [What the objective can and cannot explain](benchmarks.md#what-the-objective-can-and-cannot-explain) |
| `--spill 0.1` (0.05 in the benchmarks) | Overlap buys recall with redundancy and, past a few percent, hides what optimiser and representation actually contribute. The CLI default stays at 0.1 for everyday use; the matrix was measured at 0.05 on purpose, so the other axes stay readable. | [What spill costs](pipeline.md#what-spill-costs) |
| `--anchors 1` | Anchors raise recall per candidate but cost K times the descent; widening the beam buys the same points cheaper. The option stays, the recommendation does not. | [What anchors actually buy](benchmarks.md#what-anchors-actually-buy) |

The two budget parameters scale with the knowledge base rather than being
fixed numbers: `steps = n/3` and `hold = n/10`, where **n is the total chunk
count** — not the size of the current group. Both are the value for level 0
and halve with depth from there, so the budget follows the group sizes down
the tree. The GUI pre-fills both fields as soon as it knows the chunk count;
overwrite them and your value is kept. The result is a scheme that adapts to
a knowledge base of any size without recalibration.

The rest follows from where the difficulty actually sits in the tree:

**The top levels are not the problem.** There the algorithm converges clearly
— measured, all Lloyd multi-starts at the root fall into the same basin, with
an energy spread of 0.0005 %. Spending trials there buys nothing.

**One Lloyd start is enough.** The first Lloyd start already converges to the
minimum in almost all cases, so a single trial — nudged a little by the
plateau — does the job. That is what `hold` is for: instead of many restarts,
one chain that gets a genuine exploration phase before the temperature starts
falling. This is where variance can still be won: **in the middle levels**,
which is exactly where the descent errors were measured (depth 2/4/5, ~5 %
inconsistent assignments each).

**`p_hot = 0.4`, `p_cold = 0.02` and `kEsc = 5`** are set so the chain keeps
room to move at those middle levels: a hotter start, an end that does not
freeze quite so hard, and an E_Offset five times stronger than the earlier
fixed value, so a stuck trial escapes after a handful of rejected steps
instead of dozens.

**`--lloyd 20`** gives Lloyd enough iterations to actually converge before the
DA takes over; it stops early when it does.

With these settings the tree is good and the scheme is dynamic — the ANN tree
approximates well for knowledge bases of any size.

Exact semantics of the schedule slots, the `da_gp` calibration and the three
exploration knobs (`hold_steps`, `offset_k_escape`, `p_hot`/`γ`) are documented
in the optimizer's [graph-partitioning section](../../annealing-qubo-optimizer/README.md#graph-partitioning-specifics-gp) — see [the `da_gp` contract](../../annealing-qubo-optimizer/README.md#the-da_gp-contract-slots), [calibration](../../annealing-qubo-optimizer/README.md#calibration) and [the three exploration controls](../../annealing-qubo-optimizer/README.md#the-three-exploration-controls).
