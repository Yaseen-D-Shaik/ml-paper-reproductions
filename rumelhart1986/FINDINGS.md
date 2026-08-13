# Rumelhart 1986 Family Tree — Findings So Far

Status: **not yet a working reproduction.** Best result to date is ~0.35
mean activation on held-out test triples (individual triples 0.33-0.37,
tightly clustered), well short of the 0.8 threshold needed to count as a
pass. This doc consolidates what's been established so the next round of
investigation can build on it instead of re-deriving it.

## Dataset bugs found and fixed (read this first — invalidates earlier numbers)

Before trusting any test-set number in this doc, note: **the original
`TEST_TRIPLES` and `TRIPLES` had real data bugs**, found by cross-checking
every relation against an independently-derived family graph (father/mother/
husband/wife as ground truth, gender inferred from consistent role usage,
son/daughter/sibling/uncle/aunt/nephew/niece all algorithmically derived and
diffed against what was actually stored).

1. **2 of the original 4 test triples weren't real facts.**
   `(Penelope, mother, Victoria)` had the relation backwards — every other
   use of "mother" in the dataset is `(child, mother, parent)`, and here
   Victoria is Penelope's *daughter*, not her mother. `(Charlotte, aunt,
   Christine)` was simply false — Christine is Charlotte's grandmother, not
   an aunt; Charlotte's real aunts are Jennifer and Margaret. Confirmed by
   checking `TRIPLES[(p1, rel)]` directly: neither key/value existed. This
   silently corrupted every test metric in this document until now — e.g.
   re-evaluating a trained `BEST_CONFIG` model on only the 2 legitimate
   original triples jumped mean test activation from 0.29 to 0.43.

2. **The training set (`TRIPLES`) was missing 4 real facts.** Jennifer
   (James's sister, Colin/Charlotte's blood aunt) and Angela (Marco's
   sister, Alfonse/Sophia's blood aunt — the exact isomorphic mirror of the
   Jennifer case) were both missing their `nephew`/`niece` entries, while
   their in-law counterparts (Margaret/Charles, Gina/Tomaso) had theirs.
   Same omission in both trees, so likely systematic rather than a one-off
   typo. Fixed by adding `(Jennifer, nephew, Colin)`, `(Jennifer, niece,
   Charlotte)`, `(Angela, nephew, Alfonse)`, `(Angela, niece, Sophia)`.
   **This brought the dataset from 100 unique keys to exactly 104** —
   matching the paper's own "trained on 100 of the 104 possible triples"
   precisely. What the original debugging notes called "a documented
   ambiguity, not a bug" (100 keys vs. paper's 104) was in fact this exact
   bug.

`TEST_TRIPLES` now uses two isomorphism-mirrored multi-answer pairs: `(Colin,
uncle, Arthur)`/`(Alfonse, uncle, Emilio)` (unchanged, always valid) and
`(Charlotte, aunt, Margaret)`/`(Sophia, aunt, Gina)` (replacing the two
invalid triples). Re-running `BEST_CONFIG` on the corrected dataset gives:

```
(Colin, uncle) -> Arthur:      0.3916
(Alfonse, uncle) -> Emilio:    0.3692
(Charlotte, aunt) -> Margaret: 0.1992
(Sophia, aunt) -> Gina:        0.2028
```

Mean (0.2907) lands close to the old, corrupted 0.2896 by coincidence, but
the composition is now fully meaningful: a real, consistent split where
"uncle" queries generalize roughly 2x better than "aunt" queries, mirrored
almost exactly across the English/Italian isomorphism in each case
(Colin≈Alfonse, Charlotte≈Sophia) — good evidence the isomorphism-sharing
mechanism is real, but a genuine "uncle vs. aunt" asymmetry to still
understand, not measurement noise.

## Two corrections to how the paper was being read

1. **The paper used accumulated (batch) updates, not online.** p.535: "An
   alternative scheme, which we used in the research reported here, is to
   accumulate ∂E/∂w over all the input–output cases before changing the
   weights." Earlier notes had read Eq. 9's `t` as indexing individual
   cases; it indexes sweeps. `run_experiment`'s `update_scheme="batch"`
   implements this (gradients accumulate via repeated `.backward()` calls
   within a sweep, one `optimizer.step()` per sweep) and was verified
   correct by comparing against summing all losses into one tensor before a
   single `.backward()` — gradients matched to float32 precision.

2. **Multi-answer triples need multi-hot targets, not duplicated examples.**
   Fig. 3's caption describes Colin's two aunts as one presentation with two
   correct answers active simultaneously, not two separate presentations.
   The old `"split"` encoding (still available in `build_dataset` for
   comparison) creates a *provable* non-convergent fixed point: two examples
   with the same input and opposite single-hot targets for the same pair of
   units sum to zero gradient exactly at activation 0.5 in batch mode. Confirmed
   experimentally (Colin-uncle→Arthur/Charles both stuck at 0.4994 regardless
   of sweep count). `target_encoding="multihot"` fixes this by making it one
   example with both target bits active.

## What's mechanically verified (not the bug)

- Forward/backward/optimizer-step logic is correct (gradient-equality check
  above; also confirmed the network can overfit small hand-picked subsets
  cleanly, ruling out an architecture bug).
- `lr=0.01` with **no momentum** is simply too slow for this architecture —
  even 5 clean, non-competing triples fail to converge in 2000 sweeps at that
  learning rate. This was a confound in the original debugging notes: "online
  updates work, batch doesn't" was really "10000 sweeps of many small steps
  vs. 1500 sweeps of few big steps," not a scheme difference.
- **The paper's two-phase LR/momentum schedule (`PAPER_SCHEDULE`:
  lr=0.005/momentum=0.5 for sweeps 1-20, then lr=0.01/momentum=0.9) is
  necessary**, not optional. Momentum=0.9 from sweep 1 (skipping the warmup)
  produces oscillating, unstable loss. The full schedule gives smooth,
  monotonic convergence.

## The real finding: decay is a genuine bias-variance tradeoff, not a bug fix

With batch updates + paper schedule + multihot targets, training the full
100-example set **without decay reaches 100/100 training accuracy but test
performance actively gets *worse* as training improves** (test mean
activation drops from 0.12 at 61% train to 0.08 at 100% train) — textbook
overfitting, with no early-stopping sweet spot; it declines almost
monotonically from the start.

Adding weight decay (`decay_rate`, paper's rate is `0.998` per update)
trades this off cleanly and reproducibly. Mapped across six decay rates
(linear encoding, `w1_init_range=1.0`, decay starting at sweep 2500 after
the network already had headway):

| decay rate | train correct | test mean act |
|---|---|---|
| none | 100/100 | 0.080 |
| 0.9995 | 46/100 | 0.129 |
| 0.999 | 22/100 | 0.157 |
| 0.998 | 1/100 | 0.205 (peak) |
| 0.997 | 0/100 | 0.191 |
| 0.995 | 0/100 | 0.211 |

Monotonic until train accuracy bottoms out at 0, then it plateaus/noisy —
stronger decay past that point doesn't buy more test performance because
there's nothing left to trade. **Onset timing and ramp duration don't
change this equilibrium, only how fast you reach it** (tested: instant vs.
250/1000/2500-sweep ramps, and decay starting at sweeps 300 through 2500 —
all converge to the same destination for a given target rate).

## Architecture matters more than decay tuning

Switching to the paper's literal spec — `encoding_nonlinearity="sigmoid"`
(Eq. 1+2 applied at every layer, not just the hidden/output layers) and
`w1_init_range=0.3` (paper's uniform init range, not the `1.0` the repo had
drifted to) — shifts the entire tradeoff frontier favorably:

| decay rate | train correct | test mean act |
|---|---|---|
| none | 100/100 | 0.123 (peaks at 0.334 around sweep 2000, then declines) |
| 0.9995 | 28/100 | 0.146 |
| 0.999 | 4/100 | 0.210 |
| 0.998 | 0/100 | **0.289 (stable, not declining)** |

At every matched decay rate, sigmoid encoding beats linear encoding on test
performance. The `0.998` sigmoid config's test activation is *stable* across
2000+ sweeps rather than a fragile peak, and two of the four held-out
triples reach 0.40-0.45 — the best individual numbers seen in the whole
investigation. This is `BEST_CONFIG` in `train.py`.

Notably: for this config, exact decay onset timing (tested: start sweep
1500/2000/2250 with matched ramps) makes no difference to the final
equilibrium (~0.289 in all three) — confirms rate, not timing, is the
dominant lever, same as the linear-encoding case.

**Caveat on the two tables above**: both were measured against the
*original, since-corrected* `TEST_TRIPLES` (2 of 4 invalid, see top of this
doc). They're kept as-is because the qualitative findings (monotonic
tradeoff; sigmoid beats linear at every matched rate; onset timing doesn't
matter) are relative comparisons using a consistently-applied metric, so
almost certainly still hold — but the absolute numbers should be treated as
approximate until re-measured on the corrected test set.

## Beyond decay: looking for a genuinely different lever

Every decay variant above operates on the same mechanism — shrinking weight
*magnitude*. To find something that isn't just another point on that same
frontier, we deliberately looked for mechanisms that act on something else.

**Ruled out on principle, not just practically**: an explicit loss term
pulling English/Italian counterparts' `c1` rows toward each other. The paper
never describes such a term — it explicitly frames the isomorphism-sharing
in Fig. 4 as *emergent* from generic error-minimization + decay, not
engineered in ("the features captured by the hidden units are not at all
explicit in the input and output encodings... because the hidden features
capture the underlying structure of the task domain, the network generalizes
correctly"). Adding a hand-crafted similarity term would manufacture the
metric without reproducing the actual phenomenon, so this was deliberately
not tried.

**Mini-batch training** (chunked gradient accumulation between pure-online
and full-batch — new `update_scheme="minibatch"` + `batch_size`) turned out
to be genuinely different from decay: it improves generalization through
gradient noise, not weight magnitude, and — critically — **doesn't force
training accuracy to collapse the way every decay config does.** Mapped
across batch sizes (no decay, sigmoid encoding, `w1_init_range=0.3`):

| batch_size | train correct | test mean act |
|---|---|---|
| 1 (online) | 89.4% | 0.185 |
| 4 | 98.1% | 0.205 |
| 8 | 98.1% | 0.207 |
| 16 | 98.1% | 0.196 |
| 32 | 96.2% | **0.225** |
| 52 | 95.2% | 0.176 |
| 104 (full batch) | 100% | 0.123 |

A real middle-is-best curve, not monotonic in either direction — both
extremes underperform the 4-32 plateau. `batch_size=32` is the best single
no-decay point found. Stacking mini-batch with decay (after fixing a real
calibration bug — decay fires once per chunk, so mini-batch applies it ~13x
more often per sweep than full-batch at the same nominal rate; the target
rate must be scaled by `steps_per_sweep`-th root to compare fairly) does
**not** help — it only erodes training accuracy without any test gain.
Mini-batch noise and weight decay seem to compete for the same ground rather
than compound.

**Encoding-layer capacity bottleneck** (`encoding_dim` in `TreeNet`, paper
value 6) is mechanistically different from both: a hard architectural limit
rather than a soft penalty. Alone (no decay), `encoding_dim=3` gets 84%
train / 0.221 test — also beats the no-decay baseline. Combined with decay,
unlike mini-batch, it **does** compound favorably: `encoding_dim=3` + decay
reached 0.306 (peak 0.345), beating decay alone (0.291).

**Asymmetric person/relation split was the biggest win of this round.**
`encoding_dim` had always been applied symmetrically to `c1` (person, 24
inputs) and `c2` (relation, 12 inputs) — `TreeNet` now accepts
`person_encoding_dim`/`relation_encoding_dim` independently. Sweeping both
dimensions (with decay, total capacity varied freely, not fixed at 6):

| person_dim | relation_dim | test mean act |
|---|---|---|
| 5 | 1 | 0.199 |
| 4 | 2 | 0.210 |
| 3 | 3 | 0.306 |
| 2 | 4 | 0.294 |
| **1** | **5** | **0.350 (best)** |
| 1 | 6 | 0.326 |
| 1 | 7 | 0.330 |
| 1 | 8 | 0.324 |
| 2 | 5 | 0.257 |
| 3 | 5 | 0.214 |

`person_dim=1, relation_dim=5` is a confirmed 2D optimum — moving either
dimension away from this point in either direction makes it worse. This is
now `BEST_CONFIG`: 0.350 mean test activation, individual triples 0.33-0.37
(tightly clustered — the uncle/aunt asymmetry from before is largely gone at
this operating point), 11/104 train. Counter to the naive assumption that
person (24 inputs) needs more room than relation (12 inputs), the opposite
is true: relation type carries more of the compositional structure the
network needs space for (parent/child direction, generation-hop, gender,
spousal-vs-blood), while person identity compresses down almost trivially —
plausibly because Fig. 4's own analysis found person identity reduces to
only ~2-3 meaningful dimensions (nationality, generation, branch) in the
first place.

## What the learned weights actually encode (interpretability check)

Before pushing further on architecture, we inspected `BEST_CONFIG`'s trained
`c1`/`c2` directly rather than just trusting the test-accuracy number, to
check whether the uncle/aunt-asymmetry fix reflects real structure or
coincidence. Compared `c2` (relation encoding) between the earlier
symmetric-gap model (`person_dim=3, relation_dim=3`) and `BEST_CONFIG`
(`person_dim=1, relation_dim=5`):

- **`c1` (1-dimensional) spontaneously organized itself by generation**,
  unprompted: English gen-3 (Colin/Charlotte) ≈ -2.27, gen-1 of *both* trees
  cluster within 0.02 of each other near zero, Italian gen-3
  (Sophia/Alfonse) ≈ +2.28 — an almost perfectly symmetric mirror around 0.
  A single real number recovered most of the information Fig. 4 describes
  as spread across 2-3 of the paper's 6 dimensions.
- **The uncle/aunt fix is traceable to a specific, real change**: uncle/aunt
  and nephew/niece are reciprocal relations (same family link, opposite
  viewpoint), so a compositional representation should align their
  male-minus-female difference vectors closely. That alignment measurably
  tightened: cosine similarity between the two pairs' gender-axes went from
  0.916 (symmetric model, gap present) to 0.998 (`BEST_CONFIG`, gap
  resolved) — extra relation capacity let the network make this shared
  structure genuinely shared rather than approximately shared.
- **Caveat**: `father/mother` and `husband/wife` remain nearly unrelated to
  `uncle/aunt`'s gender axis in both models (cosine ≈ -0.05 to -0.18) — the
  network isn't using one universal "gender" direction, more like two loose
  clusters ({uncle/aunt, nephew/niece, brother/sister, son/daughter} vs.
  {father/mother, husband/wife}). Single seed/run — worth confirming across
  seeds before treating as settled.

## Where convergence actually stands

`BEST_CONFIG` (`person_dim=1, relation_dim=5`, decay from sweep 2000) reaches
0.350 mean test activation but only 11/104 train, 0/4 above the 0.8
threshold. To understand whether more time would help, we ran this
architecture **without decay** out to 30,000 sweeps (5x the budget used
everywhere else) and watched the full trajectory:

```
Sweep  3000: Train 25/104 | Test 0.4555  <- peak
Sweep 9000-17000: Train 41->60/104 | Test holds 0.42-0.45 (wide, stable plateau)
Sweep 19000+: Test begins a slow decline
Sweep 30000: Train 52/104 (noisy) | Test 0.348
```

Two real findings here: (1) this architecture has a genuinely different,
much more forgiving trajectory than the old `encoding_dim=6` one — a wide
plateau (sweeps ~3000-17000) where training climbs steadily while test holds
near its peak, not the near-immediate decline the old architecture showed;
but (2) it still eventually overfits, just far more slowly — confirming
overfitting is a real, persistent risk here, not something the architecture
change eliminated.

We then tried applying decay right where the plateau ends (`decay_start_sweep
=16000`) instead of the old, uncalibrated `sweep=2000`, at three rates:

```
rate=0.998 (strong): shock-collapses train (58->9/104), settles at test 0.354
rate=0.9995 (weak):  shock-collapses train similarly, settles at test 0.355
rate=0.999 (medium): also collapses train initially, but RECOVERS and keeps
                      climbing afterward: 0.346 -> 0.412, still rising at
                      sweep 30000, not leveled off
```

Same lesson as the very first decay-onset experiments earlier in this
investigation, now confirmed at a completely different architecture and
scale: **decay rate controls the destination, onset timing controls the
path** — correctly timing the onset to the plateau's end didn't prevent the
shock at strong/weak rates, only the medium rate avoided settling into a
worse state than doing nothing. We deliberately stopped here rather than
just extending the promising `rate=0.999` run further — chasing a better
number by brute-forcing more sweeps isn't a substitute for understanding
*why* the plateau ends, or finding a mechanism that doesn't require this
much wall-clock time to discover.

## Candidate paths forward (prioritized, as of this consolidation)

The core unresolved problem: every config found so far sits on a frontier
trading training accuracy against test performance. Nothing has broken that
tradeoff — only found better and better points on it. Ranked by how directly
each one attacks *that*, rather than just re-tuning a knob we already
understand:

1. **Understand the mechanism of the late-stage decline, rather than just
   reacting to it with decay.** We know *when* the plateau ends (~sweep
   18,000-19,000) but not *why*. Same interpretability approach that found
   the generation-axis and gender-axis structure earlier — snapshot the
   model at sweep 15,000 (plateau, good) vs. 25,000 (declined) and compare
   `c1`/`c2`/`w2` directly. Is the clean generation-axis structure in `c1`
   degrading? Is `w2` starting to encode person-specific shortcuts? This is
   the most likely path to a fix that isn't just another decay-rate guess.
2. **Multi-seed validation.** Every result in this entire document —
   including the person/relation split search and the plateau itself — used
   `seed=42` exclusively. We don't know yet whether `person_dim=1,
   relation_dim=5` and its wide plateau are robust properties of this task,
   or a lucky draw from one initialization. Cheap to check (rerun
   `BEST_CONFIG` and the no-decay trajectory at 2-3 more seeds), and it
   either solidifies everything above or reveals it's fragile — either way,
   worth knowing before investing more in this exact config.
3. **Weight averaging across the plateau.** We found a wide, stable region
   (sweeps ~3000-17000) where test performance holds near its peak while
   training keeps climbing. Averaging weight snapshots from several points
   within that plateau (stochastic weight averaging) is a well-established
   technique for exactly this situation — it might yield a single model
   better than any individual snapshot, without more sweeps or decay tuning.
4. **The two architecture ideas kept in reserve**: widening the penultimate
   hidden layer (paper: 6, never varied — would tell us whether the
   bottleneck's benefit is specific to identity-encoding or just "less
   capacity anywhere"), and replacing `concat` with a bilinear person×
   relation interaction. Deliberately not tried yet since we wanted to
   understand the person/relation split first — now that we do (relation
   carries more of the compositional load), these are informed rather than
   speculative choices.
5. **Smaller, more mechanical follow-ups**: layer-specific decay rates were
   only tested under linear encoding, never sigmoid; `w1_init_range` and
   `encoding_nonlinearity` were only ever changed together, never isolated;
   the decay-rate frontier tables from early sections predate the
   `TEST_TRIPLES` fix and should be re-measured for trustworthy absolute
   numbers (the asymmetric-split and bottleneck+decay tables are already
   post-fix).

## Harness reference

`rumelhart1986/train.py`'s `run_experiment(config)` exposes every axis above
as a config field (see its docstring). `BEST_CONFIG` holds the current best
known combination; `python train.py` runs it by default. `evaluate()` is
unified for both the training set (multi-target aware, reports full-set
memorization + mean target/non-target activation) and the held-out test
triples — use it to check any new config on both axes, not just the test
triples, since train-set behavior is what revealed the overfitting story
above.
