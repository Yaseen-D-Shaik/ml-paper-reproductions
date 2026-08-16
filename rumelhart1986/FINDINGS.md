# Rumelhart 1986 Family Tree — Findings So Far

Status: **not yet a working reproduction, but the first genuine improvement
on both axes at once, and now measured against a real external benchmark.**
Best result to date is ~0.48 mean activation on held-out test triples
(individual triples 0.43-0.52, tightly clustered) with 22/104 training
accuracy, using Grokfast-accelerated extended training (see "Grokking
hypothesis" below) — still short of any test triple crossing 0.8. Research
into other reproduction attempts of this same paper (see "External
validation" below) found none converge in the paper's stated 1500 sweeps
either, all needed substantial deviations, and the realistic bar to beat is
their **1.9-2/4 average / 3/4 best**, not an assumed 4/4 — this project's
own goal is to exceed that. This doc consolidates what's been established so
the next round of investigation can build on it instead of re-deriving it.

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

## Multi-seed validation (done before investing further)

Every result in this document up to this point used `seed=42` exclusively.
Reran `BEST_CONFIG` (`person_dim=1, relation_dim=5`) and the symmetric
`person_dim=3, relation_dim=3` baseline at 3 additional seeds:

| seed | (1,5) test mean act | (3,3) test mean act |
|---|---|---|
| 42 | 0.350 | 0.306 |
| 1 | 0.277 | 0.210 |
| 7 | 0.278 | 0.278 (tied) |
| 123 | 0.315 | 0.304 |

**The direction holds**: `(1,5)` never underperforms `(3,3)` across any
seed, so "relation needs more capacity than person" is a real, robust
pattern, not a seed-42 coincidence. **The magnitude doesn't**: the advantage
ranges from a clear ~14% relative improvement (seed 42) down to a dead tie
(seed 7). Absolute test performance also swings substantially by seed
regardless of architecture (0.21-0.35) — initialization has a large effect
on final outcome in general. Takeaway: treat `0.350` as an upper bound on
what this config typically achieves, not its expected performance — the
4-seed average (~0.30) is the more honest number to compare future configs
against.

## Mechanism of the late-stage decline (interpretability, no-decay run)

Snapshotted the `person_dim=1, relation_dim=5` no-decay model at sweeps
3000 (peak), 15000 (plateau), 19000/25000/30000 (declining) and compared
weights directly rather than guessing:

- **`b_w2` (output bias) grows continuously and never plateaus**: mean
  -1.59 -> -9.72, std 0.98 -> 8.75 across the run, even as `c1`/`w1` level
  off after their initial growth. A bias term doesn't depend on input at
  all, so growing it is the "cheap" way to fit training examples (memorize
  each output unit's marginal frequency) without any relational reasoning.
- **Sibling pairs that should encode almost identically drift apart in
  `c1`.** Colin/Charlotte (same generation, same branch, nearly identical
  facts except the relations that must distinguish them — `sister`/
  `brother`, `nephew`/`niece`-as-target) start at c1 gap 0.26 (sweep 3000)
  and reach gap 5.34 by sweep 30000; the isomorphic Alfonse/Sophia pair
  shows the same pattern (0.08 -> 4.70). Plausible mechanism: masking
  silences gradient from their many *shared* facts once satisfied, leaving
  the few facts that *require* differentiation to dominate the remaining
  active gradient later in training — eroding the shared representation
  that the test triples (themselves shared-type `uncle`/`aunt` queries)
  depend on for generalization. (Reasoned through but not directly
  verified against per-fact mask-crossing timestamps — see caveat below.)
- The uncle/aunt-vs-nephew/niece gender-axis alignment (same metric used to
  explain the earlier person/relation-split fix) **degrades in lockstep
  with test performance**: cosine similarity 0.863 (sweep 3000, peak) ->
  0.606 (sweep 25000, declined).

**Two follow-up experiments, both negative results — ruled out rather than
fixed:**

- **Disabling masking** (`use_masked_loss=False`) lets training accuracy
  climb higher/faster (86/104 vs. ~56-60/104) but generalization collapses
  far harder: test mean act 0.065 at 30000 sweeps, two of four triples
  driven to *exactly* 0.0000. Masking is confirmed necessary, not part of
  the problem — it was already doing real, load-bearing work.
- **Decay applied only to `b_w2`** (rate 0.999, everything else
  unconstrained) reaches an even higher peak than doing nothing (0.474 vs.
  0.456) but still declines afterward, settling at 0.259 by sweep 30000 —
  *worse* than the plain no-decay run at the same sweep count. Bias growth
  alone isn't sufficient to explain the decline; `c1`/`c2`/`w1`/`w2` growing
  unconstrained reproduces the same pattern on their own.

**Where this leaves the "understand before fixing" thread**: real
mechanistic understanding was gained (bias-driven memorization is real and
visible; masking is necessary; sibling-pair drift is a plausible but
unverified contributor), but neither follow-up experiment produced an
actual improvement. Diminishing returns on tuning the loss/decay mechanism
as a category — the two untried items below are different in kind, not
another variation on this theme.

## Candidate paths forward (prioritized, as of this consolidation)

Goal is now explicit (see "External validation" above): beat the 1.9-2/4
average / 3/4 best that other faithful reproductions of this paper achieve —
`BEST_CONFIG`'s best single triple (0.524) is already ~65% of the way to a
first pass. Settled/closed threads: multi-seed validation (pre-Grokfast),
late-stage-decline mechanism, the Grokfast pre-decay stall (real, but not
worth removing), `hidden_dim`/`tanh`, and Grokfast's own `alpha`/`lambda`
(all real ideas, none combine with or improve on the current recipe — five
independent directions now converge on `BEST_CONFIG` being a genuine local
optimum, see sections above). Remaining, ranked by expected leverage:

1. **[Current priority] Multi-seed validation of the *Grokfast* `BEST_CONFIG`
   specifically.** The earlier multi-seed check (0.28-0.35 range) was on the
   pre-Grokfast config; given how seed-sensitive every external reproduction
   also turned out to be (40-60% success rates) and how consistently narrow
   this local optimum has proven, it's not yet known whether 0.476 is
   typical or a favorable draw for this exact recipe — important to know
   before investing further in it specifically.
2. **The bilinear person×relation interaction** — the one remaining
   genuinely different *architectural* idea (replacing `concat` with a
   multiplicative interaction) rather than another nudge around the current
   optimum. Now informed by both the person/relation asymmetry finding and
   five negative nudge-results suggesting small parameter changes won't be
   enough.
3. **Try Xavier/Glorot init** instead of the paper's uniform `[-0.3, 0.3]` —
   cheap, single-parameter, and `cybertronai`'s reproduction used it as part
   of what got them to 1.9/4.
4. **Pei Guo's "hard label" trick** — targets of 1.1/-0.1 instead of 1.0/0.0
   (forces more separation before the mask threshold's 0.8/0.2 line is
   crossed) — reported as a real, if modest, improvement in that
   reproduction. Cheap to test, touches `compute_loss`/target construction
   only.
5. **Weight averaging (SWA)** across the post-decay region — the late-
   training oscillation seen throughout the Grokfast runs (train bouncing
   sweep to sweep) is exactly the kind of noisy-but-good region SWA is meant
   to smooth into a single better model. Apply once a strong trajectory is
   confirmed stable (after 1 above), not as a first move.
6. **Smaller, more mechanical follow-ups**: layer-specific decay rates were
   only tested under linear encoding, never sigmoid; the decay-rate frontier
   tables from early sections predate the `TEST_TRIPLES` fix and should be
   re-measured for trustworthy absolute numbers (the asymmetric-split,
   bottleneck+decay, and Grokfast tables are already post-fix).

## Hidden layer width matters too (partial architecture follow-up)

Before the deep-research pivot below, a quick sweep of `hidden_dim` (the
penultimate layer, paper: 6, newly made configurable) at `BEST_CONFIG`'s
(pre-Grokfast) decay recipe found a non-monotonic curve with a peak well
above the paper's spec:

| hidden_dim | train | test mean act |
|---|---|---|
| 4 | 3/104 | 0.237 |
| 6 (paper) | 11/104 | 0.350 |
| 8 | 7/104 | 0.325 |
| 12 | 8/104 | 0.371 |
| 16 | 5/104 | 0.373 |
| 20 | 8/104 | **0.384** |
| 24 | 10/104 | 0.370 |

This answers the question the hidden-layer idea was reserved to test: the
capacity bottleneck's benefit is specific to the *identity-encoding* layer
(`c1`/`c2`), not "less capacity anywhere" — squeezing the penultimate layer
(`hidden_dim=4`) hurts, while widening it beyond the paper's 6 helps further,
peaking around 20. Plausible mechanism: compressed identity encoding forces
more of the actual relational logic (which traversal, which gender filter)
into the combination step, which benefits from more room to do that work.
Later combined with the Grokfast recipe below — see "hidden_dim and tanh
don't combine with the Grokfast recipe" — the combination underperformed,
so this finding holds only in the pre-Grokfast context it was measured in.

## Grokking hypothesis: the deep-research pivot

After the architecture/interpretability threads hit diminishing returns
(masking and bias-only-decay follow-ups both came back negative — see
above), the user asked for a literature-grounded pass rather than continued
hand-tuning: given everything tried sits on one train/test tradeoff frontier,
what does the broader ML research on this *class* of problem say?

**Finding**: this project's setup — a small, discrete, compositionally
structured dataset (104 facts) trained with weight decay on a small network
— closely matches the conditions for **"grokking"** (Power et al. 2022,
"Grokking: Generalization Beyond Overfitting on Small Algorithmic
Datasets"): networks on small algorithmic/symbolic datasets often memorize
first (training accuracy saturates, test stays low), then — often 10-100x
more steps later — undergo a comparatively sudden transition to real
generalization. Weight decay (toward the origin, which is what this project
already implements) is reported as the single largest driver of this
transition. Later mechanistic work explains why: gradient descent forms a
high-norm "memorizing circuit" and a lower-norm "generalizing circuit";
decay continuously penalizes norm, and once the generalizing circuit is
capable of solving the task, decay preferentially erodes the memorizing one
— at which point train and test move together rather than trading off.
Critically, **steps-to-grok scale inversely with decay strength** (one study:
~13k steps at strong decay vs. ~98k at weak decay).

This re-explains three previously puzzling results without new experiments:
`rate=0.998` (strong) shock-collapsed and flatlined — plausibly too strong/
fast for a smooth transition; `rate=0.999` (medium) collapsed then
*recovered and kept climbing*, still rising at sweep 30,000 — the exact
signature of a transition in progress; `rate=0.9995` (weak) collapsed and
flatlined early — consistent with needing far more steps than given.

**Grokfast** (Lee et al. 2024) accelerates grokking up to ~44x on
algorithmic data via a per-parameter EMA gradient filter applied before the
optimizer step: `μ ← α·μ + (1-α)·g; ĝ ← g + λ·μ`. Implemented in
`run_experiment` as `grokfast_alpha`/`grokfast_lambda` (default `None`/off,
verified behavior-identical to before when disabled).

### Results: real improvement, with an unexplained wrinkle

Re-ran the sweep-16000 decay-onset experiment (`person_dim=1,
relation_dim=5`) with Grokfast (`alpha=0.98, lambda=2.0`) across three decay
rates, extended to 35,000 sweeps:

| decay rate | train | test mean act (final) | best single triple |
|---|---|---|---|
| 0.998 | 12/104 (11.5%) | 0.419 | 0.446 |
| 0.999 | 24/104 (23.1%) | 0.389 | **0.505** |
| **0.9995** | **23/104 (22.1%)** | **0.476** | **0.524** |

`rate=0.9995` — previously the *weakest* performer without Grokfast — is now
the best result of the entire investigation: 0.476 mean test activation
(individual triples 0.43-0.52, tightly clustered) at 22.1% train, beating
the old `BEST_CONFIG` (0.350 test, 10.6% train) **on both axes at once**.
This is the first config in the whole project to do that rather than sit on
the same tradeoff curve. Now `BEST_CONFIG`.

**The wrinkle**: Grokfast changes the *pre-decay* dynamics unexpectedly.
Without Grokfast, this architecture normally reaches ~0.42-0.45 test by
sweep 3000 and holds a wide plateau through sweep 17000 (see "Where
convergence actually stands" above). With Grokfast active from sweep 1, test
is instead pinned at ~0.195 (the mask-collapse floor) for the *entire*
16,000-sweep pre-decay window, identically across all three decay rates —
confirming it's a Grokfast-during-undecayed-training property, not a
decay-rate artifact.

**Isolating the stall (tested, resolved as a negative result)**: tried
`grokfast_start_sweep=16000` (new config field — Grokfast only engages once
its EMA buffer starts accumulating from that sweep, matching decay's own
onset) to see if skipping the stall reaches an equally good or better result
faster. Pre-16000, this run tracks the known no-decay plateau almost exactly
(confirming the stall really is specific to Grokfast-during-undecayed-
training). But the post-decay result was *worse*, not better: train ended
higher (31/104 vs. 23/104) but test dropped (0.371 vs. 0.476) and the four
triples split apart again (one hit 0.61, our best single number ever, while
the two aunt queries collapsed back to ~0.20-0.22) — undoing the tight
clustering the sweep-1-Grokfast version achieved. **Conclusion: the
"wasted-looking" stall isn't wasted** — Grokfast's gradient-EMA state
accumulating over that long undecayed window appears to matter for the
*quality* of the post-decay transition, not just its timing. Keep Grokfast
enabled from sweep 1 (current `BEST_CONFIG` default); don't adopt the
stall-skipping variant despite it looking more "efficient" on paper.

## External validation: how other reproductions of this exact paper fared

Before pushing further, researched other attempts to reproduce this same
1986 experiment, to get real calibration on what "success" should mean
rather than assuming a strict 4/4-at-0.8 bar. Three independent sources:

- **`cybertronai/hinton-problems`** (numpy, explicitly built for
  paper-comparison metrics): used lr=0.5, momentum=0.9, **no weight decay at
  all**, 10,000 epochs, Xavier init (not the paper's uniform[-0.3,0.3]), and
  **tanh instead of sigmoid** for hidden layers, with an explicit
  justification: *"sigmoid'(0) = 0.25, so a four-layer chain shrinks
  gradients by 0.25⁴ ≈ 0.004."* Best seed: 100% train, 75% test (3/4).
  Average across 10 seeds: **1.9/4 test correct**, and only 6/10 seeds reach
  100% training accuracy. Their own stated calibration: *"Hinton (1986)
  reported 2/4 on his hand-picked test set, so we consider 1.9/4 averaged
  over random hold-outs a faithful match of the paper's generalization
  regime."* (This 2/4 claim is from a third-party repo, not verified against
  the primary Nature letter text directly — flagged as credible but not
  fully confirmed.)
- **Pei Guo's PyTorch reproduction**: used Adam/AdamW, gradient clipping, LR
  warmup, deliberately deeper/wider layers than the paper's spec. Across 50
  seeds: average accuracy 0.725, **only 20/50 (40%) reached perfect 4/4**.
- **CompCogNeuro's textbook reimplementation** (Leabra, different learning
  rule entirely) states: *"Hinton's original model used a much smaller
  number of hidden units... over a very long training time, to force the
  model to develop more systematic representations"* — an independent
  confirmation of the exact mechanism found in this project (capacity
  bottleneck + extended training), from people reproducing this for
  teaching, not something invented by drifting off course.

**Takeaway**: no serious reproduction attempt found trains anywhere close to
1,500 sweeps or uses the paper's literal setup unmodified — all needed far
more epochs and/or activation-function changes and/or abandoned weight decay
entirely, and all show substantial seed-sensitivity (40-60% full-success
rates, not 100%). Recalibrated goal: **beat the 1.9-2/4 average / 3/4 best
external benchmark**, not an assumed-but-unverified 4/4.

## The paper's literal recipe doesn't converge at all (tested directly)

Tested the most paper-literal config possible on the corrected dataset:
symmetric `encoding_dim=6`, `hidden_dim=6`, sigmoid throughout, paper's decay
(`0.998`) starting at sweep 1 with no delay/ramp, 1500 sweeps — exactly as
the paper's text describes, no bottleneck, no Grokfast. Result: **0/104
train, mean target act (0.1959) statistically indistinguishable from mean
non-target act (0.1954)** — a completely undifferentiated network, weights
crushed to near-zero (`c1` norm 0.16, `w2` norm 3.22). Decay from sweep 1
overwhelms the still-weak early gradients before any structure can form —
consistent with earlier findings about needing a delayed decay onset. This
means **the speed gap to the paper's reported 1500 sweeps isn't explained by
"unnecessary complexity" on this project's part** — the literal recipe fails
completely, more totally than any of the elaborated configs. Combined with
the external-validation section above (nobody else converges in 1500 sweeps
either), the gap looks like it reflects something about the paper's actual
implementation (loss normalization, init details, or something else not
fully specified in a one-page Nature letter) that hasn't been identified,
not a self-inflicted detour.

## hidden_dim and tanh don't combine with the Grokfast recipe (tested, negative)

Two externally-motivated ideas — `hidden_dim=20` (previously found to beat
the paper's 6 in a pre-Grokfast context) and `tanh` for the w1/hidden layer
(the fix `cybertronai`'s reproduction used for vanishing gradients through
the sigmoid chain) — were stacked onto `BEST_CONFIG` and, separately,
recalibrated with stronger decay:

| variant | decay rate | train | test mean act |
|---|---|---|---|
| `BEST_CONFIG` (sigmoid, hidden_dim=6) | 0.9995 | 22/104 (21%) | **0.476** |
| + `hidden_dim=20` | 0.9995 | 29/104 (28%) | 0.406 |
| + `tanh` | 0.9995 | 43/104 (41%) | 0.337 |
| + `tanh` | 0.999 | 26/104 (25%) | 0.332 |
| + `tanh` | 0.998 (strongest) | 23/104 (22%) | 0.309 — **stronger decay made it worse** |
| + both (`hidden_dim=20` + `tanh`) | 0.9995 | 72/104 (69%) | 0.151 |
| + both | 0.999 | 45/104 (43%) | 0.177 |
| + both | 0.998 (strongest) | 38/104 (37%) | 0.212 |

Both ideas independently increase training capacity (more capacity, easier
gradient flow) but *decrease* test performance versus `BEST_CONFIG`, even
after direct recalibration across a 3x range of decay strength. For `tanh`
alone, stronger decay made results monotonically *worse* on both axes —
the opposite of what "just needs more decay pressure" would predict. For
`both` combined, stronger decay helped somewhat (0.151→0.212) but never
approached `BEST_CONFIG`. The mid-training logs for `both` show train and
test visibly oscillating against each other sweep-to-sweep rather than
transitioning smoothly, unlike `BEST_CONFIG`'s own trajectory.

**Working hypothesis, not confirmed**: the sigmoid chain's vanishing
gradient — which other reproductions treated purely as a problem — may be
acting as an *implicit* capacity constraint in this specific recipe,
analogous in spirit to the deliberate `person_dim=1` bottleneck that was the
actual breakthrough earlier in this investigation. Removing it via `tanh`
may remove something quietly load-bearing, not just a hindrance. Not
pursued further for now — `BEST_CONFIG` (sigmoid, `hidden_dim=6`) remains
the best-tested configuration.

## Grokfast's own hyperparameters are also a local optimum (tested, negative)

`BEST_CONFIG` has used `grokfast_alpha=0.98, grokfast_lambda=2.0` — one
untuned point from the paper's recommended ranges ([0.8, 0.99], [0.1, 5.0])
— throughout every result above. Swept both directions on each:

| variant | train | test mean act |
|---|---|---|
| baseline (α=0.98, λ=2.0) | 22/104 (21%) | **0.476** |
| α=0.90 | 26/104 (25%) | 0.358 |
| α=0.99 | 34/104 (33%) | 0.285 |
| λ=0.5 | 23/104 (22%) | 0.353 |
| λ=1.0 | 24/104 (23%) | 0.374 |
| λ=5.0 | 27/104 (26%) | 0.284 |

Every direction tested — both alpha values, all three lambda values — made
test performance worse while nudging training accuracy up, the same
shape as the `tanh`/`hidden_dim` results above. `(0.98, 2.0)` looks like a
genuine local optimum, not an unexamined default. **Combined with the
`hidden_dim`/`tanh` results, this is now a consistent pattern across five
independent directions (wider hidden layer, tanh, weaker/stronger Grokfast
alpha, weaker/stronger lambda): every adjacent move away from `BEST_CONFIG`
trades test performance for more training capacity.** Further gains look
less likely to come from parameter nudges around this recipe, and more
likely to need either a genuinely different architectural change (the
bilinear interaction, still untried) or confirmation that this optimum is
even stable across seeds (multi-seed validation of the Grokfast config,
also still pending — see "Candidate paths forward").

## Multi-seed validation of the Grokfast recipe (tested, reassuring)

Reran `BEST_CONFIG` at seeds 1, 7, 123 (matching the earlier pre-Grokfast
validation seeds for direct comparison):

| seed | train | test mean act |
|---|---|---|
| 42 (original) | 22/104 (21%) | 0.476 |
| 1 | 30/104 (29%) | 0.419 |
| 7 | 28/104 (27%) | 0.406 |
| 123 | 20/104 (19%) | 0.417 |

**Much tighter and higher than the pre-Grokfast spread.** The old
`BEST_CONFIG` ranged 0.277-0.350 across these same seeds (one seed showing
zero advantage over the plain symmetric baseline); the Grokfast version
ranges 0.406-0.476 — every seed here beats every seed the old config ever
achieved, average ~0.43 vs. the old ~0.30. The Grokfast recipe's improvement
is real and robust, not a `seed=42` fluke, though 42 remains the best draw.
Still 0/4 above 0.8 on every seed — the external benchmark (at least one
triple passing) hasn't been matched yet, but the trend and stability are
both genuinely positive.

## Bilinear person x relation interaction (tested, negative — sixth confirmation)

`TreeNet` now supports `use_bilinear=True`: replaces `concat(p_repr, r_repr)
@ w1` with a learned 3-tensor contraction `einsum('bi,ijk,bj->bk', p, B, r)`,
letting each relation-feature modulate each person-feature directly instead
of only combining additively. On top of `BEST_CONFIG`: **30/104 train (29%),
0.388 test mean act** — worse than concat's 0.476, same train-up/test-down
shape as every other variant tried. This is the **sixth independent
direction** (wider hidden layer, tanh, Grokfast alpha up/down, Grokfast
lambda variations, bilinear interaction) that trades test performance for
more expressive power/training capacity, with decay/Grokfast held at their
`BEST_CONFIG` calibration. Strong, comprehensive evidence that `BEST_CONFIG`
is a genuine local optimum specifically resistant to "give the network more
capacity" changes — further gains likely need a lever that doesn't add
capacity at all (see "Candidate paths forward").

(Implementation note: adding `use_bilinear` initially changed the RNG draw
order for `c1`/`c2`/`w1`'s `nn.init.uniform_` calls, silently breaking exact
reproducibility of every prior recorded result for the same seed — caught by
the project's standing discipline of re-verifying `BEST_CONFIG`'s exact
output after every harness change, and fixed by keeping `c1`/`c2`/`w2`
initialized before `w1`/`bilinear`, matching the original order.)

## Harness reference

`rumelhart1986/train.py`'s `run_experiment(config)` exposes every axis above
as a config field (see its docstring). `BEST_CONFIG` holds the current best
known combination; `python train.py` runs it by default. `evaluate()` is
unified for both the training set (multi-target aware, reports full-set
memorization + mean target/non-target activation) and the held-out test
triples — use it to check any new config on both axes, not just the test
triples, since train-set behavior is what revealed the overfitting story
above.
