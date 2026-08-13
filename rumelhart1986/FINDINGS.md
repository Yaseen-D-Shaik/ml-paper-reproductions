# Rumelhart 1986 Family Tree — Findings So Far

Status: **not yet a working reproduction.** Best result to date is a stable
~0.29 mean activation on held-out test triples (individual triples up to
~0.45), well short of the 0.8 threshold needed to count as a pass. This doc
consolidates what's been established so the next round of investigation can
build on it instead of re-deriving it.

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

## Open questions for next time

- **Sigmoid encoding's own decay-rate frontier isn't fully mapped past
  0.998** — does going stronger (0.997, 0.995, as was tried for linear
  encoding) improve it further, or plateau the same way?
- **No config has recovered both high train accuracy and strong test
  performance simultaneously** — every result so far sits on a frontier,
  trading one for the other. Is there a fundamentally different lever
  (rather than a better point on this same frontier) that could break the
  tradeoff — e.g. a different loss formulation, a different masked-loss
  threshold, or restructuring how the isomorphic English/Italian trees share
  parameters?
- **Layer-specific decay rates (gentler on `c1`/`c2`, paper rate on
  `w1`/`w2`) were tested only under linear encoding and only made things
  moderately better (18/100 vs. 10/100 at matched sweeps), never re-tested
  under sigmoid encoding** — worth another pass given how much sigmoid
  encoding changed everything else.
- **`w1_init_range` and `encoding_nonlinearity` were only ever tested
  together** (paper values as a pair) — never isolated from each other. Not
  yet known which of the two is doing more of the work.
- Still using `n_sweeps=6000`, well beyond the paper's stated 1500 — worth
  understanding whether the paper's own 1500-sweep training was doing
  something qualitatively different, or whether it's simply a smaller
  number that happened to work for their exact setup.

## Harness reference

`rumelhart1986/train.py`'s `run_experiment(config)` exposes every axis above
as a config field (see its docstring). `BEST_CONFIG` holds the current best
known combination; `python train.py` runs it by default. `evaluate()` is
unified for both the training set (multi-target aware, reports full-set
memorization + mean target/non-target activation) and the held-out test
triples — use it to check any new config on both axes, not just the test
triples, since train-set behavior is what revealed the overfitting story
above.
