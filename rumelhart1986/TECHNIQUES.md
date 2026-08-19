# TECHNIQUES.md — Design decisions, techniques, and lessons

This document explains the *finalized* code in `model/network.py` and
`train.py`: what it does, why it's built this way, which techniques were
tried along the way (including the ones that failed), and what each is
worth remembering for future machine learning work. The full, chronological
experiment log — every number, every dead end, in the order it happened —
lives in `FINDINGS.md`. This document is the distilled, forward-looking
version of that log.

## 1. The task, in one paragraph

Two isomorphic family trees (English and Italian, 24 people, 12 relation
types: father, mother, husband, wife, son, daughter, uncle, aunt, brother,
sister, nephew, niece) give 104 `(person1, relation, person2)` facts. A
network sees `person1` and `relation` as one-hot inputs and must output
`person2` — not by looking anything up, but by inferring family structure
from raw, arbitrary symbols. Four facts (`TEST_TRIPLES`) are held out
entirely; a network that only memorizes the training set gets them wrong,
one that has actually learned the relational structure gets them right.
This is Hinton's original 1986 demonstration that backpropagation can
discover useful internal representations — generalization here is the
whole point, not an afterthought.

## 2. Code walkthrough

- **`data/family_tree.py`** — the raw facts (`TRIPLES`), and the two name
  lists (`PEOPLE`, `RELATIONSHIPS`) that define each one-hot encoding.
  Nothing here changed during optimization; it's ground truth.
- **`model/network.py`** — `TreeNet`, the architecture. A person and a
  relation are each compressed by their own small encoding layer, sigmoid
  concatenated into a central representation, passed through one hidden
  layer, and expanded back into a 24-unit output. Every dimension is a
  hardcoded constant at the top of the file, not a configurable parameter —
  this is the one architecture the whole investigation converged on, not a
  research harness for trying others.
- **`train.py`** — the training loop and evaluation. Also fully hardcoded
  (no config dict): builds the dataset, trains with SGD + Grokfast + delayed
  weight decay for 35,000 sweeps, then reports both the paper's literal
  0.8/0.2-threshold score and the argmax-in-valid-set score (see §4.13) for
  both held-out sets.
- **`FINDINGS.md`** — the full lab notebook: every experiment, in
  chronological order, including the ones this document doesn't mention
  because they didn't go anywhere useful.

## 3. Design decisions and why they hold

| Decision | Paper's spec | This project's final choice | Why |
|---|---|---|---|
| Person/relation encoding width | 6 / 6 (even) | 1 / 5 (asymmetric) | An even split leaves room to memorize per-person identity instead of compressing it; found empirically (§4.1) and never beaten by any wider or symmetric alternative. |
| Encoding-layer activation | unspecified/linear | sigmoid | Keeps the network a uniform sigmoid chain end to end — tried switching the *hidden* layer to tanh (a common fix other reproductions use) and it hurt this setup (§4.10). |
| Hidden layer | one, width 6 | one, width 6 | Widening it (up to 20) or adding a second layer both hurt test performance every time they were tried (§4.9, §4.11) despite raising training accuracy — capacity was never the bottleneck here. |
| Target encoding | — | multihot | Figure 3 describes Colin's two aunts as *one* presentation with two correct answers, not two separate presentations — multihot targets are what actually matches this. |
| Loss | — | masked squared error, 0.8/0.2 threshold | The paper's own stated rule: an output unit stops contributing error once it's confidently on the correct side. |
| Optimizer | implied SGD+momentum | SGD+momentum + Grokfast | Adam was explored extensively (it trains faster and reaches a comparable honest result) but every attempt to add weight decay on top of Adam either did nothing or destabilized training (§4.7) — the SGD+Grokfast+decay recipe is the one with a real, reproducible generalization story behind it (grokking, §4.1). |
| Held-out sets | 4 triplets | `TEST_TRIPLES` (multi-answer) + `VAL_TRIPLES` (single-answer), 4 each | Splitting them apart lets validation-based early stopping happen without ever looking at the actual test set (§4.6) — using the test set's own curve to decide when to stop is leakage. |

## 4. Techniques catalog

Each entry: what it is, why it was tried here, what actually happened, and
when it's worth reaching for in future work.

### 4.1 Weight decay as a *generalization* mechanism (grokking)

**What**: multiplying every weight by a factor slightly less than 1 after
each update (`p *= rate`), continuously shrinking weight magnitude.

**Why here**: small, discrete, compositionally-structured datasets trained
with weight decay are the textbook setting for **grokking** (Power et al.,
2022) — a network first memorizes (train accuracy saturates, test accuracy
stays low), then, often much later, undergoes a comparatively sudden
transition to real generalization as decay erodes a high-norm "memorizing"
circuit in favor of a lower-norm "generalizing" one that fits the same
training data.

**What we observed**: decay was the single most load-bearing lever in this
project — every serious result depends on it, and it never transferred
cleanly to Adam (§4.7). Onset *timing* mattered as much as the rate itself:
turning it on too early (before the network has anything worth decaying
toward) or too abruptly both hurt (§4.2).

**When to reach for it**: any small, discrete, structured task where a
network memorizes fast but generalizes slowly — weight decay isn't just
"a little L2 for safety," it can be the entire mechanism that makes
generalization happen at all, given enough steps.

### 4.2 Delayed, ramped decay onset

**What**: `decay_rate` only applies from `DECAY_START_SWEEP` (16000) onward,
linearly ramping from no decay to full strength over 1000 sweeps rather
than switching on abruptly.

**Why**: an abrupt strong decay onset shock-collapsed training in earlier
experiments (train accuracy cratering the sweep it engaged). Ramping avoids
the shock; delaying gives the network room to reach a reasonable trajectory
before decay starts pulling on it.

**When to reach for it**: any time you're adding a regularizer to a network
that's already mid-training (or already has a working optimization
trajectory) — introduce it gradually, not as a step function, and don't
assume "more regularization, sooner" is free.

### 4.3 Grokfast (gradient EMA amplification)

**What**: Lee et al. (2024). Before every optimizer step, amplify the
slow-varying component of each parameter's gradient: `mu <- alpha*mu +
(1-alpha)*g`, then `g_hat <- g + lambda*mu`. Reported to accelerate
grokking's memorize-to-generalize transition dramatically on algorithmic
tasks.

**What we observed**: real, substantial effect here too — the decay-driven
generalization transition happens with meaningfully fewer sweeps than
without it. One unexplained side effect: a long stall (test performance
pinned flat) for thousands of sweeps right before decay's transition
completes — real, reproducible, and never fully explained, but not harmful
enough to remove.

**When to reach for it**: paired with weight decay, on a task where you
suspect grokking but don't want to wait the (sometimes huge) number of
steps raw grokking can take.

### 4.4 Masked loss (paper's 0.8/0.2 rule)

**What**: an output unit's error is zeroed once it's already on the correct
side of the threshold (`>0.8` if it should be on, `<0.2` if it should be
off) — the network stops being penalized for units it's already gotten
right, rather than continuing to push them toward exactly 1.0 or 0.0.

**When to reach for it**: when you want training to spend its gradient
budget on units that are still wrong, not on squeezing marginal extra
confidence out of units that are already correct. A form of loss clipping /
early stopping applied per-output-unit rather than globally.

### 4.5 Multihot / multi-answer targets

**What**: when a query has more than one correct answer (Colin has two
aunts), the training target activates *both* simultaneously in one
presentation, instead of splitting it into two separate single-answer
presentations.

**Why**: this is literally how the paper's own Figure 3 illustrates the
task — treating it as two separate single-answer facts is a different,
easier problem than the one being reproduced.

### 4.6 Validation-based early stopping (disjoint from the test set)

**What**: `VAL_TRIPLES`, a second held-out set disjoint from `TEST_TRIPLES`,
used to pick when to stop training or which checkpoint to report — the
actual test set is only ever evaluated once, at whatever point the
validation set selected.

**Why**: picking a stopping point by watching the test set's *own*
performance curve and reporting whatever sweep looked best is leakage — it
biases the reported number upward and isn't an honest estimate of
generalization, even though it feels like "just early stopping."

**When to reach for it**: always, whenever you're tempted to pick a
stopping point, hyperparameter, or checkpoint by looking at test-set
performance directly. This project caught itself doing exactly this once
(documented in `FINDINGS.md`) before building this safeguard.

### 4.7 Optimizer swap: Adam/AdamW

**What**: tried in place of SGD+momentum, extensively, across learning
rates and both with and without decay.

**What we observed**: Adam reaches a comparable honest result in about half
the training steps SGD+Grokfast needs — genuinely useful for fast
iteration. But it **does not have built-in anti-overfitting behavior**:
without decay, Adam shows the exact same rise-then-decline overfitting
curve SGD does, just on a faster clock. And decay never transferred to it
cleanly — continuous decay from step 1 froze the network at its random-init
output distribution entirely (Adam's adaptive step size couldn't out-run a
decay pull with nothing yet learned to decay away from); delayed decay
either matched or underperformed plain early stopping.

**Lesson for future work**: don't assume a more sophisticated optimizer
subsumes regularization — Adam's adaptivity is an *optimization-efficiency*
mechanism (it reaches a solution faster), not a generalization mechanism.
If decay/regularization is doing real work in one optimizer, verify it
still does that job before assuming it transfers to another.

### 4.8 Multi-seed validation

**What**: rerunning a promising config at 3-4 different seeds before
trusting it.

**Why**: single-seed results in this project were repeatedly misleading —
a promising-looking seed-42 result (up to 0.563 mean test activation from
one fix) averaged as low as 0.415 — *below* the reference — once checked
across seeds. A separate config swung from 0.199 to 0.563 depending purely
on seed.

**When to reach for it**: before reporting or committing to *any* result
that came from a single training run, especially on a small dataset or
small network where variance is naturally higher. A great number from one
seed is a hypothesis, not a result.

### 4.9 Capacity-adding changes (wider hidden layer, bilinear interaction,
Xavier init, a second hidden layer, tanh)

**What**: five structurally different ways of giving the network more
expressive power or a different inductive bias — tried independently.

**What we observed**: **all five made test performance worse** while
raising training accuracy, every single time, on top of the same decay
recipe. This became one of the most consistent findings in the whole
project — nine total confirmations by the end (adding the depth experiment,
§4.11).

**Lesson for future work**: when you're stuck below a target and your
instinct is "give the model more room," check whether train accuracy is
already comfortably ahead of test accuracy first. If it is, more capacity
is very likely to widen that gap, not close it — the fix is somewhere else
(regularization, data, or how you're grading the result — see §4.13).

### 4.10 tanh hidden-layer activation

**What**: replacing sigmoid with tanh in the hidden layer specifically —
motivated by other reproductions' stated reasoning (`sigmoid'(0) = 0.25`,
so a multi-layer sigmoid chain compounds a severe vanishing-gradient
penalty; tanh's wider derivative range is supposed to help).

**What we observed**: hurt this setup, joining the capacity-adding list
above (it also adds effective expressive power). The vanishing-gradient
argument didn't play out negatively here in practice.

**Lesson**: a plausible, literature-backed mechanistic argument for a
change doesn't guarantee it helps on your specific setup — test it,
don't just adopt it on reasoning alone.

### 4.11 Depth (a second hidden layer)

**What**: inserted an optional second hidden layer between the existing one
and the output.

**What we observed**: hurt, and hurt *more* the wider the second layer got
— joins the capacity pattern above, but with an extra, sharper symptom:
validation and test performance came completely apart (validation reached
0.84-0.94 while test cratered to 0.12-0.26), something none of the other
capacity experiments showed as starkly. The network found a solution that
satisfied the validation-set pattern without transferring to the test-set
pattern, even though both are genuinely held-out and structurally similar.

**Lesson**: when validation and test performance diverge sharply despite
being constructed the same way, don't just distrust the model — check
whether your validation set is actually representative of what the test
set demands, especially after adding capacity.

### 4.12 Class-balanced loss / output-bias nudges (tried, failed)

**What**: two different fixes for a real, diagnosed problem — four specific
output units (Arthur, Margaret, Emilio, Gina) each lost exactly one
positive training example to the held-out split, leaving them
systematically less confident than every other output unit, across *every*
relation type that targets them, not just the held-out facts. Fix 1:
reweight each output unit's loss term by inverse training-frequency. Fix 2:
a gentler one-time bias nudge at initialization instead of a per-step loss
reweight.

**What we observed**: both failed multi-seed validation — high variance,
and in the loss-reweighting case, outright instability (units driven to
exactly 0.0000, weight norms roughly doubling, non-monotonic training
accuracy). The diagnosis was solid (confirmed by three independent,
convergent diagnostics); the fix wasn't.

**Lesson**: correctly diagnosing *why* something is failing does not
guarantee any particular fix will work — and a fix that introduces
*asymmetric* gradient scaling across parameters (favoring some over others)
is a real destabilization risk, especially stacked on top of another
technique (here, Grokfast) that amplifies whatever gradient it's handed.
Prefer fixes that don't change the relative gradient magnitude between
parameters if you can help it.

### 4.13 Stochastic Weight Averaging (tried, failed — but instructively)

**What**: average several weight snapshots taken during a late,
potentially-noisy training window into one final model, instead of trusting
a single endpoint.

**What we observed**: lost to the plain final checkpoint on every seed,
consistently (not noisily) — because the sampled window had continuous
weight decay still applying throughout, so the snapshots weren't
oscillating around a stable point (SWA's actual assumption), they were
sliding along a trend. Averaging an early, higher-norm snapshot with a
late, lower-norm one just landed somewhere worse than either endpoint.

**Lesson**: SWA (or any snapshot-averaging technique) needs a genuinely
*stable* region to average over — check whether whatever you're sampling
from is oscillating around a fixed point or drifting along a trend before
assuming averaging will help. A window with an active regularizer still
running is often the wrong one to pick.

### 4.14 Hard-label trick (tried, failed cleanly)

**What**: train against overshoot targets (1.1 / -0.1) instead of exactly
1.0/0.0, so the gradient doesn't shrink as output approaches the
unreachable sigmoid asymptote.

**What we observed**: a clean, stable, non-destabilizing negative result —
didn't help either the SGD+Grokfast or the Adam recipe. Unlike the
class-balance fixes, it changes every output unit's target uniformly, so it
never risked the asymmetric-gradient-scaling instability those did — it
just didn't move the actual bottleneck (which is which units get gradient
signal at all, not how strong the signal is once they have it).

**Lesson**: a technique can be perfectly safe and well-motivated and still
not be the right lever if it doesn't target the actual mechanism behind
your problem. Diagnose first (§4.12's interpretability work), then match
the fix to the diagnosis.

### 4.15 Argmax-in-valid-set evaluation (the decisive late finding)

**What**: instead of (or alongside) requiring an output unit to cross an
absolute activation threshold, check whether the model's single
highest-activation output is *any* of a query's true correct answers.

**Why this matters here**: this project's absolute-threshold grading (0/4)
made the network look like it hadn't learned the task. Inspecting the full
24-unit output vector for each held-out query showed the model's top-2
picks were, every time, the two *actual* correct answers — it wasn't
confused between right and wrong, it just favored whichever answer it saw
directly trained over the one it had to infer. Both external reproductions
of this paper that report beating this project's numbers grade with a
ranking-style criterion (or a much lower absolute threshold), not the
paper's literal 0.8/0.2 rule — so the "benchmark" everyone quotes and this
project's paper-literal number were never measuring the same thing.

**Lesson for future work, generalized beyond this project**: before
concluding a model has "failed," check what your grading criterion is
actually demanding, and whether the field's comparison numbers use the
same one. An absolute-confidence threshold and a relative-ranking criterion
can disagree completely on the same model — and the gap between them can
be the whole story. This is arguably the single most transferable lesson
of the entire project: **know your evaluation metric as precisely as you
know your model.**

### 4.16 Interpretability probing (causal interpolation, activation
distributions, per-target aggregation)

**What**: rather than trusting only summary accuracy numbers, this project
built small diagnostic tools: interpolating the network's person-encoding
between two people to see how sensitive an output is to identity;
histogramming every training-set output activation to check whether a
"stuck" number is a global ceiling or a localized problem; aggregating
activation by target-person across all relations to find which specific
output units were systematically weak.

**Why it mattered**: this sequence of diagnostics is what actually found
the starved-output-unit mechanism (§4.12) and, later, confirmed the
argmax finding wasn't an artifact of multi-answer ambiguity (by checking
`VAL_TRIPLES`'s genuinely single-answer facts specifically). Neither would
have been visible from accuracy numbers alone.

**Lesson**: when a model plateaus and you don't know why, look at what it's
actually outputting — the full output vector, not just the metric you're
optimizing — before reaching for another architecture or hyperparameter
change. The plateau often has a legible, specific cause.

## 5. Where this landed

Final honest numbers (see `FINDINGS.md` for the full derivation):

- **Paper-literal (0.8/0.2 absolute threshold)**: 0/4 on `TEST_TRIPLES`.
- **Argmax-in-valid-set** (matching how the field's actual benchmark
  numbers, and plausibly the original 1986 result itself, are graded):
  **4/4** on `TEST_TRIPLES`, beating the researched external benchmark
  (1.9/4 average, 3/4 best across two independent reproductions).
- **True single-answer argmax on `VAL_TRIPLES`** (no ambiguity possible):
  **4/4**, wide clean margins, confirming the network has learned real
  compositional relational structure rather than benefiting from
  multi-answer ambiguity.

The honest gap that remains: the network reliably *knows* the right answer
relative to wrong ones, but hasn't learned to be fully *confident* about
answers it only ever saw demonstrated for a person's sibling, not for that
specific person — the same starved-output-unit asymmetry diagnosed in
§4.12, for which no fix tried so far reliably closes the gap.

## 6. Takeaways for future ML work, distilled

1. **Define your evaluation criterion before judging a result.** The
   biggest single swing in this project's apparent standing — from "0/4,
   below the field" to "4/4, beating the field" — came entirely from
   clarifying what "correct" meant, not from any change to the model.
2. **Weight decay can be a generalization mechanism, not just a
   regularizer.** On small, structured, discrete tasks, look for grokking —
   memorization now, generalization much later, driven by decay.
3. **More capacity is not automatically the fix for underperformance.**
   Check the train/test gap first; widening a network that's already
   overfitting relative to its task usually makes things worse.
4. **Multi-seed validation is not optional for small-data results.** A
   single great seed is a hypothesis. This project was fooled by one at
   least twice before adopting this as standing practice.
5. **Diagnose before you fix.** The most valuable single technique in this
   project wasn't an optimizer or a regularizer — it was building small
   tools to look directly at what the model outputs, which is what found
   both the starved-output-unit mechanism and the evaluation-criterion
   mismatch.
6. **A theoretically well-motivated fix can still destabilize training** if
   it changes relative gradient magnitude between parameters — prefer
   uniform interventions (like the hard-label trick) over asymmetric ones
   (like per-unit loss reweighting) when both are on the table.
7. **A new optimizer is not a substitute for understanding what your old
   one needed.** Adam trained faster here, but silently dropped the
   regularization mechanism SGD+decay depended on — verify what
   transfers, don't assume it all does.
