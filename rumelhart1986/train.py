"""Training harness for Rumelhart et al. 1986 reproduction.

run_experiment(config) exposes each axis currently under investigation as a
config field, so different combinations (online vs. batch updates, weight
decay scope/rate, masked-loss threshold, LR/momentum schedule, target
encoding) can be run and compared on equal footing instead of requiring
hand-edits per experiment. main() runs BEST_CONFIG, the best-known
combination found so far — see FINDINGS.md for how it was derived and why
it's still a work in progress, not a finished reproduction.
"""

import copy
import random
import torch
import torch.optim as optim

from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import TreeNet, encode


# Two isomorphism-mirrored pairs: (English person, relation) and its Italian
# counterpart, each a multi-answer key so the network must predict one of two
# valid targets. Verified against TRIPLES directly (see FINDINGS.md) —
# earlier versions of this set included (Penelope, mother, Victoria) and
# (Charlotte, aunt, Christine), neither of which is a real fact: the former
# had the relation direction backwards (Victoria is Penelope's daughter, not
# her mother) and the latter names Charlotte's grandmother, not an aunt.
TEST_TRIPLES = [
    (PEOPLE.index("Colin"),     RELATIONSHIPS.index("uncle"), PEOPLE.index("Arthur")),
    (PEOPLE.index("Alfonse"),   RELATIONSHIPS.index("uncle"), PEOPLE.index("Emilio")),
    (PEOPLE.index("Charlotte"), RELATIONSHIPS.index("aunt"),  PEOPLE.index("Margaret")),
    (PEOPLE.index("Sophia"),    RELATIONSHIPS.index("aunt"),  PEOPLE.index("Gina")),
]

# Validation set: genuinely distinct facts from TEST_TRIPLES, also held out
# from training, used to pick an early-stopping point WITHOUT looking at the
# test set — using the test triples' own performance curve to decide when to
# stop is a form of leakage (the "peak" we find is biased toward whatever
# happens to look good on the exact 4 facts we report on). Two isomorphism-
# mirrored pairs, verified real against TRIPLES directly, same discipline as
# TEST_TRIPLES.
VAL_TRIPLES = [
    (PEOPLE.index("Arthur"),  RELATIONSHIPS.index("nephew"), PEOPLE.index("Colin")),
    (PEOPLE.index("Emilio"),  RELATIONSHIPS.index("nephew"), PEOPLE.index("Alfonse")),
    (PEOPLE.index("Charles"), RELATIONSHIPS.index("niece"),  PEOPLE.index("Charlotte")),
    (PEOPLE.index("Tomaso"),  RELATIONSHIPS.index("niece"),  PEOPLE.index("Sophia")),
]

# Paper's two-phase schedule (Fig. 4 caption): epsilon=0.005, alpha=0.5 for
# sweeps 1-20; epsilon=0.01, alpha=0.9 thereafter. Each tuple is
# (last_sweep_of_phase, lr, momentum).
PAPER_SCHEDULE = [(20, 0.005, 0.5), (float("inf"), 0.01, 0.9)]

# Paper's weight decay: "decrementing every weight by 0.2% after each weight
# change" -> multiply by (1 - 0.002) after each update.
PAPER_DECAY_RATE = 0.998

# Best-known config as of the Grokfast investigation (see FINDINGS.md).
# Reaches ~0.48 mean test activation (individual triples 0.43-0.52, tightly
# clustered) with 22/104 train — NOT a converged reproduction yet (0/4 test
# at the 0.8 pass threshold), but the first config to beat a previous best on
# BOTH training accuracy and test performance simultaneously rather than
# trading one for the other. Uses Grokfast (Lee et al. 2024) gradient-EMA
# amplification from sweep 1, with decay engaging at sweep 16000 once the
# no-decay plateau (found separately, see FINDINGS.md) has run its course.
# Note: Grokfast causes an unexplained ~16000-sweep stall before decay
# kicks in (test pinned at ~0.195) — the eventual result is good despite
# this, but the stall itself is not understood yet.
BEST_CONFIG = {
    "update_scheme": "batch",
    "lr_schedule": PAPER_SCHEDULE,
    "decay_rate": 0.9995,
    "decay_scope": "all",
    "decay_start_sweep": 16000,
    "decay_ramp_sweeps": 1000,
    "use_masked_loss": True,
    "target_encoding": "multihot",
    "encoding_nonlinearity": "sigmoid",
    "w1_init_range": 0.3,
    "person_encoding_dim": 1,
    "relation_encoding_dim": 5,
    "grokfast_alpha": 0.98,
    "grokfast_lambda": 2.0,
    "n_sweeps": 35000,
    "seed": 42,
}


def build_dataset(target_encoding="split", exclude_val=False):
    """
    exclude_val: if True, also hold VAL_TRIPLES's facts out of training (in
        addition to TEST_TRIPLES, always excluded). Default False so the
        training set composition — and every previously recorded result's
        exact reproducibility — is unaffected unless validation-based
        early-stopping (track_best_on_val) is actually being used.

    target_encoding:
      "split"    — each (person1, relationship, person2) answer becomes its
                   own training example with a single active target bit.
                   Held-out test facts are excluded per-fact. (108 examples
                   from 100 keys; matches previously committed behavior.)
      "multihot" — each (person1, relationship) key becomes one training
                   example with all correct, non-held-out person2's active
                   simultaneously — matching Fig. 3's description of Colin's
                   two aunts as one presentation with two correct answers.

    Returns (train_examples, TEST_TRIPLES, VAL_TRIPLES), where each train
    example is (p1_idx, rel_idx, target_idxs) and target_idxs is a tuple of
    one or more person2 indices. VAL_TRIPLES facts are held out from
    training the same way TEST_TRIPLES are, but kept separate from it — see
    VAL_TRIPLES's docstring for why (avoiding test-set leakage when picking
    an early-stopping point).
    """
    held_out = set(TEST_TRIPLES) | (set(VAL_TRIPLES) if exclude_val else set())
    train_examples = []

    for (p1_str, rel_str), p2_list in TRIPLES.items():
        p1_idx = PEOPLE.index(p1_str)
        rel_idx = RELATIONSHIPS.index(rel_str)
        p2_idxs = [PEOPLE.index(p2) for p2 in p2_list]

        if target_encoding == "split":
            for p2_idx in p2_idxs:
                if (p1_idx, rel_idx, p2_idx) in held_out:
                    continue
                train_examples.append((p1_idx, rel_idx, (p2_idx,)))
        elif target_encoding == "multihot":
            remaining = [p2 for p2 in p2_idxs if (p1_idx, rel_idx, p2) not in held_out]
            if remaining:
                train_examples.append((p1_idx, rel_idx, tuple(remaining)))
        else:
            raise ValueError(f"Unknown target_encoding: {target_encoding!r}")

    return train_examples, TEST_TRIPLES, VAL_TRIPLES


def evaluate(model, examples, label="", verbose=True):
    """
    Unified evaluator for both held-out test triples and training examples.

    examples: list of (p1_idx, rel_idx, target) where target is either a
      single person2 index (test triples: one correct answer) or a tuple of
      indices (train examples under multihot encoding: possibly several
      correct answers, e.g. Colin+uncle -> {Arthur, Charles}).

    An example counts as correct only if ALL of its target units are > 0.8.
    Also reports mean activation split between target and non-target output
    units across the whole example set — this is what reveals whether
    learning is broad-based (e.g. across all 100 training examples) rather
    than just what a single spot-check triple shows.
    """
    model.eval()
    results = []
    target_acts, nontarget_acts = [], []
    with torch.no_grad():
        for p1_idx, rel_idx, target in examples:
            target_idxs = (target,) if isinstance(target, int) else target
            target_set = set(target_idxs)

            p1t, rt = encode(p1_idx, rel_idx)
            output = model(p1t, rt)[0]

            example_acts = [output[idx].item() for idx in target_idxs]
            is_correct = all(a > 0.8 for a in example_acts)
            results.append((p1_idx, rel_idx, target_idxs, example_acts, is_correct))

            for idx in range(output.shape[0]):
                (target_acts if idx in target_set else nontarget_acts).append(output[idx].item())
    model.train()

    correct = sum(1 for *_, is_correct in results if is_correct)
    total = len(examples)
    mean_target = sum(target_acts) / len(target_acts)
    mean_nontarget = sum(nontarget_acts) / len(nontarget_acts)

    if verbose:
        title = f"{label} EVALUATION" if label else "EVALUATION"
        print("\n" + "=" * 55)
        print(f"      {title}")
        print("=" * 55)
        if total <= 10:
            for p1_idx, rel_idx, target_idxs, example_acts, is_correct in results:
                names = ", ".join(f"{PEOPLE[i]}={a:.4f}" for i, a in zip(target_idxs, example_acts))
                status = "PASSED" if is_correct else "FAILED"
                print(f"Query: ({PEOPLE[p1_idx]}, {RELATIONSHIPS[rel_idx]}) -> {names} [{status}]")
        prefix = f"{label} " if label else ""
        print(f"{prefix}Accuracy: {correct}/{total} ({correct/total*100:.1f}%) | "
              f"mean target act={mean_target:.4f} | mean non-target act={mean_nontarget:.4f}")
        print("=" * 55)

    return correct, total, mean_target, mean_nontarget


def compute_loss(output, target, use_masked_loss, target_weight=None, loss_target=None):
    """
    target: the "ground truth" 0/1 tensor — always used for the masked-loss
    should-be-on/should-be-off test below, regardless of loss_target.

    loss_target: optional tensor of the same shape as target, used in place
    of target for the actual squared-error computation (target itself still
    decides masking). Used for the hard-label trick (see hard_label config
    field) — training against e.g. 1.1/-0.1 instead of 1.0/0.0 keeps the
    gradient from shrinking as output approaches the unreachable 1.0 sigmoid
    asymptote, since the "should be on" units are pushed toward a target
    output can never fully reach. Applied uniformly to every output unit
    (unlike target_weight below), so it doesn't introduce the asymmetric
    per-unit gradient scaling that destabilized class-balanced loss.

    target_weight: optional (24,) tensor, one weight per output unit, applied
    only to "should be on" (target==1) error terms. Used for class-balanced
    loss (see class_balance_loss config field) — up-weights output units that
    appear less often as a correct target across the visible training set, to
    compensate for the reduced positive-gradient signal a person loses when
    one of their held-out facts removes a training example that would have
    targeted them. Computed purely from train_examples' own label frequency,
    never from TEST_TRIPLES/VAL_TRIPLES, so it doesn't leak which specific
    facts are held out.
    """
    error = output - (target if loss_target is None else loss_target)
    if use_masked_loss:
        # Paper: "error was considered to be zero if output units that
        # should be on had activities above 0.8 and output units that
        # should be off had activities below 0.2."
        mask = ~((target == 1.0) & (output > 0.8)) & \
               ~((target == 0.0) & (output < 0.2))
        error = error * mask.detach()
    if target_weight is not None:
        weight = torch.where(target == 1.0, target_weight, torch.ones_like(target_weight))
        error = error * weight
    return 0.5 * torch.sum(error ** 2)


def lr_momentum_for_sweep(sweep, lr_schedule):
    """
    lr_schedule: either a fixed (lr, momentum) tuple applied throughout, or
    a list of (last_sweep_of_phase, lr, momentum) phases (see PAPER_SCHEDULE).
    """
    if isinstance(lr_schedule, tuple):
        return lr_schedule
    for last_sweep, lr, momentum in lr_schedule:
        if sweep <= last_sweep:
            return lr, momentum
    last_phase = lr_schedule[-1]
    return last_phase[1], last_phase[2]


def run_experiment(config, verbose=True):
    """
    config fields (all optional; defaults reproduce the previously committed
    behavior — online updates, no decay, masked loss, fixed lr=0.01/momentum=0,
    split targets, linear encoding, w1 init range 1.0, 1500 sweeps):

      update_scheme:          "online" | "batch" | "minibatch"  (default "online")
      batch_size:              int — chunk size for "minibatch" scheme; one
                               accumulated optimizer.step() per chunk. Only
                               used when update_scheme="minibatch".
                                                                    (default None)
      decay_rate:              float, None, or a dict {"encoding": rate_or_None,
                               "output": rate_or_None} for layer-specific rates
                                                          (default None)
      decay_scope:             "all" | "output_only"      (only used when decay_rate
                               is a plain float; default "output_only")
      decay_start_sweep:       int — decay begins ramping in from this sweep
                               onward, for delayed-onset decay     (default 1)
      decay_ramp_sweeps:       int — number of sweeps over which each decay
                               rate linearly ramps from 1.0 (no decay) up to
                               its target rate, starting at decay_start_sweep.
                               0 means an instant switch to the target rate.
                                                                    (default 0)
      use_masked_loss:         bool                       (default True)
      lr_schedule:             (lr, momentum) or PAPER_SCHEDULE-style list
                                                            (default (0.01, 0.0))
      target_encoding:         "split" | "multihot"       (default "split")
      encoding_nonlinearity:   "linear" | "sigmoid"       (default "linear")
      w1_init_range:           float                      (default 1.0)
      encoding_dim:            int — width of C1/C2 (paper: 6); a capacity
                               bottleneck lever, independent of decay.
                                                                    (default 6)
      person_encoding_dim:     int or None — overrides encoding_dim for C1
                               only, to test asymmetric person/relation splits.
                                                                    (default None)
      relation_encoding_dim:   int or None — overrides encoding_dim for C2 only.
                                                                    (default None)
      hidden_dim:              int — width of the penultimate layer (paper: 6);
                               a second capacity lever, independent of encoding_dim.
                                                                    (default 6)
      hidden_dim2:             int or None — if set, inserts a second hidden
                               layer of this width between hidden_dim and the
                               output (depth instead of width — hidden_dim
                               alone only ever tested making one layer wider).
                                                                    (default None)
      hidden_nonlinearity:     "sigmoid" | "tanh" — activation for the w1
                               (central->penultimate) layer only; other
                               reproductions of this task found tanh here
                               necessary to avoid vanishing gradient through
                               a multi-layer sigmoid chain.  (default "sigmoid")
      use_bilinear:            bool — replace concat+w1 with a bilinear
                               person x relation interaction instead of an
                               additive combination.        (default False)
      init_scheme:             "uniform" (paper: fixed U(-0.3,0.3)) | "xavier"
                               (Glorot uniform, scaled per layer width).
                                                                    (default "uniform")
      grokfast_alpha:          float or None — EMA filter coefficient for
                               Grokfast gradient amplification (Lee et al.
                               2024), typical range [0.8, 0.99]. None disables
                               Grokfast entirely (default behavior unaffected).
                                                                    (default None)
      grokfast_lambda:         float — amplification strength for the EMA'd
                               gradient component, typical range [0.1, 5.0].
                               Only used when grokfast_alpha is not None.
                                                                    (default 2.0)
      grokfast_start_sweep:    int — Grokfast only engages from this sweep
                               onward (EMA buffer starts accumulating here).
                               Lets Grokfast be applied only during/after
                               decay rather than from sweep 1.
                                                                    (default 1)
      optimizer_type:          "sgd" (paper, default) | "adam" | "adamw" —
                               every external reproduction that beat a naive
                               attempt used Adam/AdamW, not SGD+momentum;
                               this is the one training-paradigm axis never
                               varied in this project. lr_schedule's momentum
                               value is ignored for adam/adamw (Adam has no
                               momentum parameter; use adam_beta1/beta2
                               instead). Custom weight decay (decay_rate)
                               still applies as its own post-step operation
                               regardless of optimizer_type — not routed
                               through AdamW's built-in decoupled decay, so
                               it stays comparable across optimizer choices.
      adam_beta1/adam_beta2/adam_eps: standard Adam hyperparameters, only
                               used when optimizer_type is "adam"/"adamw".
                                              (defaults 0.9, 0.999, 1e-8)
      track_best_on_val:       bool — if True, evaluate VAL_TRIPLES (not
                               TEST_TRIPLES) every log_every sweeps and keep
                               a snapshot of the model at whichever sweep had
                               the best validation mean activation. Choosing
                               an early-stopping point by watching the TEST
                               triples' own curve is leakage; this picks the
                               point using a disjoint set instead. Adds
                               `best_val_*` keys to the returned dict
                               (evaluated on the real test set only once,
                               at the chosen sweep) alongside the normal
                               final-sweep results.               (default False)
      n_sweeps:                int                        (default 1500)
      seed:                    int                        (default 42)
      log_every:               int                        (default 100)
      class_balance_loss:      bool — if True, weight each output unit's
                               "should be on" error term by (max_count /
                               count) where count is how often that unit is a
                               correct target across train_examples (computed
                               once, before training starts). Compensates for
                               output units that receive fewer positive
                               training examples — e.g. because one of their
                               facts fell in TEST_TRIPLES/VAL_TRIPLES — without
                               using any held-out information itself, only
                               the visible training set's own label frequency.
                                                                  (default False)
      class_balance_power:     float — exponent applied to the (max_count /
                               count) ratio; 1.0 is full inverse-frequency
                               weighting, <1.0 softens it.        (default 1.0)
      class_balance_bias_init: bool — if True, nudge each output unit's
                               initial bias (b_w2) by
                               class_balance_bias_scale * log(max_count / count),
                               a one-time initialization change rather than a
                               per-step loss reweight. Targets the same
                               starved-output-unit mechanism as
                               class_balance_loss without inflating gradient
                               magnitude during training (which is what
                               destabilized class_balance_loss under
                               SGD+Grokfast).                     (default False)
      class_balance_bias_scale: float — magnitude of the bias nudge above.
                                                                   (default 1.0)
      swa_start_sweep:         int or None — if set, start taking running-
                               average snapshots of the model's full
                               state_dict from this sweep onward (an
                               incremental mean, so memory cost is one extra
                               model regardless of how many snapshots are
                               taken). Meant for a stable-but-noisy late-
                               training region (e.g. post-decay-onset
                               oscillation) where no single checkpoint is
                               clearly best. Adds `swa_*` keys to the
                               returned dict, evaluated once on the real test
                               set.                               (default None)
      swa_every:               int — sample a snapshot every this many
                               sweeps once swa_start_sweep is reached.
                                                                   (default 500)
      hard_label:              bool — if True, train against
                               hard_label_pos/hard_label_neg instead of
                               1.0/0.0 (masking still uses 1.0/0.0 — only the
                               squared-error target changes). Keeps the
                               gradient from shrinking as output approaches
                               the unreachable 1.0 sigmoid asymptote, since
                               "should be on" units are pushed toward a
                               target output can never fully reach. Applied
                               uniformly to every output unit (Pei Guo's
                               reproduction).                     (default False)
      hard_label_pos/hard_label_neg: the overshoot targets used above.
                                                          (default 1.1, -0.1)

    Returns a dict with final_loss, test_correct, test_total, and model.
    """
    track_best_on_val = config.get("track_best_on_val", False)
    optimizer_type    = config.get("optimizer_type", "sgd")
    adam_beta1        = config.get("adam_beta1", 0.9)
    adam_beta2        = config.get("adam_beta2", 0.999)
    adam_eps          = config.get("adam_eps", 1e-8)
    update_scheme     = config.get("update_scheme", "online")
    batch_size        = config.get("batch_size", None)
    decay_rate        = config.get("decay_rate", None)
    decay_scope       = config.get("decay_scope", "output_only")
    decay_start_sweep = config.get("decay_start_sweep", 1)
    decay_ramp_sweeps = config.get("decay_ramp_sweeps", 0)
    use_masked_loss   = config.get("use_masked_loss", True)
    lr_schedule       = config.get("lr_schedule", (0.01, 0.0))
    target_encoding   = config.get("target_encoding", "split")
    encoding_nonlin   = config.get("encoding_nonlinearity", "linear")
    w1_init_range     = config.get("w1_init_range", 1.0)
    encoding_dim      = config.get("encoding_dim", 6)
    person_enc_dim    = config.get("person_encoding_dim", None)
    relation_enc_dim  = config.get("relation_encoding_dim", None)
    hidden_dim        = config.get("hidden_dim", 6)
    hidden_dim2       = config.get("hidden_dim2", None)
    hidden_nonlin     = config.get("hidden_nonlinearity", "sigmoid")
    use_bilinear      = config.get("use_bilinear", False)
    init_scheme       = config.get("init_scheme", "uniform")
    grokfast_alpha    = config.get("grokfast_alpha", None)
    grokfast_lambda   = config.get("grokfast_lambda", 2.0)
    grokfast_start    = config.get("grokfast_start_sweep", 1)
    n_sweeps          = config.get("n_sweeps", 1500)
    seed              = config.get("seed", 42)
    log_every         = config.get("log_every", 100)
    class_balance_loss  = config.get("class_balance_loss", False)
    class_balance_power = config.get("class_balance_power", 1.0)
    class_balance_bias_init  = config.get("class_balance_bias_init", False)
    class_balance_bias_scale = config.get("class_balance_bias_scale", 1.0)
    swa_start_sweep   = config.get("swa_start_sweep", None)
    swa_every         = config.get("swa_every", 500)
    hard_label        = config.get("hard_label", False)
    hard_label_pos    = config.get("hard_label_pos", 1.1)
    hard_label_neg    = config.get("hard_label_neg", -0.1)

    torch.manual_seed(seed)
    random.seed(seed)

    # Independent of track_best_on_val, so VAL_TRIPLES can be held out of
    # training without being used to select a checkpoint — needed when
    # VAL_TRIPLES itself is the thing being evaluated (using it for both
    # selection and evaluation would be a mild form of the same leakage
    # TEST_TRIPLES/VAL_TRIPLES were split apart to avoid in the first place).
    exclude_val_from_training = config.get("exclude_val_from_training", track_best_on_val)
    train_examples, test_triples, val_triples = build_dataset(target_encoding, exclude_val=exclude_val_from_training)

    counts = None
    if class_balance_loss or class_balance_bias_init:
        counts = torch.zeros(24)
        for _, _, target_idxs in train_examples:
            for idx in target_idxs:
                counts[idx] += 1
        counts = counts.clamp(min=1)

    target_weight = None
    if class_balance_loss:
        target_weight = (counts.max() / counts) ** class_balance_power
    if verbose:
        print(f"Train: {len(train_examples)} examples ({target_encoding}) | "
              f"Val: {len(val_triples)} triples | Test: {len(test_triples)} triples")
        print(f"\nStarting Training ({n_sweeps} Sweeps)...")

    model = TreeNet(encoding_nonlinearity=encoding_nonlin, w1_init_range=w1_init_range, encoding_dim=encoding_dim,
                     person_encoding_dim=person_enc_dim, relation_encoding_dim=relation_enc_dim,
                     hidden_dim=hidden_dim, hidden_nonlinearity=hidden_nonlin, use_bilinear=use_bilinear,
                     init_scheme=init_scheme, hidden_dim2=hidden_dim2)

    if class_balance_bias_init:
        # One-time output-bias nudge, not a per-step loss reweight: give
        # output units that see fewer positive training examples a head
        # start toward the 0.8 threshold, without inflating their gradient
        # magnitude during training (that's what destabilized
        # class_balance_loss under SGD+Grokfast). log() keeps the nudge
        # mild relative to the linear (max_count/count) ratio.
        with torch.no_grad():
            model.b_w2 += class_balance_bias_scale * torch.log(counts.max() / counts)

    # decay_groups: list of (params, rate) pairs. A dict decay_rate gives
    # c1/c2 and w1/w2 independent rates; a plain float applies one rate to
    # whichever group decay_scope selects.
    decay_groups = []
    if isinstance(decay_rate, dict):
        enc_rate = decay_rate.get("encoding")
        out_rate = decay_rate.get("output")
        if enc_rate is not None:
            decay_groups.append((model.encoding_params(), enc_rate))
        if out_rate is not None:
            decay_groups.append((model.output_params(), out_rate))
    elif decay_rate is not None:
        params = list(model.parameters() if decay_scope == "all" else model.output_params())
        decay_groups.append((params, decay_rate))

    def make_optimizer(lr, momentum):
        if optimizer_type == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_type == "adam":
            return optim.Adam(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps)
        elif optimizer_type == "adamw":
            return optim.AdamW(model.parameters(), lr=lr, betas=(adam_beta1, adam_beta2), eps=adam_eps,
                                weight_decay=0.0)  # custom decay_rate handles decay, not AdamW's own
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type!r}")

    current_lr, current_momentum = lr_momentum_for_sweep(1, lr_schedule)
    optimizer = make_optimizer(current_lr, current_momentum)

    # Grokfast (Lee et al. 2024): amplify the slow-varying (EMA) component of
    # each parameter's gradient before the optimizer step. mu <- alpha*mu +
    # (1-alpha)*g; g_hat <- g + lambda*mu. Accelerates the memorize-to-
    # generalize transition seen in grokking on small algorithmic datasets.
    grokfast_ema = {p: torch.zeros_like(p) for p in model.parameters()} if grokfast_alpha is not None else None

    def apply_grokfast(sweep):
        if grokfast_ema is None or sweep < grokfast_start:
            return
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                mu = grokfast_ema[p]
                mu.mul_(grokfast_alpha).add_(p.grad, alpha=1 - grokfast_alpha)
                p.grad.add_(mu, alpha=grokfast_lambda)

    spot_p1, spot_rel = encode(PEOPLE.index("Christopher"), RELATIONSHIPS.index("wife"))
    spot_target_idx = PEOPLE.index("Penelope")

    def ramped_rate(target_rate, sweep):
        if sweep < decay_start_sweep:
            return 1.0
        if decay_ramp_sweeps <= 0:
            return target_rate
        progress = min(1.0, (sweep - decay_start_sweep) / decay_ramp_sweeps)
        return 1.0 - progress * (1.0 - target_rate)

    def apply_decay(sweep):
        if sweep < decay_start_sweep:
            return
        with torch.no_grad():
            for params, target_rate in decay_groups:
                rate = ramped_rate(target_rate, sweep)
                for p in params:
                    p.mul_(rate)

    best_val_mean_act = -1.0
    best_val_sweep = None
    best_val_state_dict = None

    swa_avg_state = None
    swa_n_snapshots = 0

    final_loss = None
    for sweep in range(1, n_sweeps + 1):
        lr, momentum = lr_momentum_for_sweep(sweep, lr_schedule)
        if (lr, momentum) != (current_lr, current_momentum):
            optimizer = make_optimizer(lr, momentum)
            current_lr, current_momentum = lr, momentum

        random.shuffle(train_examples)
        total_loss = 0.0

        # Unified update scheme: chunk the sweep's examples into groups that
        # each get one accumulated optimizer.step(). "online" = chunks of 1,
        # "batch" = one chunk of everything, "minibatch" = chunks of
        # batch_size — all three are the same mechanism at different scales.
        if update_scheme == "online":
            chunk_size = 1
        elif update_scheme == "minibatch":
            chunk_size = batch_size
        elif update_scheme == "batch":
            chunk_size = len(train_examples)
        else:
            raise ValueError(f"Unknown update_scheme: {update_scheme!r}")

        for start in range(0, len(train_examples), chunk_size):
            chunk = train_examples[start:start + chunk_size]
            optimizer.zero_grad()

            for p1_idx, rel_idx, target_idxs in chunk:
                p1t, rt = encode(p1_idx, rel_idx)
                target = torch.zeros(1, 24)
                for idx in target_idxs:
                    target[0, idx] = 1.0

                loss_target = None
                if hard_label:
                    loss_target = torch.where(target == 1.0,
                                               torch.full_like(target, hard_label_pos),
                                               torch.full_like(target, hard_label_neg))

                output = model(p1t, rt)
                loss = compute_loss(output, target, use_masked_loss, target_weight, loss_target)
                loss.backward()
                total_loss += loss.item()

            apply_grokfast(sweep)
            optimizer.step()
            apply_decay(sweep)

        final_loss = total_loss

        if swa_start_sweep is not None and sweep >= swa_start_sweep and \
                (sweep - swa_start_sweep) % swa_every == 0:
            # Incremental running mean of the state dict, so we never hold
            # more than one averaged snapshot in memory regardless of how
            # many sweeps are sampled.
            snapshot = model.state_dict()
            swa_n_snapshots += 1
            if swa_avg_state is None:
                swa_avg_state = {k: v.clone() for k, v in snapshot.items()}
            else:
                for k, v in snapshot.items():
                    swa_avg_state[k] += (v - swa_avg_state[k]) / swa_n_snapshots

        if track_best_on_val and (sweep == 1 or sweep % log_every == 0):
            _, _, val_mean_target, _ = evaluate(model, val_triples, verbose=False)
            if val_mean_target > best_val_mean_act:
                best_val_mean_act = val_mean_target
                best_val_sweep = sweep
                best_val_state_dict = copy.deepcopy(model.state_dict())

        if verbose and (sweep == 1 or sweep % log_every == 0):
            model.eval()
            with torch.no_grad():
                spot = model(spot_p1, spot_rel)[0, spot_target_idx].item()
            model.train()
            train_correct, train_total, _, _ = evaluate(model, train_examples, verbose=False)
            test_correct, test_total, test_mean_target, _ = evaluate(model, test_triples, verbose=False)
            val_note = f" | Val mean act={best_val_mean_act:.4f}@{best_val_sweep}" if track_best_on_val else ""
            print(f"Sweep {sweep:4d}/{n_sweeps} | Loss: {total_loss:.4f} | "
                  f"Christopher->wife->Penelope: {spot:.4f} | Train: {train_correct}/{train_total} | "
                  f"Test: {test_correct}/{test_total} (mean act={test_mean_target:.4f}){val_note}")

    train_correct, train_total, train_mean_target, train_mean_nontarget = \
        evaluate(model, train_examples, label="TRAINING SET", verbose=verbose)
    test_correct, test_total, test_mean_target, test_mean_nontarget = \
        evaluate(model, test_triples, label="HELD-OUT TEST TRIPLES", verbose=verbose)

    central_param_name = "bilinear" if use_bilinear else "w1"
    weight_norms = {name: getattr(model, name).norm().item() for name in ["c1", "c2", central_param_name, "w2"]}
    if verbose:
        print(f"Final weight norms: {', '.join(f'{k}={v:.4f}' for k, v in weight_norms.items())}")

    result = {
        "final_loss": final_loss,
        "train_correct": train_correct,
        "train_total": train_total,
        "train_mean_target_act": train_mean_target,
        "train_mean_nontarget_act": train_mean_nontarget,
        "test_correct": test_correct,
        "test_total": test_total,
        "test_mean_target_act": test_mean_target,
        "test_mean_nontarget_act": test_mean_nontarget,
        "weight_norms": weight_norms,
        "model": model,
    }

    if track_best_on_val and best_val_state_dict is not None:
        # Evaluate the test set exactly once, at the sweep chosen purely by
        # watching VAL_TRIPLES — this is the honest, leakage-free number to
        # compare against the "final sweep" numbers above.
        best_model = TreeNet(encoding_nonlinearity=encoding_nonlin, w1_init_range=w1_init_range,
                              encoding_dim=encoding_dim, person_encoding_dim=person_enc_dim,
                              relation_encoding_dim=relation_enc_dim, hidden_dim=hidden_dim,
                              hidden_nonlinearity=hidden_nonlin, use_bilinear=use_bilinear,
                              init_scheme=init_scheme, hidden_dim2=hidden_dim2)
        best_model.load_state_dict(best_val_state_dict)
        bv_train_correct, bv_train_total, _, _ = evaluate(best_model, train_examples, verbose=False)
        bv_test_correct, bv_test_total, bv_test_mean_target, bv_test_mean_nontarget = \
            evaluate(best_model, test_triples, label="BEST-ON-VALIDATION TEST TRIPLES" if verbose else "",
                      verbose=verbose)
        if verbose:
            print(f"(best-on-validation checkpoint was sweep {best_val_sweep}, "
                  f"val mean act={best_val_mean_act:.4f})")
        result.update({
            "best_val_sweep": best_val_sweep,
            "best_val_mean_act": best_val_mean_act,
            "best_val_train_correct": bv_train_correct,
            "best_val_train_total": bv_train_total,
            "best_val_test_correct": bv_test_correct,
            "best_val_test_total": bv_test_total,
            "best_val_test_mean_target_act": bv_test_mean_target,
            "best_val_test_mean_nontarget_act": bv_test_mean_nontarget,
            "best_val_model": best_model,
        })

    if swa_avg_state is not None:
        swa_model = TreeNet(encoding_nonlinearity=encoding_nonlin, w1_init_range=w1_init_range,
                             encoding_dim=encoding_dim, person_encoding_dim=person_enc_dim,
                             relation_encoding_dim=relation_enc_dim, hidden_dim=hidden_dim,
                             hidden_nonlinearity=hidden_nonlin, use_bilinear=use_bilinear,
                             init_scheme=init_scheme, hidden_dim2=hidden_dim2)
        swa_model.load_state_dict(swa_avg_state)
        swa_train_correct, swa_train_total, _, _ = evaluate(swa_model, train_examples, verbose=False)
        swa_test_correct, swa_test_total, swa_test_mean_target, swa_test_mean_nontarget = \
            evaluate(swa_model, test_triples, label="SWA-AVERAGED TEST TRIPLES" if verbose else "",
                      verbose=verbose)
        if verbose:
            print(f"(SWA averaged {swa_n_snapshots} snapshots from sweep {swa_start_sweep} "
                  f"every {swa_every} sweeps)")
        result.update({
            "swa_n_snapshots": swa_n_snapshots,
            "swa_train_correct": swa_train_correct,
            "swa_train_total": swa_train_total,
            "swa_test_correct": swa_test_correct,
            "swa_test_total": swa_test_total,
            "swa_test_mean_target_act": swa_test_mean_target,
            "swa_test_mean_nontarget_act": swa_test_mean_nontarget,
            "swa_model": swa_model,
        })

    return result


def main():
    run_experiment(BEST_CONFIG)


if __name__ == "__main__":
    main()
