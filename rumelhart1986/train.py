"""Training harness for Rumelhart et al. 1986 reproduction.

run_experiment(config) exposes each axis currently under investigation as a
config field, so different combinations (online vs. batch updates, weight
decay scope/rate, masked-loss threshold, LR/momentum schedule, target
encoding) can be run and compared on equal footing instead of requiring
hand-edits per experiment. main() runs BEST_CONFIG, the best-known
combination found so far — see FINDINGS.md for how it was derived and why
it's still a work in progress, not a finished reproduction.
"""

import random
import torch
import torch.optim as optim

from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import TreeNet, encode


TEST_TRIPLES = [
    (PEOPLE.index("Colin"),     RELATIONSHIPS.index("uncle"),  PEOPLE.index("Arthur")),
    (PEOPLE.index("Alfonse"),   RELATIONSHIPS.index("uncle"),  PEOPLE.index("Emilio")),
    (PEOPLE.index("Penelope"),  RELATIONSHIPS.index("mother"), PEOPLE.index("Victoria")),
    (PEOPLE.index("Charlotte"), RELATIONSHIPS.index("aunt"),   PEOPLE.index("Christine")),
]

# Paper's two-phase schedule (Fig. 4 caption): epsilon=0.005, alpha=0.5 for
# sweeps 1-20; epsilon=0.01, alpha=0.9 thereafter. Each tuple is
# (last_sweep_of_phase, lr, momentum).
PAPER_SCHEDULE = [(20, 0.005, 0.5), (float("inf"), 0.01, 0.9)]

# Paper's weight decay: "decrementing every weight by 0.2% after each weight
# change" -> multiply by (1 - 0.002) after each update.
PAPER_DECAY_RATE = 0.998

# Best-known config as of the decay/architecture investigation (see
# rumelhart1986/FINDINGS.md). Reaches a stable ~0.29 mean test activation
# (individual triples up to ~0.45) but 0/100 train and 0/4 test at the 0.8
# pass threshold — NOT a converged reproduction yet. Kept here as the
# starting point for the next round of investigation, not as a final answer.
BEST_CONFIG = {
    "update_scheme": "batch",
    "lr_schedule": PAPER_SCHEDULE,
    "decay_rate": PAPER_DECAY_RATE,
    "decay_scope": "all",
    "decay_start_sweep": 2000,
    "decay_ramp_sweeps": 250,
    "use_masked_loss": True,
    "target_encoding": "multihot",
    "encoding_nonlinearity": "sigmoid",
    "w1_init_range": 0.3,
    "n_sweeps": 6000,
    "seed": 42,
}


def build_dataset(target_encoding="split"):
    """
    target_encoding:
      "split"    — each (person1, relationship, person2) answer becomes its
                   own training example with a single active target bit.
                   Held-out test facts are excluded per-fact. (108 examples
                   from 100 keys; matches previously committed behavior.)
      "multihot" — each (person1, relationship) key becomes one training
                   example with all correct, non-held-out person2's active
                   simultaneously — matching Fig. 3's description of Colin's
                   two aunts as one presentation with two correct answers.

    Returns (train_examples, TEST_TRIPLES), where each train example is
    (p1_idx, rel_idx, target_idxs) and target_idxs is a tuple of one or
    more person2 indices.
    """
    test_set = set(TEST_TRIPLES)
    train_examples = []

    for (p1_str, rel_str), p2_list in TRIPLES.items():
        p1_idx = PEOPLE.index(p1_str)
        rel_idx = RELATIONSHIPS.index(rel_str)
        p2_idxs = [PEOPLE.index(p2) for p2 in p2_list]

        if target_encoding == "split":
            for p2_idx in p2_idxs:
                if (p1_idx, rel_idx, p2_idx) in test_set:
                    continue
                train_examples.append((p1_idx, rel_idx, (p2_idx,)))
        elif target_encoding == "multihot":
            remaining = [p2 for p2 in p2_idxs if (p1_idx, rel_idx, p2) not in test_set]
            if remaining:
                train_examples.append((p1_idx, rel_idx, tuple(remaining)))
        else:
            raise ValueError(f"Unknown target_encoding: {target_encoding!r}")

    return train_examples, TEST_TRIPLES


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


def compute_loss(output, target, use_masked_loss):
    error = output - target
    if use_masked_loss:
        # Paper: "error was considered to be zero if output units that
        # should be on had activities above 0.8 and output units that
        # should be off had activities below 0.2."
        mask = ~((target == 1.0) & (output > 0.8)) & \
               ~((target == 0.0) & (output < 0.2))
        error = error * mask.detach()
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

      update_scheme:          "online" | "batch"        (default "online")
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
      n_sweeps:                int                        (default 1500)
      seed:                    int                        (default 42)
      log_every:               int                        (default 100)

    Returns a dict with final_loss, test_correct, test_total, and model.
    """
    update_scheme     = config.get("update_scheme", "online")
    decay_rate        = config.get("decay_rate", None)
    decay_scope       = config.get("decay_scope", "output_only")
    decay_start_sweep = config.get("decay_start_sweep", 1)
    decay_ramp_sweeps = config.get("decay_ramp_sweeps", 0)
    use_masked_loss   = config.get("use_masked_loss", True)
    lr_schedule       = config.get("lr_schedule", (0.01, 0.0))
    target_encoding   = config.get("target_encoding", "split")
    encoding_nonlin   = config.get("encoding_nonlinearity", "linear")
    w1_init_range     = config.get("w1_init_range", 1.0)
    n_sweeps          = config.get("n_sweeps", 1500)
    seed              = config.get("seed", 42)
    log_every         = config.get("log_every", 100)

    torch.manual_seed(seed)
    random.seed(seed)

    train_examples, test_triples = build_dataset(target_encoding)
    if verbose:
        print(f"Train: {len(train_examples)} examples ({target_encoding}) | Test: {len(test_triples)} triples")
        print(f"\nStarting Training ({n_sweeps} Sweeps)...")

    model = TreeNet(encoding_nonlinearity=encoding_nonlin, w1_init_range=w1_init_range)

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

    current_lr, current_momentum = lr_momentum_for_sweep(1, lr_schedule)
    optimizer = optim.SGD(model.parameters(), lr=current_lr, momentum=current_momentum)

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

    final_loss = None
    for sweep in range(1, n_sweeps + 1):
        lr, momentum = lr_momentum_for_sweep(sweep, lr_schedule)
        if (lr, momentum) != (current_lr, current_momentum):
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            current_lr, current_momentum = lr, momentum

        random.shuffle(train_examples)
        total_loss = 0.0

        if update_scheme == "batch":
            optimizer.zero_grad()

        for p1_idx, rel_idx, target_idxs in train_examples:
            if update_scheme == "online":
                optimizer.zero_grad()

            p1t, rt = encode(p1_idx, rel_idx)
            target = torch.zeros(1, 24)
            for idx in target_idxs:
                target[0, idx] = 1.0

            output = model(p1t, rt)
            loss = compute_loss(output, target, use_masked_loss)
            loss.backward()

            if update_scheme == "online":
                optimizer.step()
                apply_decay(sweep)

            total_loss += loss.item()

        if update_scheme == "batch":
            optimizer.step()
            apply_decay(sweep)

        final_loss = total_loss

        if verbose and (sweep == 1 or sweep % log_every == 0):
            model.eval()
            with torch.no_grad():
                spot = model(spot_p1, spot_rel)[0, spot_target_idx].item()
            model.train()
            train_correct, train_total, _, _ = evaluate(model, train_examples, verbose=False)
            test_correct, test_total, test_mean_target, _ = evaluate(model, test_triples, verbose=False)
            print(f"Sweep {sweep:4d}/{n_sweeps} | Loss: {total_loss:.4f} | "
                  f"Christopher->wife->Penelope: {spot:.4f} | Train: {train_correct}/{train_total} | "
                  f"Test: {test_correct}/{test_total} (mean act={test_mean_target:.4f})")

    train_correct, train_total, train_mean_target, train_mean_nontarget = \
        evaluate(model, train_examples, label="TRAINING SET", verbose=verbose)
    test_correct, test_total, test_mean_target, test_mean_nontarget = \
        evaluate(model, test_triples, label="HELD-OUT TEST TRIPLES", verbose=verbose)

    weight_norms = {name: getattr(model, name).norm().item() for name in ["c1", "c2", "w1", "w2"]}
    if verbose:
        print(f"Final weight norms: {', '.join(f'{k}={v:.4f}' for k, v in weight_norms.items())}")

    return {
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


def main():
    run_experiment(BEST_CONFIG)


if __name__ == "__main__":
    main()
