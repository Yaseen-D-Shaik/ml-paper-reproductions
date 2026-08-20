"""Training loop for the family-tree reproduction: SGD + Grokfast + delayed
weight decay. Run directly: `python train.py`
"""

import random
import torch
import torch.optim as optim

from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import TreeNet, encode

SEED = 42
N_SWEEPS = 35000
LOG_EVERY = 5000

# Paper's two-phase schedule (Fig. 4): epsilon=0.005, alpha=0.5 for sweeps
# 1-20, then epsilon=0.01, alpha=0.9. Each tuple is (last_sweep, lr, momentum).
LR_SCHEDULE = [(20, 0.005, 0.5), (float("inf"), 0.01, 0.9)]

# Grokfast (Lee et al. 2024): mu <- alpha*mu + (1-alpha)*g; g_hat <- g + lambda*mu
GROKFAST_ALPHA = 0.98
GROKFAST_LAMBDA = 2.0

# Decay engages at sweep 16000, ramped over 1000 sweeps — an abrupt onset
# shock-collapses training.
DECAY_RATE = 0.9995
DECAY_START_SWEEP = 16000
DECAY_RAMP_SWEEPS = 1000

# Two isomorphism-mirrored (English, Italian) pairs, each with two valid
# answers — the network must recover one it was never directly trained on.
TEST_TRIPLES = [
    (PEOPLE.index("Colin"),     RELATIONSHIPS.index("uncle"), PEOPLE.index("Arthur")),
    (PEOPLE.index("Alfonse"),   RELATIONSHIPS.index("uncle"), PEOPLE.index("Emilio")),
    (PEOPLE.index("Charlotte"), RELATIONSHIPS.index("aunt"),  PEOPLE.index("Margaret")),
    (PEOPLE.index("Sophia"),    RELATIONSHIPS.index("aunt"),  PEOPLE.index("Gina")),
]

# A second held-out set, each a genuinely single-answer fact.
VAL_TRIPLES = [
    (PEOPLE.index("Arthur"),  RELATIONSHIPS.index("nephew"), PEOPLE.index("Colin")),
    (PEOPLE.index("Emilio"),  RELATIONSHIPS.index("nephew"), PEOPLE.index("Alfonse")),
    (PEOPLE.index("Charles"), RELATIONSHIPS.index("niece"),  PEOPLE.index("Charlotte")),
    (PEOPLE.index("Tomaso"),  RELATIONSHIPS.index("niece"),  PEOPLE.index("Sophia")),
]

HELD_OUT = set(TEST_TRIPLES) | set(VAL_TRIPLES)


def build_dataset():
    """One multihot example per (person1, relation) key, minus held-out facts."""
    examples = []
    for (p1_str, rel_str), p2_list in TRIPLES.items():
        p1_idx, rel_idx = PEOPLE.index(p1_str), RELATIONSHIPS.index(rel_str)
        remaining = tuple(PEOPLE.index(p2) for p2 in p2_list
                           if (p1_idx, rel_idx, PEOPLE.index(p2)) not in HELD_OUT)
        if remaining:
            examples.append((p1_idx, rel_idx, remaining))
    return examples


def compute_loss(output, target):
    """Masked per the paper: error is zero once a unit that should be on is
    above 0.8, or one that should be off is below 0.2."""
    error = output - target
    mask = ~((target == 1.0) & (output > 0.8)) & ~((target == 0.0) & (output < 0.2))
    return 0.5 * torch.sum((error * mask.detach()) ** 2)


def lr_momentum_for_sweep(sweep):
    for last_sweep, lr, momentum in LR_SCHEDULE:
        if sweep <= last_sweep:
            return lr, momentum


def decay_rate_for_sweep(sweep):
    if sweep < DECAY_START_SWEEP:
        return 1.0
    progress = min(1.0, (sweep - DECAY_START_SWEEP) / DECAY_RAMP_SWEEPS)
    return 1.0 - progress * (1.0 - DECAY_RATE)


def evaluate(model, triples, label="", verbose=True):
    """Paper's literal threshold: correct only if every target unit is above 0.8."""
    model.eval()
    target_acts, nontarget_acts, results = [], [], []
    with torch.no_grad():
        for p1_idx, rel_idx, target in triples:
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
    mean_target = sum(target_acts) / len(target_acts)
    mean_nontarget = sum(nontarget_acts) / len(nontarget_acts)

    if verbose:
        title = f"{label} (0.8/0.2 threshold)" if label else "EVALUATION"
        print(f"\n--- {title} ---")
        for p1_idx, rel_idx, target_idxs, example_acts, is_correct in results:
            names = ", ".join(f"{PEOPLE[i]}={a:.4f}" for i, a in zip(target_idxs, example_acts))
            print(f"  ({PEOPLE[p1_idx]}, {RELATIONSHIPS[rel_idx]}) -> {names} "
                  f"[{'PASS' if is_correct else 'FAIL'}]")
        print(f"  {correct}/{len(triples)} correct | mean target act={mean_target:.4f} | "
              f"mean non-target act={mean_nontarget:.4f}")

    return correct, len(triples), mean_target, mean_nontarget


def evaluate_argmax(model, triples, label="", verbose=True):
    """Ranking-based: correct if the model's top-activation unit is any of
    the query's true valid answers, not just the one specific fact held out."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for p1_idx, rel_idx, target_idx in triples:
            valid = {PEOPLE.index(p2) for p2 in TRIPLES[(PEOPLE[p1_idx], RELATIONSHIPS[rel_idx])]}
            p1t, rt = encode(p1_idx, rel_idx)
            output = model(p1t, rt)[0]
            top = output.argmax().item()
            is_correct = top in valid
            correct += is_correct
            if verbose:
                print(f"  ({PEOPLE[p1_idx]}, {RELATIONSHIPS[rel_idx]}) -> top pick={PEOPLE[top]} "
                      f"(act={output[top].item():.4f}) | valid answers={[PEOPLE[i] for i in valid]} "
                      f"[{'PASS' if is_correct else 'FAIL'}]")
    model.train()
    if verbose:
        print(f"  {label}: {correct}/{len(triples)} correct (argmax-in-valid-set)")
    return correct, len(triples)


def train():
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_examples = build_dataset()
    model = TreeNet()
    optimizer = optim.SGD(model.parameters(), lr=LR_SCHEDULE[0][1], momentum=LR_SCHEDULE[0][2])
    current_lr, current_momentum = LR_SCHEDULE[0][1], LR_SCHEDULE[0][2]
    grokfast_ema = {p: torch.zeros_like(p) for p in model.parameters()}

    print(f"Train: {len(train_examples)} examples | Val: {len(VAL_TRIPLES)} | Test: {len(TEST_TRIPLES)}")
    print(f"Starting training ({N_SWEEPS} sweeps)...")

    for sweep in range(1, N_SWEEPS + 1):
        lr, momentum = lr_momentum_for_sweep(sweep)
        if (lr, momentum) != (current_lr, current_momentum):
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            current_lr, current_momentum = lr, momentum

        random.shuffle(train_examples)
        optimizer.zero_grad()
        total_loss = 0.0
        for p1_idx, rel_idx, target_idxs in train_examples:
            p1t, rt = encode(p1_idx, rel_idx)
            target = torch.zeros(1, 24)
            for idx in target_idxs:
                target[0, idx] = 1.0
            output = model(p1t, rt)
            loss = compute_loss(output, target)
            loss.backward()
            total_loss += loss.item()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is None:
                    continue
                mu = grokfast_ema[p]
                mu.mul_(GROKFAST_ALPHA).add_(p.grad, alpha=1 - GROKFAST_ALPHA)
                p.grad.add_(mu, alpha=GROKFAST_LAMBDA)

        optimizer.step()

        rate = decay_rate_for_sweep(sweep)
        if rate < 1.0:
            with torch.no_grad():
                for p in model.parameters():
                    p.mul_(rate)

        if sweep == 1 or sweep % LOG_EVERY == 0:
            train_correct, train_total, _, _ = evaluate(model, train_examples, verbose=False)
            test_correct, test_total, test_mean, _ = evaluate(model, TEST_TRIPLES, verbose=False)
            print(f"Sweep {sweep:5d}/{N_SWEEPS} | Loss: {total_loss:.4f} | "
                  f"Train: {train_correct}/{train_total} | "
                  f"Test: {test_correct}/{test_total} (mean act={test_mean:.4f})")

    evaluate(model, train_examples, label="TRAINING SET")
    evaluate(model, TEST_TRIPLES, label="HELD-OUT TEST TRIPLES")
    evaluate(model, VAL_TRIPLES, label="HELD-OUT VAL TRIPLES")
    print("\nArgmax-in-valid-set scoring:")
    evaluate_argmax(model, TEST_TRIPLES, label="TEST_TRIPLES")
    evaluate_argmax(model, VAL_TRIPLES, label="VAL_TRIPLES")

    return model


if __name__ == "__main__":
    train()
