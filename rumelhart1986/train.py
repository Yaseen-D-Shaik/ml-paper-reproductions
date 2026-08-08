"""Training loop for Rumelhart et al. 1986 reproduction.

Critical implementation note:
The paper uses ONLINE gradient descent — weights are updated after each
individual training triple, not after accumulating gradients over all triples.
Equation 9 states t is incremented for each input-output case, confirming
per-case updates. Full-batch accumulation causes gradient cancellation across
the 108 competing triples, preventing convergence.
"""

import random
import torch
import torch.optim as optim

from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import TreeNet, encode


def build_dataset():
    test_triples = [
        (PEOPLE.index("Colin"),     RELATIONSHIPS.index("uncle"),  PEOPLE.index("Arthur")),
        (PEOPLE.index("Alfonse"),   RELATIONSHIPS.index("uncle"),  PEOPLE.index("Emilio")),
        (PEOPLE.index("Penelope"),  RELATIONSHIPS.index("mother"), PEOPLE.index("Victoria")),
        (PEOPLE.index("Charlotte"), RELATIONSHIPS.index("aunt"),   PEOPLE.index("Christine")),
    ]
    test_set = set(test_triples)
    all_triples = []
    for (p1_str, rel_str), p2_list in TRIPLES.items():
        p1_idx  = PEOPLE.index(p1_str)
        rel_idx = RELATIONSHIPS.index(rel_str)
        for p2_str in p2_list:
            p2_idx = PEOPLE.index(p2_str)
            all_triples.append((p1_idx, rel_idx, p2_idx))
    train_triples = [t for t in all_triples if t not in test_set]
    return train_triples, test_triples


def evaluate(model, test_triples):
    model.eval()
    print("\n" + "=" * 55)
    print("      HELD-OUT TEST TRIPLES EVALUATION")
    print("=" * 55)
    correct = 0
    with torch.no_grad():
        for p1_idx, rel_idx, p2_idx in test_triples:
            p1t, rt = encode(p1_idx, rel_idx)
            output  = model(p1t, rt)
            target_activation = output[0, p2_idx].item()
            is_correct = target_activation > 0.8
            if is_correct:
                correct += 1
            status = "PASSED" if is_correct else "FAILED"
            print(f"Query: ({PEOPLE[p1_idx]}, {RELATIONSHIPS[rel_idx]}) -> {PEOPLE[p2_idx]}")
            print(f"  Activation: {target_activation:.4f} | [{status}]\n")
    print(f"Test Accuracy: {correct}/{len(test_triples)} ({correct/len(test_triples)*100:.1f}%)")
    print("=" * 55)


def main():
    torch.manual_seed(42)
    random.seed(42)

    train_triples, test_triples = build_dataset()
    n = len(train_triples)
    print(f"Train: {n} triples | Test: {len(test_triples)} triples")

    model = TreeNet()

    # Online SGD: one optimizer step per training triple per sweep
    # Use the diagnostic's stable online hyperparameters from sweep 1.
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

    print("\nStarting Training (10000 Sweeps)...")

    for sweep in range(1, 10000):

        total_loss = 0.0

        # ONLINE: update after each triple, not after full sweep
        for p1_idx, rel_idx, p2_idx in train_triples:
            optimizer.zero_grad()

            p1t, rt = encode(p1_idx, rel_idx)
            target  = torch.zeros(1, 24)
            target[0, p2_idx] = 1.0

            output = model(p1t, rt)
            loss   = 0.5 * torch.sum((output - target) ** 2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if sweep == 1 or sweep % 500 == 0:
            # Spot check a known training triple
            model.eval()
            with torch.no_grad():
                p1t, rt = encode(
                    PEOPLE.index("Christopher"),
                    RELATIONSHIPS.index("wife")
                )
                out = model(p1t, rt)
                pen = out[0, PEOPLE.index("Penelope")].item()
            model.train()
            print(f"Sweep {sweep:4d}/10000 | Loss: {total_loss:.4f} | Christopher->wife->Penelope: {pen:.4f}")

    evaluate(model, test_triples)


if __name__ == "__main__":
    main()