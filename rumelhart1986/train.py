"""Training loop for Rumelhart et al. 1986 reproduction."""

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

    # Phase 1: sweeps 1-20, low momentum warm-up (paper specification)
    # lr=0.05 chosen empirically — paper's lr=0.005 is too small relative
    # to weight decay magnitude for this PyTorch implementation
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)

    print("\nStarting Training (1500 Sweeps)...")

    for sweep in range(1, 1501):

        # Phase 2: sweeps 21-1500
        if sweep == 21:
            optimizer.param_groups[0]['lr'] = 0.1
            optimizer.param_groups[0]['momentum'] = 0.9

        optimizer.zero_grad()
        total_loss = 0.0

        for p1_idx, rel_idx, p2_idx in train_triples:
            p1t, rt = encode(p1_idx, rel_idx)
            target  = torch.zeros(1, 24)
            target[0, p2_idx] = 1.0
            output = model(p1t, rt)
            loss = 0.5 * torch.sum((output - target) ** 2)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()

        # Multiplicative weight decay — paper: 0.2% per weight update
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'b_' not in name:
                    param.mul_(0.998)

        if sweep == 1 or sweep % 100 == 0:
            print(f"Sweep {sweep:4d}/1500 | Loss: {total_loss:.4f}")

    evaluate(model, test_triples)


if __name__ == "__main__":
    main()