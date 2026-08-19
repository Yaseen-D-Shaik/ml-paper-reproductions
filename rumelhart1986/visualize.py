"""Figure 4/5 reproduction: visualizes what the trained network actually
learned. See TECHNIQUES.md sections 4.1 and 4.15 for what these two plots
show and why they matter.

Run directly: `python visualize.py` (trains a fresh model, ~5-6 minutes,
then writes figures/person_encoding.png and figures/test_results.png).
"""

import os
import matplotlib.pyplot as plt
import torch

import train
from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import encode

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

# The paper's own labels for each person — not learned, just the family
# tree structure (see data/family_tree.py's comments) — used to color and
# annotate the plot below.
GENERATION = {
    "Christopher": 1, "Penelope": 1, "Andrew": 1, "Christine": 1,
    "Roberto": 1, "Maria": 1, "Pierro": 1, "Francesca": 1,
    "Arthur": 2, "Margaret": 2, "Victoria": 2, "James": 2, "Jennifer": 2, "Charles": 2,
    "Emilio": 2, "Gina": 2, "Lucia": 2, "Marco": 2, "Angela": 2, "Tomaso": 2,
    "Colin": 3, "Charlotte": 3, "Alfonse": 3, "Sophia": 3,
}
ENGLISH = {"Christopher", "Penelope", "Andrew", "Christine", "Arthur", "Margaret",
           "Victoria", "James", "Jennifer", "Charles", "Colin", "Charlotte"}
NATIONALITY = {name: ("English" if name in ENGLISH else "Italian") for name in PEOPLE}


def plot_person_encoding(model, out_path):
    """Figure 5 analog. PERSON_DIM=1 here (not the paper's 6), so instead of
    a multi-row Hinton diagram this is a single labeled value per person —
    still enough to show the same story: the network organizes people by
    generation and nationality without ever being told either."""
    values = []
    for name in PEOPLE:
        p_onehot, _ = encode(PEOPLE.index(name), 0)
        with torch.no_grad():
            v = (p_onehot @ model.c1 + model.b_c1).item()
        values.append(v)

    order = sorted(range(len(PEOPLE)), key=lambda i: values[i])
    names_sorted = [PEOPLE[i] for i in order]
    values_sorted = [values[i] for i in order]
    colors = ["#4C72B0" if NATIONALITY[n] == "English" else "#DD8452" for n in names_sorted]
    markers = {1: "o", 2: "s", 3: "^"}

    fig, ax = plt.subplots(figsize=(6, 9))
    for y, (name, v, c) in enumerate(zip(names_sorted, values_sorted, colors)):
        ax.scatter(v, y, color=c, marker=markers[GENERATION[name]], s=80, zorder=3)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.axvline(0, color="gray", linewidth=0.8, zorder=1)
    ax.set_xlabel("learned person-encoding value (c1)")
    ax.set_title("Person encoding: organizes by generation and nationality\n"
                  "(never told either — this emerged from training)")

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0", markersize=10, label="English"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#DD8452", markersize=10, label="Italian"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Gen 1"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Gen 2"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Gen 3"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_test_results(model, out_path):
    """Visualizes the argmax-in-valid-set finding (TECHNIQUES.md 4.15): for
    each held-out test query, the model's top-3 output activations, showing
    that both real answers rank above every wrong candidate."""
    fig, axes = plt.subplots(1, len(train.TEST_TRIPLES), figsize=(14, 4), sharey=True)
    for ax, (p1_idx, rel_idx, target_idx) in zip(axes, train.TEST_TRIPLES):
        valid = {PEOPLE.index(p2) for p2 in TRIPLES[(PEOPLE[p1_idx], RELATIONSHIPS[rel_idx])]}
        p1t, rt = encode(p1_idx, rel_idx)
        with torch.no_grad():
            output = model(p1t, rt)[0]
        ranked = sorted(range(24), key=lambda i: -output[i].item())[:3]
        names = [PEOPLE[i] for i in ranked]
        acts = [output[i].item() for i in ranked]
        colors = ["#55A868" if i in valid else "#C44E52" for i in ranked]

        ax.bar(range(3), acts, color=colors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"({PEOPLE[p1_idx]}, {RELATIONSHIPS[rel_idx]})\nheld out: {PEOPLE[target_idx]}",
                     fontsize=9)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("output activation")

    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, color="#55A868", label="real correct answer"),
        plt.Rectangle((0, 0), 1, 1, color="#C44E52", label="wrong answer"),
    ]
    fig.tight_layout(rect=[0, 0, 1, 0.82])
    fig.suptitle("Top-3 outputs per held-out query — the top 2 are always the two real answers",
                 fontsize=11, y=0.99)
    fig.legend(handles=legend_elems, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model = train.train()

    person_path = os.path.join(FIGURES_DIR, "person_encoding.png")
    test_path = os.path.join(FIGURES_DIR, "test_results.png")
    plot_person_encoding(model, person_path)
    plot_test_results(model, test_path)
    print(f"\nSaved {person_path}")
    print(f"Saved {test_path}")


if __name__ == "__main__":
    main()
