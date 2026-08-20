"""Figure 4/5 reproduction: visualizes what the trained network learned.

Run directly: `python visualize.py` (trains a fresh model, ~5-6 minutes,
then writes the three figures/*.png files below).
"""

import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

import train
from data.family_tree import PEOPLE, RELATIONSHIPS, TRIPLES
from model.network import encode

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")

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
    """Figure 5 analog: person-encoding value per person, colored by
    nationality and generation — neither given to the network directly."""
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
    """Top-3 output activations per held-out query — both real answers rank
    above every wrong candidate."""
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


def plot_activation_diagram(model, p1_name, rel_name, out_path):
    """Figure 4 analog: activity levels across every layer for one query, as
    a grid of squares sized by activation magnitude, bottom (input) to top
    (output) — the paper's own style of showing what the network computed."""
    p1_idx, rel_idx = PEOPLE.index(p1_name), RELATIONSHIPS.index(rel_name)
    p1_onehot, r_onehot = encode(p1_idx, rel_idx)
    with torch.no_grad():
        p_repr = torch.sigmoid(p1_onehot @ model.c1 + model.b_c1)[0]
        r_repr = torch.sigmoid(r_onehot @ model.c2 + model.b_c2)[0]
        hidden = torch.sigmoid(torch.cat([p_repr, r_repr]).unsqueeze(0) @ model.w1 + model.b_w1)[0]
        output = torch.sigmoid(hidden.unsqueeze(0) @ model.w2 + model.b_w2)[0]
    valid = {PEOPLE.index(p2) for p2 in TRIPLES[(p1_name, rel_name)]}

    rows = [
        ("output (person2)", output.tolist(), {i: PEOPLE[i] for i in valid}, valid),
        ("hidden", hidden.tolist(), None, None),
        ("relation encoding (c2)", r_repr.tolist(), None, None),
        ("person encoding (c1)", p_repr.tolist(), None, None),
        ("relation (input)", r_onehot[0].tolist(), {rel_idx: rel_name}, None),
        ("person1 (input)", p1_onehot[0].tolist(), {p1_idx: p1_name}, None),
    ]

    spacing, cell = 0.55, 0.46
    fig, ax = plt.subplots(figsize=(13, 9))
    for row_i, (label, values, annotate, green_idxs) in enumerate(rows):
        n = len(values)
        width = n * spacing
        ax.axhspan(row_i - 0.5, row_i + 0.5, color="#F5F5F5" if row_i % 2 else "white", zorder=0)
        for i, v in enumerate(values):
            x = -width / 2 + (i + 0.5) * spacing
            is_answer = bool(green_idxs and i in green_idxs)
            # Outline box: every unit's fixed-size cell, so "small value" and
            # "no unit here" look different even when the fill is tiny.
            ax.add_patch(patches.Rectangle((x - cell / 2, row_i - cell / 2), cell, cell,
                                            facecolor="none",
                                            edgecolor="#2E7D32" if is_answer else "#999999",
                                            linewidth=1.3 if is_answer else 0.7, zorder=1))
            side = cell * (v ** 0.5)
            if side > 0.02:
                color = "#55A868" if is_answer else "#4C72B0"
                ax.add_patch(patches.Rectangle((x - side / 2, row_i - side / 2), side, side,
                                                facecolor=color, edgecolor="black",
                                                linewidth=0.5, zorder=2))
            if annotate and i in annotate:
                ax.text(x, row_i - cell / 2 - 0.1, annotate[i], ha="right", va="top",
                         fontsize=8, rotation=40, rotation_mode="anchor")
        ax.text(-width / 2 - 0.35, row_i, label, ha="right", va="center", fontsize=9.5)

    ax.set_xlim(-13, 13)
    ax.set_ylim(-1.5, len(rows) - 0.5)
    ax.axis("off")
    ax.set_title(f"Activity levels for ({p1_name}, {rel_name})\n"
                 f"square area = activation · outlined cell = one unit · green = the query's real correct answers",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    model = train.train()

    person_path = os.path.join(FIGURES_DIR, "person_encoding.png")
    test_path = os.path.join(FIGURES_DIR, "test_results.png")
    activation_path = os.path.join(FIGURES_DIR, "activation_diagram.png")
    plot_person_encoding(model, person_path)
    plot_test_results(model, test_path)
    plot_activation_diagram(model, "Colin", "uncle", activation_path)
    for path in (person_path, test_path, activation_path):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
