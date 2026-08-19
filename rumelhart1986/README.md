# Rumelhart 1986 Reproduction

Reproduction of the family-tree relational-learning experiment from
Rumelhart, Hinton & Williams (1986) / Hinton (1986) — a small network that
learns to answer relational queries (e.g. "Colin's uncle?") over two
isomorphic family trees purely from raw, arbitrary symbols.

Run it: `python train.py`

## Where to look

- `TECHNIQUES.md` — **start here.** Design decisions, every technique
  tried (with what worked, what didn't, and why), and the takeaways worth
  carrying into future ML work.
- `FINDINGS.md` — the full chronological experiment log behind those
  decisions, including every dead end.
- `train.py` — the finalized training loop and evaluation (SGD + Grokfast +
  delayed weight decay; both the paper's literal threshold and an
  argmax-in-valid-set score are reported).
- `model/network.py` — the finalized architecture.
- `data/family_tree.py` — the raw facts.

## Structure

- `data/` — family tree triples
- `model/` — network architecture
- `train.py` — training loop and evaluation
- `visualize.py` — Figure 4 reproduction (not yet built — planned once
  optimization work is complete)
- `notebooks/` — end-to-end notebook (not yet built)
