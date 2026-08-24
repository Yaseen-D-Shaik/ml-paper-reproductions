# ML Paper Reproductions

Foundational machine learning papers reproduced from scratch in PyTorch.

The goal is not to match someone else's implementation — it's to understand
each paper deeply enough to rebuild it independently, identify where the math
meets the code, and know exactly why each design decision was made.

Every reproduction here is written from the paper alone, without referencing
official implementations until after my version is working.

---

## Philosophy

Reading a paper and running official code teaches you what a model does.
Reproducing it from scratch teaches you how and why it works.

That distinction matters for anyone serious about ML research or architecture design.
This repo is the record of building that kind of understanding.

---

## Reproductions

| Paper | Year | Key Concept | Status | Notes |
|-------|------|-------------|--------|-------|
| Rumelhart et al. (1986) | 1986 | backpropagation | Completed | — |

---

## What Each Reproduction Includes

Every completed reproduction contains:

- **Clean PyTorch implementation** written from the paper alone
- **README** covering:
  - What the paper proposes and why it matters
  - Implementation decisions and where the paper is ambiguous
  - Training results vs. paper's reported numbers
  - Where results diverge and the likely reason
  - Key insights that aren't obvious from just reading the paper
- **Loss curves and evaluation metrics**
- **Comparison table** — my results vs. paper's reported results

---

## Papers in Queue

Papers I'm planning to reproduce, roughly in order:

- [x] Rumelhart et al. (1986)
- [ ] Srivastava et al. (2014)
- [ ] Ioffe & Szegedy (2015)
- [ ] Krizhevsky et al. (2012)

---

## Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

---

## Connect

Built by [Yaseen Dada Shaik](www.linkedin.com/in/yaseen-dada-shaik-27990a420) — Machine Learning Engineer, Bengaluru.
