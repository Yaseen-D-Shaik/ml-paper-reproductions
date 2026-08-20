# Rumelhart 1986: Family Trees

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=python&logoColor=white)

In 1986, Geoffrey Hinton wanted to show something that sounds obvious now
and was radical then: that a neural network handed nothing but arbitrary,
meaningless symbols could, on its own, discover the structure hiding
underneath them. He built two family trees — one English, one Italian,
otherwise identical — and taught a small network to answer questions like
"who is Colin's uncle?" The network was never told what a person, a
generation, or a nationality was. It had to invent all of that for itself,
just to answer the questions correctly. And it did.

This is that experiment, rebuilt from scratch in PyTorch and pushed further
than the original paper attempted. Below is the story of what it took to
get there, what the network actually figured out, and a discovery about
*how you grade a reproduction* that turned out to matter more than any
single hyperparameter.

## The task

Every fact about the two families is written as `(person, relation,
person)` — 104 of them in total, using relations like father, mother,
uncle, aunt, brother, and niece. The network sees a person and a relation
as raw one-hot symbols and has to produce the right person in response.
Four of those facts are deliberately hidden from it during training. A
network that just memorized the training set has no chance on those four —
it has to have actually inferred how the families are built.

## What the network figured out on its own

Nobody ever told the network which people are English and which are
Italian, or which generation anyone belongs to. And yet, watch what happens
to the single number the network learns to represent each person:

![Person encoding](figures/person_encoding.png)

It splits cleanly into two nationalities and three generations, entirely
unprompted. That's the whole point of the original experiment, reproduced
directly: understanding a domain and being *told* its structure are
different things, and a network trained the right way discovers the first
without ever receiving the second.

Here's what that looks like at the level of a single question. This is
every layer of the network lighting up in response to "Colin, uncle?" —
drawn in the same style Hinton used in his original paper, where the size
of each square shows how strongly that unit fired:

![Activation diagram](figures/activation_diagram.png)

## Getting there wasn't straightforward

Training this network well took two techniques stacked on top of plain
gradient descent:

**Weight decay, used as a mechanism for generalization, not just as a
safety net against overfitting.** This turns out to be a textbook case of
*grokking* (Power et al., 2022): the network memorizes the training set
almost immediately, then sits there for thousands of steps looking like
it's learned nothing new — until decay quietly erodes the high-magnitude
"memorizing" solution in favor of a smaller, structurally cleaner one that
fits the same data. Generalization doesn't arrive gradually here. It
arrives all at once, late, after the network looks finished. Turning decay
on too early or too abruptly collapses training outright, so it only
switches on at sweep 16,000, and even then it's ramped in gradually rather
than flipped like a switch.

**Grokfast** (Lee et al., 2024), which amplifies the slow-moving part of
the gradient before every update, shortens how long that wait actually is.

And one finding kept repeating no matter what else changed: **giving the
network more room to work with only ever made it worse.** A wider hidden
layer, a second hidden layer, a richer way of combining person and relation
signals, a different weight initialization, a different activation function
— nine separate attempts, and every single one traded away test
performance for a higher training score. This network was never
capacity-starved. Whatever it was missing, more room was never the answer.

## The twist: how you grade this changes everything

Here's the part of this project that mattered most, and it isn't a
hyperparameter.

Graded strictly by the paper's own rule — an answer only counts if the
network is at least 80% confident in it — this network gets 0 out of 4 on
the held-out questions. That sounds like failure. But look at what it
actually outputs for those four questions, not just whether it clears a
threshold:

![Test results](figures/test_results.png)

For every single one, the network's top two guesses are the two *actual*
correct answers. Colin genuinely has two uncles — Arthur and Charles — and
the network ranks both of them above every wrong answer, every time. It
isn't confused. It just trusts the answer it saw demonstrated directly a
little more than the one it had to work out by noticing Colin's sister has
the same uncles he does.

Two other public attempts to reproduce this exact experiment report
scores of "1.9 out of 4 on average" and "3 out of 4 at best" — and neither
one grades with the paper's strict 80%-confidence rule either; one uses a
much lower bar, the other simply checks whether the top guess is *a*
correct answer rather than *the* one specific fact being tested. Once this
network is graded the same way those numbers were produced, it doesn't
just match them — it answers all four questions correctly, including on a
second, entirely unambiguous set of held-out questions built specifically
to rule out this being a lucky quirk of the multi-answer setup.

The lesson underneath all of this: before deciding a model has failed,
it's worth asking exactly what "success" was defined to mean — because two
completely reasonable definitions can look at the identical model and
disagree completely.

## Try it yourself

```
pip install -r requirements.txt
python train.py       # trains the network and evaluates it, ~5-6 minutes
python visualize.py   # regenerates the three figures above
```

`train.py` holds the training loop, `model/network.py` the architecture,
and `visualize.py` the plotting code that made the figures on this page.
[`notebooks/reproduction.ipynb`](notebooks/reproduction.ipynb) walks
through all of it inline if you'd rather read it as one continuous story,
and [`data/family_tree.py`](data/family_tree.py) has the raw facts the two
families are built from.
