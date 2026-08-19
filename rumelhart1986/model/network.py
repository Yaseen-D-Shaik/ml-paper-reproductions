"""TreeNet — final architecture for the Rumelhart et al. (1986) family-tree
reproduction.

Follows Figure 3 of the paper: person1 and relationship each arrive as a
local (one-hot) code, get compressed by their own encoding layer, are
concatenated into a central representation, pass through a penultimate
layer, and expand back out into a local code over all 24 people.

Two deliberate departures from the paper's literal spec, both empirically
load-bearing rather than arbitrary (see TECHNIQUES.md for the experiments
that established them):

- PERSON_DIM/RELATION_DIM are 1 and 5, not the paper's even 6/6 split. This
  asymmetric bottleneck is what let the person encoding organize cleanly by
  generation and nationality (see TECHNIQUES.md's "capacity bottleneck"
  section) — a 6/6 split left the network too much room to memorize instead
  of compress.
- The encoding layers (c1/c2) pass through a sigmoid, not a linear
  pass-through — this keeps the network a uniform sigmoid chain end to end,
  which is what actually trains well here despite the vanishing-gradient
  risk that motivated other reproductions to switch to tanh instead (see
  TECHNIQUES.md — tanh was tried on this project's setup and hurt).
"""

import torch
import torch.nn as nn

N_PEOPLE = 24
N_RELATIONS = 12
PERSON_DIM = 1
RELATION_DIM = 5
HIDDEN_DIM = 6
INIT_RANGE = 0.3  # paper: fixed uniform(-0.3, 0.3) for every weight


class TreeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Parameter(torch.empty(N_PEOPLE, PERSON_DIM))       # person encoding
        self.c2 = nn.Parameter(torch.empty(N_RELATIONS, RELATION_DIM))  # relation encoding
        self.b_c1 = nn.Parameter(torch.zeros(PERSON_DIM))
        self.b_c2 = nn.Parameter(torch.zeros(RELATION_DIM))

        self.w2 = nn.Parameter(torch.empty(HIDDEN_DIM, N_PEOPLE))       # penultimate -> output
        self.b_w1 = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.b_w2 = nn.Parameter(torch.zeros(N_PEOPLE))

        self.w1 = nn.Parameter(torch.empty(PERSON_DIM + RELATION_DIM, HIDDEN_DIM))  # central -> penultimate

        # Init order (c1, c2, w2, w1) matches every recorded result in
        # FINDINGS.md for seed=42 — changing it changes the actual values
        # drawn for a given seed, not just cosmetically.
        for p in [self.c1, self.c2, self.w2, self.w1]:
            nn.init.uniform_(p, a=-INIT_RANGE, b=INIT_RANGE)

    def forward(self, person1, relationship):
        """
        person1:      (1, 24) one-hot tensor
        relationship: (1, 12) one-hot tensor
        returns:      (1, 24) output activations
        """
        p_repr = torch.sigmoid(person1 @ self.c1 + self.b_c1)
        r_repr = torch.sigmoid(relationship @ self.c2 + self.b_c2)
        combined = torch.cat([p_repr, r_repr], dim=1)
        hidden = torch.sigmoid(combined @ self.w1 + self.b_w1)
        return torch.sigmoid(hidden @ self.w2 + self.b_w2)


def encode(person_idx, relation_idx):
    """Convert integer indices to one-hot tensors with a batch dimension."""
    person_onehot = torch.zeros(1, N_PEOPLE)
    person_onehot[0, person_idx] = 1.0
    rel_onehot = torch.zeros(1, N_RELATIONS)
    rel_onehot[0, relation_idx] = 1.0
    return person_onehot, rel_onehot
