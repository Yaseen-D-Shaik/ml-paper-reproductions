"""TreeNet — the family-tree relation network from Rumelhart, Hinton &
Williams (1986) / Hinton (1986), Figure 3.

Two departures from the paper's literal 6/6 spec: PERSON_DIM/RELATION_DIM
are 1/5 (asymmetric), and the encoding layers pass through a sigmoid
rather than staying linear. Both are empirically necessary, not arbitrary.
"""

import torch
import torch.nn as nn

N_PEOPLE = 24
N_RELATIONS = 12
PERSON_DIM = 1
RELATION_DIM = 5
HIDDEN_DIM = 6
INIT_RANGE = 0.3  # paper: fixed uniform(-0.3, 0.3)


class TreeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.c1 = nn.Parameter(torch.empty(N_PEOPLE, PERSON_DIM))
        self.c2 = nn.Parameter(torch.empty(N_RELATIONS, RELATION_DIM))
        self.b_c1 = nn.Parameter(torch.zeros(PERSON_DIM))
        self.b_c2 = nn.Parameter(torch.zeros(RELATION_DIM))

        self.w2 = nn.Parameter(torch.empty(HIDDEN_DIM, N_PEOPLE))
        self.b_w1 = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.b_w2 = nn.Parameter(torch.zeros(N_PEOPLE))

        self.w1 = nn.Parameter(torch.empty(PERSON_DIM + RELATION_DIM, HIDDEN_DIM))

        # Order matters: changes the values drawn for a given seed.
        for p in [self.c1, self.c2, self.w2, self.w1]:
            nn.init.uniform_(p, a=-INIT_RANGE, b=INIT_RANGE)

    def forward(self, person1, relationship):
        p_repr = torch.sigmoid(person1 @ self.c1 + self.b_c1)
        r_repr = torch.sigmoid(relationship @ self.c2 + self.b_c2)
        combined = torch.cat([p_repr, r_repr], dim=1)
        hidden = torch.sigmoid(combined @ self.w1 + self.b_w1)
        return torch.sigmoid(hidden @ self.w2 + self.b_w2)


def encode(person_idx, relation_idx):
    person_onehot = torch.zeros(1, N_PEOPLE)
    person_onehot[0, person_idx] = 1.0
    rel_onehot = torch.zeros(1, N_RELATIONS)
    rel_onehot[0, relation_idx] = 1.0
    return person_onehot, rel_onehot
