"""Neural network architecture for the Rumelhart 1986 reproduction.

Architecture follows Figure 3 of Rumelhart, Hinton & Williams (1986):
  - Input: person1 (24-dim one-hot) + relationship (12-dim one-hot)
  - Layer 2: two separate encoding_dim-unit groups, one per input group
    (Eq. 1, 2). Paper uses 6; configurable here as a capacity-bottleneck lever.
  - Layer 3: central 2*encoding_dim-unit layer (concatenation of both groups)
  - Layer 4: penultimate 6-unit layer
  - Output: 24-unit layer, one per person

encoding_params() / output_params() split the parameters into two groups so
train.py can apply weight decay to either group independently — decay scope
(all params vs. output-layer-only) is one of the axes under investigation,
not a fixed architectural decision.
"""

import torch
import torch.nn as nn


class TreeNet(nn.Module):

    def __init__(self, encoding_nonlinearity="linear", w1_init_range=0.3,
                 encoding_dim=6, person_encoding_dim=None, relation_encoding_dim=None,
                 hidden_dim=6, hidden_nonlinearity="sigmoid", use_bilinear=False,
                 init_scheme="uniform"):
        """
        encoding_nonlinearity: "linear" or "sigmoid" — whether C1/C2 outputs
            pass through a sigmoid (paper's Eq. 1+2 applied uniformly to all
            layers) or stay linear (current working assumption).
        w1_init_range: uniform init bound for w1, i.e. U(-w1_init_range, w1_init_range).
            Paper specifies 0.3 for all weights; current code deviates to 1.0.
        encoding_dim: width of the C1/C2 encoding layers (paper: 6). A capacity
            bottleneck lever — smaller values force more compression than
            weight decay's magnitude penalty does, independent of it. Used
            symmetrically for both C1 and C2 unless overridden below.
        person_encoding_dim / relation_encoding_dim: override encoding_dim
            independently for C1 (person) and C2 (relation), to test whether
            an even split is actually optimal or just the default.
        hidden_dim: width of the penultimate layer (paper: 6). Varying this
            independently of encoding_dim tests whether a capacity bottleneck's
            benefit is specific to the identity-encoding layer or just "less
            capacity anywhere in the network."
        hidden_nonlinearity: "sigmoid" (paper) or "tanh" — activation for the
            central->penultimate layer (w1) only. With sigmoid encoding and
            sigmoid output, this network has a 3-deep sigmoid chain; other
            reproductions of this task found sigmoid'(0)=0.25 compounds across
            layers (0.25^4 ~ 0.004) and switched an interior layer to tanh
            (wider derivative range) to keep gradient reaching the encoding
            layer. Output stays sigmoid regardless, since compute_loss's
            0/1 targets and 0.8/0.2 masked threshold assume that range.
        use_bilinear: if True, replace concat+w1 with a bilinear person x
            relation interaction (a learned 3-tensor B[person_dim,
            relation_dim, hidden_dim] contracted against both representations)
            instead of concatenating them into one linear layer. Concatenation
            only allows additive combinations of person-features and
            relation-features; a bilinear form lets each relation-feature
            modulate each person-feature directly, which paper's Fig. 3
            "central layer" doesn't literally specify but is a structurally
            richer way to combine two heavily compressed signals.
        init_scheme: "uniform" (paper: fixed U(-0.3,0.3) everywhere,
            regardless of layer width) or "xavier" (Glorot uniform: bound
            scales with 1/sqrt(fan_in+fan_out) per layer, keeping initial
            signal variance comparable across layers of very different width
            — relevant here since person_dim/relation_dim/hidden_dim can now
            differ enormously, unlike the paper's uniform 6/6/6/24 shape).
        """
        super().__init__()
        self.encoding_nonlinearity = encoding_nonlinearity
        self.hidden_nonlinearity = hidden_nonlinearity
        self.use_bilinear = use_bilinear
        person_dim = person_encoding_dim if person_encoding_dim is not None else encoding_dim
        relation_dim = relation_encoding_dim if relation_encoding_dim is not None else encoding_dim

        # Encoding layers — one per input group (Figure 3)
        # Paper: 24 person units -> 6 units, 12 relation units -> 6 units
        self.c1 = nn.Parameter(torch.empty(24, person_dim))     # person encoding
        self.c2 = nn.Parameter(torch.empty(12, relation_dim))   # relation encoding
        self.b_c1 = nn.Parameter(torch.zeros(person_dim))
        self.b_c2 = nn.Parameter(torch.zeros(relation_dim))

        # Central and output layers (Figure 3)
        self.w2 = nn.Parameter(torch.empty(hidden_dim, 24))     # penultimate -> output
        self.b_w1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_w2 = nn.Parameter(torch.zeros(24))

        def init_(param, uniform_bound):
            if init_scheme == "xavier":
                nn.init.xavier_uniform_(param)
            else:
                nn.init.uniform_(param, a=-uniform_bound, b=uniform_bound)

        # Initialize all weights (paper: fixed U(-0.3,0.3) everywhere).
        # This loop must stay ordered/positioned exactly as before adding
        # use_bilinear, and w1/bilinear init must stay last — the RNG draw
        # order affects the actual values assigned for a given seed, and
        # every recorded result in FINDINGS.md depends on this exact sequence.
        for param in [self.c1, self.c2, self.w2]:
            init_(param, 0.3)

        if use_bilinear:
            # B[i, j, k]: contribution of person-feature i and relation-feature
            # j to hidden unit k. hidden_pre = einsum('bi,ijk,bj->bk', p, B, r)
            self.bilinear = nn.Parameter(torch.empty(person_dim, relation_dim, hidden_dim))
            init_(self.bilinear, w1_init_range)
            self.w1 = None
        else:
            # (person_dim + relation_dim) -> hidden_dim, central -> penultimate
            self.w1 = nn.Parameter(torch.empty(person_dim + relation_dim, hidden_dim))
            init_(self.w1, w1_init_range)
            self.bilinear = None

    def forward(self, person1, relationship):
        """
        person1:      (1, 24) one-hot tensor
        relationship: (1, 12) one-hot tensor
        returns:      (1, 24) output activations
        """
        # Equation 1 + 2 applied at encoding layers
        p_repr = person1 @ self.c1 + self.b_c1        # (1, person_dim)
        r_repr = relationship @ self.c2 + self.b_c2   # (1, relation_dim)
        if self.encoding_nonlinearity == "sigmoid":
            p_repr = torch.sigmoid(p_repr)
            r_repr = torch.sigmoid(r_repr)

        if self.use_bilinear:
            # Each relation-feature modulates each person-feature directly,
            # instead of the two only ever being combined additively.
            pre_hidden = torch.einsum("bi,ijk,bj->bk", p_repr, self.bilinear, r_repr) + self.b_w1
        else:
            # Concatenate into central layer input (Figure 3)
            combined = torch.cat([p_repr, r_repr], dim=1)              # (1, person_dim + relation_dim)
            pre_hidden = combined @ self.w1 + self.b_w1                # (1, hidden_dim)

        # Central -> penultimate -> output (Eq. 1, 2)
        hidden = torch.tanh(pre_hidden) if self.hidden_nonlinearity == "tanh" else torch.sigmoid(pre_hidden)
        output = torch.sigmoid(hidden @ self.w2 + self.b_w2)          # (1, 24)

        return output

    def encoding_params(self):
        """Parameters that should NOT receive weight decay."""
        return [self.c1, self.c2, self.b_c1, self.b_c2]

    def output_params(self):
        """Parameters that receive weight decay (Eq. 9 + paper decay rule)."""
        central_to_hidden = self.bilinear if self.use_bilinear else self.w1
        return [central_to_hidden, self.w2, self.b_w1, self.b_w2]


def encode(person_idx, relation_idx, n_people=24, n_relations=12):
    """Convert integer indices to one-hot tensors with batch dimension."""
    person_onehot = torch.zeros(1, n_people)
    person_onehot[0, person_idx] = 1.0
    rel_onehot = torch.zeros(1, n_relations)
    rel_onehot[0, relation_idx] = 1.0
    return person_onehot, rel_onehot
