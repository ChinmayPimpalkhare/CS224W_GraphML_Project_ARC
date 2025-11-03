import torch
import torch.nn as nn


class GraphFlix(nn.Module):
    """Skeleton for GraphFlix with user-conditioned metadata bias.

    b_meta(u,i) = beta * tanh( LN(p(u))^T W LN(phi(i)) )
    where p(u) is a recency + rating-weighted profile.
    """

    def __init__(self, dim=64, heads=4, beta_init=1.0):
        super().__init__()
        self.dim = dim
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.W = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def meta_bias(self, p_u, phi_i):
        s = (self.ln(p_u) * self.W(self.ln(phi_i))).sum(-1)
        return self.beta * torch.tanh(s)

    def score(self, u_vec, i_vec, p_u=None, phi_i=None, b_other=0.0):
        base = (u_vec * i_vec).sum(-1)
        b_meta = (
            self.meta_bias(p_u, phi_i)
            if (p_u is not None and phi_i is not None)
            else 0.0
        )
        return base + b_other + b_meta
