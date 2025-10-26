import torch, torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users:int, num_items:int, dim:int=64, n_layers:int=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.n_layers = n_layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index):
        # Placeholder: implement propagation or reuse a library implementation.
        return self.user_emb.weight, self.item_emb.weight

    def score(self, u_idx, i_idx):
        u = self.user_emb(u_idx)
        v = self.item_emb(i_idx)
        return (u * v).sum(-1)