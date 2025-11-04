# scripts/run_lightgcn.py
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import LGConv


def build_bipartite_edge_index(U, M, um_edge_index):
    src = um_edge_index[0]  # users: 0..U-1
    dst = um_edge_index[1] + U  # items shifted: U..U+M-1
    e = torch.stack([src, dst], dim=0)
    er = e.flip(0)
    return torch.cat([e, er], dim=1)  # undirected


class LightGCN(nn.Module):
    def __init__(self, num_nodes, emb_dim=64, num_layers=3):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.layers = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.L = num_layers

    def forward(self, edge_index):
        x = self.emb.weight
        out = x
        for conv in self.layers:
            x = conv(x, edge_index)
            out = out + x
        return out / (self.L + 1)


def bpr_loss(u_e, i_e, j_e, reg=1e-4):
    pos = (u_e * i_e).sum(-1)
    neg = (u_e * j_e).sum(-1)
    # per-sample BPR + per-sample L2, averaged over batch
    bpr = -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()
    l2 = (u_e.pow(2).sum(dim=1) + i_e.pow(2).sum(dim=1) + j_e.pow(2).sum(dim=1)).mean()
    return bpr + reg * l2


@torch.no_grad()
def eval_split(emb_all, U, M, split_csv, train_pos, split_name="val", K=(10, 20)):
    s = pd.read_csv(split_csv)
    truth = {}
    for _, r in s.iterrows():
        if r["split"] != split_name:
            continue
        u, i = int(r["user_idx"]), int(r["movie_idx"])
        truth.setdefault(u, []).append(i)

    user_emb = emb_all[:U]
    item_emb = emb_all[U : U + M]
    scores = user_emb @ item_emb.T

    # mask train items
    for u, items in enumerate(train_pos):
        if items:
            scores[u, torch.tensor(list(items))] = -1e9

    out = {}
    for k in K:
        topk_idx = torch.topk(scores, k, dim=1).indices
        recalls, ndcgs = [], []
        for u, items in truth.items():
            top = topk_idx[u].tolist()
            hits = sum(1 for it in items if it in top)
            recalls.append(hits / max(1, len(items)))
            dcg = 0.0
            for rank, it in enumerate(top):
                if it in items:
                    dcg += 1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
            ideal = sum(
                1.0 / torch.log2(torch.tensor(r + 2.0)).item()
                for r in range(min(len(items), k))
            )
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        out[f"recall@{k}"] = float(torch.tensor(recalls).mean())
        out[f"ndcg@{k}"] = float(torch.tensor(ndcgs).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/ml1m")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    P = Path(args.root)
    data = torch.load(P / "graph_pyg.pt")  # built earlier with train-only user→movie
    U = data["user"].num_nodes
    M = data["movie"].num_nodes

    # bipartite edges for LGConv
    um = data["user", "rates", "movie"].edge_index.to(args.device)
    E = um.size(1)
    edge_index = build_bipartite_edge_index(U, M, um).to(args.device)

    # train positives by user (CPU set for sampling)
    pos = [[] for _ in range(U)]
    um_cpu = um.cpu()
    for u, i in zip(um_cpu[0].tolist(), um_cpu[1].tolist()):
        pos[u].append(i)
    pos = [set(x) for x in pos]

    model = LightGCN(U + M, emb_dim=args.dim, num_layers=args.layers).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    g = torch.Generator(device=args.device)

    def sample_batch(B):
        #  Sample B train edges uniformly → gives (u, i_pos)
        idx = torch.randint(0, E, (B,), generator=g, device=args.device)
        u = um[0, idx]
        i_pos = um[1, idx]

        #  Sample B negatives; resample if collides with any of user's positives
        j_neg = torch.randint(0, M, (B,), generator=g, device=args.device)
        # Resample only where needed
        bad = torch.tensor(
            [(int(j_neg[k].item()) in pos[int(u[k].item())]) for k in range(B)],
            device=args.device,
            dtype=torch.bool,
        )
        while bad.any():
            kidx = bad.nonzero(as_tuple=True)[0]
            j_neg[kidx] = torch.randint(
                0, M, (kidx.numel(),), generator=g, device=args.device
            )
            bad = torch.tensor(
                [(int(j_neg[k].item()) in pos[int(u[k].item())]) for k in range(B)],
                device=args.device,
                dtype=torch.bool,
            )
        return u, i_pos, j_neg

    steps = max(1, um.size(1) // args.batch)
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(steps):
            users, i_pos, j_neg = sample_batch(args.batch)
            emb_all = model(edge_index)  # [U+M, d]
            u_e = emb_all[users]  # [B, d] (users)
            i_e = emb_all[U + i_pos]  # items are offset by U
            j_e = emb_all[U + j_neg]
            loss = bpr_loss(u_e, i_e, j_e, reg=1e-4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {ep:03d} | bpr {total/steps:.4f}")

    spl = P / "splits" / "ratings_split_reindexed.csv"

    model.eval()
    emb_all = model(edge_index).detach().cpu()  # final propagated embeddings

    val_metrics = eval_split(emb_all, U, M, spl, pos, split_name="val", K=(10, 20))
    test_metrics = eval_split(emb_all, U, M, spl, pos, split_name="test", K=(10, 20))
    print("VAL :", val_metrics)
    print("TEST:", test_metrics)

    # Save artifacts (git-ignored)
    (P / "runs").mkdir(parents=True, exist_ok=True)
    torch.save({"emb": emb_all, "U": U, "M": M}, P / "lightgcn_emb.pt")

    with open(P / "runs" / "lightgcn_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)
    print("Saved:", P / "runs" / "lightgcn_metrics.json")

    # Save embeddings (for eval script)
    # torch.save({"emb": model.emb.weight.detach().cpu(), "U": U, "M": M}, P/"lightgcn_emb.pt")
    print("Saved:", P / "lightgcn_emb.pt")


if __name__ == "__main__":
    main()
