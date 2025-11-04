# scripts/run_lightgcn.py
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


@torch.no_grad()
def eval_split(
    emb_all, U, M, split_csv, train_pos, split_name="val", K=(10, 20), use_cosine=False
):
    s = pd.read_csv(split_csv)
    truth = {}
    for _, r in s.iterrows():
        if r["split"] != split_name:
            continue
        u, i = int(r["user_idx"]), int(r["movie_idx"])
        truth.setdefault(u, []).append(i)

    user_emb = emb_all[:U]
    item_emb = emb_all[U : U + M]

    # Optional: cosine similarity
    if use_cosine:
        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)

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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--neg", type=int, default=5)  # ← MOVED HERE
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--use_cosine", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    P = Path(args.root)
    data = torch.load(P / "graph_pyg.pt")
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

    torch.manual_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    def sample_batch(B):
        idx = torch.randint(0, E, (B,), generator=g, device=args.device)
        u = um[0, idx]
        i_pos = um[1, idx]
        # K negatives per positive (collision rate ~3-5% on ML-1M, acceptable)
        j_neg = torch.randint(0, M, (B, args.neg), generator=g, device=args.device)
        return u, i_pos, j_neg

    steps = max(1, E // args.batch)
    spl = P / "splits" / "ratings_split_reindexed.csv"

    best_val_ndcg = 0.0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps):
            users, i_pos, j_neg = sample_batch(args.batch)
            emb_all = model(edge_index)  # [U+M, d]

            u_e = emb_all[users]  # [B, d]
            i_e = emb_all[U + i_pos]  # [B, d]
            j_e = emb_all[U + j_neg]  # [B, neg, d]

            pos_scores = (u_e * i_e).sum(-1, keepdim=True)  # [B, 1]
            neg_scores = (u_e.unsqueeze(1) * j_e).sum(-1)  # [B, neg]
            bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            # per-sample L2 across all negatives:
            l2 = (
                u_e.pow(2).sum(1).mean()
                + i_e.pow(2).sum(1).mean()
                + j_e.pow(2).sum(-1).mean()  # sum over d, mean over B×neg
            )
            loss = bpr + args.reg * l2

            # ← CRITICAL: Actually train!
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps
        print(f"epoch {ep:03d} | loss {avg_loss:.4f}", end="")

        # Periodic validation
        if ep % 5 == 0 or ep == 1:
            model.eval()
            emb_all = model(edge_index).detach().cpu()
            val_metrics = eval_split(
                emb_all, U, M, spl, pos, "val", K=(10,), use_cosine=args.use_cosine
            )
            val_ndcg = val_metrics["ndcg@10"]
            print(f" | val_ndcg@10={val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = ep
                # Save best checkpoint
                torch.save(
                    {"emb": emb_all, "epoch": ep, "val_ndcg": val_ndcg},
                    P / "lightgcn_best.pt",
                )
        else:
            print()

    print(f"\nBest val NDCG@10: {best_val_ndcg:.4f} at epoch {best_epoch}")

    # Final eval on test
    model.eval()
    emb_all = model(edge_index).detach().cpu()

    val_metrics = eval_split(
        emb_all, U, M, spl, pos, "val", K=(10, 20), use_cosine=args.use_cosine
    )
    test_metrics = eval_split(
        emb_all, U, M, spl, pos, "test", K=(10, 20), use_cosine=args.use_cosine
    )

    print("VAL :", val_metrics)
    print("TEST:", test_metrics)

    # Save artifacts
    (P / "runs").mkdir(parents=True, exist_ok=True)
    torch.save({"emb": emb_all, "U": U, "M": M}, P / "lightgcn_emb.pt")

    with open(P / "runs" / "lightgcn_metrics.json", "w") as f:
        json.dump(
            {
                "val": val_metrics,
                "test": test_metrics,
                "best_epoch": best_epoch,
                "best_val_ndcg": best_val_ndcg,
            },
            f,
            indent=2,
        )

    print("Saved:", P / "runs" / "lightgcn_metrics.json")
    print("Saved:", P / "lightgcn_emb.pt")


if __name__ == "__main__":
    main()
