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
    def __init__(self, num_nodes, emb_dim=64, num_layers=3, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.emb.weight, std=0.01)
        self.layers = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.L = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, edge_index, apply_dropout=True):
        x = self.emb.weight
        out = x
        for conv in self.layers:
            x = conv(x, edge_index)
            if self.dropout is not None and apply_dropout:
                x = self.dropout(x)
            out = out + x
        return out / (self.L + 1)


@torch.no_grad()
def eval_split(
    emb_all,
    U,
    M,
    split_csv,
    train_pos,
    split_name="val",
    K=(10, 20),
    use_cosine=False,
    device="cpu",
):
    # Ensure embeddings are on CPU for evaluation to avoid memory issues
    if emb_all.device.type != "cpu":
        emb_all = emb_all.cpu()

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

    # mask train items - optimize by batching
    for u, items in enumerate(train_pos):
        if items:
            scores[u, torch.tensor(list(items), dtype=torch.long)] = -1e9

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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )
    ap.add_argument("--early_stop_patience", type=int, default=20)
    ap.add_argument("--use_cosine", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # MPS-specific optimizations
    if args.device == "mps":
        # Enable MPS fallback for unsupported operations
        import os

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("✓ Enabled MPS fallback for unsupported operations")

    print("=" * 70)
    print("LightGCN Training Configuration")
    print("=" * 70)
    print(f"Device: {args.device}")
    if args.device == "mps":
        print(f"  → Using Apple Silicon GPU (Metal Performance Shaders)")
    print(f"Seed: {args.seed}")
    print(f"Embedding dim: {args.dim}")
    print(f"Num layers: {args.layers}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"Negatives per positive: {args.neg}")
    print(f"L2 regularization: {args.reg}")
    print(f"Dropout: {args.dropout}")
    print(f"Gradient clipping: {args.grad_clip}")
    print(f"LR Scheduler: {args.scheduler}")
    print(f"Early stopping patience: {args.early_stop_patience}")
    print(f"Use cosine similarity: {args.use_cosine}")
    print("=" * 70)

    P = Path(args.root)
    print(f"\nLoading data from: {P}")
    data = torch.load(P / "graph_pyg.pt", weights_only=False)
    U = data["user"].num_nodes
    M = data["movie"].num_nodes
    print(f"Users: {U:,}")
    print(f"Movies: {M:,}")

    # bipartite edges for LGConv
    um = data["user", "rates", "movie"].edge_index.to(args.device)
    E = um.size(1)
    print(f"Training edges: {E:,}")
    edge_index = build_bipartite_edge_index(U, M, um).to(args.device)
    print(f"Bipartite graph edges (undirected): {edge_index.size(1):,}")

    # train positives by user (CPU set for sampling)
    print("\nBuilding train positive sets...")
    pos = [[] for _ in range(U)]
    um_cpu = um.cpu()
    for u, i in zip(um_cpu[0].tolist(), um_cpu[1].tolist()):
        pos[u].append(i)
    pos = [set(x) for x in pos]

    print(f"Initializing LightGCN model...")
    model = LightGCN(
        U + M, emb_dim=args.dim, num_layers=args.layers, dropout=args.dropout
    ).to(args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Add learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        print(f"Using cosine annealing scheduler (eta_min={args.lr * 0.01:.2e})")
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        print(f"Using step scheduler (step_size=50, gamma=0.5)")
    else:
        print("No learning rate scheduler")

    torch.manual_seed(args.seed)

    # MPS optimization: use CPU generator and transfer to device
    # This avoids potential MPS-specific issues with torch.randint
    if args.device == "mps":
        g = torch.Generator(device="cpu").manual_seed(args.seed)

        def sample_batch(B):
            idx = torch.randint(0, E, (B,), generator=g, device="cpu").to(args.device)
            u = um[0, idx]
            i_pos = um[1, idx]
            # K negatives per positive (collision rate ~3-5% on ML-1M, acceptable)
            j_neg = torch.randint(0, M, (B, args.neg), generator=g, device="cpu").to(
                args.device
            )
            return u, i_pos, j_neg

    else:
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
    patience_counter = 0

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"Steps per epoch: {steps:,}")
    print(f"Total training samples per epoch: {steps * args.batch:,}")
    print()

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for step_idx in range(steps):
            users, i_pos, j_neg = sample_batch(args.batch)
            # Apply dropout during training
            emb_all = model(edge_index, apply_dropout=True)  # [U+M, d]

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

            opt.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            total_loss += loss.item()

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr

        # Synchronize for MPS to ensure completion
        if args.device == "mps":
            torch.mps.synchronize()

        avg_loss = total_loss / steps
        print(
            f"Epoch {ep:3d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}",
            end="",
        )

        # Periodic validation
        if ep % 5 == 0 or ep == 1:
            model.eval()
            print(" | Evaluating...", end="", flush=True)
            # No dropout during evaluation
            with torch.no_grad():
                emb_all = model(edge_index, apply_dropout=False).detach().cpu()
            val_metrics = eval_split(
                emb_all, U, M, spl, pos, "val", K=(10,), use_cosine=args.use_cosine
            )
            val_ndcg = val_metrics["ndcg@10"]
            print(
                f"\rEpoch {ep:3d}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Val NDCG@10: {val_ndcg:.4f}",
                end="",
            )

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = ep
                patience_counter = 0
                print(" ← NEW BEST", end="")
                # Save best checkpoint
                torch.save(
                    {"emb": emb_all, "epoch": ep, "val_ndcg": val_ndcg},
                    P / "lightgcn_best.pt",
                )
            else:
                patience_counter += 1
                if (
                    patience_counter >= args.early_stop_patience // 5
                ):  # Check every 5 epochs
                    print(
                        f" | Patience: {patience_counter * 5}/{args.early_stop_patience}",
                        end="",
                    )

            # Early stopping check
            if patience_counter * 5 >= args.early_stop_patience:
                print(
                    f"\n\nEarly stopping triggered after {ep} epochs (no improvement for {args.early_stop_patience} epochs)"
                )
                break
            print()
        else:
            print()

    print("\n" + "=" * 70)
    print(
        f"Training Complete! Best Val NDCG@10: {best_val_ndcg:.4f} at epoch {best_epoch}"
    )
    print("=" * 70)

    # Final eval on test - use best checkpoint
    model.eval()
    best_path = P / "lightgcn_best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        emb_all = ck["emb"]
        print(
            f"\nLoaded best checkpoint: epoch {ck.get('epoch', -1)}, "
            f"val_ndcg@10={ck.get('val_ndcg', 0.0):.4f}"
        )
    else:
        print("\nWarning: Best checkpoint not found, using final epoch embeddings")
        with torch.no_grad():
            emb_all = model(edge_index, apply_dropout=False).detach().cpu()

    print("\nComputing final metrics on validation and test sets...")
    val_metrics = eval_split(
        emb_all, U, M, spl, pos, "val", K=(10, 20), use_cosine=args.use_cosine
    )
    test_metrics = eval_split(
        emb_all, U, M, spl, pos, "test", K=(10, 20), use_cosine=args.use_cosine
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print("VALIDATION METRICS:")
    for k, v in val_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("\nTEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print("=" * 70)

    # Save artifacts
    print("\nSaving results...")
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

    print(f"✓ Saved metrics:     {P / 'runs' / 'lightgcn_metrics.json'}")
    print(f"✓ Saved embeddings:  {P / 'lightgcn_emb.pt'}")
    print(f"✓ Saved checkpoint:  {P / 'lightgcn_best.pt'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
