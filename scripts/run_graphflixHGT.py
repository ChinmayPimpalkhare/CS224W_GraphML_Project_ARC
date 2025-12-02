# scripts/run_graphflixHGT.py
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

# Disable fused segment_matmul kernel (forces naive matmul path)
if hasattr(torch_geometric, "backend") and hasattr(
    torch_geometric.backend, "use_segment_matmul"
):
    torch_geometric.backend.use_segment_matmul = False


class GraphFlixModel(nn.Module):
    """
    Heterogeneous graph Transformer (HGT-style) encoder with
    movie-metadata features and learnable metadata bias parameters.

    The actual metadata bias is applied in the BPR scoring code, not
    inside HGTConv, to keep the implementation simple and stable.
    """

    def __init__(
        self,
        metadata,
        num_nodes_dict,
        phi_movies: torch.Tensor,
        d_phi: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.metadata = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_phi = d_phi

        # --- Node-type specific embeddings (users, movies, actors, directors, genres)
        self.embeddings = nn.ModuleDict()
        for ntype, n_nodes in num_nodes_dict.items():
            emb = nn.Embedding(n_nodes, hidden_dim)
            nn.init.normal_(emb.weight, std=0.01)
            self.embeddings[ntype] = emb

        # --- Movie metadata projection: phi(j) -> hidden_dim
        self.meta_proj = nn.Linear(d_phi, hidden_dim)

        # Register phi(j) as a non-trainable buffer (fixed features)
        self.register_buffer("phi", phi_movies.float(), persistent=False)

        # --- Graph transformer layers (HGT-style)
        self.layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

        # --- Metadata bias parameters: LN + W + beta (Eq. 2 in design doc)
        self.profile_ln = nn.LayerNorm(d_phi)  # on p(u)
        self.meta_ln = nn.LayerNorm(d_phi)  # on phi(j)
        self.meta_W = nn.Linear(d_phi, d_phi, bias=False)
        # we parameterize beta via softplus(log_beta) to ensure positivity
        self.log_beta = nn.Parameter(torch.tensor(0.0))  # beta ≈ softplus(0) ≈ 0.693

    @property
    def beta(self) -> torch.Tensor:
        # ensure beta > 0
        return F.softplus(self.log_beta)

    def forward(self, data: HeteroData) -> dict:
        """
        Args:
            data: HeteroData on the same device as the model.

        Returns:
            x_dict: dict[node_type -> node_embeddings]
        """
        x_dict = {}
        device = next(self.parameters()).device

        # Initial node features per type
        for ntype, emb in self.embeddings.items():
            n_nodes = data[ntype].num_nodes
            idx = torch.arange(n_nodes, device=device)
            x = emb(idx)
            if ntype == "movie":
                # add projected metadata embedding
                if self.phi is None:
                    raise RuntimeError(
                        "Movie metadata phi is not set in GraphFlixModel."
                    )
                phi = self.phi.to(device)  # [num_movies, d_phi]
                x = x + self.meta_proj(phi)
            x_dict[ntype] = x

        # HGT layers
        for conv in self.layers:
            x_dict = conv(x_dict, data.edge_index_dict)

        return x_dict


def build_train_pos_lists(num_users: int, um_edge_index: torch.Tensor):
    """
    Build list of training positives per user (for masking at eval).
    """
    pos = [[] for _ in range(num_users)]
    um = um_edge_index.cpu()
    for u, i in zip(um[0].tolist(), um[1].tolist()):
        pos[u].append(i)
    return [set(items) for items in pos]


def sample_batch(
    um_edge_index: torch.Tensor,
    num_items: int,
    batch_size: int,
    num_neg: int,
    g: torch.Generator,
    device: str,
):
    """
    Sample a batch of (u, i_pos, j_neg) triples from train edges.
    """
    E = um_edge_index.size(1)
    idx = torch.randint(0, E, (batch_size,), generator=g, device=device)
    u = um_edge_index[0, idx]
    i_pos = um_edge_index[1, idx]
    # K negatives per positive (uniform over items)
    j_neg = torch.randint(
        0, num_items, (batch_size, num_neg), generator=g, device=device
    )
    return u, i_pos, j_neg


@torch.no_grad()
def eval_split_graphflix(
    model: GraphFlixModel,
    data: HeteroData,
    profiles_phase: torch.Tensor,
    phi_movies: torch.Tensor,
    train_pos,
    split_csv: Path,
    split_name: str = "val",
    K=(10, 20),
    use_cosine: bool = False,
):
    """
    Full-catalog ranking eval (Recall/NDCG@K) with metadata bias added at scoring.

    This is similar to run_lightgcn.py but with:
      scores = user_emb @ item_emb^T + b_meta(u, i)
    where b_meta is computed from profiles_phase and phi_movies using
    the same LN/W/beta as in training.
    """
    # device = next(model.parameters()).device

    #  Encode the full graph once
    model.eval()
    x_dict = model(data)
    user_emb = x_dict["user"].detach().cpu()  # [U, d]
    item_emb = x_dict["movie"].detach().cpu()  # [M, d]

    if use_cosine:
        user_emb = F.normalize(user_emb, dim=1)
        item_emb = F.normalize(item_emb, dim=1)

    #  Base scores from graph encoder
    scores = user_emb @ item_emb.T  # [U, M]

    #  Metadata bias matrix b_meta(u, i) on CPU using functional LN
    U, M = scores.shape
    d_phi = phi_movies.size(1)
    assert profiles_phase.shape == (U, d_phi), "profiles_phase shape mismatch"

    # Move LN/W params to CPU (via state dict / tensors)
    pln_weight = model.profile_ln.weight.detach().cpu()
    pln_bias = model.profile_ln.bias.detach().cpu()
    pln_eps = model.profile_ln.eps

    mln_weight = model.meta_ln.weight.detach().cpu()
    mln_bias = model.meta_ln.bias.detach().cpu()
    mln_eps = model.meta_ln.eps

    W_weight = model.meta_W.weight.detach().cpu()  # [d_phi, d_phi]
    beta = model.beta.detach().cpu()

    # LayerNorm on profiles and phi(j)
    P = F.layer_norm(
        profiles_phase.cpu(),
        normalized_shape=(d_phi,),
        weight=pln_weight,
        bias=pln_bias,
        eps=pln_eps,
    )  # [U, d_phi]
    phi_hat = F.layer_norm(
        phi_movies.cpu(),
        normalized_shape=(d_phi,),
        weight=mln_weight,
        bias=mln_bias,
        eps=mln_eps,
    )  # [M, d_phi]

    # Apply W to phi_hat: [M, d_phi]
    phi_W = phi_hat @ W_weight.T  # [M, d_phi]

    # Pre-tanh scores: [U, M]
    meta_raw = P @ phi_W.T
    b_meta = beta * torch.tanh(meta_raw)  # [U, M]

    scores = scores + b_meta

    # Mask training positives
    for u, items in enumerate(train_pos):
        if items:
            scores[u, torch.tensor(list(items))] = -1e9

    # Build ground truth dict from split CSV
    s = pd.read_csv(split_csv)
    truth = {}
    for _, r in s.iterrows():
        if r["split"] != split_name:
            continue
        u = int(r["user_idx"])
        i = int(r["movie_idx"])
        truth.setdefault(u, []).append(i)

    out = {}
    for k in K:
        topk_idx = torch.topk(scores, k, dim=1).indices  # [U, k]
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
    ap.add_argument(
        "--profiles_tau",
        type=int,
        default=150,
        help="Half-life tau (days) used in precompute_profiles",
    )
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--neg", type=int, default=20)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--use_cosine", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    torch.manual_seed(args.seed)
    g = torch.Generator(device=args.device).manual_seed(args.seed)

    # --- Load hetero graph
    data_path = root / "graph_pyg.pt"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run scripts/build_graph_pyg.py --use_train_only first."
        )

    data: HeteroData = torch.load(data_path, weights_only=False)
    print("Loaded graph_pyg.pt")
    print("Node types:", data.node_types)
    print("Edge types:", data.edge_types)

    num_users = data["user"].num_nodes
    num_movies = data["movie"].num_nodes

    # train user->movie edges
    um = data["user", "rates", "movie"].edge_index.to(args.device)
    num_edges = um.size(1)

    # train positives (CPU) for eval-time masking
    train_pos = build_train_pos_lists(num_users, um)

    # --- Load half-life profiles and phi(j)
    profiles_path = root / f"half_life_profiles_tau{int(args.profiles_tau)}.pt"
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"{profiles_path} not found. Run scripts/precompute_profiles.py --tau {args.profiles_tau} first."
        )

    prof = torch.load(profiles_path, weights_only=False)
    phi = prof["phi"]  # [M, d_phi]
    P_train = prof["profiles_train"]  # [U, d_phi]
    P_val = prof["profiles_val"]
    P_test = prof["profiles_test"]
    d_phi = int(prof["dims"]["d_phi"])

    assert phi.shape[0] == num_movies, "phi rows != num_movies"
    assert P_train.shape[0] == num_users, "profiles_train rows != num_users"

    # --- Instantiate GraphFlix model
    metadata = data.metadata()
    num_nodes_dict = {ntype: data[ntype].num_nodes for ntype in data.node_types}

    model = GraphFlixModel(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        phi_movies=phi,
        d_phi=d_phi,
        hidden_dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    # move HeteroData to device once
    data = data.to(args.device)

    steps_per_epoch = max(1, num_edges // args.batch)
    split_csv = root / "splits" / "ratings_split_reindexed.csv"

    best_val_ndcg = 0.0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            users, i_pos, j_neg = sample_batch(
                um_edge_index=um,
                num_items=num_movies,
                batch_size=args.batch,
                num_neg=args.neg,
                g=g,
                device=args.device,
            )

            # 1) Encode graph (full-batch) -> node embeddings
            x_dict = model(data)
            user_emb_all = x_dict["user"]  # [U, d]
            movie_emb_all = x_dict["movie"]  # [M, d]

            u_e = user_emb_all[users]  # [B, d]
            i_e = movie_emb_all[i_pos]  # [B, d]
            j_e = movie_emb_all[j_neg]  # [B, neg, d]

            # base encoder scores
            pos_scores_enc = (u_e * i_e).sum(-1, keepdim=True)  # [B, 1]
            neg_scores_enc = (u_e.unsqueeze(1) * j_e).sum(-1)  # [B, neg]

            # 2) Metadata bias for this batch (using train profiles)
            P_u = P_train[users.cpu()].to(args.device)  # [B, d_phi]
            phi_pos = phi[i_pos.cpu()].to(args.device)  # [B, d_phi]
            phi_neg = phi[j_neg.cpu()].to(args.device)  # [B, neg, d_phi]

            # LayerNorm
            P_hat = model.profile_ln(P_u)  # [B, d_phi]
            phi_pos_hat = model.meta_ln(phi_pos)  # [B, d_phi]

            B, Kneg, _ = phi_neg.shape
            phi_neg_hat = model.meta_ln(phi_neg.view(B * Kneg, -1)).view(
                B, Kneg, -1
            )  # [B, neg, d_phi]

            # Apply W
            phi_pos_W = model.meta_W(phi_pos_hat)  # [B, d_phi]
            phi_neg_W = model.meta_W(phi_neg_hat.view(B * Kneg, -1)).view(
                B, Kneg, -1
            )  # [B, neg, d_phi]

            # Raw scores and bounded bias
            meta_raw_pos = (P_hat * phi_pos_W).sum(-1, keepdim=True)  # [B, 1]
            meta_raw_neg = (P_hat.unsqueeze(1) * phi_neg_W).sum(-1)  # [B, neg]

            beta = model.beta  # scalar tensor
            b_meta_pos = beta * torch.tanh(meta_raw_pos)  # [B, 1]
            b_meta_neg = beta * torch.tanh(meta_raw_neg)  # [B, neg]

            pos_scores = pos_scores_enc + b_meta_pos
            neg_scores = neg_scores_enc + b_meta_neg

            bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

            # L2 regularization on embeddings for this batch
            l2 = (
                u_e.pow(2).sum(1).mean()
                + i_e.pow(2).sum(1).mean()
                + j_e.pow(2).sum(-1).mean()
            )
            loss = bpr + args.reg * l2

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        print(f"epoch {ep:03d} | loss {avg_loss:.4f}", end="")

        # --- Periodic validation
        if ep % 5 == 0 or ep == 1:
            # evaluate on VAL (encoder + metadata bias)
            val_metrics = eval_split_graphflix(
                model=model,
                data=data,
                profiles_phase=P_val,
                phi_movies=phi,
                train_pos=train_pos,
                split_csv=split_csv,
                split_name="val",
                K=(10,),
                use_cosine=args.use_cosine,
            )
            val_ndcg = val_metrics["ndcg@10"]
            print(f" | val_ndcg@10={val_ndcg:.4f}")

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                best_epoch = ep
                state = {
                    "model_state": model.state_dict(),
                    "epoch": ep,
                    "val_ndcg@10": val_ndcg,
                    "config": vars(args),
                }
                (root / "runs").mkdir(parents=True, exist_ok=True)
                torch.save(state, root / "runs" / "graphflix_best.pt")
        else:
            print()

    print(f"\nBest val NDCG@10: {best_val_ndcg:.4f} at epoch {best_epoch}")

    # --- Final evaluation using best checkpoint
    best_path = root / "runs" / "graphflix_best.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(
            f"Loaded best checkpoint: epoch {ck.get('epoch', -1)}, "
            f"val_ndcg@10={ck.get('val_ndcg@10', 0.0):.4f}"
        )
    else:
        print("Warning: best checkpoint not found; using final epoch parameters.")

    val_metrics = eval_split_graphflix(
        model=model,
        data=data,
        profiles_phase=P_val,
        phi_movies=phi,
        train_pos=train_pos,
        split_csv=split_csv,
        split_name="val",
        K=(10, 20),
        use_cosine=args.use_cosine,
    )
    test_metrics = eval_split_graphflix(
        model=model,
        data=data,
        profiles_phase=P_test,
        phi_movies=phi,
        train_pos=train_pos,
        split_csv=split_csv,
        split_name="test",
        K=(10, 20),
        use_cosine=args.use_cosine,
    )

    print("VAL :", val_metrics)
    print("TEST:", test_metrics)

    # Save summary metrics
    (root / "runs").mkdir(parents=True, exist_ok=True)
    out_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_val_ndcg,
    }
    with open(root / "runs" / "graphflix_HGT_v1_metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=2)

    print("Saved:", root / "runs" / "graphflix_HGT_v1_metrics.json")


if __name__ == "__main__":
    main()
