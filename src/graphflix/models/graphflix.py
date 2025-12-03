"""
GraphFlix: Graph Transformer for Movie Recommendations with Metadata Bias.

Implements the full 8-step forward pass from Section 16:
1. Sample subgraph & get features
2. Lookup profiles and metadata
3. Normalize embeddings
4. Compute metadata scores
5. Bound and scale the bias
6. Inject into attention logits (user-row-only)
7. Run Graphormer and score
8. Optimize with BPR
"""

from pathlib import Path

import torch
import torch.nn as nn

from .graphormer import GraphormerEncoder


class GraphFlix(nn.Module):
    """
    GraphFlix recommendation model with user-conditioned metadata bias.

    Key features:
    - Heterogeneous graph transformer (Graphormer)
    - Half-life user profiles p(u)
    - Movie metadata embeddings φ(j)
    - Metadata bias: b_meta(u,j) = β * tanh(LN(p(u))^T W LN(φ(j)))
    - User-row-only attention injection
    """

    def __init__(
        self,
        dim=64,
        heads=4,
        num_layers=3,
        num_node_types=5,
        beta_init=1.0,
        d_phi=192,
        max_path_distance=5,
        dropout=0.1,
        user_row_only=True,
        precomputed_data_path=None,
    ):
        """
        Args:
            dim: Hidden dimension for node embeddings
            heads: Number of attention heads
            num_layers: Number of Graphormer layers
            num_node_types: Number of node types (user, movie, actor, director, genre)
            beta_init: Initial value for metadata bias scale
            d_phi: Dimension of metadata embeddings (should match precomputed phi)
            max_path_distance: Maximum path distance for structural encoding
            dropout: Dropout rate
            user_row_only: If True, inject bias only in user's attention row
            precomputed_data_path: Path to load P and Φ matrices
        """
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.d_phi = d_phi
        self.user_row_only = user_row_only

        # Graphormer encoder
        self.graphormer = GraphormerEncoder(
            dim=dim,
            heads=heads,
            num_layers=num_layers,
            num_node_types=num_node_types,
            max_path_distance=max_path_distance,
            dropout=dropout,
        )

        # Metadata bias components
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.W = nn.Linear(d_phi, d_phi, bias=False)
        self.ln_profile = nn.LayerNorm(d_phi)
        self.ln_metadata = nn.LayerNorm(d_phi)

        # Initial node features (learnable embeddings for each node type)
        self.node_type_init = nn.Embedding(num_node_types, dim)

        # Load precomputed user profiles P and movie metadata Φ
        self.P = None  # [num_users, d_phi]
        self.Phi = None  # [num_movies, d_phi]
        self.movie_id_to_idx = None

        if precomputed_data_path is not None:
            self.load_precomputed_data(precomputed_data_path)

    def load_precomputed_data(self, data_path):
        """Load precomputed user profiles and movie metadata."""
        data_path = Path(data_path)

        # Load user profiles
        profile_path = data_path / "user_profiles.pt"
        if profile_path.exists():
            profile_data = torch.load(
                profile_path, map_location="cpu", weights_only=False
            )
            self.P = profile_data["user_profiles"]
            print(f"Loaded user profiles: {self.P.shape}")

        # Load movie metadata
        phi_path = data_path / "phi_matrix.pt"
        if phi_path.exists():
            phi_data = torch.load(phi_path, map_location="cpu", weights_only=False)
            self.Phi = phi_data["phi_matrix"]
            self.movie_id_to_idx = phi_data.get("movie_id_to_idx", {})
            print(f"Loaded movie metadata: {self.Phi.shape}")

    def compute_metadata_bias(self, user_ids, movie_ids):
        """
        Steps 2-5: Compute metadata bias for user-movie pairs.

        Args:
            user_ids: [batch_size] - user IDs
            movie_ids: [batch_size, num_movies] - movie IDs per user

        Returns:
            b_meta: [batch_size, num_movies] - metadata bias values
        """
        if self.P is None or self.Phi is None:
            # If precomputed data not loaded, return zeros
            return torch.zeros(
                user_ids.shape[0], movie_ids.shape[1], device=user_ids.device
            )

        device = user_ids.device
        batch_size = user_ids.shape[0]
        num_movies = movie_ids.shape[1]

        # Step 2: Lookup profiles and metadata
        # Get user profiles p(u) - indices must be on CPU for indexing
        P_batch = self.P[user_ids.cpu()].to(device)  # [batch_size, d_phi]

        # Get movie metadata φ(j) - indices must be on CPU for indexing
        Phi_batch = self.Phi[movie_ids.cpu()].to(
            device
        )  # [batch_size, num_movies, d_phi]

        # Step 3: Normalize embeddings
        p_u_norm = self.ln_profile(P_batch)  # [batch_size, d_phi]
        phi_j_norm = self.ln_metadata(Phi_batch)  # [batch_size, num_movies, d_phi]

        # Step 4: Compute metadata scores
        # s(u,j) = p̂(u)^T W φ̂(j)
        p_u_transformed = p_u_norm @ self.W.weight.t()  # [batch_size, d_phi]
        scores = torch.einsum(
            "bd,bmd->bm", p_u_transformed, phi_j_norm
        )  # [batch_size, num_movies]

        # Step 5: Bound and scale the bias
        b_meta = self.beta * torch.tanh(scores)

        return b_meta

    def inject_metadata_bias(self, attn_bias, batch_info, b_meta):
        """
        Step 6: Inject metadata bias into attention tensor.

        User-row-only injection: only modify B[:, i_u, j_m]

        Args:
            attn_bias: [batch_size, max_nodes, max_nodes] - existing attention bias
            batch_info: dict with 'user_indices' and 'movie_indices'
            b_meta: [batch_size, num_movies] - metadata bias

        Returns:
            attn_bias: Modified attention bias tensor
        """
        user_indices = batch_info["user_indices"]  # [batch_size]
        movie_indices = batch_info["movie_indices"]  # [batch_size, num_movies]

        batch_size = attn_bias.shape[0]

        if self.user_row_only:
            # Only modify the user's attention row
            for b in range(batch_size):
                i_u = user_indices[b]
                j_m = movie_indices[b]
                # attn_bias[b, i_u, j_m] += b_meta[b]
                # Handle variable number of movies per batch element
                num_movies = (j_m >= 0).sum()  # Count valid movie indices
                if num_movies > 0:
                    valid_indices = j_m[:num_movies]
                    attn_bias[b, i_u, valid_indices] += b_meta[b, :num_movies]
        else:
            # Broadcast to all nodes' attention to movies (ablation)
            for b in range(batch_size):
                j_m = movie_indices[b]
                num_movies = (j_m >= 0).sum()
                if num_movies > 0:
                    valid_indices = j_m[:num_movies]
                    attn_bias[b, :, valid_indices] += b_meta[b, :num_movies].unsqueeze(
                        0
                    )

        return attn_bias

    def forward(
        self,
        x,
        edge_index,
        node_types,
        batch,
        user_ids,
        movie_ids_pos,
        movie_ids_neg,
        batch_info,
    ):
        """
        Full 8-step forward pass for BPR training.

        Args:
            x: [total_nodes, dim] - initial node features
            edge_index: [2, num_edges] - graph connectivity
            node_types: [total_nodes] - node type IDs
            batch: [total_nodes] - batch assignment
            user_ids: [batch_size] - user IDs for metadata lookup
            movie_ids_pos: [batch_size] - positive movie IDs
            movie_ids_neg: [batch_size] - negative movie IDs
            batch_info: dict with 'user_indices', 'movie_pos_indices', 'movie_neg_indices'

        Returns:
            scores_pos: [batch_size] - scores for positive items
            scores_neg: [batch_size] - scores for negative items
            aux: dict with auxiliary outputs (attention weights, etc.)
        """
        device = x.device
        batch_size = user_ids.shape[0]

        # Step 1: Already done - subgraph sampled and features provided

        # Steps 2-5: Compute metadata bias
        # Combine positive and negative movies
        movie_ids_combined = torch.stack(
            [movie_ids_pos, movie_ids_neg], dim=1
        )  # [batch_size, 2]
        b_meta = self.compute_metadata_bias(
            user_ids, movie_ids_combined
        )  # [batch_size, 2]

        # Prepare batch_info for injection
        movie_indices_combined = torch.stack(
            [batch_info["movie_pos_indices"], batch_info["movie_neg_indices"]], dim=1
        )  # [batch_size, 2]

        injection_info = {
            "user_indices": batch_info["user_indices"],
            "movie_indices": movie_indices_combined,
        }

        # Step 6: Create attention bias and inject metadata bias
        # Note: Graphormer will create structural and type biases internally
        # We provide external_bias which will be added to the combined bias
        # Calculate max_nodes: maximum number of nodes in any graph in the batch
        num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        max_nodes = num_nodes_per_graph.max().item()
        external_bias = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
        external_bias = self.inject_metadata_bias(external_bias, injection_info, b_meta)

        # Step 7: Run Graphormer encoder
        x_out, mask, attn_weights = self.graphormer(
            x=x,
            edge_index=edge_index,
            node_types=node_types,
            batch=batch,
            external_bias=external_bias,
        )
        # x_out: [batch_size, max_nodes, dim]

        # Extract user and movie embeddings
        user_embeddings = self.graphormer.get_node_embeddings(
            x_out, mask, batch_info["user_indices"]
        )  # [batch_size, dim]

        movie_pos_embeddings = self.graphormer.get_node_embeddings(
            x_out, mask, batch_info["movie_pos_indices"]
        )  # [batch_size, dim]

        movie_neg_embeddings = self.graphormer.get_node_embeddings(
            x_out, mask, batch_info["movie_neg_indices"]
        )  # [batch_size, dim]

        # Compute scores: s(u,i) = z_u^T z_i
        scores_pos = (user_embeddings * movie_pos_embeddings).sum(
            dim=-1
        )  # [batch_size]
        scores_neg = (user_embeddings * movie_neg_embeddings).sum(
            dim=-1
        )  # [batch_size]

        # Step 8: BPR loss will be computed by the training loop

        # Return auxiliary information for debugging/analysis
        aux = {
            "attn_weights": attn_weights,
            "user_embeddings": user_embeddings,
            "movie_pos_embeddings": movie_pos_embeddings,
            "movie_neg_embeddings": movie_neg_embeddings,
            "b_meta": b_meta,
            "beta": self.beta.item(),
        }

        return scores_pos, scores_neg, aux

    def score_batch(
        self, x, edge_index, node_types, batch, user_ids, movie_ids, batch_info
    ):
        """
        Score multiple user-movie pairs (for evaluation).

        Args:
            x: [total_nodes, dim]
            edge_index: [2, num_edges]
            node_types: [total_nodes]
            batch: [total_nodes]
            user_ids: [batch_size]
            movie_ids: [batch_size, num_candidates]
            batch_info: dict with node indices

        Returns:
            scores: [batch_size, num_candidates]
        """
        device = x.device
        batch_size = user_ids.shape[0]
        num_candidates = movie_ids.shape[1]

        # Compute metadata bias
        b_meta = self.compute_metadata_bias(
            user_ids, movie_ids
        )  # [batch_size, num_candidates]

        # Create external bias (for simplicity, only add to user-movie pairs)
        external_bias = torch.zeros(batch_size, x.shape[0], x.shape[0], device=device)

        # Run Graphormer
        x_out, mask, _ = self.graphormer(
            x=x,
            edge_index=edge_index,
            node_types=node_types,
            batch=batch,
            external_bias=external_bias,
        )

        # Extract embeddings
        user_embeddings = self.graphormer.get_node_embeddings(
            x_out, mask, batch_info["user_indices"]
        )  # [batch_size, dim]

        # Get movie embeddings for all candidates
        movie_embeddings = []
        for i in range(num_candidates):
            movie_indices = batch_info["movie_indices"][:, i]
            movie_emb = self.graphormer.get_node_embeddings(x_out, mask, movie_indices)
            movie_embeddings.append(movie_emb)

        movie_embeddings = torch.stack(
            movie_embeddings, dim=1
        )  # [batch_size, num_candidates, dim]

        # Compute scores
        scores = torch.einsum("bd,bcd->bc", user_embeddings, movie_embeddings)

        # Add metadata bias
        scores = scores + b_meta

        return scores
