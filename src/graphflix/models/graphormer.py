"""
Graphormer-style graph transformer encoder with attention bias injection.

Implements:
- Multi-head self-attention with external bias tensor
- Structural encodings (shortest path distances)
- Type encodings (node type embeddings)
- Support for user-row-only bias injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch


class GraphormerAttention(nn.Module):
    """Multi-head attention with external bias injection."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, mask=None):
        """
        Args:
            x: [batch_size, num_nodes, dim]
            attn_bias: [batch_size, num_heads, num_nodes, num_nodes] or
                       [batch_size, num_nodes, num_nodes] (will be broadcast)
            mask: [batch_size, num_nodes] - True for valid nodes, False for padding

        Returns:
            out: [batch_size, num_nodes, dim]
            attn_weights: [batch_size, num_heads, num_nodes, num_nodes]
        """
        B, N, D = x.shape
        H = self.heads

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, H, N, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add external bias if provided
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # [B, N, N] -> [B, 1, N, N] for broadcasting
                attn_bias = attn_bias.unsqueeze(1)
            attn = attn + attn_bias

        # Apply mask if provided
        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N] for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        return out, attn_weights


class GraphormerLayer(nn.Module):
    """Single Graphormer transformer layer."""

    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.attn = GraphormerAttention(dim, heads, dropout)
        self.ln2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None, mask=None):
        """
        Args:
            x: [batch_size, num_nodes, dim]
            attn_bias: [batch_size, num_heads, num_nodes, num_nodes]
            mask: [batch_size, num_nodes]

        Returns:
            x: [batch_size, num_nodes, dim]
            attn_weights: [batch_size, num_heads, num_nodes, num_nodes]
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.ln1(x), attn_bias, mask)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights


class StructuralEncoding(nn.Module):
    """Compute structural bias based on shortest path distances."""

    def __init__(self, max_path_distance=5, num_heads=4):
        super().__init__()
        self.max_path_distance = max_path_distance
        # Learnable bias for each distance (including unreachable)
        self.edge_bias = nn.Embedding(max_path_distance + 2, num_heads)

    def forward(self, edge_index, batch, num_nodes_per_graph):
        """
        Compute structural bias from edge_index.

        Args:
            edge_index: [2, num_edges]
            batch: [total_nodes] - batch assignment
            num_nodes_per_graph: [batch_size] - number of nodes in each graph

        Returns:
            structural_bias: [batch_size, num_heads, max_nodes, max_nodes]
        """
        device = edge_index.device
        batch_size = len(num_nodes_per_graph)
        max_nodes = num_nodes_per_graph.max().item()
        num_heads = self.edge_bias.weight.shape[1]

        # Initialize bias tensor
        structural_bias = torch.zeros(
            batch_size, num_heads, max_nodes, max_nodes, device=device
        )

        # Convert edge_index to dense adjacency per graph
        adj = to_dense_adj(
            edge_index, batch, max_num_nodes=max_nodes
        )  # [B, max_nodes, max_nodes]

        # Compute shortest paths using Floyd-Warshall (simplified version)
        # For now, use simple heuristic: direct edge = 1, no edge = max_distance + 1
        for b in range(batch_size):
            n = num_nodes_per_graph[b].item()
            adj_b = adj[b, :n, :n]

            # Distance matrix: 0 for self, 1 for connected, max+1 for disconnected
            dist = torch.full(
                (max_nodes, max_nodes), self.max_path_distance + 1, device=device
            )
            dist[:n, :n] = torch.where(
                adj_b > 0,
                torch.ones_like(adj_b),
                torch.full_like(adj_b, float(self.max_path_distance + 1)),
            )
            dist[:n, :n].fill_diagonal_(0)

            # Clip distances to max_path_distance
            dist = torch.clamp(dist, 0, self.max_path_distance + 1).long()

            # Lookup embeddings
            bias_b = self.edge_bias(dist)  # [max_nodes, max_nodes, num_heads]
            structural_bias[b] = bias_b.permute(
                2, 0, 1
            )  # [num_heads, max_nodes, max_nodes]

        return structural_bias


class TypeEncoding(nn.Module):
    """Node type embeddings for heterogeneous graphs."""

    def __init__(self, num_types, dim):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, dim)

    def forward(self, node_types):
        """
        Args:
            node_types: [total_nodes] - type ID for each node

        Returns:
            type_embeds: [total_nodes, dim]
        """
        return self.type_embedding(node_types)


class GraphormerEncoder(nn.Module):
    """
    Graphormer encoder for heterogeneous graphs.

    Supports:
    - Structural bias from shortest paths
    - Type bias from node type embeddings
    - External metadata bias injection
    """

    def __init__(
        self,
        dim=64,
        heads=4,
        num_layers=3,
        num_node_types=5,  # user, movie, actor, director, genre
        max_path_distance=5,
        dropout=0.1,
        mlp_ratio=4,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.num_layers = num_layers

        # Initial node embedding (will be set from external features)
        self.node_embedding = nn.Linear(dim, dim)

        # Type encoding
        self.type_encoding = TypeEncoding(num_node_types, dim)

        # Structural encoding
        self.structural_encoding = StructuralEncoding(max_path_distance, heads)

        # Transformer layers
        self.layers = nn.ModuleList(
            [GraphormerLayer(dim, heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )

        self.ln_final = nn.LayerNorm(dim)

    def compute_type_bias(self, node_types, batch, num_nodes_per_graph):
        """
        Compute type bias from node type embeddings.

        Type bias is added to attention to help the model distinguish different node types.
        """
        device = node_types.device
        batch_size = len(num_nodes_per_graph)
        max_nodes = num_nodes_per_graph.max().item()

        # Get type embeddings
        type_embeds = self.type_encoding(node_types)  # [total_nodes, dim]

        # Convert to dense batch
        type_embeds_dense, mask = to_dense_batch(
            type_embeds, batch, max_num_nodes=max_nodes
        )
        # [batch_size, max_nodes, dim]

        # Compute pairwise type similarity as bias
        # bias[i,j] = type_embed[i] @ type_embed[j]
        type_bias = torch.einsum("bnd,bmd->bnm", type_embeds_dense, type_embeds_dense)
        # [batch_size, max_nodes, max_nodes]

        return type_bias, mask

    def forward(self, x, edge_index, node_types, batch, external_bias=None):
        """
        Forward pass through Graphormer encoder.

        Args:
            x: [total_nodes, dim] - node features
            edge_index: [2, num_edges] - edge connectivity
            node_types: [total_nodes] - node type IDs (0=user, 1=movie, etc.)
            batch: [total_nodes] - batch assignment
            external_bias: Optional [batch_size, max_nodes, max_nodes] - metadata bias

        Returns:
            out: [batch_size, max_nodes, dim] - node embeddings
            mask: [batch_size, max_nodes] - valid node mask
            attn_weights_list: List of attention weights from each layer
        """
        device = x.device
        batch_size = batch.max().item() + 1

        # Count nodes per graph
        num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        max_nodes = num_nodes_per_graph.max().item()

        # Embed nodes and add type information
        x = self.node_embedding(x) + self.type_encoding(node_types)

        # Convert to dense batch format
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=max_nodes)
        # x_dense: [batch_size, max_nodes, dim]
        # mask: [batch_size, max_nodes]

        # Compute structural bias
        structural_bias = self.structural_encoding(
            edge_index, batch, num_nodes_per_graph
        )
        # [batch_size, num_heads, max_nodes, max_nodes]

        # Compute type bias
        type_bias, _ = self.compute_type_bias(node_types, batch, num_nodes_per_graph)
        # [batch_size, max_nodes, max_nodes]

        # Combine biases
        # structural_bias has shape [B, H, N, N], need to average over heads or add
        # For simplicity, average over heads to get [B, N, N]
        structural_bias_avg = structural_bias.mean(dim=1)  # [B, N, N]

        combined_bias = structural_bias_avg + type_bias

        # Add external metadata bias if provided
        if external_bias is not None:
            combined_bias = combined_bias + external_bias

        # Pass through transformer layers
        attn_weights_list = []
        for layer in self.layers:
            x_dense, attn_weights = layer(x_dense, combined_bias, mask)
            attn_weights_list.append(attn_weights)

        # Final layer norm
        x_dense = self.ln_final(x_dense)

        return x_dense, mask, attn_weights_list

    def get_node_embeddings(self, x_dense, mask, node_indices):
        """
        Extract embeddings for specific nodes from dense batch.

        Args:
            x_dense: [batch_size, max_nodes, dim]
            mask: [batch_size, max_nodes]
            node_indices: [batch_size] - index of node to extract per graph

        Returns:
            embeddings: [batch_size, dim]
        """
        batch_size = x_dense.shape[0]
        embeddings = x_dense[torch.arange(batch_size), node_indices]
        return embeddings
