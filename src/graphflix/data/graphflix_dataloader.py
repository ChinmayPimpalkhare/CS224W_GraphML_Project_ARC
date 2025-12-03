"""
GraphFlix DataLoader with BPR triple sampling and k-hop subgraph extraction.

Implements Step 3.1: Subgraph Sampling for Training
- Sample BPR triples (u, i+, i-) from training data
- Extract k-hop ego-graphs around each triple
- Linearize nodes into token sequences
- Track node indices for attention bias injection
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import k_hop_subgraph


class GraphFlixBatch:
    """Container for a batch of GraphFlix training samples."""

    def __init__(self):
        self.x: Optional[torch.Tensor] = None  # [total_nodes, dim] - node features
        self.edge_index: Optional[torch.Tensor] = None  # [2, num_edges] - edges
        self.node_types: Optional[torch.Tensor] = None  # [total_nodes] - node type IDs
        self.batch: Optional[torch.Tensor] = None  # [total_nodes] - batch assignment

        # For metadata lookup
        self.user_ids: Optional[torch.Tensor] = None  # [batch_size] - original user IDs
        self.movie_ids_pos: Optional[torch.Tensor] = (
            None  # [batch_size] - positive movie IDs
        )
        self.movie_ids_neg: Optional[torch.Tensor] = (
            None  # [batch_size] - negative movie IDs
        )

        # For attention bias injection (indices within each graph)
        self.batch_info: Dict[str, Optional[torch.Tensor]] = {
            "user_indices": None,  # [batch_size] - user node index in each graph
            "movie_pos_indices": None,  # [batch_size] - positive movie index
            "movie_neg_indices": None,  # [batch_size] - negative movie index
        }

    def to(self, device):
        """Move all tensors to device."""
        if self.x is not None:
            self.x = self.x.to(device)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.node_types is not None:
            self.node_types = self.node_types.to(device)
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.user_ids is not None:
            self.user_ids = self.user_ids.to(device)
        if self.movie_ids_pos is not None:
            self.movie_ids_pos = self.movie_ids_pos.to(device)
        if self.movie_ids_neg is not None:
            self.movie_ids_neg = self.movie_ids_neg.to(device)
        if self.batch_info["user_indices"] is not None:
            self.batch_info["user_indices"] = self.batch_info["user_indices"].to(device)
        if self.batch_info["movie_pos_indices"] is not None:
            self.batch_info["movie_pos_indices"] = self.batch_info[
                "movie_pos_indices"
            ].to(device)
        if self.batch_info["movie_neg_indices"] is not None:
            self.batch_info["movie_neg_indices"] = self.batch_info[
                "movie_neg_indices"
            ].to(device)
        return self


class GraphFlixDataset(Dataset):
    """
    Dataset for GraphFlix training with BPR sampling.

    Each sample is a (user, positive_item, negative_item) triple.
    """

    def __init__(
        self, ratings_df, graph_data, split="train", neg_sampling="uniform", seed=42
    ):
        """
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp, split]
            graph_data: PyG Data object with the full heterogeneous graph
            split: 'train', 'val', or 'test'
            neg_sampling: 'uniform' or 'popularity'
            seed: Random seed
        """
        self.ratings_df = ratings_df
        self.graph_data = graph_data
        self.split = split
        self.neg_sampling = neg_sampling
        self.rng = np.random.RandomState(seed)

        # Filter ratings by split
        self.split_ratings = ratings_df[ratings_df["split"] == split].copy()

        # Build user-item interaction dict (for negative sampling)
        # CRITICAL FIX: Use ALL ratings (not just current split) to avoid data leakage
        # Negative samples must exclude ALL items the user has ever interacted with
        self.user_pos_items = defaultdict(set)

        # DEBUG: Print sizes to verify we're using full dataset
        print(
            f"  DEBUG: Building user_pos_items from {len(ratings_df)} total ratings (not just {len(self.split_ratings)} {split} ratings)"
        )

        for _, row in ratings_df.iterrows():  # Use full ratings_df, not split_ratings!
            self.user_pos_items[row["user_id"]].add(row["movie_id"])

        # DEBUG: Verify we have items from all splits
        total_interactions = sum(len(items) for items in self.user_pos_items.values())
        print(
            f"  DEBUG: user_pos_items contains {total_interactions} total interactions across all users"
        )

        # Get all unique movie IDs
        self.all_movies = set(ratings_df["movie_id"].unique())

        # For popularity-based negative sampling
        if neg_sampling == "popularity":
            movie_counts = ratings_df["movie_id"].value_counts()
            self.movie_popularity = movie_counts / movie_counts.sum()

        print(
            f"GraphFlixDataset ({split}): {len(self.split_ratings)} interactions, "
            f"{len(self.user_pos_items)} users"
        )

    def __len__(self):
        return len(self.split_ratings)

    def __getitem__(self, idx):
        """
        Return a BPR triple: (user_id, positive_movie_id, negative_movie_id)
        """
        row = self.split_ratings.iloc[idx]
        user_id = row["user_id"]
        pos_movie_id = row["movie_id"]

        # Sample negative movie
        neg_movie_id = self._sample_negative(user_id)

        return {
            "user_id": user_id,
            "pos_movie_id": pos_movie_id,
            "neg_movie_id": neg_movie_id,
        }

    def _sample_negative(self, user_id):
        """Sample a negative item for the user."""
        user_pos = self.user_pos_items[user_id]

        # Keep sampling until we get a negative item
        max_tries = 100
        for _ in range(max_tries):
            if self.neg_sampling == "uniform":
                neg_id = self.rng.choice(list(self.all_movies))
            else:  # popularity
                neg_id = self.rng.choice(
                    self.movie_popularity.index, p=self.movie_popularity.values
                )

            if neg_id not in user_pos:
                return neg_id

        # Fallback: random negative from all movies
        candidates = list(self.all_movies - user_pos)
        return self.rng.choice(candidates) if candidates else list(self.all_movies)[0]


class SubgraphSampler:
    """
    Sample k-hop subgraphs around BPR triples.

    For each (u, i+, i-) triple, sample a subgraph containing:
    - The user node
    - Positive and negative movie nodes
    - K-hop neighbors around these nodes
    """

    def __init__(
        self, graph_data, k_hops=2, max_neighbors_per_hop=None, node_type_map=None
    ):
        """
        Args:
            graph_data: PyG Data object with full graph
            k_hops: Number of hops for neighborhood sampling
            max_neighbors_per_hop: List of max neighbors per hop (None = no limit)
            node_type_map: Dict mapping node_id -> node_type
        """
        self.graph_data = graph_data
        self.k_hops = k_hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self.node_type_map = node_type_map or {}

        # Handle both HeteroData and Data objects
        from torch_geometric.data import HeteroData

        if isinstance(graph_data, HeteroData):
            # Heterogeneous graph - convert to homogeneous
            homo_data = graph_data.to_homogeneous()
            self.edge_index = homo_data.edge_index
            self.num_nodes = homo_data.num_nodes
        else:
            # Homogeneous graph
            self.edge_index = graph_data.edge_index
            self.num_nodes = graph_data.num_nodes

    def sample_subgraph(self, center_nodes):
        """
        Sample k-hop subgraph around center nodes.

        Args:
            center_nodes: List or tensor of node IDs

        Returns:
            subgraph: Dict with:
                - subset: Node IDs in subgraph
                - edge_index: Edges within subgraph
                - node_id_map: Mapping from original ID to subgraph index
        """
        if isinstance(center_nodes, list):
            center_nodes = torch.tensor(center_nodes, dtype=torch.long)

        # Use PyG's k_hop_subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_nodes,
            num_hops=self.k_hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes,
        )

        # Create node ID mapping (original_id -> subgraph_index)
        node_id_map = {
            int(orig_id): int(new_id) for new_id, orig_id in enumerate(subset)
        }

        return {
            "subset": subset,
            "edge_index": edge_index,
            "node_id_map": node_id_map,
            "center_mapping": mapping,  # Maps center_nodes to their new indices
        }

    def sample_batch(self, bpr_triples):
        """
        Sample subgraphs for a batch of BPR triples.

        Args:
            bpr_triples: List of dicts with 'user_id', 'pos_movie_id', 'neg_movie_id'

        Returns:
            List of subgraph dicts
        """
        subgraphs = []
        for triple in bpr_triples:
            center_nodes = [
                triple["user_id"],
                triple["pos_movie_id"],
                triple["neg_movie_id"],
            ]
            subgraph = self.sample_subgraph(center_nodes)
            subgraph["triple"] = triple
            subgraphs.append(subgraph)

        return subgraphs


def collate_graphflix_batch(
    batch_samples, graph_data, sampler, node_type_map, node_features=None
):
    """
    Collate function for GraphFlix DataLoader.

    Takes a batch of BPR triples and creates a batched PyG graph with:
    - Subgraphs around each triple
    - Proper node indexing for attention injection
    - Node features and types

    Args:
        batch_samples: List of dicts from GraphFlixDataset
        graph_data: Full PyG Data object
        sampler: SubgraphSampler instance
        node_type_map: Dict mapping node_id -> type_id
        node_features: Optional precomputed node features

    Returns:
        GraphFlixBatch object
    """
    batch_size = len(batch_samples)

    # Sample subgraphs for all triples
    subgraphs = sampler.sample_batch(batch_samples)

    # Prepare batch
    graphflix_batch = GraphFlixBatch()

    # Collect all nodes and edges across subgraphs
    all_x = []
    all_edge_index = []
    all_node_types = []
    all_batch_assignment = []

    user_ids = []
    movie_ids_pos = []
    movie_ids_neg = []
    user_indices_global = []  # Global indices (with offset)
    movie_pos_indices_global = []
    movie_neg_indices_global = []
    user_indices_local = []  # Local indices (within each graph)
    movie_pos_indices_local = []
    movie_neg_indices_local = []

    node_offset = 0

    for batch_idx, subgraph in enumerate(subgraphs):
        subset = subgraph["subset"]
        edge_index = subgraph["edge_index"]
        node_id_map = subgraph["node_id_map"]
        triple = subgraph["triple"]

        num_nodes_in_graph = len(subset)

        # Get node features for this subgraph
        if node_features is not None:
            x_subgraph = node_features[subset]
        else:
            # Initialize random features (will be learned)
            x_subgraph = torch.randn(num_nodes_in_graph, 64)

        # Get node types
        node_types_subgraph = torch.tensor(
            [node_type_map.get(int(node_id), 0) for node_id in subset], dtype=torch.long
        )

        # Adjust edge indices by offset
        edge_index_offset = edge_index + node_offset

        # Find indices of user and movie nodes in this subgraph
        user_id = triple["user_id"]
        pos_id = triple["pos_movie_id"]
        neg_id = triple["neg_movie_id"]

        user_idx_in_subgraph = node_id_map[user_id]
        pos_idx_in_subgraph = node_id_map[pos_id]
        neg_idx_in_subgraph = node_id_map[neg_id]

        # Store original IDs
        user_ids.append(user_id)
        movie_ids_pos.append(pos_id)
        movie_ids_neg.append(neg_id)

        # Store global indices (for extracting from concatenated tensors)
        user_indices_global.append(user_idx_in_subgraph + node_offset)
        movie_pos_indices_global.append(pos_idx_in_subgraph + node_offset)
        movie_neg_indices_global.append(neg_idx_in_subgraph + node_offset)

        # Store local indices (for attention bias injection within each graph)
        user_indices_local.append(user_idx_in_subgraph)
        movie_pos_indices_local.append(pos_idx_in_subgraph)
        movie_neg_indices_local.append(neg_idx_in_subgraph)

        # Batch assignment
        batch_assignment = torch.full(
            (num_nodes_in_graph,), batch_idx, dtype=torch.long
        )

        # Accumulate
        all_x.append(x_subgraph)
        all_edge_index.append(edge_index_offset)
        all_node_types.append(node_types_subgraph)
        all_batch_assignment.append(batch_assignment)

        node_offset += num_nodes_in_graph

    # Concatenate everything
    graphflix_batch.x = torch.cat(all_x, dim=0)
    graphflix_batch.edge_index = torch.cat(all_edge_index, dim=1)
    graphflix_batch.node_types = torch.cat(all_node_types, dim=0)
    graphflix_batch.batch = torch.cat(all_batch_assignment, dim=0)

    graphflix_batch.user_ids = torch.tensor(user_ids, dtype=torch.long)
    graphflix_batch.movie_ids_pos = torch.tensor(movie_ids_pos, dtype=torch.long)
    graphflix_batch.movie_ids_neg = torch.tensor(movie_ids_neg, dtype=torch.long)

    # Store local indices for attention bias injection
    graphflix_batch.batch_info["user_indices"] = torch.tensor(
        user_indices_local, dtype=torch.long
    )
    graphflix_batch.batch_info["movie_pos_indices"] = torch.tensor(
        movie_pos_indices_local, dtype=torch.long
    )
    graphflix_batch.batch_info["movie_neg_indices"] = torch.tensor(
        movie_neg_indices_local, dtype=torch.long
    )

    # Store global indices (if needed for other purposes)
    graphflix_batch.batch_info["user_indices_global"] = torch.tensor(
        user_indices_global, dtype=torch.long
    )
    graphflix_batch.batch_info["movie_pos_indices_global"] = torch.tensor(
        movie_pos_indices_global, dtype=torch.long
    )
    graphflix_batch.batch_info["movie_neg_indices_global"] = torch.tensor(
        movie_neg_indices_global, dtype=torch.long
    )

    return graphflix_batch


class GraphFlixDataLoader:
    """
    Main DataLoader for GraphFlix training.

    Combines:
    - BPR triple sampling (GraphFlixDataset)
    - K-hop subgraph sampling (SubgraphSampler)
    - Batch collation (collate_graphflix_batch)
    """

    def __init__(
        self,
        ratings_df,
        graph_data,
        node_type_map,
        split="train",
        batch_size=32,
        k_hops=2,
        neg_sampling="uniform",
        shuffle=True,
        num_workers=0,
        seed=42,
        node_features=None,
    ):
        """
        Args:
            ratings_df: DataFrame with ratings
            graph_data: PyG Data object
            node_type_map: Dict mapping node_id -> type_id
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            k_hops: Number of hops for subgraph sampling
            neg_sampling: Negative sampling strategy
            shuffle: Shuffle data
            num_workers: Number of worker processes
            seed: Random seed
            node_features: Optional precomputed node features
        """
        self.dataset = GraphFlixDataset(
            ratings_df=ratings_df,
            graph_data=graph_data,
            split=split,
            neg_sampling=neg_sampling,
            seed=seed,
        )

        self.sampler = SubgraphSampler(
            graph_data=graph_data, k_hops=k_hops, node_type_map=node_type_map
        )

        self.graph_data = graph_data
        self.node_type_map = node_type_map
        self.node_features = node_features

        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch_samples):
        """Wrapper for collate function."""
        return collate_graphflix_batch(
            batch_samples=batch_samples,
            graph_data=self.graph_data,
            sampler=self.sampler,
            node_type_map=self.node_type_map,
            node_features=self.node_features,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def load_graph_and_create_dataloader(
    data_dir, batch_size=32, k_hops=2, split="train", seed=42
):
    """
    Convenience function to load graph data and create DataLoader.

    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size
        k_hops: Number of hops for subgraph sampling
        split: 'train', 'val', or 'test'
        seed: Random seed

    Returns:
        dataloader: GraphFlixDataLoader instance
        graph_data: PyG Data object
        node_type_map: Dict mapping node_id -> type_id
    """
    data_dir = Path(data_dir)

    # Load ratings with split information
    ratings_path = data_dir / "splits" / "ratings_split_reindexed.csv"
    if not ratings_path.exists():
        ratings_path = data_dir / "splits" / "ratings_split.csv"

    print(f"Loading ratings from: {ratings_path}")
    ratings_df = pd.read_csv(ratings_path)

    # Standardize column names (reindexed files use user_idx/movie_idx)
    if "user_idx" in ratings_df.columns:
        ratings_df = ratings_df.rename(
            columns={"user_idx": "user_id", "movie_idx": "movie_id"}
        )

    # Load graph
    graph_path = data_dir / "graph_pyg.pt"
    print(f"Loading graph from: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)

    # Create node type map (assuming graph_data has node_type attribute)
    if hasattr(graph_data, "node_type"):
        node_type_map = {i: int(t) for i, t in enumerate(graph_data.node_type)}
    else:
        # Fallback: assign types based on node ID ranges
        # This is a simplified version - adjust based on your data structure
        num_users = ratings_df["user_id"].nunique()
        num_movies = ratings_df["movie_id"].nunique()

        node_type_map = {}
        for i in range(graph_data.num_nodes):
            if i < num_users:
                node_type_map[i] = 0  # user
            elif i < num_users + num_movies:
                node_type_map[i] = 1  # movie
            else:
                # Assign actor/director/genre based on ID
                node_type_map[i] = 2  # default to actor

    # Create DataLoader
    dataloader = GraphFlixDataLoader(
        ratings_df=ratings_df,
        graph_data=graph_data,
        node_type_map=node_type_map,
        split=split,
        batch_size=batch_size,
        k_hops=k_hops,
        seed=seed,
    )

    return dataloader, graph_data, node_type_map
