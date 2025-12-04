#!/usr/bin/env python3
"""
Proper evaluation with 1-vs-99 protocol (100 candidates total).

This is the CORRECT evaluation protocol for recommender systems.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graphflix.models.graphflix import GraphFlix
from src.graphflix.data.graphflix_dataloader import load_graph_and_create_dataloader


def sample_negatives_for_evaluation(ratings_df, split='test', num_negatives=99, seed=42):
    """
    Sample negatives for proper 1-vs-N evaluation.
    
    For each test interaction, sample N negative items that the user hasn't interacted with.
    
    Args:
        ratings_df: DataFrame with ratings
        split: Which split to evaluate ('test' or 'val')
        num_negatives: Number of negative samples per positive (default: 99 for 1-vs-100)
        seed: Random seed
        
    Returns:
        List of dicts with:
            - user_id: User ID
            - pos_item: Positive item ID
            - neg_items: List of negative item IDs
    """
    rng = np.random.RandomState(seed)
    
    # Get split interactions
    split_data = ratings_df[ratings_df['split'] == split]
    
    # Build user interaction history (ALL splits)
    user_items = {}
    for _, row in ratings_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        if user_id not in user_items:
            user_items[user_id] = set()
        user_items[user_id].add(movie_id)
    
    # All movies
    all_movies = list(ratings_df['movie_id'].unique())
    
    # Sample negatives for each test interaction
    eval_data = []
    for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc="Sampling negatives"):
        user_id = row['user_id']
        pos_item = row['movie_id']
        
        # Sample negatives (not in user's history)
        user_history = user_items[user_id]
        candidates = [m for m in all_movies if m not in user_history]
        
        if len(candidates) < num_negatives:
            print(f"Warning: User {user_id} has too few candidates ({len(candidates)})")
            neg_items = candidates + [candidates[0]] * (num_negatives - len(candidates))
        else:
            neg_items = rng.choice(candidates, size=num_negatives, replace=False).tolist()
        
        eval_data.append({
            'user_id': user_id,
            'pos_item': pos_item,
            'neg_items': neg_items
        })
    
    return eval_data


def evaluate_with_proper_protocol(
    model,
    eval_data,
    graph_data,
    device='cuda',
    k_list=[10, 20, 50]
):
    """
    Evaluate model with proper 1-vs-N protocol.
    
    For each test case:
    1. Score 1 positive + N negatives
    2. Rank positive among all candidates
    3. Compute metrics
    
    Args:
        model: GraphFlix model
        eval_data: List of dicts with user_id, pos_item, neg_items
        graph_data: PyG graph
        device: Device
        k_list: List of K values for metrics
        
    Returns:
        Dict of metrics
    """
    model.eval()
    
    all_ranks = []
    
    print(f"\nEvaluating {len(eval_data)} test cases...")
    print(f"Protocol: 1-vs-{len(eval_data[0]['neg_items'])} evaluation")
    print()
    
    with torch.no_grad():
        for item in tqdm(eval_data, desc="Evaluating"):
            user_id = item['user_id']
            pos_item = item['pos_item']
            neg_items = item['neg_items']
            
            # Combine positive and negatives
            all_items = [pos_item] + neg_items
            num_items = len(all_items)
            
            # Score all items for this user
            # Note: This is a simplified version - proper implementation would
            # extract subgraph and score items properly through the model
            
            # For now, we'll use a placeholder that shows the structure
            # In reality, you'd need to:
            # 1. Extract user's subgraph
            # 2. For each candidate item, create (user, item) pair
            # 3. Score through model
            # 4. Rank based on scores
            
            # Placeholder: random scores (replace with actual model scoring)
            scores = torch.rand(num_items, device=device)
            
            # Find rank of positive (index 0)
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == 0).nonzero(as_tuple=True)[0].item() + 1
            
            all_ranks.append(rank)
    
    all_ranks = np.array(all_ranks)
    
    # Compute metrics
    metrics = {}
    
    # MRR
    metrics['mrr'] = np.mean(1.0 / all_ranks)
    
    # Recall@K and NDCG@K
    for k in k_list:
        metrics[f'recall@{k}'] = np.mean(all_ranks <= k)
        
        # NDCG@K
        dcg = np.sum(np.where(all_ranks <= k, 1.0 / np.log2(all_ranks + 1), 0.0))
        idcg = 1.0  # Only one relevant item
        metrics[f'ndcg@{k}'] = (dcg / idcg) / len(all_ranks)
        
        metrics[f'hit_rate@{k}'] = np.mean(all_ranks <= k)
    
    return metrics


def print_metrics(metrics, prefix="Results"):
    """Print metrics in a nice format."""
    print()
    print(prefix)
    print("-" * 50)
    print(f"MRR: {metrics['mrr']:.4f}")
    print()
    print("Recall@K:")
    for k in [10, 20, 50]:
        if f'recall@{k}' in metrics:
            print(f"  recall@{k}: {metrics[f'recall@{k}']:.4f}")
    print()
    print("NDCG@K:")
    for k in [10, 20, 50]:
        if f'ndcg@{k}' in metrics:
            print(f"  ndcg@{k}: {metrics[f'ndcg@{k}']:.4f}")
    print()
    print("Other:")
    for k in [10, 20, 50]:
        if f'hit_rate@{k}' in metrics:
            print(f"  hit_rate@{k}: {metrics[f'hit_rate@{k}']:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Proper evaluation with 1-vs-99 protocol")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed/ml1m_10pct')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--num_negatives', type=int, default=99)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*80)
    print("GraphFlix Proper Evaluation (1-vs-N Protocol)")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Protocol: 1-vs-{args.num_negatives} evaluation")
    print()
    
    # Load data
    print("Loading data...")
    ratings_path = Path(args.data_dir) / "splits" / "ratings_split_reindexed.csv"
    ratings_df = pd.read_csv(ratings_path)
    
    # Standardize column names
    if 'user_idx' in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={'user_idx': 'user_id', 'movie_idx': 'movie_id'})
    
    # Load graph
    graph_path = Path(args.data_dir) / "graph_pyg.pt"
    graph_data = torch.load(graph_path, weights_only=False)
    
    # Sample negatives
    print(f"\nSampling {args.num_negatives} negatives per test case...")
    eval_data = sample_negatives_for_evaluation(
        ratings_df,
        split=args.split,
        num_negatives=args.num_negatives
    )
    
    print(f"Created {len(eval_data)} test cases")
    
    # Load model
    print("\nLoading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get hyperparameters
    if 'hyperparams' in checkpoint:
        hyperparams = checkpoint['hyperparams']
    else:
        hyperparams = {'dim': 64, 'heads': 4, 'num_layers': 3, 'beta_init': 1.0}
    
    model = GraphFlix(
        dim=hyperparams['dim'],
        heads=hyperparams['heads'],
        num_layers=hyperparams['num_layers'],
        beta_init=hyperparams.get('beta_init', 1.0),
        d_phi=192,
        precomputed_data_path=None
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Evaluate
    print("\n" + "="*80)
    print("⚠️  NOTE: This is a PLACEHOLDER implementation")
    print("   The actual scoring logic needs to be implemented")
    print("   Currently using random scores to show the structure")
    print("="*80)
    
    metrics = evaluate_with_proper_protocol(
        model,
        eval_data,
        graph_data,
        device=device
    )
    
    # Print results
    print_metrics(metrics, prefix=f"{args.split.upper()} Set Results (Proper Evaluation)")
    
    print()
    print("="*80)
    print("Expected results with proper evaluation:")
    print("  Recall@10: 0.15-0.25 (NOT 1.0!)")
    print("  NDCG@10: 0.08-0.12")
    print("  MRR: 0.10-0.18")
    print("="*80)
