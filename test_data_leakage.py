#!/usr/bin/env python3
"""
Test script to verify data leakage fix in negative sampling.

This script checks that negative samples don't include items
from the user's history (train + val + test).
"""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graphflix.data.graphflix_dataloader import GraphFlixDataset


def test_negative_sampling(data_dir="data/processed/ml1m", num_samples=100):
    """
    Test that negative samples don't include items from user history.
    
    Args:
        data_dir: Path to processed data
        num_samples: Number of samples to test
    """
    print("="*80)
    print("Testing Negative Sampling - Data Leakage Check")
    print("="*80)
    print()
    
    data_dir = Path(data_dir)
    
    # Load ratings
    ratings_path = data_dir / "splits" / "ratings_split_reindexed.csv"
    if not ratings_path.exists():
        ratings_path = data_dir / "splits" / "ratings_split.csv"
    
    print(f"Loading ratings from: {ratings_path}")
    ratings_df = pd.read_csv(ratings_path)
    
    # Standardize column names
    if 'user_idx' in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={'user_idx': 'user_id', 'movie_idx': 'movie_id'})
    
    # Load graph
    graph_path = data_dir / "graph_pyg.pt"
    print(f"Loading graph from: {graph_path}")
    graph_data = torch.load(graph_path, weights_only=False)
    
    print()
    print(f"Dataset: {ratings_df.shape[0]} ratings")
    print(f"  Train: {(ratings_df['split'] == 'train').sum()}")
    print(f"  Val: {(ratings_df['split'] == 'val').sum()}")
    print(f"  Test: {(ratings_df['split'] == 'test').sum()}")
    print()
    
    # Build complete user history (for verification)
    print("Building user history across all splits...")
    user_all_items = {}
    for _, row in ratings_df.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        if user_id not in user_all_items:
            user_all_items[user_id] = set()
        user_all_items[user_id].add(movie_id)
    
    print(f"  Total users: {len(user_all_items)}")
    print()
    
    # Test each split
    for split in ['train', 'val', 'test']:
        print("-"*80)
        print(f"Testing {split.upper()} split")
        print("-"*80)
        
        # Create dataset
        dataset = GraphFlixDataset(
            ratings_df=ratings_df,
            graph_data=graph_data,
            split=split,
            seed=42
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test samples
        num_to_test = min(num_samples, len(dataset))
        leakage_count = 0
        
        print(f"Testing {num_to_test} samples...")
        
        for i in range(num_to_test):
            sample = dataset[i]
            user_id = sample['user_id']
            pos_id = sample['pos_movie_id']
            neg_id = sample['neg_movie_id']
            
            # Get user's complete history
            user_history = user_all_items[user_id]
            
            # Check for leakage
            if neg_id in user_history:
                leakage_count += 1
                if leakage_count <= 3:  # Print first 3 examples
                    print(f"  ❌ LEAKAGE DETECTED (sample {i}):")
                    print(f"     User {user_id}, Pos: {pos_id}, Neg: {neg_id}")
                    print(f"     User history size: {len(user_history)}")
                    print(f"     Negative item {neg_id} is in user's history!")
        
        if leakage_count == 0:
            print(f"  ✅ PASS: No data leakage detected in {num_to_test} samples")
        else:
            print(f"  ❌ FAIL: Found {leakage_count}/{num_to_test} samples with data leakage ({leakage_count/num_to_test*100:.1f}%)")
        
        print()
    
    print("="*80)
    print("Test Complete")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test negative sampling for data leakage")
    parser.add_argument('--data_dir', type=str, default='data/processed/ml1m',
                       help='Path to processed data')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to test per split')
    
    args = parser.parse_args()
    
    test_negative_sampling(
        data_dir=args.data_dir,
        num_samples=args.num_samples
    )
