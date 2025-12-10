#!/usr/bin/env python3
"""
Compute half-life user profiles p(u) for all users.

For each user u, we aggregate their historical movie metadata with:
- Recency weighting: exp(-Δt / τ) where τ is half-life in days
- Rating weighting: max(0, r - r_bar_u) to emphasize above-average ratings

p(u) = Σ_j w(u,j) · φ(j)

where w(u,j) = recency_weight(u,j) * rating_weight(u,j)

Output: user_profiles.pt with shape [num_users × d_phi]
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def compute_recency_weights(timestamps, half_life_days=20):
    """
    Compute exponential decay weights based on time since most recent interaction.

    w_recency = exp(-Δt / τ)

    where Δt is time difference from most recent rating in days.
    """
    # Convert timestamps to datetime
    if len(timestamps) == 0:
        return np.array([])

    # Most recent timestamp is the reference point
    t_max = timestamps.max()

    # Compute time differences in seconds, convert to days
    delta_t_days = (t_max - timestamps) / (60 * 60 * 24)

    # Exponential decay
    tau = half_life_days
    weights = np.exp(-delta_t_days / tau)

    return weights


def compute_rating_weights(ratings, user_mean):
    """
    Compute rating-based weights emphasizing above-average ratings.

    w_rating = max(0, r - r_bar_u)

    If all ratings are equal to mean (uniform ratings), fall back to uniform weights.
    """
    weights = np.maximum(0, ratings - user_mean)

    # Handle edge case: user rated everything the same
    if weights.sum() == 0:
        weights = np.ones_like(ratings)

    return weights


def compute_user_profile(user_ratings, phi_matrix, half_life_days=20, normalize=True):
    """
    Compute profile for a single user.

    Args:
        user_ratings: DataFrame with columns [movie_id, rating, timestamp]
        phi_matrix: Tensor of movie metadata embeddings
        half_life_days: τ parameter for temporal decay
        normalize: Whether to L2-normalize the final profile

    Returns:
        p_u: User profile vector (d_phi dimension)
    """
    if len(user_ratings) == 0:
        # Edge case: user has no ratings (shouldn't happen but just in case)
        return torch.zeros(phi_matrix.shape[1])

    movie_ids = user_ratings["movie_id"].values
    ratings = user_ratings["rating"].values
    timestamps = user_ratings["timestamp"].values

    # Compute user mean rating
    user_mean = ratings.mean()

    # Compute weights
    recency_weights = compute_recency_weights(timestamps, half_life_days)
    rating_weights = compute_rating_weights(ratings, user_mean)

    # Combined weight
    combined_weights = recency_weights * rating_weights

    # Normalize weights to sum to 1
    if combined_weights.sum() > 0:
        combined_weights = combined_weights / combined_weights.sum()
    else:
        # Fallback: uniform weights
        combined_weights = np.ones_like(combined_weights) / len(combined_weights)

    # Aggregate movie metadata
    movie_embeddings = phi_matrix[movie_ids]
    weights_tensor = torch.from_numpy(combined_weights).float().unsqueeze(1)

    # Weighted sum: p(u) = Σ_j w(u,j) · φ(j)
    p_u = (weights_tensor * movie_embeddings).sum(dim=0)

    # Optional: L2 normalize
    if normalize:
        norm = p_u.norm()
        if norm > 0:
            p_u = p_u / norm

    return p_u


def compute_all_user_profiles(
    ratings_df,
    movies_df,
    phi_matrix,
    movie_id_to_idx,
    half_life_days=20,
    normalize=True,
):
    """
    Compute profiles for all users.

    Args:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie information
        phi_matrix: Tensor of movie metadata embeddings
        movie_id_to_idx: Dict mapping movie IDs to phi_matrix indices
        half_life_days: τ parameter for temporal decay
        normalize: Whether to L2-normalize profiles

    Returns:
        user_profiles: Tensor of shape [num_users × d_phi]
        user_stats: Dict with statistics about profile computation
    """
    # Get unique users
    unique_users = sorted(ratings_df["user_id"].unique())
    num_users = len(unique_users)
    d_phi = phi_matrix.shape[1]

    print(f"Computing profiles for {num_users} users...")

    # Initialize profile matrix
    user_profiles = torch.zeros(num_users, d_phi)

    # Statistics
    stats = {
        "num_ratings_per_user": [],
        "profile_norms": [],
        "users_with_uniform_ratings": 0,
        "avg_recency_weight": [],
        "avg_rating_weight": [],
    }

    for idx, user_id in enumerate(unique_users):
        # Get user's ratings
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]

        # Compute profile
        p_u = compute_user_profile(
            user_ratings=user_ratings,
            phi_matrix=phi_matrix,
            half_life_days=half_life_days,
            normalize=normalize,
        )

        user_profiles[idx] = p_u

        # Collect stats
        stats["num_ratings_per_user"].append(len(user_ratings))
        stats["profile_norms"].append(p_u.norm().item())

        # Check for uniform ratings
        ratings = user_ratings["rating"].values
        if len(set(ratings)) == 1:
            stats["users_with_uniform_ratings"] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{num_users} users")

    print(f"  Processed {num_users}/{num_users} users")

    return user_profiles, stats


def print_statistics(stats):
    """Print summary statistics about user profiles."""
    print("\nProfile Statistics:")
    print(f"  Total users: {len(stats['num_ratings_per_user'])}")
    print(f"  Avg ratings per user: {np.mean(stats['num_ratings_per_user']):.2f}")
    print(f"  Min ratings per user: {np.min(stats['num_ratings_per_user'])}")
    print(f"  Max ratings per user: {np.max(stats['num_ratings_per_user'])}")
    print(f"  Users with uniform ratings: {stats['users_with_uniform_ratings']}")
    print(f"  Avg profile norm: {np.mean(stats['profile_norms']):.4f}")
    print(f"  Std profile norm: {np.std(stats['profile_norms']):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute half-life user profiles p(u) for all users"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model/graphflix.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ml1m",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--half_life_days",
        type=int,
        default=20,
        help="Half-life τ for temporal decay (in days)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize user profiles",
    )
    parser.add_argument(
        "--use_train_only",
        action="store_true",
        default=False,
        help="Use only training ratings (from split file)",
    )
    parser.add_argument(
        "--use_reindexed",
        action="store_true",
        default=True,
        help="Use reindexed data (0-based IDs)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (not used, for compatibility)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load config if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            if "profile" in config:
                if "half_life_days" in config["profile"]:
                    args.half_life_days = config["profile"]["half_life_days"]

    print("STEP 1.2: Compute Half-Life User Profiles p(u)")
    print(f"Data directory: {data_dir}")
    print(f"Half-life τ: {args.half_life_days} days")
    print(f"Normalize profiles: {args.normalize}")
    print(f"Use train only: {args.use_train_only}")
    print()

    # Load φ matrix
    phi_path = data_dir / "phi_matrix.pt"
    if not phi_path.exists():
        raise FileNotFoundError(
            f"φ matrix not found at {phi_path}. "
            "Run precompute_metadata.py first (Step 1.1)."
        )

    print(f"Loading φ matrix from: {phi_path}")
    phi_data = torch.load(phi_path, weights_only=False)
    phi_matrix = phi_data["phi_matrix"]
    movie_id_to_idx = phi_data.get("movie_id_to_idx", {})
    print(f"  φ matrix shape: {phi_matrix.shape}")
    print(f"  Movie ID mapping: {len(movie_id_to_idx)} movies")
    print()

    # Load ratings
    if args.use_train_only:
        if args.use_reindexed:
            ratings_path = data_dir / "splits" / "ratings_split_reindexed.csv"
        else:
            ratings_path = data_dir / "splits" / "ratings_split.csv"
        print(f"Loading training ratings from: {ratings_path}")
        ratings_df = pd.read_csv(ratings_path, header=0)
        # Standardize column names (reindexed files use user_idx/movie_idx)
        if "user_idx" in ratings_df.columns:
            ratings_df = ratings_df.rename(
                columns={"user_idx": "user_id", "movie_idx": "movie_id"}
            )
        # Filter to training only
        ratings_df = ratings_df[ratings_df["split"] == "train"]
        print(f"  Training ratings: {len(ratings_df)}")
    else:
        if args.use_reindexed:
            ratings_path = data_dir / "ratings_reindexed.csv"
        else:
            ratings_path = data_dir / "ratings.csv"
        print(f"Loading all ratings from: {ratings_path}")
        ratings_df = pd.read_csv(ratings_path, header=0)
        print(f"  Total ratings: {len(ratings_df)}")
    print()

    # Standardize column names (reindexed files use user_idx/movie_idx)
    if "user_idx" in ratings_df.columns:
        ratings_df = ratings_df.rename(
            columns={"user_idx": "user_id", "movie_idx": "movie_id"}
        )

    # Load movies
    movies_path = data_dir / "movies.csv"
    print(f"Loading movies from: {movies_path}")
    movies_df = pd.read_csv(movies_path, header=0)
    print(f"  Total movies: {len(movies_df)}")
    print()

    # Verify movie IDs match
    max_movie_id = ratings_df["movie_id"].max()
    if max_movie_id >= len(phi_matrix):
        print(
            f"WARNING: Max movie_id ({max_movie_id}) >= φ matrix size ({len(phi_matrix)})"
        )
        print("This might cause indexing errors. Consider using reindexed data.")

    # Compute user profiles
    print("Computing user profiles...")
    user_profiles, stats = compute_all_user_profiles(
        ratings_df=ratings_df,
        movies_df=movies_df,
        phi_matrix=phi_matrix,
        movie_id_to_idx=movie_id_to_idx,
        half_life_days=args.half_life_days,
        normalize=args.normalize,
    )
    print()

    # Print statistics
    print_statistics(stats)
    print()

    # Verification
    print("Verification:")
    print(f"  User profiles shape: {user_profiles.shape}")
    num_users = ratings_df["user_id"].nunique()
    print(f"  Expected: [{num_users} × {phi_matrix.shape[1]}]")
    print(f"  Contains NaN: {torch.isnan(user_profiles).any().item()}")
    print(f"  Contains Inf: {torch.isinf(user_profiles).any().item()}")
    print(f"  Mean norm: {user_profiles.norm(dim=1).mean().item():.4f}")
    print(f"  Std norm: {user_profiles.norm(dim=1).std().item():.4f}")
    print()

    # Test: Check that recent ratings have higher weight
    print("Sanity Check - Temporal Weighting:")
    sample_user = (
        ratings_df.groupby("user_id").size().idxmax()
    )  # User with most ratings
    sample_ratings = ratings_df[ratings_df["user_id"] == sample_user].sort_values(
        "timestamp"
    )
    if len(sample_ratings) >= 5:
        print(f"  Sample user {sample_user} has {len(sample_ratings)} ratings")
        timestamps = sample_ratings["timestamp"].values
        recency_weights = compute_recency_weights(timestamps, args.half_life_days)
        print(f"  Oldest rating weight: {recency_weights[0]:.4f}")
        print(f"  Newest rating weight: {recency_weights[-1]:.4f}")
        print(
            f"  Ratio (newest/oldest): {recency_weights[-1] / (recency_weights[0] + 1e-8):.2f}x"
        )
    print()

    # Save
    output_path = data_dir / "user_profiles.pt"
    print(f"Saving user profiles to: {output_path}")
    torch.save(
        {
            "user_profiles": user_profiles,
            "num_users": num_users,
            "d_phi": user_profiles.shape[1],
            "half_life_days": args.half_life_days,
            "normalize": args.normalize,
            "stats": stats,
        },
        output_path,
    )



if __name__ == "__main__":
    main()
