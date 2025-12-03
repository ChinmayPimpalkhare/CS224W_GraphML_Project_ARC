#!/usr/bin/env python3
"""
Create subsampled versions of MovieLens dataset for faster training.

Options:
- 10% sample: ~10x faster, good for testing/debugging
- 25% sample: ~4x faster, good quality model
- 50% sample: ~2x faster, near-full quality
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_subsample(data_dir, sample_ratio=0.25, seed=42):
    """
    Create a subsampled version of the dataset.

    Strategy:
    1. Sample users (to maintain user history)
    2. Keep all their interactions
    3. Maintain temporal splits

    Args:
        data_dir: Path to processed data directory
        sample_ratio: Fraction of users to keep (0.1, 0.25, 0.5)
        seed: Random seed
    """
    data_dir = Path(data_dir)
    np.random.seed(seed)

    print(f"=" * 80)
    print(f"Creating {int(sample_ratio*100)}% Subsample of Dataset")
    print(f"=" * 80)
    print()

    # Load original data
    print("Loading original data...")
    ratings_path = data_dir / "splits" / "ratings_split_reindexed.csv"
    ratings_df = pd.read_csv(ratings_path)

    # Standardize column names
    if "user_idx" in ratings_df.columns:
        ratings_df = ratings_df.rename(
            columns={"user_idx": "user_id", "movie_idx": "movie_id"}
        )

    print(f"  Original: {len(ratings_df)} interactions")
    print(f"  Users: {ratings_df['user_id'].nunique()}")
    print(f"  Movies: {ratings_df['movie_id'].nunique()}")
    print()

    # Sample users
    all_users = ratings_df["user_id"].unique()
    n_sample = int(len(all_users) * sample_ratio)
    sampled_users = np.random.choice(all_users, size=n_sample, replace=False)

    print(f"Sampling {len(sampled_users)} users ({sample_ratio*100:.0f}%)...")

    # Filter ratings to sampled users
    sampled_ratings = ratings_df[ratings_df["user_id"].isin(sampled_users)].copy()

    # Get movies that appear in sampled ratings
    sampled_movies = sampled_ratings["movie_id"].unique()

    print(f"  Sampled: {len(sampled_ratings)} interactions")
    print(f"  Users: {sampled_ratings['user_id'].nunique()}")
    print(f"  Movies: {len(sampled_movies)}")
    print()

    # Create output directory
    output_dir = data_dir.parent / f"ml1m_{int(sample_ratio*100)}pct"
    output_dir.mkdir(exist_ok=True)

    # Save subsampled ratings
    output_path = output_dir / "splits"
    output_path.mkdir(exist_ok=True)

    print(f"Saving to: {output_dir}")

    # Rename back to original column names if needed
    if "user_id" in sampled_ratings.columns:
        sampled_ratings = sampled_ratings.rename(
            columns={"user_id": "user_idx", "movie_id": "movie_idx"}
        )

    sampled_ratings.to_csv(output_path / "ratings_split_reindexed.csv", index=False)

    # Copy/filter other necessary files
    print("\nCopying metadata files...")

    # Copy movies (filter to sampled movies)
    if (data_dir / "movies.csv").exists():
        movies_df = pd.read_csv(data_dir / "movies.csv")
        sampled_movies_df = movies_df[movies_df["movie_id"].isin(sampled_movies)]
        sampled_movies_df.to_csv(output_dir / "movies.csv", index=False)
        print(f"  ✓ movies.csv: {len(sampled_movies_df)} movies")

    # Copy users (filter to sampled users)
    if (data_dir / "users.csv").exists():
        users_df = pd.read_csv(data_dir / "users.csv")
        sampled_users_df = users_df[users_df["user_id"].isin(sampled_users)]
        sampled_users_df.to_csv(output_dir / "users.csv", index=False)
        print(f"  ✓ users.csv: {len(sampled_users_df)} users")

    # Copy metadata files (genres, actors, directors)
    for file in ["genres.csv", "actors.csv", "directors.csv"]:
        src = data_dir / file
        if src.exists():
            dst = output_dir / file
            pd.read_csv(src).to_csv(dst, index=False)
            print(f"  ✓ {file}")

    # Copy edge files (filter to sampled movies)
    edge_files = [
        "movie_genre_edges_reindexed.csv",
        "movie_actor_edges_reindexed.csv",
        "movie_director_edges_reindexed.csv",
    ]

    for file in edge_files:
        src = data_dir / file
        if src.exists():
            edges_df = pd.read_csv(src)
            if "movie_idx" in edges_df.columns:
                sampled_edges = edges_df[edges_df["movie_idx"].isin(sampled_movies)]
            else:
                sampled_edges = edges_df[edges_df["movie_id"].isin(sampled_movies)]
            sampled_edges.to_csv(output_dir / file, index=False)
            print(f"  ✓ {file}: {len(sampled_edges)} edges")

    # Copy mappings directory (needed by build_graph_pyg.py)
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(exist_ok=True)
    if (data_dir / "mappings").exists():
        # Filter user mappings
        if (data_dir / "mappings/user_index.csv").exists():
            user_mapping = pd.read_csv(data_dir / "mappings/user_index.csv")
            if "user_id" in user_mapping.columns:
                sampled_user_mapping = user_mapping[
                    user_mapping["user_id"].isin(sampled_users)
                ]
            else:
                sampled_user_mapping = user_mapping[
                    user_mapping.iloc[:, 0].isin(sampled_users)
                ]
            sampled_user_mapping.to_csv(mappings_dir / "user_index.csv", index=False)
            print(f"  ✓ mappings/user_index.csv: {len(sampled_user_mapping)} users")

        # Filter movie mappings
        if (data_dir / "mappings/movie_index.csv").exists():
            movie_mapping = pd.read_csv(data_dir / "mappings/movie_index.csv")
            if "movie_id" in movie_mapping.columns:
                sampled_movie_mapping = movie_mapping[
                    movie_mapping["movie_id"].isin(sampled_movies)
                ]
            else:
                sampled_movie_mapping = movie_mapping[
                    movie_mapping.iloc[:, 0].isin(sampled_movies)
                ]
            sampled_movie_mapping.to_csv(mappings_dir / "movie_index.csv", index=False)
            print(f"  ✓ mappings/movie_index.csv: {len(sampled_movie_mapping)} movies")

    # Copy ratings_reindexed.csv (needed by build_graph_pyg.py)
    if (data_dir / "ratings_reindexed.csv").exists():
        # Use the already filtered sampled_ratings
        # Ensure it has user_idx/movie_idx columns (not user_id/movie_id)
        sampled_ratings_copy = sampled_ratings.copy()
        if (
            "user_id" in sampled_ratings_copy.columns
            and "user_idx" not in sampled_ratings_copy.columns
        ):
            sampled_ratings_copy = sampled_ratings_copy.rename(
                columns={"user_id": "user_idx", "movie_id": "movie_idx"}
            )

        # Keep only the columns that exist in the original ratings_reindexed.csv
        original_cols = pd.read_csv(
            data_dir / "ratings_reindexed.csv", nrows=0
        ).columns.tolist()
        available_cols = [
            col for col in original_cols if col in sampled_ratings_copy.columns
        ]
        sampled_ratings_copy[available_cols].to_csv(
            output_dir / "ratings_reindexed.csv", index=False
        )
        print(f"  ✓ ratings_reindexed.csv")

    # Create info file
    info = {
        "sample_ratio": sample_ratio,
        "original_users": int(len(all_users)),
        "sampled_users": int(len(sampled_users)),
        "original_interactions": int(len(ratings_df)),
        "sampled_interactions": int(len(sampled_ratings)),
        "original_movies": int(ratings_df["movie_id"].nunique()),
        "sampled_movies": int(len(sampled_movies)),
        "speedup_factor": 1.0 / sample_ratio,
        "seed": seed,
    }

    with open(output_dir / "subsample_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print()
    print("=" * 80)
    print("✓ Subsample Created Successfully!")
    print("=" * 80)
    print()
    print("Statistics:")
    print(f"  Sample ratio: {sample_ratio*100:.0f}%")
    print(f"  Users: {info['sampled_users']:,} / {info['original_users']:,}")
    print(
        f"  Interactions: {info['sampled_interactions']:,} / {info['original_interactions']:,}"
    )
    print(f"  Movies: {info['sampled_movies']:,} / {info['original_movies']:,}")
    print(f"  Expected speedup: ~{info['speedup_factor']:.1f}x")
    print()
    print("To use this dataset:")
    print(f"  python scripts/train_graphflix.py --data_dir {output_dir}")
    print()
    print("Note: You'll need to rebuild graph and precompute features for this subset:")
    print(f"  python scripts/build_graph_pyg.py --data_dir {output_dir}")
    print(f"  python scripts/precompute_metadata.py --data_dir {output_dir}")
    print(f"  python scripts/compute_user_profiles.py --data_dir {output_dir}")
    print()

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Create subsampled dataset for faster training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ml1m",
        help="Original data directory",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.25,
        choices=[0.1, 0.25, 0.5],
        help="Fraction of users to sample (0.1, 0.25, or 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    create_subsample(args.data_dir, args.sample_ratio, args.seed)


if __name__ == "__main__":
    main()
