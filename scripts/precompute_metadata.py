#!/usr/bin/env python3
"""
Precompute metadata embeddings φ(j) for all movies.

For each movie j, we aggregate embeddings from:
- Connected actors
- Connected directors  
- Connected genres

Output: phi_matrix.pt with shape [num_movies × d_phi]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_metadata_entities(data_dir):
    """Load actors, directors, and genres CSV files."""
    actors_df = pd.read_csv(
        data_dir / "actors.csv", names=["actor_id", "tmdb_id", "name"], skiprows=0
    )

    directors_df = pd.read_csv(data_dir / "directors.csv", skiprows=1)  # has header

    genres_df = pd.read_csv(
        data_dir / "genres.csv", names=["genre_id", "tmdb_id", "name"], skiprows=0
    )

    return actors_df, directors_df, genres_df


def load_movie_metadata_edges(data_dir):
    """Load movie-actor, movie-director, movie-genre edge lists.

    Handles both reindexed and non-reindexed files.
    """
    # Movie-Actor edges - try reindexed first
    actor_file = data_dir / "movie_actor_edges_reindexed.csv"
    if not actor_file.exists():
        actor_file = data_dir / "movie_actor_edges.csv"

    movie_actor_df = pd.read_csv(actor_file)
    # Standardize column names
    if "movie_idx" in movie_actor_df.columns:
        movie_actor_df = movie_actor_df.rename(
            columns={"movie_idx": "movie_id", "actor_idx": "actor_id"}
        )

    # Movie-Director edges - try reindexed first
    director_file = data_dir / "movie_director_edges_reindexed.csv"
    if not director_file.exists():
        director_file = data_dir / "movie_director_edges.csv"

    movie_director_df = pd.read_csv(director_file)
    if "movie_idx" in movie_director_df.columns:
        movie_director_df = movie_director_df.rename(
            columns={"movie_idx": "movie_id", "director_idx": "director_id"}
        )

    # Movie-Genre edges - try reindexed first
    genre_file = data_dir / "movie_genre_edges_reindexed.csv"
    if not genre_file.exists():
        genre_file = data_dir / "movie_genre_edges.csv"

    movie_genre_df = pd.read_csv(genre_file)
    if "movie_idx" in movie_genre_df.columns:
        movie_genre_df = movie_genre_df.rename(
            columns={"movie_idx": "movie_id", "genre_idx": "genre_id"}
        )

    return movie_actor_df, movie_director_df, movie_genre_df


def create_metadata_embeddings(
    num_actors, num_directors, num_genres, embed_dim, seed=42
):
    """Initialize random embeddings for actors, directors, and genres.

    In v1, these are fixed (not learned). Future versions could train these end-to-end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Xavier/Glorot initialization for better gradient flow if we later make them learnable
    E_actor = torch.randn(num_actors, embed_dim) * (2.0 / embed_dim) ** 0.5
    E_director = torch.randn(num_directors, embed_dim) * (2.0 / embed_dim) ** 0.5
    E_genre = torch.randn(num_genres, embed_dim) * (2.0 / embed_dim) ** 0.5

    return E_actor, E_director, E_genre


def aggregate_movie_metadata(
    movies_df,
    movie_actor_edges,
    movie_director_edges,
    movie_genre_edges,
    E_actor,
    E_director,
    E_genre,
    aggregation="mean",
):
    """
    For each movie j, compute φ(j) by aggregating metadata embeddings.

    φ(j) = [mean(actor_embeds); mean(director_embeds); mean(genre_embeds)]

    Returns:
        phi_matrix: Tensor of shape [max_movie_id+1 × (3 * embed_dim)]
                    Indexed by movie_id directly
    """
    num_movies = len(movies_df)
    max_movie_id = movies_df["movie_id"].max()
    embed_dim = E_actor.shape[1]
    d_phi = 3 * embed_dim  # concatenate actor, director, genre embeddings

    # Create matrix large enough to index by movie_id directly (0 to max_movie_id)
    phi_matrix = torch.zeros(max_movie_id + 1, d_phi)

    print(f"Aggregating metadata for {num_movies} movies...")
    print(
        f"  Creating phi_matrix of size [{max_movie_id + 1} × {d_phi}] for direct movie_id indexing"
    )

    # Create lookup dictionaries for faster access
    actor_dict = movie_actor_edges.groupby("movie_id")["actor_id"].apply(list).to_dict()
    director_dict = (
        movie_director_edges.groupby("movie_id")["director_id"].apply(list).to_dict()
    )
    genre_dict = movie_genre_edges.groupby("movie_id")["genre_id"].apply(list).to_dict()

    for idx, row in movies_df.iterrows():
        movie_id = row["movie_id"]

        # Get connected actors, directors, genres from lookup dicts
        actor_ids = actor_dict.get(movie_id, [])
        director_ids = director_dict.get(movie_id, [])
        genre_ids = genre_dict.get(movie_id, [])

        # Aggregate actor embeddings (with bounds checking)
        if len(actor_ids) > 0:
            valid_actor_ids = [aid for aid in actor_ids if 0 <= aid < E_actor.shape[0]]
            if len(valid_actor_ids) > 0:
                actor_indices = torch.tensor(valid_actor_ids, dtype=torch.long)
                actor_embeds = E_actor[actor_indices]
                if aggregation == "mean":
                    actor_agg = actor_embeds.mean(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                actor_agg = torch.zeros(embed_dim)
        else:
            actor_agg = torch.zeros(embed_dim)

        # Aggregate director embeddings (with bounds checking)
        if len(director_ids) > 0:
            valid_director_ids = [
                did for did in director_ids if 0 <= did < E_director.shape[0]
            ]
            if len(valid_director_ids) > 0:
                director_indices = torch.tensor(valid_director_ids, dtype=torch.long)
                director_embeds = E_director[director_indices]
                if aggregation == "mean":
                    director_agg = director_embeds.mean(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                director_agg = torch.zeros(embed_dim)
        else:
            director_agg = torch.zeros(embed_dim)

        # Aggregate genre embeddings (with bounds checking)
        if len(genre_ids) > 0:
            valid_genre_ids = [gid for gid in genre_ids if 0 <= gid < E_genre.shape[0]]
            if len(valid_genre_ids) > 0:
                genre_indices = torch.tensor(valid_genre_ids, dtype=torch.long)
                genre_embeds = E_genre[genre_indices]
                if aggregation == "mean":
                    genre_agg = genre_embeds.mean(dim=0)
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                genre_agg = torch.zeros(embed_dim)
        else:
            genre_agg = torch.zeros(embed_dim)

        # Concatenate: φ(j) = [actor_agg; director_agg; genre_agg]
        # Index by movie_id directly (not DataFrame row index)
        phi_matrix[movie_id] = torch.cat([actor_agg, director_agg, genre_agg])

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{num_movies} movies")

    return phi_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Precompute metadata embeddings φ(j) for all movies"
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
        "--embed_dim",
        type=int,
        default=64,
        help="Embedding dimension for each metadata type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load config if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
            if "metadata" in config and "actor_embed_dim" in config["metadata"]:
                args.embed_dim = config["metadata"]["actor_embed_dim"]

    print("=" * 80)
    print("STEP 1.1: Precompute Metadata Embeddings φ(j)")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Random seed: {args.seed}")
    print()

    # Load metadata entities
    print("Loading metadata entities...")
    actors_df, directors_df, genres_df = load_metadata_entities(data_dir)
    print(f"  Actors: {len(actors_df)}")
    print(f"  Directors: {len(directors_df)}")
    print(f"  Genres: {len(genres_df)}")
    print()

    # Load movie-metadata edges
    print("Loading movie-metadata edges...")
    movie_actor_edges, movie_director_edges, movie_genre_edges = (
        load_movie_metadata_edges(data_dir)
    )
    print(f"  Movie-Actor edges: {len(movie_actor_edges)}")
    print(f"  Movie-Director edges: {len(movie_director_edges)}")
    print(f"  Movie-Genre edges: {len(movie_genre_edges)}")
    print()

    # Load movies
    print("Loading movies...")
    movies_df = pd.read_csv(data_dir / "movies.csv", header=0)  # has header row
    # Rename if needed
    if "movie_id" not in movies_df.columns:
        movies_df.columns = ["movie_id", "title", "genres_str"]
    print(f"  Total movies: {len(movies_df)}")
    print()

    # Create metadata embeddings
    print("Creating metadata embeddings...")
    E_actor, E_director, E_genre = create_metadata_embeddings(
        num_actors=len(actors_df),
        num_directors=len(directors_df),
        num_genres=len(genres_df),
        embed_dim=args.embed_dim,
        seed=args.seed,
    )
    print(f"  E_actor shape: {E_actor.shape}")
    print(f"  E_director shape: {E_director.shape}")
    print(f"  E_genre shape: {E_genre.shape}")
    print()

    # Aggregate to compute φ(j) for each movie
    print("Computing φ(j) for each movie...")
    phi_matrix = aggregate_movie_metadata(
        movies_df=movies_df,
        movie_actor_edges=movie_actor_edges,
        movie_director_edges=movie_director_edges,
        movie_genre_edges=movie_genre_edges,
        E_actor=E_actor,
        E_director=E_director,
        E_genre=E_genre,
        aggregation="mean",
    )
    print()

    # Verification
    print("Verification:")
    print(f"  φ matrix shape: {phi_matrix.shape}")
    print(f"  Matrix can be indexed by movie_id from 0 to {phi_matrix.shape[0]-1}")
    print(f"  Movies with metadata: {len(movies_df)}")
    print(f"  Contains NaN: {torch.isnan(phi_matrix).any().item()}")
    print(f"  Contains Inf: {torch.isinf(phi_matrix).any().item()}")
    # Only compute norms for non-zero rows
    non_zero_mask = phi_matrix.norm(dim=1) > 0
    if non_zero_mask.any():
        print(
            f"  Mean norm (non-zero): {phi_matrix[non_zero_mask].norm(dim=1).mean().item():.4f}"
        )
        print(
            f"  Std norm (non-zero): {phi_matrix[non_zero_mask].norm(dim=1).std().item():.4f}"
        )
    print()

    # Create movie_id to index mapping
    movie_id_to_idx = {row["movie_id"]: idx for idx, row in movies_df.iterrows()}

    # Save
    output_path = data_dir / "phi_matrix.pt"
    print(f"Saving φ matrix to: {output_path}")
    torch.save(
        {
            "phi_matrix": phi_matrix,
            "movie_id_to_idx": movie_id_to_idx,
            "num_movies": len(movies_df),
            "d_phi": phi_matrix.shape[1],
            "embed_dim": args.embed_dim,
            "E_actor": E_actor,
            "E_director": E_director,
            "E_genre": E_genre,
            "metadata": {
                "num_actors": len(actors_df),
                "num_directors": len(directors_df),
                "num_genres": len(genres_df),
            },
        },
        output_path,
    )

    print("✓ Step 1.1 complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
