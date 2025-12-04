# scripts/build_graph_pyg.py
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData


def make_edge_index(df, src_col, dst_col):
    src = torch.as_tensor(df[src_col].to_numpy(), dtype=torch.long)
    dst = torch.as_tensor(df[dst_col].to_numpy(), dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/ml1m")
    ap.add_argument(
        "--use_train_only",
        action="store_true",
        help="Use only train user->movie edges from the reindexed split",
    )
    args = ap.parse_args()
    P = Path(args.root)

    # --- Node counts from mapping tables / node CSVs
    n_users = len(pd.read_csv(P / "mappings/user_index.csv"))
    n_movies = len(pd.read_csv(P / "mappings/movie_index.csv"))
    n_actors = len(pd.read_csv(P / "actors.csv"))
    n_directors = len(pd.read_csv(P / "directors.csv"))
    n_genres = len(pd.read_csv(P / "genres.csv"))

    data = HeteroData()
    data["user"].num_nodes = n_users
    data["movie"].num_nodes = n_movies
    data["actor"].num_nodes = n_actors
    data["director"].num_nodes = n_directors
    data["genre"].num_nodes = n_genres

    # --- User->Movie edges (train-only by default to avoid leakage in message passing)
    if args.use_train_only and (P / "splits/ratings_split_reindexed.csv").exists():
        s = pd.read_csv(P / "splits/ratings_split_reindexed.csv")
        # Handle both column name variations
        if "user_idx" not in s.columns and "user_id" in s.columns:
            s = s.rename(columns={"user_id": "user_idx", "movie_id": "movie_idx"})
        train_pairs = s[s["split"] == "train"][["user_idx", "movie_idx"]]
        um = train_pairs
    else:
        um = pd.read_csv(P / "ratings_reindexed.csv")
        # Handle both column name variations
        if "user_idx" not in um.columns and "user_id" in um.columns:
            um = um.rename(columns={"user_id": "user_idx", "movie_id": "movie_idx"})
        um = um[["user_idx", "movie_idx"]]

    ei_um = make_edge_index(um, "user_idx", "movie_idx")
    data["user", "rates", "movie"].edge_index = ei_um
    # add reverse relation (useful for GNN message passing / LightGCN-like propagation)
    data["movie", "rev_rates", "user"].edge_index = ei_um.flip(0)

    # --- Movie->Actor / Director / Genre (from reindexed edges)
    ma = pd.read_csv(P / "movie_actor_edges_reindexed.csv")
    if "movie_idx" not in ma.columns and "movie_id" in ma.columns:
        ma = ma.rename(columns={"movie_id": "movie_idx", "actor_id": "actor_idx"})
    ma = ma[["movie_idx", "actor_idx"]]

    md = pd.read_csv(P / "movie_director_edges_reindexed.csv")
    if "movie_idx" not in md.columns and "movie_id" in md.columns:
        md = md.rename(columns={"movie_id": "movie_idx", "director_id": "director_idx"})
    md = md[["movie_idx", "director_idx"]]

    mg = pd.read_csv(P / "movie_genre_edges_reindexed.csv")
    if "movie_idx" not in mg.columns and "movie_id" in mg.columns:
        mg = mg.rename(columns={"movie_id": "movie_idx", "genre_id": "genre_idx"})
    mg = mg[["movie_idx", "genre_idx"]]

    data["movie", "stars", "actor"].edge_index = make_edge_index(
        ma, "movie_idx", "actor_idx"
    )
    data["actor", "rev_stars", "movie"].edge_index = data[
        "movie", "stars", "actor"
    ].edge_index.flip(0)
    data["movie", "directed_by", "director"].edge_index = make_edge_index(
        md, "movie_idx", "director_idx"
    )
    data["director", "rev_directed_by", "movie"].edge_index = data[
        "movie", "directed_by", "director"
    ].edge_index.flip(0)
    data["movie", "has_genre", "genre"].edge_index = make_edge_index(
        mg, "movie_idx", "genre_idx"
    )
    data["genre", "rev_has_genre", "movie"].edge_index = data[
        "movie", "has_genre", "genre"
    ].edge_index.flip(0)

    # --- integrity checks
    def n_edges(etype):
        return data[etype].edge_index.size(1)

    print("HeteroData summary:")
    print("  users     :", n_users)
    print("  movies    :", n_movies)
    print("  actors    :", n_actors)
    print("  directors :", n_directors)
    print("  genres    :", n_genres)
    print("  rates     :", n_edges(("user", "rates", "movie")))
    print("  stars     :", n_edges(("movie", "stars", "actor")))
    print("  directed_by:", n_edges(("movie", "directed_by", "director")))
    print("  has_genre :", n_edges(("movie", "has_genre", "genre")))

    # --- Save
    out_path = P / "graph_pyg.pt"
    torch.save(data, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
