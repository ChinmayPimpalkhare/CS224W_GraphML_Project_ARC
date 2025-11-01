import argparse
import json
import pathlib

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/ml1m")
    args = ap.parse_args()
    P = pathlib.Path(args.root)

    # Load tables
    movies = pd.read_csv(P / "movies.csv")
    ratings = pd.read_csv(P / "ratings.csv")
    users = pd.read_csv(P / "users.csv")
    actors = pd.read_csv(P / "actors.csv")
    directors = pd.read_csv(P / "directors.csv")
    genres = pd.read_csv(P / "genres.csv")
    e_ma = pd.read_csv(P / "movie_actor_edges.csv")
    e_md = pd.read_csv(P / "movie_director_edges.csv")
    e_mg = pd.read_csv(P / "movie_genre_edges.csv")

    # ID maps (assuming IDs are already 0..N-1 in your CSVs; if not, map them)
    uid_set = set(users["user_id"])
    mid_set = set(movies["movie_id"])
    aid_set = set(actors["actor_id"])
    did_set = set(directors["director_id"])
    gid_set = set(genres["genre_id"])

    # Keep only edges with valid endpoints
    ratings = ratings[
        ratings["user_id"].isin(uid_set) & ratings["movie_id"].isin(mid_set)
    ]
    e_ma = e_ma[e_ma["movie_id"].isin(mid_set) & e_ma["actor_id"].isin(aid_set)]
    e_md = e_md[e_md["movie_id"].isin(mid_set) & e_md["director_id"].isin(did_set)]
    e_mg = e_mg[e_mg["movie_id"].isin(mid_set) & e_mg["genre_id"].isin(gid_set)]

    stats = {
        "nodes": {
            "user": len(uid_set),
            "movie": len(mid_set),
            "actor": len(aid_set),
            "director": len(did_set),
            "genre": len(gid_set),
        },
        "edges": {
            "rates": len(ratings),
            "stars": len(e_ma),
            "directed_by": len(e_md),
            "has_genre": len(e_mg),
        },
    }
    P.mkdir(parents=True, exist_ok=True)
    with open(P / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved:", P / "graph_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
