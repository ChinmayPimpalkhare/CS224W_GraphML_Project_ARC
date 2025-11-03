# scripts/reindex_ids.py
from pathlib import Path

import pandas as pd

P = Path("data/processed/ml1m")
SPLITS = P / "splits" / "ratings_split.csv"

# --- Load node tables
users = pd.read_csv(P / "users.csv")
movies = pd.read_csv(P / "movies.csv")


# --- Build stable mappings (sorted unique -> 0..N-1)
def make_map(df, col, idx_col):
    uniq = sorted(df[col].dropna().astype("int64").unique().tolist())
    return pd.DataFrame({col: uniq, idx_col: range(len(uniq))})


map_user = make_map(users, "user_id", "user_idx")
map_movie = make_map(movies, "movie_id", "movie_idx")

# Save mappings for reproducibility / debugging
M = P / "mappings"
M.mkdir(parents=True, exist_ok=True)
map_user.to_csv(M / "user_index.csv", index=False)
map_movie.to_csv(M / "movie_index.csv", index=False)


# --- Helper to apply mapping with a left join (detects missing)
def apply_map(df, left_col, mapper, right_col, out_col):
    out = df.merge(mapper, how="left", left_on=left_col, right_on=right_col)
    if out[out_col].isna().any():
        missing = int(out[out_col].isna().sum())
        raise RuntimeError(
            f"Missing {out_col} while mapping {left_col} in {df.shape}: {missing} rows"
        )
    return out


# --- Reindex ratings (user,movie)
ratings = pd.read_csv(P / "ratings.csv")
r = apply_map(ratings, "user_id", map_user, "user_id", "user_idx")
r = apply_map(r, "movie_id", map_movie, "movie_id", "movie_idx")
r[["user_idx", "movie_idx", "rating", "timestamp"]].to_csv(
    P / "ratings_reindexed.csv", index=False
)

# --- Reindex movie->actor
e_ma = pd.read_csv(P / "movie_actor_edges.csv")
actors = pd.read_csv(
    P / "actors.csv"
)  # actors are already 0-based contiguous; we keep actor_id as actor_idx
ma = apply_map(e_ma, "movie_id", map_movie, "movie_id", "movie_idx")
ma.rename(columns={"actor_id": "actor_idx"}, inplace=True)
ma[["movie_idx", "actor_idx", "character"]].to_csv(
    P / "movie_actor_edges_reindexed.csv", index=False
)

# --- Reindex movie->director
e_md = pd.read_csv(P / "movie_director_edges.csv")
directors = pd.read_csv(P / "directors.csv")  # already 0-based contiguous
md = apply_map(e_md, "movie_id", map_movie, "movie_id", "movie_idx")
md.rename(columns={"director_id": "director_idx"}, inplace=True)
md[["movie_idx", "director_idx"]].to_csv(
    P / "movie_director_edges_reindexed.csv", index=False
)

# --- Reindex movie->genre
e_mg = pd.read_csv(P / "movie_genre_edges.csv")
genres = pd.read_csv(P / "genres.csv")  # already 0-based contiguous
mg = apply_map(e_mg, "movie_id", map_movie, "movie_id", "movie_idx")
mg.rename(columns={"genre_id": "genre_idx"}, inplace=True)
mg[["movie_idx", "genre_idx"]].to_csv(
    P / "movie_genre_edges_reindexed.csv", index=False
)

# --- Reindex the split file so you can filter train edges by indices
if SPLITS.exists():
    splits = pd.read_csv(SPLITS)  # columns: user_id, movie_id, rating, timestamp, split
    s = apply_map(splits, "user_id", map_user, "user_id", "user_idx")
    s = apply_map(s, "movie_id", map_movie, "movie_id", "movie_idx")
    out_path = P / "splits" / "ratings_split_reindexed.csv"
    s[["user_idx", "movie_idx", "rating", "timestamp", "split"]].to_csv(
        out_path, index=False
    )
    print("Wrote", out_path)
else:
    print("splits/ratings_split.csv not found; skip reindexing splits.")

print("Reindex complete.")
print("  - data/processed/ml1m/mappings/{user_index.csv,movie_index.csv}")
print("  - data/processed/ml1m/ratings_reindexed.csv")
print("  - data/processed/ml1m/movie_actor_edges_reindexed.csv")
print("  - data/processed/ml1m/movie_director_edges_reindexed.csv")
print("  - data/processed/ml1m/movie_genre_edges_reindexed.csv")
