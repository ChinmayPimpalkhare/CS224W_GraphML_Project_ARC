from pathlib import Path

import pandas as pd

P = Path("data/processed/ml1m")
U = pd.read_csv(P / "mappings/user_index.csv").shape[0]
M = pd.read_csv(P / "mappings/movie_index.csv").shape[0]
A = pd.read_csv(P / "actors.csv").shape[0]
D = pd.read_csv(P / "directors.csv").shape[0]
G = pd.read_csv(P / "genres.csv").shape[0]


def check(df, col, N, label):
    mi, mx = int(df[col].min()), int(df[col].max())
    assert mi == 0 and mx < N, f"{label}:{col} out of range (min={mi}, max={mx}, N={N})"
    assert df[col].isna().sum() == 0, f"{label}:{col} has NaNs"
    print(f"{label}:{col} OK  (min={mi}, max={mx}, N={N})")


r = pd.read_csv(P / "ratings_reindexed.csv")
ma = pd.read_csv(P / "movie_actor_edges_reindexed.csv")
md = pd.read_csv(P / "movie_director_edges_reindexed.csv")
mg = pd.read_csv(P / "movie_genre_edges_reindexed.csv")

check(r, "user_idx", U, "ratings")
check(r, "movie_idx", M, "ratings")
check(ma, "movie_idx", M, "movie_actor")
check(ma, "actor_idx", A, "movie_actor")
check(md, "movie_idx", M, "movie_director")
check(md, "director_idx", D, "movie_director")
check(mg, "movie_idx", M, "movie_genre")
check(mg, "genre_idx", G, "movie_genre")

# Optional: check the reindexed split if present
SP = P / "splits" / "ratings_split_reindexed.csv"
if SP.exists():
    s = pd.read_csv(SP)
    check(s, "user_idx", U, "split")
    check(s, "movie_idx", M, "split")
    print("Split labels:", s["split"].value_counts().to_dict())
