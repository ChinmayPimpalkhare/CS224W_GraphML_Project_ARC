from pathlib import Path

import pandas as pd

P = Path("data/processed/ml1m")

# Load node tables
users = pd.read_csv(P / "users.csv")
movies = pd.read_csv(P / "movies.csv")
actors = pd.read_csv(P / "actors.csv")
directors = pd.read_csv(P / "directors.csv")
genres = pd.read_csv(P / "genres.csv")

# Load edge tables
ratings = pd.read_csv(P / "ratings.csv")  # user_id, movie_id, rating, timestamp
e_ma = pd.read_csv(P / "movie_actor_edges.csv")  # movie_id, actor_id, character
e_md = pd.read_csv(P / "movie_director_edges.csv")  # movie_id, director_id
e_mg = pd.read_csv(P / "movie_genre_edges.csv")  # movie_id, genre_id


def check_id_space(df, col, name):
    u = pd.Index(df[col].dropna().astype("int64").unique())
    u = u.sort_values()
    contiguous_0 = (u.min() == 0) and (len(u) == int(u.max()) + 1)
    contiguous_1 = (u.min() == 1) and (len(u) == int(u.max()))
    report = {
        "n_unique": len(u),
        "min": int(u.min()),
        "max": int(u.max()),
        "is_zero_based_contiguous": contiguous_0,
        "is_one_based_contiguous": contiguous_1,
    }
    print(f"[{name}] {col}: {report}")
    return set(u)


# Check node ID spaces
U = check_id_space(users, "user_id", "user")
M = check_id_space(movies, "movie_id", "movie")
A = check_id_space(actors, "actor_id", "actor")
D = check_id_space(directors, "director_id", "director")
G = check_id_space(genres, "genre_id", "genre")


def check_edge_coverage(edge_df, src_col, src_set, dst_col, dst_set, label):
    missing_src = (~edge_df[src_col].isin(src_set)).sum()
    missing_dst = (~edge_df[dst_col].isin(dst_set)).sum()
    print(
        f"[edges:{label}] rows={len(edge_df)}  missing {src_col}={missing_src}  missing {dst_col}={missing_dst}"
    )


# Edge coverage against node sets
check_edge_coverage(ratings, "user_id", U, "movie_id", M, "rates (ratings)")
check_edge_coverage(e_ma, "movie_id", M, "actor_id", A, "stars (movie->actor)")
check_edge_coverage(
    e_md, "movie_id", M, "director_id", D, "directed_by (movie->director)"
)
check_edge_coverage(e_mg, "movie_id", M, "genre_id", G, "has_genre (movie->genre)")
