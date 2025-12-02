# scripts/precompute_profilesHGT.py
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SECONDS_PER_DAY = 60 * 60 * 24
LN2 = math.log(2.0)


def build_metadata_embeddings(
    root: Path, d_dir: int, d_actor: int, d_genre: int, seed: int = 42
):
    """
    Build random-but-fixed embeddings for directors/actors/genres and aggregate them
    into per-movie metadata vectors phi(j).

    Returns:
        phi: [num_movies, d_phi] float32 tensor
        dims: dict with d_dir, d_actor, d_genre, d_phi
    """
    rng = torch.Generator().manual_seed(seed)

    actors = pd.read_csv(root / "actors.csv")
    directors = pd.read_csv(root / "directors.csv")
    genres = pd.read_csv(root / "genres.csv")

    num_movies = len(pd.read_csv(root / "mappings" / "movie_index.csv"))
    num_actors = len(actors)
    num_directors = len(directors)
    num_genres = len(genres)

    # Random frozen metadata tables (GraphFlix v1)
    E_actor = torch.randn(num_actors, d_actor, generator=rng) / math.sqrt(d_actor)
    E_dir = torch.randn(num_directors, d_dir, generator=rng) / math.sqrt(d_dir)
    E_genre = torch.randn(num_genres, d_genre, generator=rng) / math.sqrt(d_genre)

    phi_dir = torch.zeros(num_movies, d_dir)
    phi_actor = torch.zeros(num_movies, d_actor)
    phi_genre = torch.zeros(num_movies, d_genre)

    cnt_dir = torch.zeros(num_movies, 1)
    cnt_actor = torch.zeros(num_movies, 1)
    cnt_genre = torch.zeros(num_movies, 1)

    # movie->director
    md = pd.read_csv(root / "movie_director_edges_reindexed.csv")
    for row in md.itertuples(index=False):
        m = int(row.movie_idx)
        d = int(row.director_idx)
        if 0 <= m < num_movies and 0 <= d < num_directors:
            phi_dir[m] += E_dir[d]
            cnt_dir[m] += 1.0

    # movie->actor
    ma = pd.read_csv(root / "movie_actor_edges_reindexed.csv")
    for row in ma.itertuples(index=False):
        m = int(row.movie_idx)
        a = int(row.actor_idx)
        if 0 <= m < num_movies and 0 <= a < num_actors:
            phi_actor[m] += E_actor[a]
            cnt_actor[m] += 1.0

    # movie->genre
    mg = pd.read_csv(root / "movie_genre_edges_reindexed.csv")
    for row in mg.itertuples(index=False):
        m = int(row.movie_idx)
        g = int(row.genre_idx)
        if 0 <= m < num_movies and 0 <= g < num_genres:
            phi_genre[m] += E_genre[g]
            cnt_genre[m] += 1.0

    # average where we have at least one metadata entry
    for phi, cnt in [
        (phi_dir, cnt_dir),
        (phi_actor, cnt_actor),
        (phi_genre, cnt_genre),
    ]:
        mask = cnt.squeeze(-1) > 0
        if mask.any():
            phi[mask] /= cnt[mask]

    phi = torch.cat([phi_dir, phi_actor, phi_genre], dim=1).float()
    dims = {
        "d_dir": d_dir,
        "d_actor": d_actor,
        "d_genre": d_genre,
        "d_phi": d_dir + d_actor + d_genre,
    }
    return phi, dims


def compute_profiles_for_phase(
    movie_idx: np.ndarray,
    ratings: np.ndarray,
    timestamps: np.ndarray,
    t_now: float,
    phi: torch.Tensor,
    tau_days: float,
    eps_min: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a single user profile p(u) for a given reference time t_now.
    movie_idx, ratings, timestamps are 1D numpy arrays for that user (train interactions only).
    """
    if len(movie_idx) == 0:
        return torch.zeros(phi.size(1), dtype=torch.float32)

    # rating-based weights: max(0, r_uj - r̄_u)
    r_bar = ratings.mean()
    w_rating = np.maximum(0.0, ratings - r_bar)

    # time-based weights (half-life, Eq. 1)
    delta_days = (t_now - timestamps) / SECONDS_PER_DAY
    delta_days = np.maximum(delta_days, 0.0)
    w_time = np.exp(-(LN2 / tau_days) * delta_days)

    w = w_time * w_rating
    Z = w.sum()

    # fallback if rating weights are all zero (degenerate)
    if Z < eps_min:
        w = w_time
        Z = w.sum()

    # if still degenerate, fall back to uniform weights
    if Z < eps_min:
        w = np.ones_like(w)
        Z = w.sum()

    w_norm = w / (Z + 1e-8)

    # aggregate phi(j) with weights
    phi_user = phi[movie_idx]  # [H, d_phi]
    w_t = torch.from_numpy(w_norm).float().unsqueeze(1)  # [H, 1]
    profile = (w_t * phi_user).sum(dim=0)
    return profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/ml1m")
    ap.add_argument("--tau", type=float, default=150.0, help="Half-life in days")
    ap.add_argument("--d_dir", type=int, default=64)
    ap.add_argument("--d_actor", type=int, default=64)
    ap.add_argument("--d_genre", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    splits_path = root / "splits" / "ratings_split_reindexed.csv"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found. Run scripts/split_temporal.py and scripts/reindex_ids.py first."
        )

    print(f"Loading splits from {splits_path} ...")
    s = pd.read_csv(splits_path)
    # ensure expected columns
    expected_cols = {"user_idx", "movie_idx", "rating", "timestamp", "split"}
    if not expected_cols.issubset(set(s.columns)):
        raise ValueError(f"Unexpected columns in {splits_path}: {s.columns.tolist()}")

    num_users = len(pd.read_csv(root / "mappings" / "user_index.csv"))
    num_movies = len(pd.read_csv(root / "mappings" / "movie_index.csv"))

    print("Building movie metadata embeddings phi(j) ...")
    phi, dims = build_metadata_embeddings(
        root,
        d_dir=args.d_dir,
        d_actor=args.d_actor,
        d_genre=args.d_genre,
        seed=args.seed,
    )
    assert phi.shape[0] == num_movies, "phi(j) rows != num_movies"

    # Compute per-user train history and t_now for each phase
    train = s[s["split"] == "train"].copy()
    val = s[s["split"] == "val"].copy()
    test = s[s["split"] == "test"].copy()

    # Sanity: exactly one val/test per user
    per_user = s.groupby("user_idx")["split"].value_counts().unstack(fill_value=0)
    if not ((per_user["val"] == 1).all() and (per_user["test"] == 1).all()):
        raise RuntimeError("Expected exactly 1 val and 1 test interaction per user.")

    # t_now for each phase
    t_train_now = train.groupby("user_idx")["timestamp"].max().to_dict()
    t_val = val.set_index("user_idx")["timestamp"].to_dict()
    t_test = test.set_index("user_idx")["timestamp"].to_dict()

    # global popularity for p_global
    movie_freq = train.groupby("movie_idx")["user_idx"].size()
    freq = torch.zeros(num_movies, dtype=torch.float32)
    freq[movie_freq.index.to_numpy(dtype=int)] = torch.from_numpy(
        movie_freq.to_numpy(dtype=np.float32)
    )

    profiles_train = torch.zeros(num_users, dims["d_phi"], dtype=torch.float32)
    profiles_val = torch.zeros_like(profiles_train)
    profiles_test = torch.zeros_like(profiles_train)

    # groupby over train interactions
    grouped = train.groupby("user_idx")

    print("Computing half-life profiles per user (train/val/test) ...")
    eps_min = 1e-6
    n_fallback = 0

    for u, g in grouped:
        u_int = int(u)
        movie_idx = g["movie_idx"].to_numpy(dtype=int)
        ratings = g["rating"].to_numpy(dtype=float)
        timestamps = g["timestamp"].to_numpy(dtype=float)

        # training profile: t_now = last train interaction
        t_now_train = float(t_train_now[u_int])
        p_train = compute_profiles_for_phase(
            movie_idx,
            ratings,
            timestamps,
            t_now_train,
            phi,
            tau_days=args.tau,
            eps_min=eps_min,
        )

        # validation/test profiles: same Hu, different t_now
        t_now_val = float(t_val[u_int])
        t_now_test = float(t_test[u_int])

        p_val = compute_profiles_for_phase(
            movie_idx,
            ratings,
            timestamps,
            t_now_val,
            phi,
            tau_days=args.tau,
            eps_min=eps_min,
        )
        p_test = compute_profiles_for_phase(
            movie_idx,
            ratings,
            timestamps,
            t_now_test,
            phi,
            tau_days=args.tau,
            eps_min=eps_min,
        )

        # track degenerate profiles (norm ~ 0)
        if p_train.abs().sum().item() < 1e-6:
            n_fallback += 1

        profiles_train[u_int] = p_train
        profiles_val[u_int] = p_val
        profiles_test[u_int] = p_test

    print(f"Users with (near-)degenerate train profiles: {n_fallback}/{num_users}")

    # cold-start users (no train interactions): fallback to global profile
    # (in our temporal-LOO setup this should not happen, but handle just in case)
    mask_cold = profiles_train.abs().sum(dim=1) < 1e-6
    if mask_cold.any():
        print(
            f"Cold-start users without train interactions: {int(mask_cold.sum().item())}"
        )
        # global profile: popularity-weighted average over movies
        if freq.sum() > 0:
            w = freq / freq.sum()
            p_global = (w.unsqueeze(1) * phi).sum(dim=0)
        else:
            p_global = torch.zeros_like(phi[0])
        profiles_train[mask_cold] = p_global
        profiles_val[mask_cold] = p_global
        profiles_test[mask_cold] = p_global
    else:
        # still compute p_global for potential downstream use
        if freq.sum() > 0:
            w = freq / freq.sum()
            p_global = (w.unsqueeze(1) * phi).sum(dim=0)
        else:
            p_global = torch.zeros_like(phi[0])

    out = {
        "tau_days": float(args.tau),
        "dims": dims,
        "phi": phi,  # [num_movies, d_phi]
        "profiles_train": profiles_train,  # [num_users, d_phi]
        "profiles_val": profiles_val,
        "profiles_test": profiles_test,
        "p_global": p_global,
        "meta": {
            "root": str(root),
            "num_users": int(num_users),
            "num_movies": int(num_movies),
        },
    }

    out_path = root / f"half_life_profiles_tau{int(args.tau)}.pt"
    torch.save(out, out_path)
    print(f"✓ Saved profiles and phi to {out_path}")


if __name__ == "__main__":
    main()
