"""Per-user temporal leave-one-out split (train, val, test)."""

import argparse
import pathlib

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False)
    p.add_argument("--ratings", type=str, default="output/ratings.csv")
    p.add_argument("--out", type=str, default="data/processed/ml1m/splits")
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # NOTE: Replace with your real ratings path. This is a placeholder.
    if not pathlib.Path(args.ratings).exists():
        print("Ratings file not found. Generate CSVs first (download_movielens.py).")
        return
    df = pd.read_csv(args.ratings)
    df = df.sort_values(["user_id", "timestamp"])
    # simple per-user last/second-last split
    parts = []
    for uid, g in df.groupby("user_id"):
        if len(g) < 3:
            continue
        test = g.iloc[[-1]].assign(split="test")
        val = g.iloc[[-2]].assign(split="val")
        train = g.iloc[:-2].assign(split="train")
        parts.extend([train, val, test])
    split = pd.concat(parts, ignore_index=True)
    split.to_csv(out / "ratings_split.csv", index=False)
    print(f"Wrote {out/'ratings_split.csv'}")
    


if __name__ == "__main__":
    main()
