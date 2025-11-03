# scripts/run_mostpop.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_splits(root: Path):
    s = pd.read_csv(root / "splits" / "ratings_split_reindexed.csv")
    train = s[s["split"] == "train"][["user_idx", "movie_idx"]].to_numpy()
    val = s[s["split"] == "val"][["user_idx", "movie_idx"]].to_numpy()
    test = s[s["split"] == "test"][["user_idx", "movie_idx"]].to_numpy()
    return train, val, test


def build_user_pos(train, num_users):
    pos = [set() for _ in range(num_users)]
    for u, i in train:
        pos[int(u)].add(int(i))
    return pos


def global_rank_by_popularity(train, num_items):
    # count interactions per item on TRAIN only
    counts = np.zeros(num_items, dtype=np.int64)
    for _, i in train:
        counts[int(i)] += 1
    # sort items by count desc, then by id for deterministic ties
    items = np.arange(num_items)
    order = np.lexsort((-items, -counts))  # counts desc, id desc as tie-breaker
    return items[order].tolist(), counts


def eval_split(global_rank, user_pos_train, pairs, K=(10, 20)):
    # pairs: array of [u, i] truth for this split
    truth = {}
    for u, i in pairs:
        truth.setdefault(int(u), []).append(int(i))
    out = {}
    for k in K:
        recalls, ndcgs = [], []
        # we will filter the top-k list per user to exclude their train items
        topk_global = global_rank  # reuse
        for u, items in truth.items():
            banned = user_pos_train[u]
            picked = []
            for it in topk_global:
                if it not in banned:
                    picked.append(it)
                if len(picked) == k:
                    break
            # compute hits & NDCG@k
            hits = 0
            dcg = 0.0
            for rank, it in enumerate(picked):
                if it in items:
                    hits += 1
                    dcg += 1.0 / np.log2(rank + 2.0)
            ideal = sum(1.0 / np.log2(r + 2.0) for r in range(min(len(items), k)))
            recalls.append(hits / max(1, len(items)))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
        out[f"recall@{k}"] = float(np.mean(recalls))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/processed/ml1m")
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()
    P = Path(args.root)

    U = len(pd.read_csv(P / "mappings" / "user_index.csv"))
    M = len(pd.read_csv(P / "mappings" / "movie_index.csv"))

    train, val, test = load_splits(P)
    user_pos = build_user_pos(train, U)
    rank, counts = global_rank_by_popularity(train, M)

    # Metrics
    val_metrics = eval_split(rank, user_pos, val, K=(10, 20))
    test_metrics = eval_split(rank, user_pos, test, K=(10, 20))
    print("Most-Pop VAL :", val_metrics)
    print("Most-Pop TEST:", test_metrics)

    # Save
    (P / "runs").mkdir(parents=True, exist_ok=True)
    with open(P / "runs" / "mostpop_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    # Show top-10 popular items (ids)
    print("Top-10 item ids by train popularity:", rank[:10])


if __name__ == "__main__":
    main()
