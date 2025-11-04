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

    # ===== COMPREHENSIVE DIAGNOSTICS =====
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Users:                           {U:,}")
    print(f"Movies:                          {M:,}")
    print(f"Train edges:                     {len(train):,}")
    print(f"Val edges:                       {len(val):,}")
    print(f"Test edges:                      {len(test):,}")
    print(f"Avg train interactions per user: {len(train) / U:.1f}")
    print(f"Avg train interactions per item: {len(train) / M:.1f}")
    print(f"Train sparsity:                  {100 * (1 - len(train)/(U*M)):.4f}%")

    print(f"\n{'='*70}")
    print("TOP-10 MOST POPULAR ITEMS (BY TRAIN COUNT)")
    print(f"{'='*70}")
    print(f"{'Rank':<6} {'Item ID':<10} {'Count':<10} {'% of Train':<12}")
    print(f"{'-'*70}")
    total_train = len(train)
    for i in range(10):
        item_id = rank[i]
        count = counts[item_id]
        pct = 100.0 * count / total_train
        print(f"{i+1:<6} {item_id:<10} {count:<10} {pct:>10.3f}%")

    # Analyze test set
    test_items = test[:, 1].astype(int)
    test_item_counts = [counts[item] for item in test_items]

    print(f"\n{'='*70}")
    print("TEST SET ANALYSIS")
    print(f"{'='*70}")
    print(f"Unique users in test:            {len(set(test[:, 0]))}")
    print(f"Unique items in test:            {len(set(test_items))}")
    print(f"Total test interactions:         {len(test)}")

    print("\nTest Item Popularity (based on train counts):")
    print(f"  Min count:    {min(test_item_counts):>6}")
    print(f"  25th %ile:    {int(np.percentile(test_item_counts, 25)):>6}")
    print(f"  Median:       {int(np.median(test_item_counts)):>6}")
    print(f"  75th %ile:    {int(np.percentile(test_item_counts, 75)):>6}")
    print(f"  Max count:    {max(test_item_counts):>6}")
    print(f"  Mean:         {np.mean(test_item_counts):>6.1f}")

    # Long-tail analysis
    lt10 = sum(1 for c in test_item_counts if c < 10)
    lt50 = sum(1 for c in test_item_counts if c < 50)
    lt100 = sum(1 for c in test_item_counts if c < 100)
    lt200 = sum(1 for c in test_item_counts if c < 200)

    print("\nLong-Tail Breakdown:")
    print(
        f"  Test items with <10 interactions:   {lt10:>4} ({100*lt10/len(test_item_counts):>5.1f}%)"
    )
    print(
        f"  Test items with <50 interactions:   {lt50:>4} ({100*lt50/len(test_item_counts):>5.1f}%)"
    )
    print(
        f"  Test items with <100 interactions:  {lt100:>4} ({100*lt100/len(test_item_counts):>5.1f}%)"
    )
    print(
        f"  Test items with <200 interactions:  {lt200:>4} ({100*lt200/len(test_item_counts):>5.1f}%)"
    )

    # Check top-K coverage in test
    top10_set = set(rank[:10])
    top50_set = set(rank[:50])
    top100_set = set(rank[:100])
    test_items_set = set(test_items)

    overlap10 = len(top10_set & test_items_set)
    overlap50 = len(top50_set & test_items_set)
    overlap100 = len(top100_set & test_items_set)

    print("\nTop-K Popular Items Appearing in Test Set:")
    print(f"  Items in top-10:    {overlap10:>3} / 10  ({100*overlap10/10:>5.1f}%)")
    print(f"  Items in top-50:    {overlap50:>3} / 50  ({100*overlap50/50:>5.1f}%)")
    print(f"  Items in top-100:   {overlap100:>3} / 100 ({100*overlap100/100:>5.1f}%)")

    if overlap10 > 0:
        print(
            f"\n  Top-10 items appearing in test: {sorted(top10_set & test_items_set)}"
        )

    # User coverage
    val_users = set(val[:, 0])
    test_users = set(test[:, 0])
    print("\nUser Coverage:")
    print(f"  Users in val:  {len(val_users):>4} / {U} ({100*len(val_users)/U:>5.1f}%)")
    print(
        f"  Users in test: {len(test_users):>4} / {U} ({100*len(test_users)/U:>5.1f}%)"
    )

    print(f"\n{'='*70}\n")
    # ===== END DIAGNOSTICS =====

    # Metrics
    val_metrics = eval_split(rank, user_pos, val, K=(10, 20))
    test_metrics = eval_split(rank, user_pos, test, K=(10, 20))

    print("RESULTS:")
    print(f"  Most-Pop VAL : {val_metrics}")
    print(f"  Most-Pop TEST: {test_metrics}")

    # Save
    (P / "runs").mkdir(parents=True, exist_ok=True)
    with open(P / "runs" / "mostpop_metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    print(f"\nSaved: {P / 'runs' / 'mostpop_metrics.json'}")


if __name__ == "__main__":
    main()
