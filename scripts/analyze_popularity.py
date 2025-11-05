#!/usr/bin/env python3
"""
Analyze test set popularity distribution to validate protocol difficulty claims.
"""
from pathlib import Path

import pandas as pd


def main():
    P = Path("data/processed/ml1m")
    spl = pd.read_csv(P / "splits" / "ratings_split_reindexed.csv")

    train = spl[spl["split"] == "train"]
    test = spl[spl["split"] == "test"]

    # Item popularity in training set
    freq = train["movie_idx"].value_counts()

    # Test item popularity; count how many train interactions each test item has
    test_items = test["movie_idx"]
    test_counts = freq.reindex(test_items, fill_value=0)

    # Basic counts
    total_test = int(test_items.shape[0])
    unique_test = int(test_items.nunique())
    cold_start = int((test_counts == 0).sum())

    # Stats over test items that do exist in TRAIN
    nonzero = test_counts[test_counts > 0]
    median = float(nonzero.median()) if not nonzero.empty else 0.0
    mean = float(nonzero.mean()) if not nonzero.empty else 0.0
    minv = int(nonzero.min()) if not nonzero.empty else 0
    maxv = int(nonzero.max()) if not nonzero.empty else 0

    # Head overlap
    top100 = set(freq.index[:100])
    test_set = set(test_items.unique())
    overlap = len(top100 & test_set)
    all_top_in_test = overlap == len(top100)

    print("\n" + "=" * 60)
    print("TEST SET POPULARITY ANALYSIS")
    print("=" * 60)
    print(f"Total test events:   {total_test}")
    print(f"Unique test items:   {unique_test}")
    print(f"Cold-start items:    {cold_start}")
    print("\nTest items WITH train interactions:")
    print(f"  Median count:      {median:.1f}")
    print(f"  Mean count:        {mean:.1f}")
    print(f"  Min/Max count:     {minv} / {maxv}")
    print("\nTop-100 popular items (by TRAIN counts):")
    print(f"  All appear in test: {all_top_in_test}")
    print(f"  Overlap:            {overlap} / 100")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
