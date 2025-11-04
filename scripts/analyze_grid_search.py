import sys

import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python analyze_grid_search.py <summary.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])

print("\n" + "=" * 70)
print("GRID SEARCH RESULTS ANALYSIS")
print("=" * 70)

print(f"\nTotal configs: {len(df)}")
print(f"Best Test NDCG@10: {df['test_ndcg10'].max():.4f}")
print(f"Best Test Recall@10: {df['test_recall10'].max():.4f}")

print("\n" + "-" * 70)
print("TOP 10 CONFIGS BY TEST NDCG@10")
print("-" * 70)
top10 = df.nlargest(10, "test_ndcg10")
print(
    top10[
        ["config", "test_recall10", "test_ndcg10", "val_ndcg10", "best_epoch"]
    ].to_string(index=False)
)

print("\n" + "-" * 70)
print("PARAMETER IMPACT ANALYSIS")
print("-" * 70)

if "layers" in df.columns:
    print("\nBy Layers:")
    print(
        df.groupby("layers")[["test_ndcg10", "test_recall10"]]
        .mean()
        .sort_values("test_ndcg10", ascending=False)
    )

if "dim" in df.columns:
    print("\nBy Embedding Dimension:")
    print(
        df.groupby("dim")[["test_ndcg10", "test_recall10"]]
        .mean()
        .sort_values("test_ndcg10", ascending=False)
    )

if "lr" in df.columns:
    print("\nBy Learning Rate:")
    print(
        df.groupby("lr")[["test_ndcg10", "test_recall10"]]
        .mean()
        .sort_values("test_ndcg10", ascending=False)
    )

if "neg" in df.columns:
    print("\nBy # Negatives:")
    print(
        df.groupby("neg")[["test_ndcg10", "test_recall10"]]
        .mean()
        .sort_values("test_ndcg10", ascending=False)
    )

if "use_cosine" in df.columns:
    print("\nBy Cosine Scoring:")
    print(
        df.groupby("use_cosine")[["test_ndcg10", "test_recall10"]]
        .mean()
        .sort_values("test_ndcg10", ascending=False)
    )

print("\n" + "=" * 70)
