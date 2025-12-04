"""
Proper evaluation dataset that samples multiple negatives (1-vs-100 protocol).
"""

from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


class ProperEvalDataset(Dataset):
    """
    Dataset for proper evaluation with multiple negatives.

    For each test interaction (user, positive_item), samples N negative items
    that the user has never interacted with (across all splits).

    Standard protocol: N=99 for 1-vs-100 evaluation.
    """

    def __init__(self, ratings_df, graph_data, split="test", num_negatives=99, seed=42):
        """
        Args:
            ratings_df: DataFrame with all ratings (train+val+test)
            graph_data: PyG Data object
            split: Which split to evaluate ('test' or 'val')
            num_negatives: Number of negative samples per positive (default: 99)
            seed: Random seed
        """
        self.ratings_df = ratings_df
        self.graph_data = graph_data
        self.split = split
        self.num_negatives = num_negatives
        self.rng = np.random.RandomState(seed)

        # Get split ratings
        self.split_ratings = ratings_df[ratings_df["split"] == split].copy()

        # Build user interaction history (ALL splits - prevents data leakage)
        print(f"  Building user history from ALL splits for proper evaluation...")
        self.user_all_items = defaultdict(set)
        for _, row in ratings_df.iterrows():
            self.user_all_items[row["user_id"]].add(row["movie_id"])

        total_interactions = sum(len(items) for items in self.user_all_items.values())
        print(
            f"  User history: {total_interactions} total interactions across {len(self.user_all_items)} users"
        )

        # Get all movies
        self.all_movies = list(ratings_df["movie_id"].unique())

        # Pre-sample negatives for efficiency
        print(f"  Pre-sampling {num_negatives} negatives per test case...")
        self.negatives_cache = {}
        for i, (_, row) in enumerate(self.split_ratings.iterrows()):
            user_id = row["user_id"]
            pos_item = row["movie_id"]

            # Sample negatives for this user
            neg_items = self._sample_negatives(user_id)
            self.negatives_cache[i] = neg_items  # Use sequential index

        print(f"ProperEvalDataset ({split}): {len(self.split_ratings)} test cases")
        print(f"  Protocol: 1-vs-{num_negatives} evaluation (proper RecSys evaluation)")

    def _sample_negatives(self, user_id):
        """Sample N negative items for user (not in their history)."""
        user_history = self.user_all_items[user_id]

        # Get candidate negatives (movies not in user's history)
        candidates = [m for m in self.all_movies if m not in user_history]

        if len(candidates) < self.num_negatives:
            # Edge case: user has interacted with most movies
            # Sample with replacement
            neg_items = self.rng.choice(
                candidates, size=self.num_negatives, replace=True
            ).tolist()
        else:
            # Normal case: sample without replacement
            neg_items = self.rng.choice(
                candidates, size=self.num_negatives, replace=False
            ).tolist()

        return neg_items

    def __len__(self):
        return len(self.split_ratings)

    def __getitem__(self, idx):
        """
        Return a test case with 1 positive and N negatives.

        Returns:
            dict with:
                - user_id: User ID
                - pos_movie_id: Positive item ID
                - neg_movie_ids: List of N negative item IDs
        """
        row = self.split_ratings.iloc[idx]
        user_id = row["user_id"]
        pos_movie_id = row["movie_id"]

        # Get pre-sampled negatives
        neg_movie_ids = self.negatives_cache[idx]

        return {
            "user_id": user_id,
            "pos_movie_id": pos_movie_id,
            "neg_movie_ids": neg_movie_ids,  # List of N negatives
        }
