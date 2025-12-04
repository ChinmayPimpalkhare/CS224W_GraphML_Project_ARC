# GraphFlix Explained: A Deep Dive

This document provides a detailed explanation of the GraphFlix model, a graph-based recommendation system designed for movie recommendations. We will explore its core components, with a special focus on how it processes graph information, handles temporal signals, and integrates user preferences with item metadata.

## 1. High-Level Architecture

GraphFlix is a recommendation model that leverages a **Graph Transformer (Graphormer)** to learn from a heterogeneous graph of users, movies, and metadata (like actors, directors, and genres).

Its key innovation lies in how it injects a **metadata-aware bias** directly into the attention mechanism of the graph transformer. This bias is conditioned on two pre-computed elements:

1.  **Half-Life User Profiles (`p(u)`):** A vector for each user that summarizes their historical preferences, giving more weight to recent and highly-rated interactions.
2.  **Movie Metadata Embeddings (`φ(j)`):** A vector for each movie derived from its associated metadata (genres, actors, etc.).

The model combines these two elements to predict a user's affinity for a movie's characteristics, and uses this signal to guide the message-passing process within the graph.

## 2. Time Decay: The Half-Life User Profile `p(u)`

A user's taste evolves. GraphFlix captures this by computing a user profile `p(u)` that explicitly models time decay. Instead of treating all past ratings equally, it prioritizes more recent interactions.

The profile `p(u)` is a weighted average of the metadata embeddings (`φ`) of the movies `j` that user `u` has rated:

**p(u) = Σ_j w(u,j) · φ(j)**

The weight `w(u,j)` is a product of two components: a recency weight and a rating weight.

### Recency Weight

This weight is calculated using an exponential decay function, where `τ` is the "half-life" in days. A more recent rating will have a weight closer to 1, while older ratings decay towards 0.

`w_recency = exp(-Δt / τ)`

-   `Δt`: The time elapsed between the user's *most recent* rating and the rating for movie `j`.
-   `τ`: The half-life period. A `τ` of 20 days means a rating's influence is halved every 20 days.

Here is the implementation from `scripts/compute_user_profiles.py`:

```python
def compute_recency_weights(timestamps, half_life_days=20):
    """
    Compute exponential decay weights based on time since most recent interaction.
    """
    # Most recent timestamp is the reference point
    t_max = timestamps.max()

    # Compute time differences in seconds, convert to days
    delta_t_days = (t_max - timestamps) / (60 * 60 * 24)

    # Exponential decay
    tau = half_life_days
    weights = np.exp(-delta_t_days / tau)

    return weights
```

### Rating Weight

This weight emphasizes movies that the user rated more highly than their personal average. It captures the strength of their preference.

`w_rating = max(0, r - r_bar_u)`

-   `r`: The user's rating for movie `j`.
-   `r_bar_u`: The user's average rating across all movies.

```python
def compute_rating_weights(ratings, user_mean):
    """
    Compute rating-based weights emphasizing above-average ratings.
    """
    weights = np.maximum(0, ratings - user_mean)

    # Handle edge case: user rated everything the same
    if weights.sum() == 0:
        weights = np.ones_like(ratings)

    return weights
```

By combining these weights, the user profile `p(u)` becomes a rich summary of a user's current tastes, emphasizing the movies they recently enjoyed the most.

## 3. Graph Message Aggregation & Edge Processing

GraphFlix uses a **Graphormer** encoder to perform message passing on the graph. Graphormer is a powerful Graph Transformer architecture that learns node representations by attending to their neighbors.

The unique aspect of GraphFlix is not in changing the Graphormer architecture itself, but in **influencing its attention mechanism**. This is where "edge processing" becomes highly specialized.

### The Metadata Bias `b_meta(u,j)`

Instead of treating all connections equally, GraphFlix computes a bias term `b_meta(u,j)` for each potential user-movie interaction within a subgraph. This bias represents a pre-calculated affinity score based on the user's profile and the movie's metadata.

The calculation follows these steps, as seen in `src/graphflix/models/graphflix.py`:

1.  **Lookup:** For a user `u` and a set of movies `j`, retrieve the user's profile `p(u)` and the movies' metadata embeddings `φ(j)`.
2.  **Normalize:** Apply Layer Normalization to both `p(u)` and `φ(j)` to stabilize training.
3.  **Compute Score:** Project the normalized profile `p̂(u)` through a learned linear transformation `W` and compute the dot product with the normalized metadata `φ̂(j)`.
    `s(u,j) = p̂(u)^T W φ̂(j)`
4.  **Bound and Scale:** Pass the score through a `tanh` function to bound it between [-1, 1] and scale it by a learnable parameter `β`.

**b_meta(u,j) = β * tanh(s(u,j))**

Here is the implementation:

```python
# from src/graphflix/models/graphflix.py

def compute_metadata_bias(self, user_ids, movie_ids):
    # ... (lookup P and Phi) ...

    # Step 3: Normalize embeddings
    p_u_norm = self.ln_profile(P_batch)
    phi_j_norm = self.ln_metadata(Phi_batch)

    # Step 4: Compute metadata scores
    p_u_transformed = p_u_norm @ self.W.weight.t()
    scores = torch.einsum("bd,bmd->bm", p_u_transformed, phi_j_norm)

    # Step 5: Bound and scale the bias
    b_meta = self.beta * torch.tanh(scores)

    return b_meta
```

### Injecting Bias into Attention

This computed bias `b_meta` is then **added directly to the attention logits** inside the Graphormer encoder.

Crucially, this is done with **"user-row-only" injection**. In the attention matrix of the subgraph, the bias is only added to the rows corresponding to the central `user` node.

This means when the user node is calculating its new representation, the attention it pays to different `movie` nodes is biased by their pre-computed metadata affinity. A positive bias encourages the user node to attend more to a movie, while a negative bias discourages it.

```python
# from src/graphflix/models/graphflix.py

def inject_metadata_bias(self, attn_bias, batch_info, b_meta):
    # ...
    if self.user_row_only:
        # Only modify the user's attention row
        for b in range(batch_size):
            i_u = user_indices[b]  # Index of the user node in the subgraph
            j_m = movie_indices[b] # Indices of the movie nodes
            # ...
            attn_bias[b, i_u, valid_indices] += b_meta[b, :num_movies]
    # ...
```

This mechanism acts as a powerful inductive bias, guiding the model to favor items that align with a user's temporally-aware preferences *before* the full message-passing even occurs.

## 4. Summary: The Full Forward Pass

The entire process can be summarized in the following steps:

1.  **Subgraph Sampling:** For a given user-movie pair, sample a k-hop subgraph from the full heterogeneous graph.
2.  **Lookup Profiles & Metadata:** Retrieve the pre-computed profile `p(u)` for the user and metadata `φ(j)` for all movies in the subgraph.
3.  **Compute Metadata Bias:** Calculate `b_meta` for the user and movies as described above.
4.  **Inject Bias:** Add `b_meta` to the Graphormer's attention bias tensor (user-row-only).
5.  **Graphormer Encoding:** Pass the subgraph nodes and the modified attention bias through the Graphormer encoder to get final node embeddings.
6.  **Score Prediction:** The final recommendation score is typically the dot product of the user's final embedding and the movie's final embedding.
7.  **Optimization:** The model is trained using a ranking loss like BPR (Bayesian Personalized Ranking), which aims to score observed interactions higher than unobserved ones.

By combining a powerful graph transformer with an explicit, temporally-aware metadata bias, GraphFlix creates a nuanced and effective recommendation system.

## 5. Data Partitioning for Faster Iteration (10% & 25% Subsets)

Training graph-based models on large datasets can be computationally expensive and time-consuming. To facilitate faster experimentation, debugging, and hyperparameter tuning, we created smaller subsets of the full dataset, specifically 10% and 25% samples.

### The Partitioning Strategy

A naive random sampling of individual ratings would destroy the integrity of user histories, making it impossible to build meaningful user profiles. Therefore, we adopted a **user-centric sampling strategy**:

1.  **Sample Users:** A fixed percentage (10% or 25%) of unique users were randomly selected from the dataset.
2.  **Keep Full Histories:** For each selected user, we kept *all* of their historical rating interactions.
3.  **Filter Entities:** The `movies` table and other metadata tables (actors, directors, genres) were then filtered to only include entities that appeared in the interaction history of the sampled users.

This approach ensures that each user in the subsample has a complete and realistic interaction history, preserving the data's structural and temporal properties. The implementation for this can be found in `scripts/create_subsample.py`.

```python
# from scripts/create_subsample.py

def create_subsample(data_dir, sample_ratio=0.25, seed=42):
    # ...
    # Sample users
    all_users = ratings_df["user_id"].unique()
    n_sample = int(len(all_users) * sample_ratio)
    sampled_users = np.random.choice(all_users, size=n_sample, replace=False)

    # Filter ratings to sampled users
    sampled_ratings = ratings_df[ratings_df["user_id"].isin(sampled_users)].copy()
    # ...
```

### The "Out-of-Index" Problem and Re-indexing

This sampling strategy introduces a critical technical challenge: **discontiguous IDs**.

After sampling, the `user_id` and `movie_id` fields are no longer a contiguous sequence from `0` to `N-1`. For example, the user IDs in the 10% sample might be `[5, 12, 88, 103, ...]`.

Embedding layers in neural networks (like `torch.nn.Embedding`) require dense, 0-indexed integer inputs. Attempting to use the original, sparse IDs would immediately result in `IndexError: index out of range` because the model would try to access an embedding at an index that doesn't exist in a smaller, 0-indexed embedding table.

#### The Solution: Re-indexing

To solve this, we implemented a **re-indexing** step, found in `scripts/reindex_ids.py`. This script is run on the *full* dataset before any subsampling occurs.

1.  **Create Mappings:** It scans all users and movies and creates a mapping from the original, sparse ID (e.g., `user_id: 88`) to a new, dense, 0-based index (e.g., `user_idx: 2`).
2.  **Save Mappings:** These mappings are saved to files (e.g., `mappings/user_index.csv`) to ensure reproducibility and allow for debugging.
3.  **Apply Mappings:** The script then replaces the original ID columns in all data files (`ratings.csv`, edge files, etc.) with the new `_idx` columns.

```python
# from scripts/reindex_ids.py

def make_map(df, col, idx_col):
    uniq = sorted(df[col].dropna().astype("int64").unique().tolist())
    return pd.DataFrame({col: uniq, idx_col: range(len(uniq))})

map_user = make_map(users, "user_id", "user_idx")
map_movie = make_map(movies, "movie_id", "movie_idx")
```

When the `create_subsample.py` script runs, it works with the re-indexed data and ensures that all generated files for the 10% or 25% subsets remain consistent. This guarantees that when the model trains on a smaller data partition, it is provided with dense, 0-indexed `user_idx` and `movie_idx` values, preventing any out-of-index errors.