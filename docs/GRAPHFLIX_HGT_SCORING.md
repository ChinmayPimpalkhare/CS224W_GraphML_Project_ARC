# GraphFlixHGT (Scoring‑Layer Metadata Bias)

This document explains how to run the **GraphFlixHGT (scoring)** variant in this repo:

* It uses a **heterogeneous HGT encoder** (`HGTConv`) on `graph_pyg.pt`.
* It uses **half‑life user profiles** and **movie metadata embeddings** from a precompute step.
* It injects the **metadata bias at the scoring layer**
  (adds `b_meta(u, j)` to the user–movie score) instead of inside the attention logits.

This variant is completely **additive**:

* It does **not** modify any of the baseline scripts (`run_lightgcn.py`, etc.). 
* It only **reads** the existing split + graph files and writes its own artifacts under `data/processed/ml1m/runs/`.

---

## Files / What’s new

These two scripts live under `scripts/`:

* `scripts/precompute_profilesHGT.py` 

  * Builds **movie metadata embeddings** $(\phi(j))$ using random‑but‑fixed director/actor/genre embeddings.
  * Builds **half‑life user profiles** $(p_{\text{train}}(u),\ p_{\text{val}}(u),\ p_{\text{test}}(u))$
 using the temporal LOO split.

* `scripts/run_graphflixHGT.py` 

  * Defines `GraphFlixModel` (HGTConv encoder + metadata projection + bias params).
  * Trains with **BPR** on the train split using the heterograph in `graph_pyg.pt`.
  * Adds **metadata bias at scoring time**:
    $s(u,j) = \langle z_u, z_j\rangle + b_{\text{meta}}(u,j)$
.
  * Evaluates with full‑catalog ranking (masking train positives) on **val**/**test**.

These scripts are **standalone**—they don’t import or overwrite any other project code. They only rely on:

* the preprocessed ML‑1M CSVs,
* the temporal split,
* and the heterograph `graph_pyg.pt` that you already build for LightGCN. 

---

## 1. Prerequisites

Follow `docs/GETTING_STARTED.md` first so you have:

1. A working Python env with PyTorch + PyG installed. 

2. Processed ML‑1M CSVs under:

   ```text
   data/processed/ml1m/
     ratings.csv
     users.csv
     movies.csv
     actors.csv
     directors.csv
     genres.csv
     movie_actor_edges.csv
     movie_director_edges.csv
     movie_genre_edges.csv
   ```

3. Temporal leave‑one‑out split and reindexed IDs:

   ```bash
   python scripts/split_temporal.py \
     --ratings data/processed/ml1m/ratings.csv \
     --out     data/processed/ml1m/splits

   python scripts/reindex_ids.py
   ```

   This should create `data/processed/ml1m/splits/ratings_split_reindexed.csv`.

4. Heterogeneous PyG graph (train‑only user→movie edges):

   ```bash
   python scripts/build_graph_pyg.py \
     --root data/processed/ml1m \
     --use_train_only
   ```

   This writes `data/processed/ml1m/graph_pyg.pt` with node types
   `{user, movie, actor, director, genre}` and edge types `{rates, stars, directed_by, has_genre}` + reverses.

> **Important:** GraphFlixHGT assumes the same temporal LOO protocol and graph as LightGCN, so the results are directly comparable. 

---

## 2. Step 1 – Precompute metadata and half‑life profiles

This step builds:

* `phi[j]` – a **metadata embedding for each movie** (j),
* `profiles_train[u], profiles_val[u], profiles_test[u]` – **time‑aware profiles** for each user (u).

Run from repo root:

```bash
python scripts/precompute_profilesHGT.py \
  --root data/processed/ml1m \
  --tau 150 \
  --d_dir 64 --d_actor 64 --d_genre 32
```

Key flags:

* `--root` – processed ML‑1M directory (default: `data/processed/ml1m`).
* `--tau` – **half‑life in days**; here `150` ⇒ “interactions 150 days ago count half as much”.
* `--d_dir`, `--d_actor`, `--d_genre` – dimensions for director/actor/genre embeddings. Total `d_phi = d_dir + d_actor + d_genre`.

Outputs (single `.pt` file):

```text
data/processed/ml1m/half_life_profiles_tau150.pt
  - phi:            [num_movies, d_phi]         # metadata embeddings ϕ(j)
  - profiles_train: [num_users, d_phi]         # p_train(u)
  - profiles_val:   [num_users, d_phi]         # p_val(u)
  - profiles_test:  [num_users, d_phi]         # p_test(u)
  - dims:           {d_dir, d_actor, d_genre, d_phi}
  - tau_days:       150.0
  - p_global:       [d_phi]                    # global popularity profile
  - meta:           {root, num_users, num_movies}
```

You can re‑run this with different `--tau`; each run writes a new file with the corresponding suffix, so it **does not overwrite** previous ones.

---

## 3. Step 2 – Train and evaluate GraphFlixHGT

Once profiles are precomputed, you can train the HGT scoring model.

Basic command:

```bash
python scripts/run_graphflixHGT.py \
  --root data/processed/ml1m \
  --profiles_tau 150 \
  --epochs 100 \
  --dim 128 \
  --layers 2 \
  --heads 4 \
  --batch 4096 \
  --neg 20 \
  --lr 1e-3 \
  --device cuda
```

Important flags (see `--help` for the full list):

* `--root` – same as above.
* `--profiles_tau` – must match the `--tau` you used in `precompute_profilesHGT.py` so the script loads the correct `half_life_profiles_tau{τ}.pt`.
* `--dim` – hidden dimension for all node embeddings.
* `--layers` – number of `HGTConv` layers.
* `--heads` – attention heads per HGT layer.
* `--batch` – train batch size (number of positive edges per step).
* `--neg` – negatives per positive for BPR.
* `--lr` – learning rate.
* `--device` – `cuda` or `cpu`.

During training you’ll see logs like:

```text
epoch 079 | loss 0.1652
epoch 080 | loss 0.1643 | val_ndcg@10=0.0364
...
epoch 100 | loss 0.1478 | val_ndcg@10=0.0363

Best val NDCG@10: 0.0364 at epoch 80
Loaded best checkpoint: epoch 80, val_ndcg@10=0.0364
VAL : {'recall@10': 0.0783, 'ndcg@10': 0.0364, 'recall@20': 0.1386, 'ndcg@20': 0.0516}
TEST: {'recall@10': 0.0667, 'ndcg@10': 0.0328, 'recall@20': 0.1242, 'ndcg@20': 0.0471}
Saved: data/processed/ml1m/runs/graphflix_HGT_v1_metrics.json
```

Artifacts:

* A **best checkpoint** (by val NDCG@10) saved under `data/processed/ml1m/runs/`
  (see `run_graphflixHGT.py` for the exact filename). 
* A JSON with **val/test Recall@K / NDCG@K** and the best epoch, e.g.:

  ```text
  data/processed/ml1m/runs/graphflix_HGT_v1_metrics.json
  ```

These are separate from the LightGCN artifacts (`lightgcn_best.pt`, `lightgcn_metrics.json`), so you won’t overwrite your baseline runs.

---

## 4. What the model is actually doing (short version)

If you want the 10‑second mental model when you read the code:

1. **Hetero HGT encoder**

   * `GraphFlixModel` builds per‑type embeddings for `{user, movie, actor, director, genre}` and runs `L` layers of `HGTConv` over `graph_pyg.pt`.
   * Movies get an extra feature: a linear projection of `ϕ(j)` added to their learned embedding.

2. **Precomputed metadata**

   * `phi[j]` and user profiles `p(u)` are loaded from `half_life_profiles_tau{τ}.pt`.
   * Profiles use **half‑life decay + rating weights**, computed once from the train split.

3. **Scoring with metadata bias**

* Encoder scores:  
  $s_{\text{enc}}(u,j) = \langle z_u, z_j \rangle$

* Metadata bias (stabilized):  
  $b_{\text{meta}}(u,j) = \beta \,\tanh\big(\mathrm{LN}(p(u))^\top W\,\mathrm{LN}(\phi(j))\big)$

* Final score used in BPR + evaluation:  
  $s(u,j) = s_{\text{enc}}(u,j) + b_{\text{meta}}(u,j)$


   * This is computed **batch‑wise** during training, and as a full matrix `B_meta` during eval.

4. **Evaluation protocol**

   * Same as `run_lightgcn.py`:

     * temporal LOO (`ratings_split_reindexed.csv`),
     * train on **train** edges only,
     * full‑catalog ranking with **train positives masked** per user,
     * metrics: Recall@10/20, NDCG@10/20.

For a deeper dive (half‑life math, LN/tanh/β, how to move the bias into attention logits instead of scoring), see the longer final project notes **“Half_Life_Profiles_and_Metadata_Aware_Graphormer_Attention.pdf”**.

---

## 5. Integration / “Will this overwrite anything?”

Short answer: **no** – not by itself.

* The HGT scripts **only read**:

  * `data/processed/ml1m/…` CSVs,
  * `splits/ratings_split_reindexed.csv`,
  * `graph_pyg.pt`,
  * and the `half_life_profiles_tau{τ}.pt` file they themselves create.
* They **write** only into:

  * `data/processed/ml1m/half_life_profiles_tau*.pt`,
  * `data/processed/ml1m/runs/graphflix_*` (checkpoint + metrics).

If you want absolute isolation, you can also:

* Point `--root` to another directory (e.g. `data/processed/ml1m_hgt`).
* Or rename the output filenames inside `run_graphflixHGT.py` to something like `graphflix_HGT_v1_best.pt` / `graphflix_HGT_v1_metrics.json` (purely cosmetic, the README doesn’t assume a specific name).

---
