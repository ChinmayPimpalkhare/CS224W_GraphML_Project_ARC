# Getting Started (GraphFlix)

This guide gets you from zero → runnable baselines (**Most‑Popular** and **LightGCN**) quickly and reproducibly.

> **TL;DR Fast Path**
>
> 1. Create a Python venv and install PyTorch + PyG.
> 2. **Download the processed ML‑1M bundle** from OneDrive and unzip into `data/processed/ml1m/`.
> 3. Run `scripts/run_mostpop.py` and `scripts/run_lightgcn.py`.

---

## Prerequisites

* **OS:** Linux or macOS (Windows: use **WSL2**).
* **Python:** 3.10 (recommended).
* **GPU:** NVIDIA with CUDA‑12.1 capable driver (optional, recommended).
* **Git:** installed and configured (HTTPS or SSH).
* **Disk:** ~2–4 GB free for data + artifacts.

---

## 1) Clone the repository

```bash
# HTTPS (easiest)
git clone https://github.com/ChinmayPimpalkhare/CS224W_GraphML_Project_ARC.git
cd CS224W_GraphML_Project_ARC
```

> If SSH gives `Permission denied (publickey)`, either add your SSH key to GitHub or use HTTPS.

---

## 2) Python environment (pip + venv; GPU‑ready)

We use **pip + venv** (not conda) to avoid known CUDA/MKL issues.

```bash
python3 -m venv ~/venvs/graphflix-pip
source ~/venvs/graphflix-pip/bin/activate
python -m pip install --upgrade pip
```

### Install PyTorch

* **GPU build (CUDA 12.1):**

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

* **CPU‑only:**

```bash
pip install torch torchvision torchaudio
```

**Verify:**

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

### Install PyG (PyTorch Geometric)

Use wheels matching your Torch/cu121:

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### Project dependencies (and optional hooks)

```bash
pip install -r requirements.txt
# Optional local formatting/lint hooks
pip install pre-commit
python -m pre_commit install
```

> If `pre-commit` modifies files, `git add -A` and re-run; once hooks pass, commit.

---

## 3) Quickstart (recommended): Use the processed bundle

**Download** (Stanford OneDrive, team‑shared):

* [**ml1m_processed_20251109.zip**](https://office365stanford-my.sharepoint.com/:u:/r/personal/amulyasp_stanford_edu/Documents/CS224w-project/ml1m_processed_20251109.zip?csf=1&web=1&e=XB8YiG)

**Unzip into repo root**:

```bash
mkdir -p data/processed/ml1m
unzip ~/Downloads/ml1m_processed_20251109.zip -d .
# If it unzips into a nested folder, move contents into data/processed/ml1m/
```

**(Optional) Verify checksum** (if you have one):

```bash
md5sum -c ml1m_processed.md5     # or: md5 ml1m_processed_20251109.zip (macOS)
```

**You should now have (typical):**

```
data/processed/ml1m/
  actors.csv                    movies.csv          ratings.csv          users.csv
  directors.csv                 movies_enriched.csv movie_actor_edges.csv
  genres.csv                    mappings/{user_index.csv,movie_index.csv}
  ratings_reindexed.csv         movie_actor_edges_reindexed.csv
  movie_director_edges*.csv     movie_genre_edges*.csv
  splits/ratings_split_reindexed.csv
  graph_stats.json              graph_pyg.pt        (~32 MB)
  runs/{mostpop_metrics.json, lightgcn_metrics.json}
  lightgcn_best.pt             lightgcn_emb.pt
```

> If `graph_pyg.pt` is missing, you can build it in step 6.

---

## 4) Sanity checks

Run from the repo root:

```bash
# If you ever see "ModuleNotFoundError: No module named 'src'"
export PYTHONPATH=$PWD:$PYTHONPATH

# Inspect the PyG graph (node/edge counts, types)
python scripts/inspect_graph_pyg.py
```

Expected counts (ML‑1M):

* nodes: users=6040, movies=3883, actors=8569, directors=2002, genres=19
* train edges: rates≈988,129; stars=18,008; directed_by=3,875; has_genre=8,875

---

## 5) Run baselines

### A) Most‑Popular

```bash
python scripts/run_mostpop.py --root data/processed/ml1m
```

###  B) LightGCN (LGConv + BPR)

> **Recommended:** dot-product scoring (no cosine normalization).  
> In our ablations on ML‑1M temporal LOO, cosine underperformed dot by ~15–25% NDCG@10.  
> LightGCN + BPR benefits from vector **magnitudes** (e.g., popular items learn larger norms); cosine discards that signal.

#### Quick recipes

**Fast sanity check (GPU or CPU)**
```bash
python scripts/run_lightgcn.py \
  --root data/processed/ml1m \
  --epochs 40 --layers 3 --dim 128 \
  --batch 4096 --lr 0.003 --neg 10 \
  --seed 42 --device cuda   # omit --device cuda for CPU
````

**Strong baseline (GPU) — our best config**

```bash
python scripts/run_lightgcn.py \
  --root data/processed/ml1m \
  --epochs 100 --layers 3 --dim 256 \
  --batch 4096 --lr 0.001 --neg 20 \
  --seed 42 --device cuda
```

**CPU‑friendly (smaller model)**

```bash
python scripts/run_lightgcn.py \
  --root data/processed/ml1m \
  --epochs 40 --layers 2 --dim 64 \
  --batch 2048 --lr 0.003 --neg 10 \
  --seed 42
```

#### Notes & expectations

* We **do not pass `--use_cosine`** (dot‑product is the default and performed best).
* The script automatically saves `lightgcn_best.pt` and uses it for final test evaluation.
* With the strong baseline, you should see ≈ **R@10: 0.08–0.09** and **NDCG@10: 0.040–0.042** on TEST in our protocol.
* Ensure your graph was built with **train‑only user→movie edges** (see §6 Regenerate artifacts).
* During eval, we mask each user’s **training positives** from the ranking list (implemented in the script).

#### If metrics look too low

* Check you’re on **dot‑product** (no `--use_cosine`).
* Train long enough: **epochs ≥ 80–100** for D=256, L=3, neg=20.
* Use **negatives≥10** (we found **20** slightly better at high capacity).
* Verify the pipeline order (§6): `split_temporal.py` → `reindex_ids.py` → `build_graph_pyg.py --use_train_only`.
* Confirm that `data/processed/ml1m/splits/ratings_split_reindexed.csv` exists and looks sane.

---
## 6) (Optional) Regenerate artifacts from CSVs

If you need to rebuild the evaluation split and graph locally, **follow this exact order**:

> **Important:** `split_temporal.py` must receive the **original‑ID** ratings file `ratings.csv` (columns: `user_id, movie_id, rating, timestamp`).
> Do **not** pass `ratings_reindexed.csv` (it has `user_idx, movie_idx`)—that’s what caused the `KeyError: 'user_id'`.

```bash
# A) Create temporal LOO split (deterministic, no seed needed)
python scripts/split_temporal.py \
  --ratings data/processed/ml1m/ratings.csv \
  --out     data/processed/ml1m/splits
# → writes data/processed/ml1m/splits/ratings_split.csv (user_id, movie_id, ...)

# B) Reindex to 0..N-1 AND reindex the split
python scripts/reindex_ids.py
# → writes data/processed/ml1m/ratings_reindexed.csv
# → writes data/processed/ml1m/splits/ratings_split_reindexed.csv

python scripts/validate_reindexed.py   # quick consistency checks (optional)

# C) Build the PyG HeteroData graph using TRAIN-ONLY user→movie edges
python scripts/build_graph_pyg.py --root data/processed/ml1m --use_train_only
# → writes data/processed/ml1m/graph_pyg.pt
```

You can then run baselines as in §5.

---

## 7) Troubleshooting

**Missing**: `data/processed/ml1m/splits/ratings_split_reindexed.csv`

* Run the regeneration sequence in §6 (A → B → C).
  Using `ratings_reindexed.csv` with `split_temporal.py` will fail with `KeyError: 'user_id'`.

**`split_temporal.py: error: unrecognized arguments: --seed`**

* The script doesn’t take `--seed` (temporal LOO is deterministic). For exact timestamp ties, it sorts stably by `(user_id, timestamp, movie_id)`.

**`ModuleNotFoundError: No module named 'src'`**

* Run from repo root, and set `export PYTHONPATH=$PWD:$PYTHONPATH`.

**Pre‑commit fails/absent**

* `pip install pre-commit` and then `python -m pre_commit run --all-files`. Re‑add files and retry.

**Git push blocked (email privacy GH007)**

* Configure identity to a GitHub‑verified or no‑reply email:

```bash
git config --global user.name  "Your Name"
git config --global user.email "yourid@users.noreply.github.com"
```

---

## 8) Recommended Git workflow (short version)

```bash
# Sync main
git checkout main && git fetch --prune && git pull --ff-only

# Work on a feature branch
git checkout -b baseline/lightgcn-tuning

# Format / lint
python -m pre_commit run --all-files

# Commit & push
git add -A
git commit -m "baseline: LightGCN tuning (neg=10, dim=128, cosine)"
git push -u origin HEAD

# Open PR → review → squash & merge
```

---

## 9) Repo layout

```
CS224W_GraphML_Project_ARC/
  configs/
  data/
    processed/ml1m/          ← all processed CSVs, splits, graph files (git-ignored)
  docs/
    GETTING_STARTED.md
    GPU_SETUP.md
  scripts/
    run_mostpop.py
    run_lightgcn.py
    split_temporal.py         ← builds splits/ratings_split.csv from ratings.csv (original IDs)
    reindex_ids.py            ← writes splits/ratings_split_reindexed.csv (0..N-1 IDs)
    build_graph_pyg.py        ← builds graph_pyg.pt (train-only user→movie)
    validate_* / analyze_* / run_* helpers
  src/graphflix/...
  requirements*.txt
  pyproject.toml
```

---

### Why we split on `ratings.csv` and only then reindex

* Splitting uses **original IDs** (`user_id`, `movie_id`) and timestamps to pick each user’s last two interactions (val/test) → deterministic, human‑checkable.
* Reindexing (`reindex_ids.py`) converts those splits and all edges to **0..N‑1** for efficient tensor indexing in PyTorch/PyG.
* The graph builder then includes **only TRAIN** user→movie edges to prevent message‑passing leakage.
