# Getting Started (GraphFlix)

This guide gets you from zero → runnable baselines (**Most‑Popular** and **LightGCN**) as quickly and reproducibly as possible.

> **TL;DR (Fast Path)**: Create a Python venv, install PyTorch + PyG, **download the processed ML‑1M bundle** from OneDrive, unzip into `data/processed/ml1m/`, then run `scripts/run_mostpop.py` and `scripts/run_lightgcn.py`.

---

## 0) Prerequisites

* **OS**: Linux or macOS (Windows users: use **WSL2**).
* **Python**: 3.10 (recommended).
* **GPU (optional but recommended)**: NVIDIA with CUDA 12.1 capable driver.
* **Git**: installed and configured (HTTPS or SSH).
* **Disk**: ~2–4 GB free for data + artifacts.

---

## 1) Clone the repository

```bash
# HTTPS (easiest)
git clone https://github.com/ChinmayPimpalkhare/CS224W_GraphML_Project_ARC.git
cd CS224W_GraphML_Project_ARC
```

> If you prefer SSH and see `Permission denied (publickey)`, add your SSH key to GitHub or switch to HTTPS.

---

## 2) Python environment (GPU‑ready, pip venv)

> We recommend **pip + venv** because it avoids Conda/Intel MKL issues (e.g., `iJIT_NotifyEvent`), and it matched our successful GPU setup.

```bash
python3 -m venv ~/venvs/graphflix-pip
source ~/venvs/graphflix-pip/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Install PyTorch (CUDA 12.1 build) + optional CPU fallback

* **GPU build (recommended)**:

  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```

* **CPU‑only fallback**:

  ```bash
  pip install torch torchvision torchaudio
  ```

**Verify Torch**:

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

> Use wheels matching Torch **2.4/2.5 + cu121**. These worked for us:

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

### Project dependencies + dev tools

```bash
pip install -r requirements.txt
# Optional: local hooks for formatting/linting
pip install pre-commit
python -m pre_commit install
```

> If you later run `pre-commit` and it modifies files, just `git add -A` and re-run; once hooks pass, commit.

---

## 3) **Quickstart (Recommended): Use processed dataset bundle**

This keeps everyone on identical data and avoids TMDb API rate/variance.

**Download** (Stanford OneDrive, team‑shared):

* **ml1m_processed_20251109.zip**
  [https://office365stanford-my.sharepoint.com/:u:/r/personal/amulyasp_stanford_edu/Documents/CS224w-project/ml1m_processed_20251109.zip?csf=1&web=1&e=uRR0TU](https://office365stanford-my.sharepoint.com/:u:/r/personal/amulyasp_stanford_edu/Documents/CS224w-project/ml1m_processed_20251109.zip?csf=1&web=1&e=uRR0TU)

**Unzip into the repo**:

```bash
mkdir -p data/processed/ml1m
unzip ~/Downloads/ml1m_processed_20251109.zip -d .
# If it unzips into a nested folder, move contents so they live in data/processed/ml1m/
```

**(Optional) Verify checksum** (if `ml1m_processed.md5` was shared alongside the zip):

```bash
md5sum -c ml1m_processed.md5    # or: md5 ml1m_processed_20251109.zip (macOS)
```

**You should now have (approximate list)**:

```
data/processed/ml1m/
  actors.csv
  directors.csv
  genres.csv
  movies.csv
  movies_enriched.csv
  ratings.csv
  users.csv
  movie_actor_edges.csv
  movie_director_edges.csv
  movie_genre_edges.csv

  mappings/
    user_index.csv
    movie_index.csv

  ratings_reindexed.csv
  movie_actor_edges_reindexed.csv
  movie_director_edges_reindexed.csv
  movie_genre_edges_reindexed.csv

  splits/
    ratings_split_reindexed.csv        # may be regenerated if omitted

  graph_stats.json
  graph_pyg.pt                         # ~32 MB (PyG HeteroData, uses TRAIN edges for user→movie)
  runs/
    mostpop_metrics.json
    lightgcn_metrics.json
  lightgcn_best.pt
  lightgcn_emb.pt
```

> If `graph_pyg.pt` is missing, you can build it later via `scripts/build_graph_pyg.py` once data is present.

---

## 4) Sanity checks (optional but useful)

Make sure you’re running commands **from the repo root**.

```bash
# If you ever see "ModuleNotFoundError: No module named 'src'", set:
export PYTHONPATH=$PWD:$PYTHONPATH

# Inspect the PyG graph (node/edge counts, types)
python scripts/inspect_graph_pyg.py
```

Expected node/edge counts (ML‑1M):

* nodes: users=6040, movies=3883, actors=8569, directors=2002, genres=19
* edges (train graph): rates≈988,129; stars=18,008; directed_by=3,875; has_genre=8,875

---

## 5) Run baselines

### A) Most‑Popular

```bash
python scripts/run_mostpop.py --root data/processed/ml1m
```

You’ll see VAL/TEST **Recall@10/20** and **NDCG@10/20**.
On our temporal LOO protocol, Most‑Popular Recall@10 is typically ~**0.04** (it’s a strict, realistic setup).

### B) LightGCN (LGConv + BPR)

```bash
# Example config (GPU):
python scripts/run_lightgcn.py \
  --root data/processed/ml1m \
  --epochs 40 \
  --layers 3 \
  --dim 128 \
  --batch 4096 \
  --lr 0.003 \
  --neg 10 \
  --use_cosine \
  --seed 42 \
  --device cuda
```

Artifacts go to `data/processed/ml1m/runs/` and `data/processed/ml1m/lightgcn_*.pt`.
With proper training, LightGCN should **significantly beat Most‑Popular** on Recall/NDCG (our current runs: Recall@10 ~0.06–0.10 depending on hyperparams).

> Tip: For longer/tuned runs, see `scripts/run_grid_search.sh` and `scripts/analyze_grid_search.py`.

---

## 6) (Optional) Build the dataset from scratch

If you need to regenerate everything locally:

```bash
# 1) (Optional) Download raw ML-1M if your scripts require it
python scripts/download_movielens.py --out data/raw/ml1m

# 2) Temporal LOO split (1 val + 1 test per user; rest train)
python scripts/split_temporal.py \
  --ratings data/processed/ml1m/ratings.csv \
  --out     data/processed/ml1m/splits

# 3) Reindex IDs (user/movie → 0..N-1) and remap edges
python scripts/reindex_ids.py
python scripts/validate_reindexed.py   # checks 0-based, contiguous

# 4) (PyG) Build graph (user→movie uses TRAIN‑only edges to avoid leakage)
python scripts/build_graph_pyg.py --root data/processed/ml1m --use_train_only

# 5) Inspect
python scripts/inspect_graph_pyg.py
```

> **TMDb enrichment** (actors/directors/genres) requires a `.env` with `TMDB_API_KEY` and adds significant runtime + rate‑limit handling. Most teammates should **skip** this and use the shared processed bundle instead.

---

## 7) Troubleshooting

* **`ModuleNotFoundError: No module named 'src'`**
  From repo root: `export PYTHONPATH=$PWD:$PYTHONPATH`.

* **Torch CUDA load errors / `iJIT_NotifyEvent`**
  Use the **pip venv** instructions above (not Conda), and install PyTorch from the **cu121 index**.

* **`pre-commit` fails or isn’t found**
  `pip install pre-commit && python -m pre_commit run --all-files` (inside your venv).

* **Git push blocked (email privacy / GH007)**
  Set your Git identity to a GitHub‑verified address (or no‑reply):

  ```bash
  git config --global user.name  "Your Name"
  git config --global user.email "yourid@users.noreply.github.com"
  ```

* **SSH clone says Permission denied**
  Add your SSH key to GitHub **or** use the HTTPS clone URL.

---

## 8) Recommended Git workflow (1‑minute crash course)

```bash
# Always start from an up-to-date main
git checkout main
git fetch --prune
git pull --ff-only

# Make a feature branch for your work
git checkout -b baseline/lightgcn-tuning

# Do work, then format / lint (if you installed hooks)
python -m pre_commit run --all-files

# Stage, commit, push
git add -A
git commit -m "baseline: LightGCN tuning (negatives=10, dim=128, cosine)"
git push -u origin baseline/lightgcn-tuning

# Open a PR to main on GitHub, request a review, squash & merge after approval
```

---

## 9) Directory layout (what goes where)

```
CS224W_GraphML_Project_ARC/
  configs/                 # YAML config samples
  data/
    processed/ml1m/        # ← all processed CSVs, splits, and graph files live here (git-ignored)
  docs/
    GETTING_STARTED.md     # Instructions for getting started
    GPU_SETUP.md           # GPU/PyG setup details (advanced)
  scripts/
    run_mostpop.py         # Most-Popular baseline
    run_lightgcn.py        # LightGCN baseline (LGConv + BPR)
    run_grid_search.sh     # grid runner
    analyze_grid_search.py # summarize grid logs
    build_*.py, validate_*.py, verify_*.py, etc.
  src/graphflix/...        # package code
  requirements*.txt
  pyproject.toml
```

---

## 10) Expected first run checklist

* `python scripts/inspect_graph_pyg.py` prints the correct node/edge counts.
* `python scripts/run_mostpop.py` prints VAL/TEST metrics.
* `python scripts/run_lightgcn.py` runs and improves over Most‑Popular.
---
