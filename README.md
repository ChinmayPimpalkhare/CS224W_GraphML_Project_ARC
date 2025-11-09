# GraphFlix (CS224W Project)

Config‑driven scaffold for a graph‑transformer recommender on MovieLens‑1M (GraphFlix).
We use a heterogeneous graph, temporal leave‑one‑out splits, BPR training, and top‑K metrics.

## Team & Ownership
- **Data/Infra**: <Name A> — `scripts/`, `src/graphflix/data`, `configs/data`
- **Modeling**:  <Name B> — `src/graphflix/models`, `src/graphflix/training`
- **Eval/Blog**: <Name C> — `src/graphflix/evaluation`, `docs/`, Medium

> **New to the repo? Start here:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
---

## 1) Setup (Conda + pip)

> Recommended: Create a Conda env, install **PyTorch via Conda** (CPU or GPU), then install the remaining Python packages with **pip**.

### A) Create & activate the Conda env

**macOS / Linux**
```bash
conda env create -f environment.yml    # creates env "graphflix"
conda activate graphflix
# If environment.yml isn't present:
# conda create -n graphflix python=3.10 -y
# conda activate graphflix
```

**Windows (PowerShell)**
```powershell
conda env create -f environment.yml
conda activate graphflix
# If activation is blocked:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### B) Install PyTorch

**CPU only (simple)**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**GPU (CUDA)** — choose the CUDA version that matches your driver (example: CUDA 12.1)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### C) Install project Python packages (pip)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: dev tools (lint/test/hooks)
pre-commit install                    # optional: auto-format on commit
```

> We intentionally **do not** install `torch` via pip to avoid CUDA wheel mismatches with Conda.

---

## 2) Secrets

Create a local `.env` (do **not** commit it):
```bash
cp .env.example .env      # Windows: copy .env.example .env
# then edit .env and set:
# TMDB_API_KEY=YOUR_TMDB_KEY
```

We auto‑load `.env` in scripts via `python-dotenv`. To verify:
```bash
python scripts/check_env.py
```

---

## 3) Data placement (enriched CSVs)

Place your 10 CSVs here (they are **git‑ignored**):
```
data/processed/ml1m/
  actors.csv
  directors.csv
  genres.csv
  movies.csv
  movies_enriched.csv
  movie_actor_edges.csv
  movie_director_edges.csv
  movie_genre_edges.csv
  ratings.csv
  users.csv
```
See `data/README.md` for schemas and join keys.

---

## 4) Configuration

Open `configs/data/movielens_1m.yaml` and confirm:
```yaml
data:
  name: "movielens-1m"
  processed_dir: "data/processed/ml1m"
  min_user_interactions: 3

split:
  strategy: "temporal_leave_one_out"  # per user: val=second-last, test=last
  val_holdout: 1
  test_holdout: 1
  filter_unknown_users: true
  seed: 42
```

---

## 5) Run the first steps

**Make the temporal split (per-user LOO)**
```bash
python scripts/split_temporal.py --ratings data/processed/ml1m/ratings.csv --out data/processed/ml1m/splits
```

**Validate the split**
```bash
python scripts/validate_split.py
```

**(Stub) Run a baseline**
```bash
python scripts/run_baselines.py --config configs/model/lightgcn.yaml
```

---

## 6) Repository layout

```
.
├─ .github/                     # issue/PR templates
├─ configs/
│  ├─ data/movielens_1m.yaml    # paths & split strategy
│  └─ model/{lightgcn,graphflix}.yaml
├─ data/                        # (ignored) place CSVs under processed/ml1m/
├─ docs/                        # architecture notes, flowchart, onboarding
├─ scripts/                     # download/build/split/run baselines, validators
├─ src/graphflix/               # data, models, training, evaluation, utils
├─ tests/                       # unit tests
├─ .env.example                 # example secrets (do not commit .env)
├─ .gitignore
├─ environment.yml              # Conda environment spec
├─ pyproject.toml               # black/isort/flake8 config
├─ requirements.txt             # pip packages
├─ requirements-dev.txt         # dev tools
└─ README.md
```

---

## 7) Git workflow

- **Branches:** small, focused (`feature/lightgcn`, `data/split-loo`, `docs/blog`).
- **PRs:** at least one reviewer; use the PR template.
- **Pre‑commit:** run `pre-commit install` once; formatting/linting on every commit.
- **No large files in git:** `data/` and `output/` are ignored.

---

## 8) Troubleshooting

- **pandas not found** → `pip install -r requirements.txt` in the **activated** env.
- **`.env` not picked up** → ensure `.env` exists; or set env var in shell:
  - macOS/Linux: `export TMDB_API_KEY=...`
  - Windows PS: `$env:TMDB_API_KEY="..."`
  - Persistent Windows: `setx TMDB_API_KEY "..."` (restart terminal)
- **PyTorch Geometric** → install after PyTorch; follow PyG’s wheel selector for your Torch/CUDA.