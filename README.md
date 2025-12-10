# GraphFlix

GraphFlix is a graph-based recommendation system built on the MovieLens 1M dataset.  
It combines:

- A **LightGCN** collaborative filtering baseline
- A **GraphFlixHGT** variant that adds a metadata-aware bias at the *scoring* layer
- The full **GraphFlix** model, which injects a learnable metadata bias directly into a Graphormer-style attention encoder

This repository is designed to be fully reproducible and public so that anyone can download the data, set up the environment, and run all models end-to-end.

---

## Quick Start

Most users should start with the **Getting Started** guide, which walks through downloading the data, setting up the environment, and running the LightGCN baseline.

### 1. Environment, Data, and LightGCN Baseline

All of these steps live in the same document; the links below all point to `docs/GETTING_STARTED.md` but highlight the main tasks:

- **Environment & installation (Python, PyTorch, PyG, etc.)**  
  [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

- **Preparing the MovieLens 1M data & preprocessing pipeline**  
  [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

- **Running the LightGCN baseline and evaluating it**  
  [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)

Follow that document first; once you can run LightGCN successfully, you’re ready to move on to the GraphFlix variants.

---

### 2. Training the GraphFlix Model (Subsampled Datasets)

To train the full **GraphFlix** model on 10% or 25% subsamples of MovieLens 1M (faster runs with proper evaluation and data-leakage prevention), see:

[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)

That guide explains:

- How to launch training with:
  - `./train_10pct_fixed.sh`
  - `./train_25pct_proper.sh`
- How the scripts automatically:
  - Create subsampled datasets (10% / 25%)
  - Build graphs and precompute metadata + user profiles
  - Run `scripts/train_graphflix.py` with the correct config
  - Perform a proper **1-vs-100** evaluation at the end of training
- Where to find logs, checkpoints, and evaluation results under `runs/…`

If you want to understand or tweak training hyperparameters (e.g., learning rate, batch size, epochs), that document also points to `configs/model/graphflix_full.yaml` and explains the most important fields.

---

### 3. Running the GraphFlixHGT (Scoring-Layer Metadata Bias) Variant

The **GraphFlixHGT** variant keeps the metadata bias outside the attention layers and applies it at the scoring layer on top of an HGT encoder.  
To train and evaluate this model (which is directly comparable to LightGCN under the same temporal LOO protocol), see:

[`docs/GRAPHFLIX_HGT_SCORING.md`](docs/GRAPHFLIX_HGT_SCORING.md)

That document covers:

- How to precompute movie metadata and half-life user profiles for HGT
- How to run `scripts/precompute_profilesHGT.py` and `scripts/run_graphflixHGT.py`
- Full-catalog ranking evaluation with Recall@K and NDCG@K

---

### 4. Understanding β (Beta) and the Final GraphFlix Model

The **β (beta)** parameter controls how strongly the model relies on metadata versus pure graph structure. It scales a metadata-based bias term:

- When **β → 0**: metadata is effectively ignored (pure collaborative filtering)
- When **β ≈ 1**: metadata and graph signals contribute roughly equally
- When **β is large**: metadata dominates the prediction

For an in-depth explanation of:

- What β is and how it’s learned
- How GraphFlix combines:
  - Graph encodings `s_enc(u, j) = ⟨z_u, z_j⟩`
  - Metadata bias `b_meta(u, j) = β · tanh(LN(p(u))ᵀ W LN(φ(j)))`
- How this bias is injected into the attention mechanism in the full GraphFlix model

…see:

[`docs/UNDERSTANDING_BETA.md`](docs/UNDERSTANDING_BETA.md)

This document is the conceptual companion to the final GraphFlix implementation and will be updated to include a more step-by-step “run the full GraphFlix model” guide as the project evolves.

---

## Repository Layout (High Level)

Some useful top-level paths:

- `docs/`
  - `GETTING_STARTED.md` – env setup, data prep, and LightGCN baseline
  - `GRAPHFLIX_HGT_SCORING.md` – how to run the GraphFlixHGT scoring-layer variant
  - `TRAINING_GUIDE.md` – training GraphFlix on 10% / 25% subsamples, with proper evaluation
  - `UNDERSTANDING_BETA.md` – deep-dive on the β parameter and metadata bias in GraphFlix
- `scripts/`
  - Data preprocessing, graph construction, and model runners (LightGCN, GraphFlixHGT, GraphFlix)
- `configs/`
  - YAML config files (e.g., `configs/model/graphflix_full.yaml`) controlling model and training hyperparameters
- `data/processed/`
  - Expected location for processed MovieLens 1M data, splits, graphs, and precomputed features (not distributed in this repo)

---

## Data & Licensing

This project expects the **MovieLens 1M** dataset, which is **not** bundled with the repository.  
Please download it from the official GroupLens website and follow the instructions in `docs/GETTING_STARTED.md` for preprocessing and usage.

---

If you’re new to the repo, the recommended path is:

1. Read **`docs/GETTING_STARTED.md`** and run the LightGCN baseline.  
2. Use **`docs/TRAINING_GUIDE.md`** to train GraphFlix on a subsampled dataset.  
3. Explore **`docs/GRAPHFLIX_HGT_SCORING.md`** and **`docs/UNDERSTANDING_BETA.md`** to dive deeper into the metadata-aware models.

---

### Use of AI Tools

We used AI-assisted tools (such as ChatGPT/Claude) in a limited way to:
- polish and clarify written instructions and documentation,
- suggest refactorings and minor optimizations to improve code readability and runtime,
- improve error messages and logging, and
- assist with debugging and error analysis.

All modeling choices, algorithms, experiments, and reported results are our own work. Any AI-generated suggestions were manually reviewed, edited, and tested by the authors before inclusion.

