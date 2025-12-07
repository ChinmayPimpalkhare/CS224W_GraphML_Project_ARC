# GraphFlix

GraphFlix is a CS224W course project that turns MovieLens‑1M into a heterogeneous user–movie–actor–director–genre graph and trains graph-based recommenders for top‑K movie recommendation. The final model is a Graphormer‑style graph transformer with user‑conditioned metadata bias built from half‑life–decayed user profiles, alongside a strong LightGCN baseline and an HGT-based scoring variant.

---

## 1. Getting started & LightGCN baseline

If you're just trying to get the code running, start here—**all environment, data prep, and baseline instructions live in `GETTING_STARTED.md`.**

The same doc covers three things you’ll probably want to do first:

- **Environment setup (Python, PyTorch, PyG, etc.)**  
  See: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

- **Download & preprocess MovieLens‑1M, build the heterogeneous graph**  
  See: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

- **Run the LightGCN baseline**  
  See: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

---

## 2. GraphFlixHGT: Heterogeneous Graph Transformer with scoring‑layer metadata bias

To train and evaluate the `GraphFlixHGT` variant (HGT encoder + metadata bias added at the **scoring layer**):

- **[`docs/GRAPHFLIX_HGT_SCORING.md`](docs/GRAPHFLIX_HGT_SCORING.md)** explains:
  - prerequisites (processed MovieLens‑1M data and `graph_pyg.pt`)
  - how to precompute half‑life user profiles and movie metadata
  - the training / evaluation commands for `run_graphflixHGT.py`
  - where checkpoints and metrics are written

Use this if you want a metadata‑aware graph baseline that stays very close to LightGCN’s evaluation protocol.

---

## 3. Full GraphFlix model (Graphormer + attention‑level metadata bias)

The final GraphFlix model moves the metadata bias **inside** the Graphormer attention logits and learns a global scale parameter **β** that balances graph‑structure signals with content‑based metadata:

-  **[`docs/UNDERSTANDING_BETA.md`](docs/UNDERSTANDING_BETA.md)** currently documents:
  - how β scales the metadata bias relative to the graph encoder scores
  - how half‑life user profiles and movie metadata embeddings are constructed and used
  - interpretability tips (e.g., reading off what β is doing during training)
  - with the full training / evaluation commands for the end‑to‑end GraphFlix model

This README will stay as the main entry point. Follow the links above for the most up‑to‑date per‑model instructions.

---

## 4. Reproducibility

All models in this repo (LightGCN, GraphFlixHGT, and the full GraphFlix transformer) share:

- the same MovieLens‑1M preprocessing pipeline,
- a temporal leave‑one‑out split (train / val / test),
- and top‑K ranking metrics (Recall@K, NDCG@K),

so results are directly comparable across architectures.
