# ONBOARDING — GraphFlix

This guide walks you from a blank machine to running the project.

## 0) Prereqs
- Git
- Python 3.10+ (conda or system python)
- (Optional) GPU with CUDA and matching PyTorch

## 1) Clone and environment

### macOS / Linux
```bash
git clone <YOUR-REPO-SSH-OR-HTTPS>
cd CS224W_GraphML_Project_ARC
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional
pre-commit install                   # optional
```

### Windows (PowerShell)
```powershell
git clone <YOUR-REPO-SSH-OR-HTTPS>
cd CS224W_GraphML_Project_ARC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional
pre-commit install                   # optional
```
If activation fails, run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

## 2) Secrets
Copy `.env.example` → `.env` and set:
```
TMDB_API_KEY=YOUR_TMDB_KEY
```
We auto‑load `.env` in scripts via `python-dotenv`.

## 3) Data placement
Place the 10 enriched CSVs at `data/processed/ml1m/`. See `data/README.md` for schemas and keys.

## 4) Temporal split
```bash
python scripts/split_temporal.py --ratings data/processed/ml1m/ratings.csv --out data/processed/ml1m/splits
```

## 5) Validate split
```bash
python scripts/validate_split.py
```

## 6) Baseline (stub)
```bash
python scripts/run_baselines.py --config configs/model/lightgcn.yaml
```

## 7) Common issues
- `pip: command not found` → ensure Python is in PATH or use `py -m pip` (Windows).
- SSL/cert errors on macOS → run `Install Certificates.command` from your Python folder.
- Pre-commit not formatting → run `pre-commit run --all-files` once.

## 8) Next steps
- Build graph from CSVs in `scripts/build_graph.py`.
- Train LightGCN baseline end‑to‑end.
- Implement GraphFlix with user‑conditioned metadata bias and BPR.