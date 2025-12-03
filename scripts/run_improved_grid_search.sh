#!/bin/bash
# Grid search script for LightGCN hyperparameter tuning

# Base directory
ROOT="data/processed/ml1m"
DEVICE="auto"
EPOCHS=200
SEED=42

echo "========================================================================"
echo "LightGCN Grid Search - Improved Hyperparameters"
echo "========================================================================"

# Function to run experiment
run_experiment() {
    local dim=$1
    local layers=$2
    local lr=$3
    local reg=$4
    local neg=$5
    local dropout=$6
    local batch=$7
    local scheduler=$8
    
    echo ""
    echo "--------------------------------------------------------------------"
    echo "Running: dim=${dim}, layers=${layers}, lr=${lr}, reg=${reg}, neg=${neg}, dropout=${dropout}, batch=${batch}, scheduler=${scheduler}"
    echo "--------------------------------------------------------------------"
    
    python scripts/run_lightgcn.py \
        --root ${ROOT} \
        --epochs ${EPOCHS} \
        --dim ${dim} \
        --layers ${layers} \
        --lr ${lr} \
        --batch ${batch} \
        --neg ${neg} \
        --reg ${reg} \
        --dropout ${dropout} \
        --grad_clip 1.0 \
        --scheduler ${scheduler} \
        --early_stop_patience 20 \
        --device ${DEVICE} \
        --seed ${SEED}
}

# Best configuration from analysis (baseline for comparison)
echo ""
echo "========================================================================"
echo "BASELINE: Improved Configuration"
echo "========================================================================"
run_experiment 64 3 1e-3 1e-3 10 0.1 2048 "cosine"

# Experiment 1: Try higher embedding dimension
echo ""
echo "========================================================================"
echo "EXPERIMENT 1: Higher Embedding Dimension"
echo "========================================================================"
run_experiment 128 3 1e-3 1e-3 10 0.1 2048 "cosine"

# Experiment 2: Try more layers
echo ""
echo "========================================================================"
echo "EXPERIMENT 2: More Layers"
echo "========================================================================"
run_experiment 64 4 1e-3 1e-3 10 0.1 2048 "cosine"

# Experiment 3: Stronger regularization
echo ""
echo "========================================================================"
echo "EXPERIMENT 3: Stronger Regularization"
echo "========================================================================"
run_experiment 64 3 1e-3 5e-3 10 0.1 2048 "cosine"

# Experiment 4: More negatives
echo ""
echo "========================================================================"
echo "EXPERIMENT 4: More Negatives"
echo "========================================================================"
run_experiment 64 3 1e-3 1e-3 20 0.1 2048 "cosine"

# Experiment 5: Higher dropout
echo ""
echo "========================================================================"
echo "EXPERIMENT 5: Higher Dropout"
echo "========================================================================"
run_experiment 64 3 1e-3 1e-3 10 0.2 2048 "cosine"

# Experiment 6: Smaller batch size (more updates)
echo ""
echo "========================================================================"
echo "EXPERIMENT 6: Smaller Batch Size"
echo "========================================================================"
run_experiment 64 3 1e-3 1e-3 10 0.1 1024 "cosine"

# Experiment 7: Step scheduler instead of cosine
echo ""
echo "========================================================================"
echo "EXPERIMENT 7: Step Scheduler"
echo "========================================================================"
run_experiment 64 3 1e-3 1e-3 10 0.1 2048 "step"

# Experiment 8: Best combo - high dim, more layers
echo ""
echo "========================================================================"
echo "EXPERIMENT 8: High Dim + More Layers"
echo "========================================================================"
run_experiment 128 4 1e-3 1e-3 10 0.1 2048 "cosine"

# Experiment 9: Conservative - lower LR, higher reg
echo ""
echo "========================================================================"
echo "EXPERIMENT 9: Conservative (Lower LR, Higher Reg)"
echo "========================================================================"
run_experiment 64 3 5e-4 5e-3 10 0.1 2048 "cosine"

echo ""
echo "========================================================================"
echo "Grid Search Complete!"
echo "========================================================================"
echo "Use analyze_grid_search.py to compare results"
