#!/bin/bash
# End-to-end GraphFlix training pipeline
# 
# This script orchestrates the full GraphFlix workflow:
# 1. Precompute metadata embeddings (if not exists)
# 2. Compute user profiles (if not exists)
# 3. Train GraphFlix model
# 4. Evaluate on test set

set -e  # Exit on error

# Configuration
DATA_DIR="data/processed/ml1m"
CONFIG="configs/model/graphflix_full.yaml"
SAVE_DIR="runs"

echo "================================================================================"
echo "GraphFlix End-to-End Training Pipeline"
echo "================================================================================"
echo "Data directory: $DATA_DIR"
echo "Config: $CONFIG"
echo "Save directory: $SAVE_DIR"
echo ""

# Step 1: Check/Create metadata embeddings
PHI_FILE="$DATA_DIR/phi_matrix.pt"
if [ ! -f "$PHI_FILE" ]; then
    echo "Step 1: Precomputing metadata embeddings..."
    echo "--------------------------------------------------------------------------------"
    python scripts/precompute_metadata.py \
        --data_dir "$DATA_DIR" \
        --embed_dim 64 \
        --seed 42
    echo ""
else
    echo "Step 1: Metadata embeddings found: $PHI_FILE"
    echo ""
fi

# Step 2: Check/Create user profiles
PROFILE_FILE="$DATA_DIR/user_profiles.pt"
if [ ! -f "$PROFILE_FILE" ]; then
    echo "Step 2: Computing user profiles..."
    echo "--------------------------------------------------------------------------------"
    python scripts/compute_user_profiles.py \
        --data_dir "$DATA_DIR" \
        --half_life_days 20 \
        --seed 42
    echo ""
else
    echo "Step 2: User profiles found: $PROFILE_FILE"
    echo ""
fi

# Step 3: Train model
echo "Step 3: Training GraphFlix model..."
echo "--------------------------------------------------------------------------------"
python scripts/train_graphflix.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR"

echo ""
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo "Results saved to: $SAVE_DIR/graphflix_*"
echo ""
echo "To view results:"
echo "  - Training history: $SAVE_DIR/graphflix_*/history.json"
echo "  - Best model: $SAVE_DIR/graphflix_*/best.pt"
echo "  - Config: $SAVE_DIR/graphflix_*/config.yaml"
echo ""
