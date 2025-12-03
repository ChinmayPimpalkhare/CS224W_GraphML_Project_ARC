#!/bin/bash
# Quick training script for GraphFlix on MovieLens 1M
# This will run a complete training session with reasonable defaults

set -e

echo "=========================================="
echo "GraphFlix Training on MovieLens 1M"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="data/processed/ml1m"
CONFIG="configs/model/graphflix_full.yaml"
SAVE_DIR="runs"

# Check if preprocessed data exists
echo "Checking preprocessed data..."
if [ ! -f "$DATA_DIR/phi_matrix.pt" ]; then
    echo "Metadata embeddings not found. Running preprocessing..."
    python scripts/precompute_metadata.py \
        --data_dir "$DATA_DIR" \
        --embed_dim 64 \
        --seed 42
else
    echo "✓ Metadata embeddings found"
fi

if [ ! -f "$DATA_DIR/user_profiles.pt" ]; then
    echo "User profiles not found. Running preprocessing..."
    python scripts/compute_user_profiles.py \
        --data_dir "$DATA_DIR" \
        --half_life_days 150 \
        --seed 42
else
    echo "✓ User profiles found"
fi

echo ""
echo "Starting training..."
echo "=========================================="

# Read actual values from config file
BATCH_SIZE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['batch_size'])")
EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['epochs'])")
LR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['lr'])")
K_HOPS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['training']['k_hops'])")

echo "Configuration:"
echo "  - Data: $DATA_DIR"
echo "  - Config: $CONFIG"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs: $EPOCHS"
echo "  - Learning rate: $LR"
echo "  - K-hops: $K_HOPS"
echo "=========================================="
echo ""

# Run training
python scripts/train_graphflix.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Results saved to the latest run directory in: $SAVE_DIR"
echo ""
echo "To view results:"
echo "  ls -lt $SAVE_DIR/graphflix_*"
echo ""
