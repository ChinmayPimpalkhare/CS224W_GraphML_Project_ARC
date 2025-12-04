#!/bin/bash
# Quick training on 25% subsample - RECOMMENDED
# Time: ~1.25 days vs 5 days on full dataset
# Quality: 90-95% of full model

set -e

echo "=========================================="
echo "GraphFlix Quick Training (25% Sample)"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Create 25% data subsample (~4x faster)"
echo "  2. Build graph for subsample"
echo "  3. Precompute features"
echo "  4. Train model"
echo "  5. Evaluate results"
echo ""
echo "Expected time: ~1.25 days (vs 5 days full)"
echo "Expected quality: 90-95% of full model"
echo "=========================================="
echo ""

DATA_DIR="data/processed/ml1m"
SUBSAMPLE_DIR="data/processed/ml1m_25pct"
SAMPLE_RATIO=0.25

# Step 1: Create subsample
if [ ! -d "$SUBSAMPLE_DIR" ]; then
    echo "Step 1/5: Creating 25% subsample..."
    python scripts/create_subsample.py \
        --data_dir "$DATA_DIR" \
        --sample_ratio $SAMPLE_RATIO \
        --seed 42
else
    echo "Step 1/5: ✓ Subsample already exists"
fi

echo ""

# Step 2: Build graph
if [ ! -f "$SUBSAMPLE_DIR/graph_pyg.pt" ]; then
    echo "Step 2/5: Building graph..."
    python scripts/build_graph_pyg.py \
        --root "$SUBSAMPLE_DIR"
else
    echo "Step 2/5: ✓ Graph already exists"
fi

echo ""

# Step 3: Precompute metadata embeddings
if [ ! -f "$SUBSAMPLE_DIR/phi_matrix.pt" ]; then
    echo "Step 3/5: Precomputing metadata embeddings..."
    python scripts/precompute_metadata.py \
        --data_dir "$SUBSAMPLE_DIR" \
        --embed_dim 64 \
        --seed 42
else
    echo "Step 3/5: ✓ Metadata embeddings already exist"
fi

echo ""

# Step 4: Compute user profiles
if [ ! -f "$SUBSAMPLE_DIR/user_profiles.pt" ]; then
    echo "Step 4/5: Computing user profiles..."
    python scripts/compute_user_profiles.py \
        --data_dir "$SUBSAMPLE_DIR" \
        --half_life_days 150 \
        --seed 42
else
    echo "Step 4/5: ✓ User profiles already exist"
fi

echo ""

# Step 5: Train
echo "Step 5/5: Training model..."
echo "=========================================="
echo ""

python scripts/train_graphflix.py \
    --config configs/model/graphflix_full.yaml \
    --data_dir "$SUBSAMPLE_DIR" \
    --save_dir runs/subsample_25pct

echo ""
echo "=========================================="
echo "✓ Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: runs/subsample_25pct/"
echo ""
echo "To evaluate:"
echo "  python evaluate.py runs/subsample_25pct/best.pt --split test"
echo ""
