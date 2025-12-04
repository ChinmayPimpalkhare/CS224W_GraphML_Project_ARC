#!/bin/bash
#
# Train GraphFlix on 10% subsample with proper data leakage prevention
# 
# This script:
# 1. Creates 10% subsample (if not exists)
# 2. Ensures proper train/val/test splits
# 3. Verifies no data leakage in negative sampling
# 4. Trains model with better hyperparameters (20 epochs)
# 5. Evaluates with proper 1-vs-100 protocol
#

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     GraphFlix Training on 10% Subsample (Data Leakage Prevention)    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will:"
echo "  1. Create/verify 10% subsample with proper splits"
echo "  2. Build graph and precompute features"
echo "  3. Train model (20 epochs for better convergence)"
echo "  4. Evaluate with proper 1-vs-100 protocol"
echo ""
echo "Expected time: ~12 hours"
echo "Expected performance: Recall@10 of ~0.10"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

DATA_DIR="data/processed/ml1m"
SUBSAMPLE_DIR="data/processed/ml1m_10pct"
SAMPLE_RATIO=0.10

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/6: Creating 10% subsample with proper splits..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -d "$SUBSAMPLE_DIR" ]; then
    echo "Creating 10% subsample..."
    python scripts/create_subsample.py \
        --data_dir "$DATA_DIR" \
        --sample_ratio $SAMPLE_RATIO \
        --seed 42
    
    echo ""
    echo "✅ 10% subsample created"
else
    echo "✅ 10% subsample already exists at: $SUBSAMPLE_DIR"
fi

# Verify splits exist
if [ ! -f "$SUBSAMPLE_DIR/splits/ratings_split_reindexed.csv" ]; then
    echo "⚠️  WARNING: Splits directory not found!"
    echo "   Creating splits..."
    
    mkdir -p "$SUBSAMPLE_DIR/splits"
    
    # Check if ratings has split column
    python -c "
import pandas as pd
df = pd.read_csv('$SUBSAMPLE_DIR/ratings_reindexed.csv')
if 'split' in df.columns:
    df.to_csv('$SUBSAMPLE_DIR/splits/ratings_split_reindexed.csv', index=False)
    print('✅ Created splits from ratings_reindexed.csv')
else:
    print('❌ ERROR: No split column in ratings file')
    print('   You need to create splits manually')
    exit(1)
" || exit 1
else
    echo "✅ Splits directory exists"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/6: Building graph..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$SUBSAMPLE_DIR/graph_pyg.pt" ]; then
    echo "Building graph for 10% subsample..."
    python scripts/build_graph_pyg.py \
        --root "$SUBSAMPLE_DIR"
    
    echo ""
    echo "✅ Graph built"
else
    echo "✅ Graph already exists"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/6: Precomputing metadata embeddings..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$SUBSAMPLE_DIR/phi_matrix.pt" ]; then
    echo "Computing metadata embeddings..."
    python scripts/precompute_metadata.py \
        --data_dir "$SUBSAMPLE_DIR" \
        --embed_dim 64 \
        --seed 42
    
    echo ""
    echo "✅ Metadata embeddings computed"
else
    echo "✅ Metadata embeddings already exist"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/6: Computing user profiles..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "$SUBSAMPLE_DIR/user_profiles.pt" ]; then
    echo "Computing user profiles..."
    python scripts/compute_user_profiles.py \
        --data_dir "$SUBSAMPLE_DIR" \
        --half_life_days 20 \
        --seed 42
    
    echo ""
    echo "✅ User profiles computed"
else
    echo "✅ User profiles already exist"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5/6: Verifying data leakage prevention..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Clear Python cache to ensure latest code is used
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

echo "Testing data leakage fix..."
python test_data_leakage.py --data_dir "$SUBSAMPLE_DIR" --num_samples 50

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data leakage test failed!"
    echo "   Please check the negative sampling implementation."
    exit 1
fi

echo ""
echo "✅ Data leakage prevention verified!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6/6: Training model (20 epochs)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Training with improved hyperparameters:"
echo "  • Epochs: 20 (for better convergence)"
echo "  • Learning rate: 0.001 (can tune later)"
echo "  • Dataset: 10% subsample"
echo ""
echo "Expected time: ~12 hours"
echo "Target loss: 0.3-0.5 (vs 0.69 random baseline)"
echo ""

# Create run directory with timestamp
RUN_NAME="graphflix_10pct_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="runs/$RUN_NAME"

echo "Checkpoint will be saved to: $SAVE_DIR"
echo ""

python scripts/train_graphflix.py \
    --config configs/model/graphflix_full.yaml \
    --data_dir "$SUBSAMPLE_DIR" \
    --save_dir "$SAVE_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed!"
    echo "   Check error messages above"
    exit 1
fi

echo ""
echo "✅ Training complete!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7/6 (Bonus): Evaluating with proper 1-vs-100 protocol..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CHECKPOINT="$SAVE_DIR/best.pt"

if [ -f "$CHECKPOINT" ]; then
    echo "Running proper evaluation..."
    python evaluate_proper_final.py "$CHECKPOINT" \
        --data_dir "$SUBSAMPLE_DIR" \
        --split test \
        --num_negatives 99
    
    echo ""
    echo "✅ Evaluation complete!"
    echo ""
    echo "Results saved to: $SAVE_DIR/test_results_proper.txt"
else
    echo "⚠️  Checkpoint not found: $CHECKPOINT"
    echo "   Evaluation skipped"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                          ✅ ALL DONE!                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  • Dataset: 10% subsample"
echo "  • Training: 20 epochs"
echo "  • Data leakage: PREVENTED (verified)"
echo "  • Evaluation: Proper 1-vs-100 protocol"
echo ""
echo "Results:"
echo "  • Training checkpoint: $SAVE_DIR/best.pt"
echo "  • Evaluation results: $SAVE_DIR/test_results_proper.txt"
echo ""
echo "To view results:"
echo "  cat $SAVE_DIR/test_results_proper.txt"
echo ""
echo "Expected performance:"
echo "  Recall@10 = 0.101"
echo ""
echo "Next steps:"
echo "  1. Check if loss decreased properly (target: 0.3-0.5)"
echo "  2. Compare metrics with other baselines"
echo "  3. If needed, tune hyperparameters and retrain"
echo ""
