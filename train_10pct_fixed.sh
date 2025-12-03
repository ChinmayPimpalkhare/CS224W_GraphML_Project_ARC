#!/bin/bash
# Training GraphFlix on 10% Subsample with Data Leakage Fix
# Expected time: ~12 hours (vs 35% = days, full = even longer)

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         Training GraphFlix on 10% Subsample (Data Leakage Fixed)     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Verify the fix works for 10% subsample
echo "Step 1: Verifying data leakage fix for 10% subsample..."
python test_data_leakage.py --data_dir data/processed/ml1m_10pct --num_samples 50

if [ $? -ne 0 ]; then
    echo "❌ Data leakage test failed! Please check the fix."
    exit 1
fi

echo ""
echo "✅ Data leakage fix verified for 10% subsample"
echo ""

# Step 2: Train on 10% subsample
echo "Step 2: Training model on 10% subsample..."
echo "Expected time: ~12 hours"
echo ""

python scripts/train_graphflix.py \
    --data_dir data/processed/ml1m_10pct \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001 \
    --seed 42

echo ""
echo "✅ Training complete!"
echo ""

# Step 3: Evaluate
echo "Step 3: Evaluating on test set..."
LATEST_RUN=$(ls -td runs/graphflix_* | head -1)
echo "Using checkpoint: $LATEST_RUN/best.pt"
echo ""

python evaluate.py $LATEST_RUN/best.pt \
    --data_dir data/processed/ml1m_10pct \
    --split test \
    --batch_size 8 \
    --k_hops 1

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Training and evaluation complete!                                   ║"
echo "║  Check results above - should be realistic (0.15-0.25 Recall@10)    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
