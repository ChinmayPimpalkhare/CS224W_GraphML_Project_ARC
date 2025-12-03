#!/bin/bash
# Quick test of improved LightGCN with better hyperparameters

echo "========================================================================"
echo "Testing Improved LightGCN Configuration"
echo "========================================================================"
echo "Key improvements:"
echo "  - Learning rate: 3e-3 → 1e-3"
echo "  - Batch size: 4096 → 2048"
echo "  - Negatives: 5 → 10"
echo "  - Regularization: 1e-4 → 1e-3"
echo "  - Added dropout: 0.1"
echo "  - Added gradient clipping: 1.0"
echo "  - Added cosine LR scheduler"
echo "  - Early stopping: patience=20"
echo "  - Epochs: 50 → 200 (with early stopping)"
echo "========================================================================"
echo ""

python scripts/run_lightgcn.py \
    --root data/processed/ml1m \
    --epochs 200 \
    --dim 64 \
    --layers 3 \
    --lr 1e-3 \
    --batch 2048 \
    --neg 10 \
    --reg 1e-3 \
    --dropout 0.1 \
    --grad_clip 1.0 \
    --scheduler cosine \
    --early_stop_patience 20 \
    --device auto \
    --seed 42

echo ""
echo "========================================================================"
echo "Training complete! Check results in data/processed/ml1m/runs/"
echo "========================================================================"
