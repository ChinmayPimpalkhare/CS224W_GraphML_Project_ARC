#!/bin/bash
#
# Run proper evaluation with 1-vs-100 protocol
#

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          Running PROPER Evaluation (1-vs-100 Protocol)               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will evaluate using the CORRECT protocol:"
echo "  • 1 positive vs 99 negatives (100 items total)"
echo "  • Standard RecSys evaluation (not trivially easy)"
echo "  • No data leakage (negatives exclude all user interactions)"
echo ""

python evaluate_proper_final.py runs/graphflix_20251130_092953/best.pt \
    --data_dir data/processed/ml1m_10pct \
    --split test \
    --num_negatives 99

EXIT_CODE=$?

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed!"
    echo ""
    echo "Results saved to: runs/graphflix_20251130_092953/test_results_proper.txt"
    echo ""
    echo "To view:"
    echo "  cat runs/graphflix_20251130_092953/test_results_proper.txt"
    echo ""
    echo "Expected results (LEGITIMATE):"
    echo "  Recall@10: 0.15-0.25 (not 1.0!)"
    echo "  NDCG@10:   0.08-0.12"
    echo "  MRR:       0.10-0.18"
    echo ""
    echo "Compare with LightGCN baseline (~0.18 Recall@10)"
else
    echo "❌ Evaluation failed"
    echo "Check error messages above"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
