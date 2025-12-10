#!/bin/bash
set -e

ROOT="data/processed/ml1m"
RESULTS_DIR="results/focused_search_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR/logs"
SUMMARY="$RESULTS_DIR/summary.csv"
echo "config,test_recall10,test_ndcg10,val_ndcg10" > "$SUMMARY"

run() {
    local name=$1
    shift
    echo "========== $name =========="
    python scripts/run_lightgcn.py "$@" --device cuda --seed 42 2>&1 | tee "$RESULTS_DIR/logs/${name}.log"
    local log="$RESULTS_DIR/logs/${name}.log"
    local tr10=$(grep "TEST:" "$log" | grep -oP "recall@10': \K[0-9.]+")
    local tn10=$(grep "TEST:" "$log" | grep -oP "ndcg@10': \K[0-9.]+")
    local vn10=$(grep "Best val" "$log" | awk '{print $4}')
    echo "$name,$tr10,$tn10,$vn10" >> "$SUMMARY"
    echo "$name: R@10=$tr10, NDCG@10=$tn10"
}

# Your current best from grid (baseline to beat)
run "baseline_L2_D64" --epochs 100 --layers 2 --dim 64 --lr 0.001 --neg 10

# Scale up dimensions (keeping best hyperparams)
run "L2_D128" --epochs 100 --layers 2 --dim 128 --lr 0.001 --neg 10
run "L2_D256" --epochs 100 --layers 2 --dim 256 --lr 0.001 --neg 10

# Try more layers (with best hyperparams)
run "L3_D128" --epochs 100 --layers 3 --dim 128 --lr 0.001 --neg 10
run "L3_D256" --epochs 100 --layers 3 --dim 256 --lr 0.001 --neg 10
run "L4_D128" --epochs 100 --layers 4 --dim 128 --lr 0.001 --neg 10

# Try 20 negatives with best config so far
run "L3_D128_N20" --epochs 100 --layers 3 --dim 128 --lr 0.001 --neg 20
run "L3_D256_N20" --epochs 100 --layers 3 --dim 256 --lr 0.001 --neg 20

echo "========== DONE =========="
(head -1 "$SUMMARY" && tail -n +2 "$SUMMARY" | sort -t',' -k3 -rn) | column -t -s','
