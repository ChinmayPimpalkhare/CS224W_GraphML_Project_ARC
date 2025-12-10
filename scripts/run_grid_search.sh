#!/bin/bash
# scripts/run_grid_search.sh
# Grid search for LightGCN hyperparameters

set -e  # Exit on error

# Configuration
ROOT="data/processed/ml1m"
DEVICE="cuda"
SEED=42
RESULTS_DIR="results/grid_search_$(date +%Y%m%d_%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR/checkpoints"

# Log file for summary
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
echo "config,layers,dim,lr,neg,use_cosine,best_epoch,val_recall10,val_ndcg10,test_recall10,test_ndcg10,train_time_sec" > "$SUMMARY_FILE"

# Counter for tracking progress
TOTAL_CONFIGS=0
COMPLETED_CONFIGS=0

# Function to run a single config
run_config() {
    local layers=$1
    local dim=$2
    local lr=$3
    local neg=$4
    local use_cosine=$5
    local epochs=$6
    
    # Create config name
    local config_name="L${layers}_D${dim}_LR${lr}_N${neg}_cosine${use_cosine}"
    local log_file="$RESULTS_DIR/logs/${config_name}.log"
    local start_time=$(date +%s)
    
    echo ""
    echo "=========================================="
    echo "Running config: $config_name"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Build command
    local cmd="python scripts/run_lightgcn.py \
        --root $ROOT \
        --epochs $epochs \
        --layers $layers \
        --dim $dim \
        --lr $lr \
        --batch 4096 \
        --neg $neg \
        --reg 1e-4 \
        --device $DEVICE \
        --seed $SEED"
    
    if [ "$use_cosine" = "true" ]; then
        cmd="$cmd --use_cosine"
    fi
    
    # Run and capture output
    if $cmd 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Extract metrics from log
        local best_epoch=$(grep "Best val NDCG@10:" "$log_file" | awk '{print $NF}' | sed 's/epoch//' | tr -d ' ')
        local val_recall=$(grep "VAL :" "$log_file" | grep -oP "recall@10': \K[0-9.]+")
        local val_ndcg=$(grep "VAL :" "$log_file" | grep -oP "ndcg@10': \K[0-9.]+")
        local test_recall=$(grep "TEST:" "$log_file" | grep -oP "recall@10': \K[0-9.]+")
        local test_ndcg=$(grep "TEST:" "$log_file" | grep -oP "ndcg@10': \K[0-9.]+")
        
        # Append to summary
        echo "$config_name,$layers,$dim,$lr,$neg,$use_cosine,$best_epoch,$val_recall,$val_ndcg,$test_recall,$test_ndcg,$duration" >> "$SUMMARY_FILE"
        
        # Move artifacts
        if [ -f "$ROOT/lightgcn_best.pt" ]; then
            mv "$ROOT/lightgcn_best.pt" "$RESULTS_DIR/checkpoints/${config_name}_best.pt"
        fi
        
        echo "Config completed in ${duration}s"
        echo "  Test Recall@10: $test_recall"
        echo "  Test NDCG@10: $test_ndcg"
        
        COMPLETED_CONFIGS=$((COMPLETED_CONFIGS + 1))
    else
        echo "Config FAILED - see $log_file"
        
    fi
    
    echo "Progress: $COMPLETED_CONFIGS / $TOTAL_CONFIGS configs completed"
}

# =============================================================================
# GRID CONFIGURATION
# =============================================================================

# Define grid (edit these arrays to customize)
LAYERS_GRID=(2 3 4)
DIM_GRID=(64 128 256)
LR_GRID=(0.001 0.003 0.005)
NEG_GRID=(5 10 20)
COSINE_GRID=(true false)
EPOCHS=100

# Count total configs
for layers in "${LAYERS_GRID[@]}"; do
    for dim in "${DIM_GRID[@]}"; do
        for lr in "${LR_GRID[@]}"; do
            for neg in "${NEG_GRID[@]}"; do
                for cosine in "${COSINE_GRID[@]}"; do
                    TOTAL_CONFIGS=$((TOTAL_CONFIGS + 1))
                done
            done
        done
    done
done

echo "=========================================="
echo "GRID SEARCH START"
echo "=========================================="
echo "Total configurations: $TOTAL_CONFIGS"
echo "Results directory: $RESULTS_DIR"
echo "Epochs per config: $EPOCHS"
echo "Estimated time: $((TOTAL_CONFIGS * 15 / 60)) hours (assuming 15min/config)"
echo ""
read -p "Press ENTER to start or Ctrl+C to cancel..."

# Run grid search
for layers in "${LAYERS_GRID[@]}"; do
    for dim in "${DIM_GRID[@]}"; do
        for lr in "${LR_GRID[@]}"; do
            for neg in "${NEG_GRID[@]}"; do
                for cosine in "${COSINE_GRID[@]}"; do
                    run_config $layers $dim $lr $neg $cosine $EPOCHS
                done
            done
        done
    done
done

# =============================================================================
# POST-PROCESSING
# =============================================================================

echo ""
echo "=========================================="
echo "GRID SEARCH COMPLETE"
echo "=========================================="
echo "Completed: $COMPLETED_CONFIGS / $TOTAL_CONFIGS"
echo "Results: $RESULTS_DIR"
echo ""

# Find best config
if [ -f "$SUMMARY_FILE" ]; then
    echo "Top 5 Configs by Test NDCG@10:"
    echo ""
    (head -1 "$SUMMARY_FILE" && tail -n +2 "$SUMMARY_FILE" | sort -t',' -k10 -rn | head -5) | column -t -s','
    
    echo ""
    echo "Full results saved to: $SUMMARY_FILE"
fi

echo ""
echo "To analyze results:"
echo "  cat $SUMMARY_FILE | column -t -s','"
echo "  python scripts/analyze_grid_search.py $SUMMARY_FILE"
