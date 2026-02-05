#!/bin/bash
# =============================================================================
# Master Script: Train All DNNs + K-Sweep Ricci Analysis
# =============================================================================
# This script trains all 45 network configurations (excluding 3-layer) and
# runs k-sweep analysis with K = [250, 350, 450, 500, 600]
#
# Usage: bash run_full_experiment.sh

set -e  # Exit on error

cd /Users/cihan/Documents/GitHub/Ricci-Flow-Deep-Learning---Graduation-Project

echo "============================================================"
echo "FULL EXPERIMENT: 45 Networks × 5 K-values"
echo "============================================================"

# Clear previous outputs
rm -rf output_k_sweep output_k_sweep_analysis

# Network configurations (excluding 3-layer networks)
# Flat: depths 4-12, widths 16, 32, 64, 128
# Bottleneck: depths 4-12, width 128

FLAT_DEPTHS=(4 5 6 7 8 9 10 11 12)
FLAT_WIDTHS=(16 32 64 128)
BOTTLENECK_DEPTHS=(4 5 6 7 8 9 10 11 12)
EPOCHS=50

echo ""
echo "[PHASE 1] Training Flat Networks..."
echo "============================================================"

for depth in "${FLAT_DEPTHS[@]}"; do
    for width in "${FLAT_WIDTHS[@]}"; do
        echo "Training: flat_${depth}_${width}"
        python training_v2.py --architecture flat --depth $depth --width $width --epochs $EPOCHS
    done
done

echo ""
echo "[PHASE 2] Training Bottleneck Networks..."
echo "============================================================"

for depth in "${BOTTLENECK_DEPTHS[@]}"; do
    echo "Training: bottleneck_${depth}_128"
    python training_v2.py --architecture bottleneck --depth $depth --width 128 --epochs $EPOCHS
done

echo ""
echo "[PHASE 3] Running K-Sweep Ricci Analysis..."
echo "============================================================"

python knn_fixed_2_1.py \
    --input-dir output_k_sweep \
    --output-dir output_k_sweep_analysis \
    --k-values 250 350 450 500 600

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE!"
echo "============================================================"
echo "Results saved to:"
echo "  - output_k_sweep_analysis/K_SWEEP_MASTER_SUMMARY.csv"
echo "  - output_k_sweep_analysis/BEST_K_SUMMARY.csv"
echo ""
