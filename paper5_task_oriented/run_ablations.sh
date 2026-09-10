#!/bin/bash
# =============================================================================
# Paper 5 — S2 ablations (run AFTER the main matrix completes).
#
# Motivation (2026-09-02 interim read): at lambda_ser=1.0, sigma2=0.1 the
# soft-SER term is nearly flat (val ser_soft 2.95 -> 2.84 over 100 epochs)
# and overall SER/BER barely move.  Two levers, single seed (42) each:
#   - lambda_ser in {0.3, 3.0}   (strength)
#   - ser_sigma2  in {0.05, 0.2} (hardness/saturation of the soft logits)
# Plus the Pareto sweep point needed for the money figure.
#
# Usage: bash run_ablations.sh
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH
mkdir -p results/s2_abl

run_one () {  # name lambda_ser ser_sigma2
    NAME=$1; LSER=$2; SIG=$3
    echo ""
    echo "=== Ablation $NAME (lambda_ser=$LSER sigma2=$SIG), seed 42 ==="
    python3 train.py --arch slot \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed 42 \
        --lambda_ser $LSER --ser_sigma2 $SIG --name $NAME
    CKPT=$(ls -t checkpoints/slot_h64_l4_k13_bs16_lr0.001_${NAME}_s42_best.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 100 \
            --ser_comp --out_dir results/s2_abl 2>&1 | tail -4
    else
        echo "WARNING: no checkpoint for $NAME"
    fi
}

run_one ser_l0.3 0.3 0.1
run_one ser_l3.0 3.0 0.1
run_one ser_s0.05 1.0 0.05
run_one ser_s0.2 1.0 0.2

echo ""
echo "=== Ablations done. See results/s2_abl/ ==="
