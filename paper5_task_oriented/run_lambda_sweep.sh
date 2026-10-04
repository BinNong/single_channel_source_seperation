#!/bin/bash
# =============================================================================
# Paper 5 — reviewer-2 lambda_ser sweep (WCL letter v2, MC3).
#
# Trains the slot arch at lambda_mse=1.0 with lambda_ser in {0.01, 0.1, 10}
# (3 seeds 42-44), extending the existing fair-additive points:
#   lambda_ser=0   -> s2 mse     (5 seeds, results/s2)
#   lambda_ser=1   -> s2 ser_mse (5 seeds, results/s2)
# Answers: does the per-K sign-structured effect (or the pooled null)
# change with the task-loss weight?  Supports (or forces further
# softening of) the "not a better loss weight" prescription.
#
# GPU server: nohup bash run_lambda_sweep.sh > results/lambda_sweep.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH

SEEDS="42 43 44"
# name lambda_mse lambda_ser
CONFIGS=(
    "lser0p01 1.0 0.01"
    "lser0p1  1.0 0.1"
    "lser10   1.0 10.0"
)

mkdir -p results/lambda_sweep
for CFG in "${CONFIGS[@]}"; do
    read -r NAME LMSE LSER <<< "$CFG"
    for SEED in $SEEDS; do
        echo ""
        echo "=== Config $NAME (lambda_mse=$LMSE lambda_ser=$LSER), seed $SEED ==="
        python3 train.py --arch slot \
            --epochs 100 --train_samples 2000 --val_samples 400 \
            --batch_size 16 --lr 1e-3 --seed $SEED \
            --lambda_mse $LMSE --lambda_ser $LSER --name $NAME

        CKPT=$(ls -t checkpoints/slot_h64_l4_k13_bs16_lr0.001_${NAME}_s${SEED}_best.pt 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "=== Evaluating $CKPT (SER+BER compensated, n_per_cell=100) ==="
            python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 100 \
                --ser_comp --out_dir results/lambda_sweep
        else
            echo "WARNING: no checkpoint found for $NAME seed $SEED — skipping eval"
        fi
    done
done

echo "=== lambda sweep done; aggregate with results/s2 points ==="
