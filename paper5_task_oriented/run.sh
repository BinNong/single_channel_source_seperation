#!/bin/bash
# =============================================================================
# Paper 5 — Task-Oriented SC-BSS: S2 training + evaluation pipeline.
#
# S2 main matrix (docs/PAPER5_TASK_ORIENTED_PLAN.md v2), slot arch only:
#   (a) sisdr   : -SI-SDR                      (paper1-style baseline)
#   (b) mse     : -SI-SDR + lambda_mse 1.0     (paper4 recipe, reference)
#   (c) ser     : -SI-SDR + lambda_ser 1.0     (proposed)
#   (d) ser_mse : -SI-SDR + lambda_ser + MSE   (combination ablation)
#   (e) pureser : lambda_sep 0 + lambda_ser 1.0 (probe: no waveform loss at all)
#
# Usage:
#   bash run.sh           # full S2 matrix: (a)-(d) x 5 seeds
#   bash run.sh 1         # single seed (42 only)
#   bash run.sh smoke     # MVP smoke (1 epoch, tiny eval, lambda_ser on/off)
#   bash run.sh e         # config (e) probe, single seed (42)
#
# Runs locally or on the GPU server; requires Python 3.10+ with PyTorch.
# =============================================================================
set -e

cd "$(dirname "$0")"
# Server venv (RTX 4060 box); harmless locally (path simply won't exist).
export PATH=/data/experiment/venv_bss/bin:$PATH

MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
    # MVP: 1 epoch, tiny sample counts — verifies both loss code paths and
    # the SER/BER evaluation wiring end to end.
    for LSER in 0.0 1.0; do
        echo "=== Smoke test (slot, lambda_ser=$LSER, 1 epoch, seed 42) ==="
        python3 train.py --arch slot --lambda_ser $LSER \
            --epochs 1 --train_samples 64 --val_samples 32 \
            --batch_size 8 --seed 42 --num_workers 0 --name smoke_lser$LSER
    done
    # pure task-oriented config (e): checkpoint selection by val soft-SER
    echo "=== Smoke test (slot, lambda_sep=0 lambda_ser=1.0, seed 42) ==="
    python3 train.py --arch slot --lambda_sep 0.0 --lambda_ser 1.0 \
        --epochs 1 --train_samples 64 --val_samples 32 \
        --batch_size 8 --seed 42 --num_workers 0 --name smoke_pureser
    for CKPT in checkpoints/slot_*smoke*_best.pt; do
        echo "=== Smoke evaluation ($CKPT) ==="
        python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 4 --ser_comp
    done
    exit 0
fi

if [ "$MODE" = "e" ]; then
    # Config (e) probe: pure soft-SER, single seed.  Promote to 5 seeds only
    # if the probe converges (plan v2.4).
    python3 train.py --arch slot --lambda_sep 0.0 --lambda_ser 1.0 \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed 42 --name pureser
    CKPT=$(ls -t checkpoints/slot_h64_l4_k13_bs16_lr0.001_pureser_s42_best.pt 2>/dev/null | head -1)
    python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 100 --ser_comp \
        --out_dir results/s2
    exit 0
fi

if [ "$MODE" = "1" ]; then
    SEEDS="42"
else
    SEEDS="42 43 44 45 46"
fi

# name lambda_mse lambda_ser
CONFIGS=(
    "sisdr 0.0 0.0"
    "mse 1.0 0.0"
    "ser 0.0 1.0"
    "ser_mse 1.0 1.0"
)

mkdir -p results/s2
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
                --ser_comp --out_dir results/s2
        else
            echo "WARNING: no checkpoint found for $NAME seed $SEED — skipping eval"
        fi
    done
done

echo ""
echo "=== Multi-seed aggregation (results/s2) ==="
python3 aggregate_results.py results/s2

echo ""
echo "=== All done. See results/s2/ for per-seed JSON summaries. ==="
