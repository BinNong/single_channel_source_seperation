#!/bin/bash
# =============================================================================
# Paper 3 — Open-Set SC-BSS: One-shot training + evaluation pipeline.
#
# Usage:
#   bash run.sh                      # full pipeline (5 seeds, train + eval)
#   bash run.sh 1                    # single seed (42 only)
#   bash run.sh smoke                # MVP smoke test (1 epoch, 1 seed)
#
# Runs locally; requires Python 3.10+ with PyTorch (GPU recommended,
# CPU works but is much slower). See README.md for setup.
# =============================================================================
set -e

cd "$(dirname "$0")"

MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
    # MVP: 1 epoch, 1 seed, tiny sample counts — for verifying the pipeline
    echo "=== Smoke test (1 epoch, seed 42) ==="
    python3 train.py \
        --epochs 1 --train_samples 64 --val_samples 32 \
        --batch_size 8 --seed 42 --name smoke
    echo "=== Smoke evaluation ==="
    LATEST=$(ls -t checkpoints/openset_cse_*smoke*_best.pt 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        python3 evaluate.py --checkpoint "$LATEST" \
            --n_per_snr 20 --n_per_snr_uu 10
    else
        echo "No checkpoint found under checkpoints/ — smoke training may have failed"
    fi
    exit 0
fi

if [ "$MODE" = "1" ]; then
    SEEDS="42"
else
    SEEDS="42 43 44 45 46"
fi

for SEED in $SEEDS; do
    echo ""
    echo "=== Seed $SEED ==="
    python3 train.py \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed $SEED

    # Best checkpoint for this seed
    CKPT=$(ls -t checkpoints/openset_cse_h32_l4_bs16_lr0.001_*_seed${SEED}_best.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "=== Evaluating $CKPT ==="
        python3 evaluate.py --checkpoint "$CKPT" \
            --n_per_snr 200 --n_per_snr_uu 100
    else
        echo "WARNING: no checkpoint found for seed $SEED — skipping eval"
    fi
done

echo ""
echo "=== All done. See results/ for per-seed summaries and OOD score dumps. ==="