#!/bin/bash
# Center-loss experiment: 5 seeds x 100 epochs, lambda_c=0.1
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for SEED in 42 43 44 45 46; do
    echo ""
    echo "=== Seed $SEED (lambda_c=0.1) ==="
    $PY train.py \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed $SEED \
        --loss_lambda_center 0.1 --name lc01
    CKPT=$(ls -t checkpoints/openset_cse_h32_l4_bs16_lr0.001_*lc0.1*_seed${SEED}_lc01_best.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "=== Evaluating $CKPT ==="
        $PY evaluate.py --checkpoint "$CKPT" --n_per_snr 200 --n_per_snr_uu 100
    else
        echo "WARNING: no checkpoint found for seed $SEED — skipping eval"
    fi
done
echo ""
echo "=== Center-loss 5-seed run complete ==="
