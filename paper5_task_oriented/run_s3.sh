#!/bin/bash
# =============================================================================
# Paper 5 — S3: joint soft-information demodulation (JointLLRHead).
#
# Fixed K=2 (k_min=k_max=2; k_slots stays 4 so the full test grid still
# builds), slot arch, lambda_llr=1.0 + lambda_mse 1.0 — the waveform losses
# are kept so the SAME checkpoint also supports the route-A/B waveform-demod
# comparison; the joint head adds ser_joint/ber_joint in evaluate.py.
#
# Usage:
#   bash run_s3.sh           # 5 seeds
#   bash run_s3.sh 1         # seed 42 only
#   bash run_s3.sh smoke     # 1 epoch, tiny eval
# =============================================================================
set -e
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH
MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
    python3 train.py --arch slot --k_min 2 --k_max 2 \
        --lambda_llr 1.0 --lambda_mse 1.0 \
        --epochs 1 --train_samples 64 --val_samples 32 \
        --batch_size 8 --seed 42 --num_workers 0 --name s3smoke
    CKPT=$(ls -t checkpoints/slot_*s3smoke*_best.pt | head -1)
    python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 4 --ser_comp
    exit 0
fi

if [ "$MODE" = "1" ]; then
    SEEDS="42"
else
    SEEDS="42 43 44 45 46"
fi

mkdir -p results/s3
for SEED in $SEEDS; do
    echo ""
    echo "=== S3 joint head (fixed K=2), seed $SEED ==="
    python3 train.py --arch slot --k_min 2 --k_max 2 \
        --lambda_llr 1.0 --lambda_mse 1.0 \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed $SEED --name jointK2
    CKPT=$(ls -t checkpoints/slot_h64_l4_k22_bs16_lr0.001_jointK2_s${SEED}_best.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "=== Evaluating $CKPT (SER+BER, both routes) ==="
        python3 evaluate.py --checkpoint "$CKPT" --n_per_cell 100 \
            --ser_comp --out_dir results/s3
    else
        echo "WARNING: no checkpoint for jointK2 seed $SEED"
    fi
done
echo ""
echo "=== S3 done. See results/s3/ ==="
