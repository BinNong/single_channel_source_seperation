#!/bin/bash
# Embedding-dim ablation: 16/32/128 (64 = baseline, done) x 3 seeds
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for DIM in 16 32 128; do
  for SEED in 42 43 44; do
    echo ""
    echo "=== embed_dim=$DIM seed=$SEED ==="
    $PY train.py \
        --epochs 100 --train_samples 2000 --val_samples 400 \
        --batch_size 16 --lr 1e-3 --seed $SEED \
        --embed_dim $DIM --name emb$DIM
    CKPT=$(ls -t checkpoints/openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed${SEED}_emb${DIM}_best.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
        echo "=== Evaluating $CKPT ==="
        $PY evaluate.py --checkpoint "$CKPT" --n_per_snr 200 --n_per_snr_uu 100
    else
        echo "WARNING: no checkpoint for dim=$DIM seed=$SEED — skipping eval"
    fi
  done
done
echo ""
echo "=== Embed-dim ablation complete ==="
