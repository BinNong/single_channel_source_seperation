#!/usr/bin/env bash
# Paper 6 — E-C (review 3) full run: demodulation-aware fine-tuning control.
# Server-side driver (GPU server, /data/experiment/paper6_sync_jd).
# 10 fine-tune runs (5 seeds x lambda_ser in {0.1, 1.0}) + K=2 evaluation.
set -uo pipefail
cd /data/experiment/paper6_sync_jd
PY=/data/experiment/venv_bss/bin/python

for seed in 42 43 44 45 46; do
  for lser in 0.1 1.0; do
    echo "=== fine-tune seed=$seed lambda_ser=$lser ($(date)) ==="
    $PY train_finetune_demod.py --seed "$seed" --lambda_ser "$lser" \
        --epochs 15 --lr 1e-4
  done
done

echo "=== evaluation (K=2, test seed 99999, n_per_cell=100) ($(date)) ==="
$PY eval_finetune_demod.py --n_per_cell 100 --seeds 42 43 44 45 46 \
    --lser_values 0.1 1.0 --variants sym psp

echo "ALL DONE ($(date))"
