#!/bin/bash
set -u
cd /data/experiment/paper3_open_set
PY=/data/experiment/venv_bss/bin/python
for S in 42 43 44; do
  echo "[$(date "+%F %T")] eval_robustness seed $S (fixed labels)"
  nice -n 10 $PY eval_robustness.py --checkpoint checkpoints/openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed${S}_best.pt --n_per_snr 200     || echo "FAILED robustness seed $S"
done
echo "ROBUSTNESS_ALL_DONE"
