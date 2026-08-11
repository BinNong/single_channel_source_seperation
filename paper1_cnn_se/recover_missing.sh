#!/bin/bash
# Quick recovery: retrain Real s42,s43 and ConvTasNet s42 (checkpoints lost to naming bug)
set -e
source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
cd /data/work/comm_bss_project

EPOCHS=25
LR=5e-3
SAMPLES=15000
VAL=3000

echo "=== Recovery: real_baseline s42 ==="
python train.py --model real_baseline --epochs $EPOCHS --loss combined --lr $LR --hidden 64 --train_samples $SAMPLES --val_samples $VAL --seed 42 --freq_gap 5 --name pub_s42

echo "=== Recovery: real_baseline s43 ==="
python train.py --model real_baseline --epochs $EPOCHS --loss combined --lr $LR --hidden 64 --train_samples $SAMPLES --val_samples $VAL --seed 43 --freq_gap 5 --name pub_s43

echo "=== Recovery: conv_tasnet s42 ==="
python train.py --model conv_tasnet --epochs $EPOCHS --loss combined --lr $LR --hidden 64 --train_samples $SAMPLES --val_samples $VAL --seed 42 --freq_gap 5 --name pub_s42

echo "=== Recovery complete ==="
ls -lt /data/work/comm_bss_project/checkpoints/real_baseline_*pub_s4*_best.pt /data/work/comm_bss_project/checkpoints/conv_tasnet_*pub_s4*_best.pt 2>/dev/null