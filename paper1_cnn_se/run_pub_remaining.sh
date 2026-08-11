#!/bin/bash
# Resume remaining experiments after server restart.
# Skips already-completed checkpoints.
set -e
source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
cd /data/work/comm_bss_project

EPOCHS=25
LR=5e-3
TRAIN_SAMPLES=15000
VAL_SAMPLES=3000
HIDDEN=64

echo "=============================================="
echo "Resuming experiments at $(date)"
echo "=============================================="

# Helper: skip if best.pt exists for this seed
should_skip() {
    local model=$1
    local seed=$2
    local ckpt="checkpoints/${model}_h64_l4_bs16_lr${LR}_combined_pub_s${seed}_best.pt"
    if [ -f "$ckpt" ]; then
        echo "  [SKIP] $ckpt exists"
        return 0
    fi
    return 1
}

# Run model+seed combos
run_combo() {
    local model=$1
    local seed=$2
    if should_skip $model $seed; then return; fi
    LOG="logs/${model}_s${seed}.log"
    echo "[$(date)] Starting: $model seed=$seed (25 epochs)"
    python train.py \
        --model $model \
        --epochs $EPOCHS \
        --loss combined \
        --lr $LR \
        --hidden $HIDDEN \
        --train_samples $TRAIN_SAMPLES \
        --val_samples $VAL_SAMPLES \
        --seed $seed \
        --freq_gap 5 \
        --name pub \
        2>&1 | tee -a $LOG
    echo "[$(date)] Completed: $model seed=$seed"
}

# Run gap experiment (only 1 seed, no n_seeds)
run_gap() {
    local gap=$1
    local ckpt="checkpoints/complex_cnn_se_h64_l4_bs16_lr${LR}_combined_gap${gap}_best.pt"
    if [ -f "$ckpt" ]; then
        echo "  [SKIP] gap=$gap exists"
        return
    fi
    LOG="logs/se_gap${gap}.log"
    echo "[$(date)] Starting: SE gap=${gap}Hz"
    python train.py \
        --model complex_cnn_se \
        --epochs $EPOCHS \
        --loss combined \
        --lr $LR \
        --hidden $HIDDEN \
        --train_samples $TRAIN_SAMPLES \
        --val_samples $VAL_SAMPLES \
        --freq_gap $gap \
        --name "gap${gap}" \
        2>&1 | tee -a $LOG
    echo "[$(date)] Completed: SE gap=${gap}Hz"
}

# ---- Phase A: NoSE remaining seeds ----
for SEED in 43 44; do
    run_combo complex_cnn_no_se $SEED
done

# ---- Phase B: Real baseline 3 seeds ----
for SEED in 42 43 44; do
    run_combo real_baseline $SEED
done

# ---- Phase C: Conv-TasNet 3 seeds ----
for SEED in 42 43 44; do
    run_combo conv_tasnet $SEED
done

# ---- Phase D: Frequency-gap experiments (SE model) ----
for GAP in 10 50 100 200 500; do
    run_gap $GAP
done

echo "=============================================="
echo "All remaining experiments completed: $(date)"
echo "=============================================="