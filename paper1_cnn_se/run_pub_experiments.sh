#!/bin/bash
# Full publication experiment pipeline (optimized)
# Reduced samples/epochs for faster turnaround while maintaining statistical validity.
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
echo "Publication Experiment Pipeline (Optimized)"
echo "Date: $(date)"
echo "Models: complex_cnn_se, complex_cnn_no_se, real_baseline, conv_tasnet"
echo "Config: epochs=${EPOCHS}, lr=${LR}, samples=${TRAIN_SAMPLES}, loss=combined"
echo "=============================================="

# ---- Phase A: Multi-model training, 5 Hz gap, 3 seeds ----
echo ""
echo "===== PHASE A: Baseline Training (5 Hz gap, 3 seeds) ====="

for MODEL in complex_cnn_se complex_cnn_no_se real_baseline conv_tasnet; do
    LOG="logs/${MODEL}_pub.log"
    echo "[$(date)] Starting: $MODEL (3 seeds, ${EPOCHS} epochs)"
    python train.py \
        --model $MODEL \
        --epochs $EPOCHS \
        --loss combined \
        --lr $LR \
        --hidden $HIDDEN \
        --train_samples $TRAIN_SAMPLES \
        --val_samples $VAL_SAMPLES \
        --n_seeds 3 \
        --freq_gap 5 \
        --name pub \
        2>&1 | tee -a $LOG
    echo "[$(date)] Completed: $MODEL"
done

# ---- Phase B: Frequency-gap experiments (SE model only) ----
echo ""
echo "===== PHASE B: Frequency-Gap Experiments (SE model) ====="

for GAP in 10 50 100 200 500; do
    LOG="logs/se_gap${GAP}.log"
    echo "[$(date)] Starting: SE model at ${GAP} Hz gap"
    python train.py \
        --model complex_cnn_se \
        --epochs $EPOCHS \
        --loss combined \
        --lr $LR \
        --hidden $HIDDEN \
        --train_samples $TRAIN_SAMPLES \
        --val_samples $VAL_SAMPLES \
        --freq_gap $GAP \
        --name "gap${GAP}" \
        2>&1 | tee -a $LOG
    echo "[$(date)] Completed: SE gap=${GAP}Hz"
done

echo ""
echo "=============================================="
echo "All training completed: $(date)"
echo "=============================================="
