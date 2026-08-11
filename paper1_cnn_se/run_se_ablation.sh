#!/bin/bash
# SE scale_mode ablation experiment
# Compares: real (baseline, already done), complex_mean, separate
# Each variant: 2 seeds × 25 epochs, hidden=64, combined loss, lr=0.005

set -e

source /data/miniconda/etc/profile.d/conda.sh
conda activate torch

cd /data/work/comm_bss_project

echo "============================================"
echo "SE Scale Mode Ablation Experiment"
echo "Started at: $(date)"
echo "============================================"

MODES=("complex_mean" "separate")
SEEDS=(42 43)

for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> Training: se_scale_mode=${MODE}, seed=${SEED}"
        python3 train.py \
            --model complex_cnn_se \
            --hidden 64 \
            --layers 4 \
            --batch_size 16 \
            --lr 0.005 \
            --loss combined \
            --epochs 25 \
            --train_samples 15000 \
            --val_samples 3000 \
            --se_scale_mode ${MODE} \
            --name "ablation_${MODE}" \
            --seed ${SEED}

        echo ">>> Evaluating: se_scale_mode=${MODE}, seed=${SEED}"
        CKPT="checkpoints/complex_cnn_se_h64_l4_bs16_lr0.005_combined_sc${MODE}_ablation_${MODE}_s${SEED}_best.pt"

        if [ -f "$CKPT" ]; then
            python3 evaluate.py \
                --model complex_cnn_se \
                --hidden 64 \
                --layers 4 \
                --checkpoint "$CKPT" \
                --per_mod \
                --output_dir "results/pub_2026/per_mod_fixed/complex_cnn_se_sc${MODE}_s${SEED}"

            echo ">>> Frequency-offset eval: ${MODE}, seed=${SEED}"
            python3 eval_freq_offset.py \
                --checkpoint "$CKPT" \
                --model complex_cnn_se \
                --hidden 64 \
                --layers 4 \
                --output_dir "results/pub_2026/freq_offset/complex_cnn_se_sc${MODE}_s${SEED}"
        else
            echo "ERROR: Checkpoint not found: $CKPT"
        fi
    done
done

echo ""
echo "============================================"
echo "SE Scale Mode Ablation Complete!"
echo "Finished at: $(date)"
echo "============================================"
