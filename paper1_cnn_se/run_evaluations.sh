#!/bin/bash
# Full evaluation: per-mod, freq-offset, SER
set -e
source /data/miniconda/etc/profile.d/conda.sh
conda activate torch
cd /data/work/comm_bss_project

OUT="results/pub_2026"
mkdir -p $OUT $OUT/freq_offset $OUT/per_mod

HIDDEN=64

# ============ Per-modulation evaluation for each model+seed ============
for MODEL in complex_cnn_se complex_cnn_no_se real_baseline conv_tasnet; do
    for SEED in 42 43 44; do
        case $MODEL in
            complex_cnn_se)       PREFIX="complex_cnn_se" ;;
            complex_cnn_no_se)    PREFIX="complex_cnn_no_se" ;;
            real_baseline)        PREFIX="real_baseline" ;;
            conv_tasnet)          PREFIX="conv_tasnet" ;;
        esac
        # For Real s42, s43 we used --name pub_s42, pub_s43; for s44 and others we used pub
        CKPT_44="checkpoints/${PREFIX}_h${HIDDEN}_l4_bs16_lr0.005_combined_pub_s44_best.pt"
        CKPT_DEFAULT="checkpoints/${PREFIX}_h${HIDDEN}_l4_bs16_lr0.005_combined_pub_s${SEED}_best.pt"
        if [ "$SEED" = "44" ] && [ -f "$CKPT_44" ]; then
            CKPT=$CKPT_44
        elif [ -f "$CKPT_DEFAULT" ]; then
            CKPT=$CKPT_DEFAULT
        else
            CKPT=$(ls -t checkpoints/${PREFIX}_*pub*s${SEED}*_best.pt 2>/dev/null | head -1)
        fi
        if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
            echo "WARN: no checkpoint found for $MODEL s$SEED"
            continue
        fi
        echo "=== Eval: $CKPT ==="
        python evaluate.py --model $MODEL --checkpoint $CKPT \
            --hidden $HIDDEN --per_mod \
            --output_dir "$OUT/per_mod/${MODEL}_s${SEED}" 2>&1 | tail -3
        # Freq-offset robustness
        python eval_freq_offset.py --model $MODEL --checkpoint $CKPT \
            --hidden $HIDDEN --separations 5 10 50 100 200 500 \
            --output_dir "$OUT/freq_offset/${MODEL}_s${SEED}" 2>&1 | tail -3
    done
done

# ============ Gap-trained models evaluation ============
for GAP in 10 50 100 200 500; do
    CKPT="checkpoints/complex_cnn_se_h${HIDDEN}_l4_bs16_lr0.005_combined_gap${GAP}_best.pt"
    if [ ! -f "$CKPT" ]; then continue; fi
    echo "=== Eval gap=$GAP ==="
    python eval_freq_offset.py --model complex_cnn_se --checkpoint $CKPT \
        --hidden $HIDDEN --separations 5 10 50 100 200 500 \
        --output_dir "$OUT/freq_offset/gap${GAP}" 2>&1 | tail -3
done

echo "=== ALL EVALUATIONS DONE ==="
ls $OUT