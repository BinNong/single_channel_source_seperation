#!/bin/bash
# =============================================================================
# One-click training and evaluation script for Communication Signal BSS
# =============================================================================

set -e

echo "============================================"
echo "Comm-BSS: Complex Lightweight CNN Training"
echo "============================================"

# Create directories
mkdir -p checkpoints results runs

# ---- Configuration ----
EPOCHS=100
BATCH_SIZE=16
LR=1e-3
HIDDEN=32
LAYERS=4
LOSS="mse"

# ---- Train Proposed Model (Complex CNN + SE) ----
echo ""
echo "[Step 1] Training Proposed Model: Complex CNN + SE"
echo "--------------------------------------------"
python train.py \
    --model complex_cnn_se \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden $HIDDEN \
    --layers $LAYERS \
    --loss $LOSS \
    --name run1

# ---- Train Ablation: Complex CNN without SE ----
echo ""
echo "[Step 2] Training Ablation: Complex CNN without SE"
echo "--------------------------------------------"
python train.py \
    --model complex_cnn_no_se \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --hidden $HIDDEN \
    --layers $LAYERS \
    --loss $LOSS \
    --name run1

# ---- Train Baseline: Real-valued CNN ----
echo ""
echo "[Step 3] Training Baseline: Real-valued CNN"
echo "--------------------------------------------"
python train.py \
    --model real_baseline \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --loss $LOSS \
    --name run1

# ---- Evaluate All Models ----
echo ""
echo "[Step 4] Evaluating All Models"
echo "--------------------------------------------"

for model in complex_cnn_se complex_cnn_no_se real_baseline; do
    CKPT="checkpoints/${model}_h${HIDDEN}_l${LAYERS}_bs${BATCH_SIZE}_lr${LR}_${LOSS}_run1_best.pt"
    if [ -f "$CKPT" ]; then
        echo "Evaluating $model..."
        python evaluate.py \
            --model $model \
            --checkpoint $CKPT \
            --hidden $HIDDEN \
            --layers $LAYERS \
            --output_dir results/$model
    fi
done

echo ""
echo "============================================"
echo "All done! Check results/ for outputs."
echo "============================================"
