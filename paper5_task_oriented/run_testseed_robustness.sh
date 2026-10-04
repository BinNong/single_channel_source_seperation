#!/bin/bash
# =============================================================================
# Paper 5 — reviewer-2 multi-test-seed robustness (WCL letter v2, MC5).
#
# The headline grid is deterministic at test seed 99999; the reviewer
# asks whether the S1 flat-gap and the S2 per-K sign structure survive
# independent test populations (different channels/symbols/fading).
# Eval-only, no training: re-scores existing checkpoints on three fresh
# deterministic test grids (seeds 100001/100002/100003).
#
#   - mixture baseline (ser_comp.py --test_seed)
#   - S1 honest pool: 10 paper4 slot/specialist MSE checkpoints
#   - S2 fair-additive pair: (b) mse + (d) ser_mse, 5 seeds each
#
# Outputs: results/testseed_robustness/ (baseline_ts*.json, eval_*_ts*.json)
#
# GPU server: nohup bash run_testseed_robustness.sh > results/testseed_robustness.log 2>&1 &
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH

N_PER_CELL=100
P4_CKPT_DIR=../paper4_open_world/checkpoints
OUT=results/testseed_robustness
mkdir -p "$OUT"

for TS in 100001 100002 100003; do
    echo ""
    echo "##################### test seed $TS #####################"

    echo "=== mixture baseline (test seed $TS) ==="
    python ser_comp.py --n_per_cell "$N_PER_CELL" --test_seed "$TS" \
        --out "$OUT/baseline_ts${TS}.json" 2>&1 | tail -2

    for ckpt in "$P4_CKPT_DIR"/slot_*_mse_s4[2-6]_best.pt \
                "$P4_CKPT_DIR"/specialist_*_mse_s4[2-6]_best.pt; do
        [ -e "$ckpt" ] || continue
        name=$(basename "$ckpt" _best.pt)
        out="$OUT/eval_${name}_ts${TS}.json"
        if [[ -f "$out" ]]; then echo "skip $name ts$TS (done)"; continue; fi
        echo "=== $name (ts$TS) ==="
        python evaluate.py --checkpoint "$ckpt" --ser_comp \
            --n_per_cell "$N_PER_CELL" --test_seed "$TS" \
            --out_dir "$OUT" 2>&1 | tail -3
    done

    for cfg in mse ser_mse; do
        for SEED in 42 43 44 45 46; do
            ckpt="checkpoints/slot_h64_l4_k13_bs16_lr0.001_${cfg}_s${SEED}_best.pt"
            [ -e "$ckpt" ] || { echo "WARNING: missing $ckpt"; continue; }
            name=$(basename "$ckpt" _best.pt)
            out="$OUT/eval_${name}_ts${TS}.json"
            if [[ -f "$out" ]]; then echo "skip $name ts$TS (done)"; continue; fi
            echo "=== $name (ts$TS) ==="
            python evaluate.py --checkpoint "$ckpt" --ser_comp \
                --n_per_cell "$N_PER_CELL" --test_seed "$TS" \
                --out_dir "$OUT" 2>&1 | tail -3
        done
    done
done

echo "=== multi-test-seed robustness done ==="
