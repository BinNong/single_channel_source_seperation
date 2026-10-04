#!/bin/bash
# =============================================================================
# Paper 5 — reviewer-2 multi-test-seed: SUPPLEMENT for the S2 (b) mse config.
#
# run_testseed_robustness.sh had a name collision: paper4's slot mse
# checkpoints and paper5's own S2 (b) mse checkpoints share the SAME
# basename (slot_h64_l4_k13_bs16_lr0.001_mse_s4X_best.pt), so the S2-loop
# outputs were skipped as "(done)".  The paper5 mse evals are re-run here
# into a SEPARATE out_dir (results/testseed_robustness_s2b/) — merge at
# aggregation time.
#
# Queued on the server to start only after run_lambda_sweep.sh finishes
# (GPU memory is tight: 5.2GB resident services + training).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH

N_PER_CELL=100
OUT=results/testseed_robustness_s2b
mkdir -p "$OUT"

for TS in 100001 100002 100003; do
    for SEED in 42 43 44 45 46; do
        ckpt="checkpoints/slot_h64_l4_k13_bs16_lr0.001_mse_s${SEED}_best.pt"
        [ -e "$ckpt" ] || { echo "WARNING: missing $ckpt"; continue; }
        name=$(basename "$ckpt" _best.pt)
        out="$OUT/eval_${name}_ts${TS}.json"
        if [[ -f "$out" ]]; then echo "skip $name ts$TS (done)"; continue; fi
        echo "=== S2(b) $name (ts$TS) ==="
        python evaluate.py --checkpoint "$ckpt" --ser_comp \
            --n_per_cell "$N_PER_CELL" --test_seed "$TS" \
            --out_dir "$OUT" 2>&1 | tail -3
    done
done
echo "=== S2(b) supplement done ==="
