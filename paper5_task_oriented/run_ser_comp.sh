#!/bin/bash
# =============================================================================
# Paper 5 — S1 mismatch quantification (no retraining; server-side).
#
# Re-scores the 15 paper4 MSE-anchored checkpoints (which live in
# ../paper4_open_world/checkpoints/ on the GPU server) with the paper-5
# evaluate.py, which reports the compensated SER **and** the Gray-mapped
# BER (ser_comp.py, same decisions).  Plus the compensated mixture
# baseline on the same grid.
#
# Results go to results/s1_mismatch/ (SER+BER, n_per_cell=100 per plan v2
# O8); the paper4 raw/compensated JSONs are untouched.
#
# Usage: bash run_ser_comp.sh [n_per_cell]     (default 100)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export PATH=/data/experiment/venv_bss/bin:$PATH
N_PER_CELL=${1:-100}
P4_CKPT_DIR=../paper4_open_world/checkpoints
mkdir -p results/s1_mismatch results/ser

echo "=== Compensated mixture baseline, SER+BER (n_per_cell=$N_PER_CELL) ==="
python ser_comp.py --n_per_cell "$N_PER_CELL"

for ckpt in "$P4_CKPT_DIR"/*_mse_s4[2-6]_best.pt; do
    name=$(basename "$ckpt" _best.pt)
    out="results/s1_mismatch/eval_${name}.json"
    if [[ -f "$out" ]]; then echo "skip $name (done)"; continue; fi
    echo "=== $name (n_per_cell=$N_PER_CELL, compensated SER+BER) ==="
    python evaluate.py --checkpoint "$ckpt" --ser_comp \
        --n_per_cell "$N_PER_CELL" --out_dir results/s1_mismatch 2>&1 | tail -5
done
echo "All S1 mismatch evals done."
