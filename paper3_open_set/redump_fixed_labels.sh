#!/bin/bash
# Re-dump ALL per-SNR-contaminated eval artifacts with the fixed
# repeat->tile known-SNR labels (evaluate.py / refpool_dump.py / odin_dump.py,
# fixed 2026-09-21).  Archives the contaminated dumps to
# results/archive_buggy_snr_labels/ first (move, don't delete).
# Runs 5 nice'd lanes in parallel; does NOT touch the running revision driver.
set -u
cd /data/experiment/paper3_open_set
PY=/data/experiment/venv_bss/bin/python
NICE="nice -n 10"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ---- 1. archive contaminated artifacts --------------------------------------
ARCH=results/archive_buggy_snr_labels
mkdir -p "$ARCH"
log "archiving contaminated dumps to $ARCH"
for pat in '*_ood_scores.npz' '*_summary.json' '*_refpool.npz' \
           '*_odin_eps0.005_T1000.npz' 'robustness_*.json' \
           'lomo_*.json' 'lomo_*_scores.npz'; do
  for f in results/$pat; do
    [ -e "$f" ] && mv "$f" "$ARCH/" && log "  archived $(basename "$f")"
  done
done

BASE=openset_cse_h32_l4_bs16_lr0.001_alpha1.0

laneA() {
  for S in 42 43 44 45 46; do
    log "A: evaluate baseline seed $S"
    $NICE $PY evaluate.py --checkpoint checkpoints/${BASE}_seed${S}_best.pt \
        --batch_size 16 || log "FAILED baseline eval seed $S"
  done
  for S in 42 43 44 45 46; do
    log "A: refpool seed $S"
    $NICE $PY refpool_dump.py --checkpoint checkpoints/${BASE}_seed${S}_best.pt \
        --batch_size 16 || log "FAILED refpool seed $S"
  done
}

laneB() {
  for S in 42 43 44 45 46; do
    log "B: evaluate lc01 seed $S"
    $NICE $PY evaluate.py \
        --checkpoint checkpoints/${BASE}_lc0.1_seed${S}_lc01_best.pt \
        --n_per_snr 200 --n_per_snr_uu 100 || log "FAILED lc01 eval seed $S"
  done
  for S in 42 43 44 45 46; do
    log "B: odin seed $S"
    $NICE $PY odin_dump.py --checkpoint checkpoints/${BASE}_seed${S}_best.pt \
        --eps 0.005 --batch_size 16 || log "FAILED odin seed $S"
  done
}

laneC1() {
  for GAP in 10 50; do
    for S in 42 43 44 45 46; do
      log "C1: evaluate gap $GAP seed $S"
      $NICE $PY evaluate.py --checkpoint checkpoints/${BASE}_seed${S}_best.pt \
          --batch_size 16 --carrier_gap $GAP || log "FAILED gap $GAP seed $S"
    done
  done
}

laneC2() {
  for GAP in 100 500; do
    for S in 42 43 44 45 46; do
      log "C2: evaluate gap $GAP seed $S"
      $NICE $PY evaluate.py --checkpoint checkpoints/${BASE}_seed${S}_best.pt \
          --batch_size 16 --carrier_gap $GAP || log "FAILED gap $GAP seed $S"
    done
  done
}

laneD() {
  for DIM in 16 32 128; do
    for S in 42 43 44; do
      log "D: evaluate emb$DIM seed $S"
      $NICE $PY evaluate.py \
          --checkpoint checkpoints/${BASE}_seed${S}_emb${DIM}_best.pt \
          --n_per_snr 200 --n_per_snr_uu 100 || log "FAILED emb$DIM seed $S"
    done
  done
  log "D: LOMO re-eval seed 42 (was dumped 18:25 with buggy labels)"
  $NICE $PY eval_lomo.py \
      --checkpoint checkpoints/${BASE}_kmQPSK-8PSK-16QAM_seed42_best.pt \
      --n_per_snr 200 || log "FAILED lomo re-eval seed 42"
}

log "launching lanes A B C1 C2 D"
laneA  & PA=$!
laneB  & PB=$!
laneC1 & PC1=$!
laneC2 & PC2=$!
laneD  & PD=$!
wait $PA $PB $PC1 $PC2 $PD
log "REDUMP_ALL_DONE"
