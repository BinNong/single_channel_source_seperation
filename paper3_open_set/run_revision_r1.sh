#!/bin/bash
# =============================================================================
# Paper 3 — Post-rejection revision driver (2026-09-20).
#
# Runs SEQUENTIALLY on the lab RTX 4060 (single GPU):
#   Phase 1  R1-6 robustness evals (eval_robustness.py) on the EXISTING
#            baseline checkpoints, seeds 42/43/44 — SIR sweep {-6..+6} dB,
#            timing offset on, CFO jitter 25 Hz.  No retraining.
#   Phase 2  R1-5/R2-3 LOMO: train 3-known-mod models (held-out in
#            {BPSK,QPSK,8PSK,16QAM}) x seeds 42/43/44 = 12 runs (baseline
#            recipe), each followed by eval_lomo.py (pseudo-OOD validation
#            seed 77777 picks the routing boundary; test seed 99999 with 5
#            unknown classes never touches boundary selection).
#   Phase 3  R1-8 no-SE backbone: train x seeds 42/43/44 = 3 runs, each
#            followed by the standard evaluate.py + ensemble_analysis.py
#            routed-profile analysis.
#
# Estimated wall time ~22-25 h.  Launch with:
#   nohup bash run_revision_r1.sh >> results/revision_r1_driver.log 2>&1 &
# Checkpoint names are tagged (_km<...>_ / _nose_) and never collide with
# the baseline checkpoints.  Nothing is deleted by this script.
# =============================================================================
set -u
cd /data/experiment/paper3_open_set
PY=/data/experiment/venv_bss/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Phase 1: R1-6 robustness evals (baseline ckpts, seeds 42-44) ==="
for S in 42 43 44; do
  CKPT=checkpoints/openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed${S}_best.pt
  if [ ! -f "$CKPT" ]; then log "MISSING $CKPT — skip seed $S"; continue; fi
  log "eval_robustness seed $S ($CKPT)"
  $PY eval_robustness.py --checkpoint "$CKPT" --n_per_snr 200 \
      || log "FAILED robustness seed $S (continuing)"
done

log "=== Phase 2: LOMO (4 held-out mods x seeds 42-44 = 12 train+eval) ==="
for HELD in BPSK QPSK 8PSK 16QAM; do
  case $HELD in
    BPSK)  KM="QPSK,8PSK,16QAM";;
    QPSK)  KM="BPSK,8PSK,16QAM";;
    8PSK)  KM="BPSK,QPSK,16QAM";;
    16QAM) KM="BPSK,QPSK,8PSK";;
  esac
  KMTAG=${KM//,/-}
  for S in 42 43 44; do
    log "LOMO train held=$HELD seed=$S known=$KM"
    $PY train.py --known_mods "$KM" --epochs 100 --train_samples 2000 \
        --val_samples 400 --batch_size 16 --lr 1e-3 --seed $S \
        || { log "FAILED train held=$HELD seed=$S (continuing)"; continue; }
    CKPT=$(ls -t checkpoints/openset_cse_h32_l4_bs16_lr0.001_alpha1.0_km${KMTAG}_seed${S}_best.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then log "NO CKPT held=$HELD seed=$S — skip eval"; continue; fi
    log "LOMO eval held=$HELD seed=$S ($CKPT)"
    $PY eval_lomo.py --checkpoint "$CKPT" --n_per_snr 200 \
        || log "FAILED lomo eval held=$HELD seed=$S (continuing)"
  done
done

log "=== Phase 3: no-SE backbone (R1-8), seeds 42-44 = 3 train+eval ==="
for S in 42 43 44; do
  log "no-SE train seed $S"
  $PY train.py --no_se --epochs 100 --train_samples 2000 --val_samples 400 \
      --batch_size 16 --lr 1e-3 --seed $S \
      || { log "FAILED no-SE train seed $S (continuing)"; continue; }
  CKPT=$(ls -t checkpoints/openset_cse_h32_l4_bs16_lr0.001_alpha1.0_nose_seed${S}_best.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then log "NO CKPT no-SE seed $S — skip eval"; continue; fi
  log "no-SE standard eval seed $S ($CKPT)"
  $PY evaluate.py --checkpoint "$CKPT" --n_per_snr 200 --n_per_snr_uu 100 \
      || { log "FAILED evaluate no-SE seed $S (continuing)"; continue; }
  NPZ="results/$(basename "${CKPT%.pt}")_ood_scores.npz"
  if [ -f "$NPZ" ]; then
    $PY ensemble_analysis.py "$NPZ" --threshold 0 \
        || log "FAILED ensemble no-SE seed $S (continuing)"
  else
    log "NO NPZ $NPZ — skip ensemble"
  fi
done

log "=== ALL PHASES DONE ==="
