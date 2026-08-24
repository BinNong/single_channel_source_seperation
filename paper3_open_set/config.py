"""
Paper 3 — Open-Set SC-BSS: Central configuration.

Single source of truth for all hyperparameters. Read at runtime by:
  - train.py  : training loop + checkpointing
  - evaluate.py: dataset construction + inference
  - models.py  : ModulationHead dim + number of known classes
  - ood_scores.py / open_set_metrics.py : thresholds and number of classes
"""

from __future__ import annotations

from data_generator_extended import MOD_KNOWN, MOD_UNKNOWN, NUM_MOD


# ============================================================================
# Signal processing
# ============================================================================
SIGNAL_LENGTH  = 4096
SAMPLE_RATE    = 16000
N_SYMBOLS      = 256
CARRIER_FREQ_1 = 2000.0
CARRIER_FREQ_2 = 2005.0    # default co-frequency gap = 5 Hz
ROLL_OFF       = 0.35
NUM_TAPS       = 64
FADING_TAPS    = 3
SNR_TRAIN_RANGE = (-5.0, 20.0)   # dB
SNR_TEST_POINTS = [-10, -5, 0, 5, 10, 15, 20]


# ============================================================================
# Modulation vocabulary
# ============================================================================
# During training only the 4 known modulations are used; the 4 unknown
# modulations appear ONLY in the test set (per the design decision:
# "只用已知 4 类训练 head").
NUM_KNOWN_CLASSES = len(MOD_KNOWN)     # 4  — classifier output dim
NUM_ALL_MODS      = NUM_MOD             # 8  — only used for metric labelling


# ============================================================================
# OpenSetCSE model
# ============================================================================
# Backbone (C-SE) hyperparameters — start with the Paper-1 defaults; can be
# shrunk for the "smaller C-SE" validation step.
BACKBONE_HIDDEN_CHANNELS = 32
BACKBONE_N_LAYERS        = 4
BACKBONE_KERNEL_ENC      = 7
BACKBONE_KERNEL_HIDDEN   = 3
BACKBONE_KERNEL_DEC      = 7
BACKBONE_USE_SE          = True
BACKBONE_SE_REDUCTION    = 4

# Modulation classification head.
# Input: complex masked-bottleneck features [B, hidden, T] -> global pool ->
# embedding -> logits.  The embedding is the input to Prototype and VOS OOD
# scoring; the logits feed the Energy OOD score.
EMBED_DIM = 64        # size of per-source embedding


# ============================================================================
# Training
# ============================================================================
SEED              = 42
BATCH_SIZE        = 16
NUM_EPOCHS        = 100
LEARNING_RATE     = 1e-3
LOSS_ALPHA_CLS    = 1.0       # weight on classification CE loss vs SI-SDR
                              # alpha=1.0 means CE and SI-SDR contribute
                              # equally (in their natural units). Tune by
                              # monitoring val/sep_loss vs val/cls_loss.
WEIGHT_DECAY      = 0.0
GRAD_CLIP_NORM    = 5.0
LOG_EVERY         = 50        # batches
VAL_EVERY        = 1         # epochs
EARLY_STOP_PATIENCE = 20      # epochs without val improvement


# ============================================================================
# OOD scoring
# ============================================================================
ENERGY_TEMPERATURE = 1.0      # Energy-score softmax temperature
VOS_ALPHA          = 2.0      # VOS extrapolation distance (in std units)
VOS_N_SYNTHETIC    = 100      # synthetic outliers per known class


# ============================================================================
# Paths
# ============================================================================
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
RUN_DIR        = os.path.join(ROOT_DIR, 'runs')
RESULTS_DIR    = os.path.join(ROOT_DIR, 'results')

for _d in (CHECKPOINT_DIR, RUN_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)