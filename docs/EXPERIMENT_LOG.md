# Experiment Log — Paper1 (C-SE for SC-BSS)

This document records every training run, hyperparameter, and result that backs a number in the paper. Use it to reproduce.

---

## Phase 0 — Setup

| Item | Value |
|------|-------|
| Server | Lab GPU server (8 GB NVIDIA GeForce RTX 4060) |
| Python | 3.12.3 |
| PyTorch | 2.4.0 + CUDA 12.1 |
| venv | project-local virtual environment |
| Working directory | `paper1_cnn_se/` |
| Seed (default) | 42 |

---

## Phase 1 — Original 4 baselines (3 seeds)

Command template:
```bash
python train.py --model {MODEL} --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 --freq_gap 5 --n_seeds 3 --name pub
```

| Model | `--hidden` | `--layers` | Checkpoint pattern | SDR @ 10 dB (mean ± std, 3 seeds) |
|-------|------------|------------|-------------------|------------------------------------|
| `complex_cnn_se` (proposed) | 64 | 4 | `..._pub_s{42,43,44}_best.pt` | **2.75 ± 0.03** |
| `complex_cnn_no_se` | 64 | 4 | `..._pub_s{42,43,44}_best.pt` | 1.56 ± 0.18 |
| `real_baseline` | 64 | 6 (default) | `..._pub_s{42,43,44}_best.pt` | 1.95 ± 0.12 |
| `conv_tasnet` | n/a | n/a | `..._pub_s{42,43,44}_best.pt` | 3.23 ± 0.40 |

**Phase 1 was misleading on variance.** Three seeds happen to give a small std for the proposed model. Five seeds reveal a much wider spread.

---

## Phase 2 — Param-matched baselines (5 seeds each)

### 2a. `complex_cnn_no_se` matched (H=70, ~242K params)

```bash
python train.py --model complex_cnn_no_se --hidden 70 --layers 4 \
    --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 --n_seeds 5 --freq_gap 5 --name pm235k
```

Per-seed SDR:

| Seed | SDR (dB) | SI-SDR (dB) | SIR (dB) | Regime |
|------|----------|-------------|----------|--------|
| 42 | 1.59 | -1.02 | 20.66 | collapse |
| 43 | 2.66 | -0.99 | 5.92 | working |
| 44 | 1.59 | -1.02 | 20.80 | collapse |
| 45 | 1.62 | -0.91 | 22.19 | collapse |
| 46 | 1.60 | -0.96 | 20.51 | collapse |

Mean ± std over 5 seeds: **SDR = 1.81 ± 0.48 dB**, **SIR = 18.01 ± 6.79 dB**.

### 2b. `real_baseline` matched (H=80, L=12, ~237K params)

```bash
python train.py --model real_baseline --baseline_hidden 80 --baseline_layers 12 \
    --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 --n_seeds 5 --freq_gap 5 --name pm235k
```

Per-seed SDR:

| Seed | SDR (dB) | SIR (dB) | Regime |
|------|----------|----------|--------|
| 42 | 2.66 | 5.73 | working |
| 43 | 2.65 | 5.35 | working |
| 44 | 2.59 | 5.70 | working |
| 45 | 1.56 | 20.23 | collapse |
| 46 | 1.55 | 21.00 | collapse |

Mean ± std over 5 seeds: **SDR = 2.20 ± 0.59 dB**, **SIR = 11.60 ± 8.24 dB**.

---

## Phase 3 — Proposed C-SE (5 seeds)

```bash
python train.py --model complex_cnn_se --hidden 64 --layers 4 \
    --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 --n_seeds 5 --freq_gap 5 --name pub
```

Per-seed SDR:

| Seed | SDR (dB) | SI-SDR (dB) | SIR (dB) | Regime |
|------|----------|-------------|----------|--------|
| 42 | 2.79 | -0.63 | 5.20 | working |
| 43 | 2.74 | -0.65 | 5.73 | working |
| 44 | 2.74 | -0.53 | 5.71 | working |
| 45 | 1.62 | -0.78 | 19.55 | collapse |
| 46 | 1.64 | -0.69 | 20.57 | collapse |

Mean ± std over 5 seeds: **SDR = 2.31 ± 0.62 dB**, **SIR = 11.35 ± 7.96 dB**.

---

## Phase 4 — Conv-TasNet (3 seeds)

```bash
python train.py --model conv_tasnet \
    --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 --n_seeds 3 --freq_gap 5 --name pub
```

| Seed | SDR (dB) | SIR (dB) | Note |
|------|----------|----------|------|
| 42 | 3.43 | 5.54 | working |
| 43 | 2.77 | 6.69 | weak but not collapsed |
| 44 | 3.50 | 5.42 | working |

Mean ± std over 3 seeds: **SDR = 3.23 ± 0.40 dB**.

---

## Phase 5 — Pooling ablation (2 seeds per variant, due to compute budget)

```bash
for PM in power magnitude mean_power; do
    python train.py --model complex_cnn_se --hidden 64 --layers 4 \
        --epochs 25 --loss combined --lr 5e-3 \
        --train_samples 15000 --val_samples 3000 --freq_gap 5 \
        --se_pooling_mode $PM --n_seeds 2 --name pool_$PM
done
```

| Variant | SDR (s42) | SDR (s43) | Mean | SIR (mean) |
|---------|-----------|-----------|------|-------------|
| `power` | 2.78 | 2.71 | **2.75** | 5.61 |
| `magnitude` | 2.77 | 2.74 | **2.76** | 5.44 |
| `mean+power` (collapse) | 1.66 | 1.58 | **1.62** | 20.67 |

The `mean+power` variant consistently collapses with the same SIR signature as the no-SE baseline.

---

## Phase 6 — Micro-frequency generalisation (Δf ∈ U(0,5) Hz)

### 6a. Proposed C-SE trained on U(0,5) Hz

```bash
python train.py --model complex_cnn_se --hidden 64 --layers 4 \
    --epochs 25 --loss combined --lr 5e-3 \
    --train_samples 15000 --val_samples 3000 \
    --freq_gap_min 0.0 --freq_gap_max 5.0 \
    --name micro_gap0_5
```

Evaluation at extended gaps:

| Gap (Hz) | SDR (dB) | SI-SDR (dB) | SIR (dB) |
|----------|----------|-------------|----------|
| 0.0 | 3.33 | 0.74 | 6.47 |
| 0.1 | 3.31 | 0.67 | 6.53 |
| 0.5 | 3.33 | 0.75 | 6.43 |
| 1.0 | 3.33 | 0.74 | 6.44 |
| 2.0 | 3.28 | 0.63 | 6.50 |
| 5.0 | 3.30 | 0.66 | 6.55 |
| 10.0 | 3.29 | 0.66 | 6.42 |
| 50.0 | 3.35 | 0.76 | 6.49 |
| 100.0 | 3.21 | 0.46 | 6.63 |
| 500.0 | 2.91 | -0.12 | 7.57 |

The trained checkpoint holds $\text{SDR}\approx\SI{3.3}{dB}$ across the entire $[0, \SI{500}{Hz}]$ range, including the unseen co-frequency limit $\Delta f=\SI{0}{Hz}$.

### 6b. no-SE matched (H=70) trained on U(0,5) Hz

Same training pattern with `--model complex_cnn_no_se --hidden 70 --layers 4`.

Output: `results/phase_results/microfreq_complex_cnn_no_se/complex_cnn_no_se_freq_offset.json`

Result: SDR ≈ 1.89 dB across all gaps; SIR ≈ 21.5 dB (collapsed output, consistent with Table 1).

---

## Phase 7 — Cross-domain baselines (3 seeds each)

### 7a. CNSE (Hou & Gao 2022), scaled-down to fit 8 GB GPU

- Implementation: `paper1_cnn_se/models.py::CNSE`
- Architecture follows Figs. 1-5 of the original paper (3 Conv1D(256, k=16) encoder, 3 stacked blocks of [3 SepBlocks + 1 SEBlock], 3 ConvTranspose1D decoder)
- Hyperparameter changes from original: `hidden=256` (vs 512), `kernel_size=3` (vs 2; preserves length with `padding=dilation`)
- LR: original paper used 1e-3; **lr=5e-3 caused NaN losses**, so we use 1e-3

```bash
for SEED in 42 43 44; do
    python train.py --model cnse --cnse_hidden 256 \
        --epochs 25 --loss combined --lr 1e-3 \
        --train_samples 15000 --val_samples 3000 --freq_gap 5 \
        --name baseline --seed $SEED
done
```

Per-seed SDR:

| Seed | SDR (dB) | SIR (dB) | Regime |
|------|----------|----------|--------|
| 42 | 3.42 | 6.65 | working |
| 43 | 1.89 | 21.58 | collapse |
| 44 | 1.81 | 21.71 | collapse |

Mean ± std over 3 seeds: **SDR = 2.38 ± 0.91 dB**, **SIR = 16.65 ± 8.65 dB**.

### 7b. S4-UNET (Gao et al. 2026), scaled-down to fit 8 GB GPU

- Implementation: `paper1_cnn_se/models.py::S4UNET`
- Architecture: U-Net with TSEM (Temporal State Enhancement Module) + S4D state-space blocks
- Hyperparameter changes from original: `base_channels=16` (vs 32), `state_dim=16` (vs 64), 4+3 stages (vs 5+4); S4D simplified to diagonal-A FFT convolution

```bash
for SEED in 42 43 44; do
    python train.py --model s4unet --s4_base_channels 16 --s4_state_dim 16 \
        --epochs 25 --loss combined --lr 5e-3 \
        --train_samples 15000 --val_samples 3000 --freq_gap 5 \
        --name baseline --seed $SEED
done
```

Per-seed SDR:

| Seed | SDR (dB) | SIR (dB) | Regime |
|------|----------|----------|--------|
| 42 | 3.06 | 5.83 | working |
| 43 | 3.05 | 5.91 | working |
| 44 | 3.05 | 5.58 | working |

Mean ± std over 3 seeds: **SDR = 3.05 ± 0.00 dB**, **SIR = 5.77 ± 0.17 dB**.

S4-UNET is the only model in this study that did **not** collapse on any seed.

---

## Aggregated results (paper Table 1, 5-seed)

| Model | Params | N | SI-SDR (dB) | SDR (dB) | SIR (dB) |
|-------|--------|---|-------------|----------|----------|
| Complex CNN + SE (Proposed) | 235 K | 5 | -0.66 ± 0.09 | **2.31 ± 0.62** | 11.35 ± 7.96 |
| Complex CNN no-SE (matched, H=70) | 242 K | 5 | -0.98 ± 0.06 | 1.81 ± 0.48 | 18.01 ± 6.79 |
| Real-Valued CNN (matched, H=80, L=12) | 237 K | 5 | -0.94 ± 0.07 | 2.20 ± 0.59 | 11.60 ± 8.24 |
| Complex Conv-TasNet | 817 K | 3 | 0.58 ± 0.92 | 3.23 ± 0.40 | 5.88 ± 0.70 |
| CNSE (scaled, Hou & Gao 2022) | 6.69 M | 3 | 1.99 ± 0.21 | 2.38 ± 0.91 | 16.65 ± 8.65 |
| S4-UNET (scaled, Gao et al. 2026) | 1.57 M | 3 | 0.01 ± 0.12 | **3.05 ± 0.00** | 5.77 ± 0.17 |

---

## Statistical significance (Wilcoxon signed-rank, two-sided, on per-seed SDR)

| Comparison | Δ SDR (dB) | p-value | Significant at α=0.05? |
|------------|-----------|---------|------------------------|
| C-SE vs no-SE matched (5 vs 5) | +0.49 | 0.125 | No |
| C-SE vs Real matched (5 vs 5) | +0.11 | 0.062 | Borderline |
| C-SE vs Conv-TasNet (5 vs 3) | -0.92 | 0.250 | No |
| C-SE vs CNSE (5 vs 3) | -0.07 | 0.500 | No |
| C-SE vs S4-UNET (5 vs 3) | -0.74 | 0.250 | No |

**Honest conclusion**: No comparison reaches α=0.05. C-SE is competitive but not statistically dominant.

---

## Resource accounting

| Phase | Runs | Wall-clock (8 GB RTX 4060) |
|-------|------|----------------------------|
| Phase 1 (original 4 × 3 seeds) | 12 | ~10 hours |
| Phase 2 (matched × 5 seeds × 2) | 10 | ~15 hours |
| Phase 3 (C-SE × 5 seeds) | 5 | ~7 hours |
| Phase 4 (Conv-TasNet × 3 seeds) | 3 | ~5 hours |
| Phase 5 (pooling × 2 seeds × 3 variants) | 6 | ~9 hours |
| Phase 6 (micro-freq × 1 seed × 2 models) | 2 | ~3 hours |
| Phase 7 (CNSE × 3 seeds, S4-UNET × 3 seeds) | 6 | ~4 hours |
| **Total** | **44** | **~53 hours** |

(Reproducible from `cd paper1_cnn_se && bash run_pmatch_full.sh; bash run_next_phases.sh; bash run_5seeds.sh; bash eval_5seeds_full.sh; cd .. && bash paper/build.sh`)