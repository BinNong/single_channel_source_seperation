# Paper 3 — Open-Set Single-Channel BSS for Communication Signals

> **Status:** experiments COMPLETE for ICASSP scope (2026-08-21). Main result:
> pooled OOD AUROC ≈ 0.50 is an SNR-averaging artifact; a training-free
> **SNR-routed ensemble** (Energy ≤ 0 dB, Prototype ≥ 5 dB) reaches
> weighted-avg AUROC **0.625 ± 0.031** (oracle 0.641) and is robust to
> σ ≤ 6 dB SNR-estimation error.  Full history in `EXPERIMENT_LOG.md`;
> paper narrative in `OUTLINE.md`.
> **Target venue:** Physical Communication (Elsevier).
> **Backbone:** Paper 1's ComplexLightweightSepNet (C-SE) + per-source
> ModulationHead, 69K params total.

---

## What this paper adds

Existing SC-BSS literature assumes both sources come from a fixed modulation set known at training time. **This paper** relaxes that assumption:

- **Closed-set training:** the separation backbone and the per-source modulation head are trained only on the 4 *known* modulations — BPSK, QPSK, 8PSK, 16QAM.
- **Open-set test:** at inference, the model must (a) still separate two overlapping signals, (b) correctly identify each source's modulation if known, and (c) flag each source as *unknown* if it belongs to a modulation the model has never seen.

Unknown modulations used in this paper:
- **64QAM** — higher-order extension of the known PSK-QAM family
- **π/4-DQPSK** — differential QPSK with phase rotation
- **MSK** — continuous-phase FSK (reuses paper1's generator)
- **OFDM-QPSK** — multi-carrier modulation (IFFT of QPSK subcarriers + CP)

We propose **per-source** open-set detection (each separated source gets its own OOD score), as opposed to mixture-level OOD used in prior work.

### OOD scoring methods
1. **Energy score** — `-logsumexp(logits/T)` baseline (ODIN-style).
2. **Prototype distance** — distance from each per-source embedding to the nearest known-class prototype.
3. **VOS** — Virtual Outlier Synthesis; min distance to extrapolated virtual outliers (Du et al., ICLR 2022).

**Key finding:** all three are at chance (AUROC ≈ 0.50) when pooled across
SNR, but strongly complementary per SNR — Prototype/VOS work at ≥ 5 dB,
Energy at ≤ 0 dB.  The paper's proposed method is therefore the
**SNR-routed ensemble** (`ensemble_analysis.py`): Energy for SNR ≤ 0 dB,
Prototype otherwise; training-free, weighted-avg AUROC 0.625.

---

## Project structure

```
paper3_open_set/
├── README.md                  # this file
├── run.sh                     # one-shot training + evaluation pipeline
├── config.py                  # central hyperparameters
├── data_generator_extended.py # paper1 base + 3 new modulations + open-set protocols
├── models.py                  # OpenSetCSE (C-SE backbone + ModulationHead)
├── losses.py                  # pit_multi_task_loss (joint SI-SDR + CE, PIT-aligned)
├── ood_scores.py              # energy / prototype / VOS scoring
├── open_set_metrics.py        # AUROC / AUPR / FPR@95 / OSCR
├── train.py                   # training loop with TensorBoard (+ --loss_lambda_center ablation)
├── evaluate.py                # 4-part evaluation (closed-set, OOD, per-SNR, per-mod)
├── ensemble_analysis.py       # SNR-routed ensemble + SNR-noise sensitivity (main method)
├── make_figs.py               # regenerate paper figures from results/*.npz
├── run_center_loss.sh         # center-loss ablation (5 seeds, negative result)
├── run_embed_dim_ablation.sh  # embedding-dim ablation (16/32/128 × 3 seeds)
├── figures/                   # generated paper figures (pdf + png)
├── checkpoints/               # training outputs
├── runs/                      # TensorBoard logs
├── results/                   # evaluation outputs (CSV + JSON + .npz)
└── utils.py → ../paper1_cnn_se/utils.py  # symlink for SI-SDR / SDR / SIR metrics
```

---

## Quick start

```bash
# 0. Install dependencies (from the repository root)
pip install -r requirements.txt

# 1. Smoke test the pipeline (1 epoch, tiny dataset)
cd paper3_open_set
bash run.sh smoke

# 2. Single-seed full training + evaluation
bash run.sh 1

# 3. Full 5-seed run
bash run.sh
```

### Manual steps

```bash
# Train
python train.py --epochs 100 --batch_size 16 --seed 42

# Evaluate a trained checkpoint
python evaluate.py --checkpoint checkpoints/openset_cse_..._best.pt
```

### Hyperparameter reference (defaults in `config.py`)

| Flag | Default | Notes |
|---|---|---|
| `--hidden` | 32 | C-SE hidden channels (try 16 for "smaller" variant) |
| `--layers` | 4 | C-SE residual blocks |
| `--embed_dim` | 64 | Per-source embedding size for Prototype/VOS |
| `--loss_alpha` | 1.0 | CE vs SI-SDR weighting in joint loss |
| `--epochs` | 100 | |
| `--batch_size` | 16 | |
| `--lr` | 1e-3 | |

---

## Training data

- The training set (`CommBSSOpenSetDataset`) only emits the 4 known modulations (BPSK, QPSK, 8PSK, 16QAM) at SNR ∈ [-5, 20] dB, ~2000 samples per training run.
- Validation is held out from the same pool, 400 samples, seed+1000.
- **Unknown modulations never appear during training**, per the design decision "只用已知 4 类训练 head".

## Test data

Three protocols exposed by `CommBSSOpenSetTestDataset`:

| Protocol | Source 1 | Source 2 | Use case |
|---|---|---|---|
| `kk` | known | known | Closed-set sanity |
| `ku` | known | unknown | **Core novel scenario**: per-source OOD detection |
| `uu` | unknown | unknown | Extreme OOD stress test |

Each protocol uses a fixed deterministic seed (`seed=99999`) and a per-SNR sample count (`--n_per_snr` / `--n_per_snr_uu`).

---

## Evaluation outputs

`python evaluate.py --checkpoint <best.pt>` writes to `results/`:

| File | Contents |
|---|---|
| `<run_name>_summary.json` | Closed-set SI-SDR, per-class accuracy, per-SNR breakdown; OOD AUROC/AUPR/FPR@95/OSCR for each of energy / prototype / vos |
| `<run_name>_ood_scores.npz` | Raw per-sample OOD scores for all 3 methods (known + unknown pools) + raw embeddings + prototypes; useful for t-SNE / paper figures |

---

## Reproducibility

- Per-seed checkpoints saved as `<run_name>_<seed>_best.pt`.
- Training logs (TensorBoard) under `runs/<run_name>/`.
- See `EXPERIMENT_LOG.md` (symlink, once written) for the canonical run log with exact commands and per-seed numbers.

---

## Design decisions (summary)

| Decision | Choice | Rationale |
|---|---|---|
| Unknown modulation set | 64QAM / π/4-DQPSK / MSK / OFDM-QPSK | Diverse: same family, differential, continuous-phase, multi-carrier |
| OOD scoring methods | Prototype + VOS (main); Energy (ablation) | Prototype = metric learning baseline; VOS = strong SOTA |
| Backbone | C-SE (Paper 1) | Smallest, fastest iteration for validating the open-set concept |
| Training data | Only the 4 known modulations | Avoids leaking OOD knowledge; tests true open-set generalization |

---

## Related code

- `paper1_cnn_se/` — Paper 1 (C-SE for SC-BSS); shared `data_generator.py`, `utils.py`
- `paper2_dp_mamba/` — Paper 2 (CDP-Mamba); symlinks the same shared files
- `paper1/website/`, `paper2/website/` — Single-page visualization sites

The OpenSetCSE backbone mirrors Paper 1's ComplexLightweightSepNet (same complex-conv + SE recipe) but adds a per-source modulation head for OOD scoring. The head is shared across sources (permutation-invariant training).