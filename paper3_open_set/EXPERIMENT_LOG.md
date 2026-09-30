# Paper 3 — Open-Set SC-BSS: Experiment Log

> **Status:** experiments COMPLETE for the Physical Communication
> submission scope (2026-08-21).  Append entries after each run on the
> remote GPU server.

## Format

```
| Run ID | Date | Seed | Epochs | Notes | Train sep | Train cls | Val SI-SDR | Val cls_acc | Checkpoint |
```

## Entries

<!-- Add entries below.  Use the format above, one row per training run. -->

| Run ID | Date | Seed | Epochs | Notes | Train sep_loss | Train cls_loss | Val SI-SDR (dB) | Val cls_acc | Checkpoint |
|--------|------|------|--------|-------|----------------|----------------|-----------------|-------------|------------|
| _empty_ |  |  |  |  |  |  |  |  |  |

## Evaluation table template

```
| Score method | AUROC | AUPR_in | FPR@95 | OSCR |
|--------------|-------|---------|--------|------|
| energy       |       |         |        |      |
| prototype    |       |         |        |      |
| vos          |       |         |        |      |
```

## Notes / observations

### 2026-07-29 — MVP smoke test (1 epoch, seed 42)

Verified end-to-end pipeline runs cleanly.  1 epoch / 64 samples is too
little to train — all metrics land near their random baseline, as
expected.  This entry exists to confirm the infrastructure is sound
before kicking off the 5-seed full run.

| Run ID | Date | Seed | Epochs | Train sep_loss | Train cls_loss | Val SI-SDR (dB) | Val cls_acc | Checkpoint |
|--------|------|------|--------|----------------|----------------|-----------------|-------------|------------|
| smoke  | 2026-07-29 | 42 | 1 | 9.35 | 2.77 | -2.31 | 0.375 | openset_cse_h32_l4_bs8_lr0.001_alpha1.0_seed42_smoke_best.pt |

Smoke-test evaluation (n_per_snr=20, n_per_snr_uu=10):

| Score method | AUROC | AUPR_in | FPR@95 | OSCR |
|--------------|-------|---------|--------|------|
| energy       | 0.498 | 0.329   | 0.933  | 0.123 |
| prototype    | 0.502 | 0.346   | 0.938  | 0.127 |
| vos          | 0.503 | 0.348   | 0.938  | 0.128 |

Closed-set (kk): SI-SDR = -4.26 ± 4.39 dB, cls_acc = 0.250, per-class:
BPSK=0.143, QPSK=0.357, 8PSK=0.500, 16QAM=0.000.

Infrastructure fixes applied during smoke test:
1. `data_generator_extended.py`: changed `sys.path.insert(0, ...)` to
   `sys.path.append(...)` so paper1 doesn't shadow paper3's modules.
2. Same file: handled 64QAM / π/4-DQPSK in a local
   `_apply_paper1_pipeline` so paper1's hard-coded `generate_symbols`
   (which doesn't know them) is bypassed.  OFDM_QPSK already had its
   own path.
3. `train.py` / `evaluate.py`: defensive `sys.path.insert(0, HERE)` so
   `from models import OpenSetCSE` resolves to OUR models.py.
4. `evaluate.py`: all tensors moved to device before `torch.where`; all
   tensor → numpy via `.detach().cpu().numpy()`.

### 2026-07-29 — Full 5-seed run COMPLETE

`bash run.sh` finished all 5 seeds (42-46) × 100 epochs × 16 batch on
the RTX 4060 in ≈7 hours.

**Aggregate results (5 seeds, n_per_snr=200, n_per_snr_uu=100):**

| seed | closed SI-SDR (dB) | closed cls_acc | energy AUROC | prototype AUROC | vos AUROC |
|------|-------------------:|---------------:|-------------:|----------------:|----------:|
| 42   | -0.819             | 0.428          | 0.489        | 0.515           | 0.513     |
| 43   | -0.847             | 0.422          | 0.469        | 0.491           | 0.491     |
| 44   | -0.825             | 0.423          | 0.493        | 0.498           | 0.499     |
| 45   | -0.863             | 0.412          | 0.455        | 0.514           | 0.515     |
| 46   | -1.052             | 0.347          | 0.506        | 0.529           | 0.528     |
| **mean** | **-0.881**     | **0.407**      | **0.482**    | **0.509**       | **0.509** |
| **std**  |  0.087          |  0.030         |  0.018       |  0.013          |  0.013    |

**Per-SNR SI-SDR (seed 42, representative):**

| SNR | -10 | -5 | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| SI-SDR (dB) | -4.86 | -2.09 | -0.74 | -0.03 | +0.56 | +0.66 | +0.77 |

**Headline finding**: **OOD detection is essentially random across all
seeds and all three scoring methods** (AUROC ≈ 0.50, std < 0.02).  This
is a clear negative result: the per-source embedding produced by
mask-weighted bottleneck features is NOT well-structured enough to
separate known vs unknown modulations.

The closed-set separation works (SI-SDR turns positive at SNR ≥ 10 dB)
and classification accuracy is above random (0.41 average vs 0.25 random
chance), but the embedding does NOT transfer to OOD detection.

### 2026-07-29 — Per-SNR OOD analysis (corrects the headline)

`evaluate.py` was extended to track per-source SNR for both the known
(kk) and unknown (ku) pools, and to bucket AUROC by SNR.  Re-running
on all 5 seeds reveals that **the aggregate AUROC ≈ 0.50 is a mixture
of a strong signal at moderate-to-high SNR and a below-random
contribution at low SNR**.

**Prototype OOD AUROC by SNR (5-seed mean ± std):**

| SNR (dB) | prototype AUROC | vos AUROC | energy AUROC |
|----------:|----------------:|----------:|-------------:|
| -10       | 0.42 ± 0.13     | 0.41 ± 0.13 | 0.50 ± 0.14 |
| -5        | 0.35 ± 0.10     | 0.35 ± 0.10 | **0.74 ± 0.03** |
| 0         | 0.35 ± 0.10     | 0.35 ± 0.10 | **0.68 ± 0.04** |
| 5         | **0.62 ± 0.03** | **0.61 ± 0.03** | 0.38 ± 0.04 |
| **10**    | **0.84 ± 0.02** | **0.84 ± 0.02** | 0.22 ± 0.04 |
| 15        | **0.62 ± 0.05** | **0.62 ± 0.05** | 0.42 ± 0.03 |
| 20        | 0.56 ± 0.05     | 0.56 ± 0.05 | 0.46 ± 0.03 |

Two non-obvious patterns:

1. **Prototype / VOS work well at moderate-to-high SNR (5-20 dB)**,
   peaking at AUROC ≈ 0.84 at SNR=10 dB.  The aggregate AUROC masks
   this because low-SNR AUROC pulls it down.
2. **Energy score shows the OPPOSITE pattern**: best at low SNR
   (≈ 0.7 at -5 to 0 dB), worst at high SNR (the model is over-confident
   on every input, so logit spread collapses and OOD cannot be told from
   in-dist).

**Revised narrative for the paper**: per-source OOD detection is an
**SNR-dependent phenomenon**.  Prototype / VOS fail at low SNR (where
embeddings of all signals collapse to noise) but work strongly at
moderate-to-high SNR (where the separation backbone has learned
modulation-discriminative features).  Energy score has the opposite
SNR profile.

This suggests an **SNR-adaptive ensemble** (Prototype at high SNR,
Energy at low SNR) as a future direction — likely to push the average
AUROC well above 0.5.

### Diagnosis and proposed next steps

The most likely root cause for the low-SNR failure: **CE-only training
does not shape the embedding space for OOD**.  The classification head
(Linear 64→4) is trained on logits, not embeddings, so the 64-d
embedding can solve the classification task without becoming
class-clustered.  At low SNR, embeddings collapse to noise-like
vectors where known and unknown look the same — but at high SNR, the
backbone has learned features that are more discriminative and the
prototype score captures this.

Proposed fixes (in order of expected impact):

1. **Add an embedding-space metric loss** (Center Loss or supervised
   contrastive).  Goal: explicitly cluster known embeddings, push the
   prototype AUROC up at low SNR too.  Expected: low-SNR prototype
   AUROC rises from 0.35 → 0.55+, aggregate rises accordingly.
2. **SNR-adaptive ensemble** (Prototype at high SNR, Energy at low
   SNR).  Cheap, no retraining, just a routing rule based on the
   observed per-SNR profile.
3. **Train VOS on synthetic OOD** (which the design decision "只用已知
   4 类训练 head" deliberately forbade) — at minimum as an ablation.
4. **Increase embedding dim** from 64 to 128 or 256 for more capacity.
5. **Use the bottleneck features directly** (not mask-weighted) for
   classification — the mask may collapse for unknown inputs, which
   would also collapse the masked features.

### 2026-08-20 — Fix #1 tested: Center Loss (λ=0.1), 5 seeds — NO aggregate gain

`CenterLoss` added to `losses.py`, wired into `train.py` behind
`--loss_lambda_center` (loss = PIT multi-task + 0.5·λ·mean||emb − c_y||²,
centres are learnable parameters in the optimizer).  Run on remote RTX
4060 via `run_center_loss.sh`: 5 seeds (42–46) × 100 epochs, same
hyperparameters as the baseline run (bs 16, lr 1e-3).  ≈7 h total.
Smoke test confirmed centre-loss magnitude is sane (0.13 at init →
0.016 converged vs cls ≈ 2.8).

**Aggregate results (5 seeds, n_per_snr=200, n_per_snr_uu=100) vs baseline:**

| metric | baseline (no center loss) | λ=0.1 center loss |
|--------|--------------------------:|------------------:|
| closed SI-SDR (dB) | -0.881 ± 0.087 | -0.897 ± 0.132 |
| closed cls_acc     |  0.407 ± 0.030 |  0.404 ± 0.042 |
| energy AUROC       |  0.482 ± 0.018 |  0.468 ± 0.010 |
| prototype AUROC    |  0.509 ± 0.013 |  0.496 ± 0.011 |
| vos AUROC          |  0.509 ± 0.013 |  0.496 ± 0.012 |

**Per-SNR prototype AUROC (5-seed mean), baseline → λ=0.1:**

| SNR (dB) | -10 | -5 | 0 | 5 | 10 | 15 | 20 |
|----------|-----|-----|-----|-----|------|-----|-----|
| baseline | 0.42 | 0.35 | 0.35 | **0.62** | **0.84** | **0.62** | **0.56** |
| λ=0.1    | **0.46** | **0.36** | **0.46** | 0.56 | 0.75 | 0.56 | 0.53 |

**Verdict: negative result.**  Center loss slightly helps at the
lowest SNRs (-10/0 dB: +0.04/+0.11, though seed variance is large,
±0.11–0.17) but costs more at the mid-SNR peak (10 dB: 0.84 → 0.75),
so the aggregate AUROC does not move (0.509 → 0.496) and closed-set
performance is unchanged.  The energy score's SNR profile is also
unchanged.  A single-seed run (s42) looked promising at -10 dB
(0.42 → 0.61) but did not replicate across seeds — per-seed variance
at low SNR dwarfs the effect.

Interpretation: explicitly shrinking intra-class embedding spread does
not make unknown-modulation embeddings land farther from the known
prototypes; the prototypes themselves also shift during training, so
the known/unknown margin is not directly optimised.  The SNR-dependent
pattern (prototype works ≥ 5 dB, energy works ≤ 0 dB) is a property of
the backbone features, not of the head's training loss.

**Revised priorities:**

1. **SNR-adaptive ensemble** (fix #2) is now the top candidate — it is
   the only proposal that directly exploits the (stable, reproducible)
   complementary SNR profiles, needs no retraining, and both profiles
   survived the center-loss intervention unchanged.
2. Supervised contrastive loss (instead of center loss) could still be
   tried, but expectation is now lower given this result.
3. Fixes #3–#5 unchanged.

### 2026-08-20 — Fix #2 tested: SNR-adaptive ensemble — POSITIVE result

`ensemble_analysis.py` (new, no training needed) routes between scorers
by per-sample SNR using the a-priori rule read off the baseline
per-SNR profile: **energy for SNR ≤ 0 dB, prototype for SNR ≥ 5 dB**.
AUROC is computed per SNR bin from the saved global-prototype score
arrays (the deployable setting — one set of prototypes, one routing
rule, no per-bin refitting) and averaged across bins weighted by
n_known × n_unknown pairs.  Per-bin values reproduce the earlier
per-SNR table closely, so the numbers are comparable.

**Weighted-average OOD AUROC (5-seed mean ± std):**

| scorer | baseline model | λ=0.1 center-loss model |
|--------|---------------:|------------------------:|
| energy only         | 0.487 ± 0.032 | 0.465 ± 0.011 |
| prototype only      | 0.502 ± 0.008 | 0.497 ± 0.009 |
| vos only            | 0.502 ± 0.008 | 0.497 ± 0.009 |
| **SNR-routed (rule)**   | **0.625 ± 0.031** | 0.568 ± 0.021 |
| oracle (best per bin, post-hoc upper bound) | 0.641 ± 0.018 | 0.598 ± 0.029 |

Findings:

1. **Routing lifts the baseline model from 0.50 → 0.625**, nearly
   reaching the oracle bound (0.641) — the simple threshold rule
   captures almost all of the available complementarity, at zero
   training cost.
2. **Center loss hurts the ensemble too** (0.568 vs 0.625), consistent
   with the earlier negative result: it weakens the high-SNR prototype
   signal that the router relies on.  Recommendation: drop the
   training-loss direction, keep the baseline model + inference-time
   routing as the paper's method.
3. Note: the routed AUROC uses per-bin scoring (both pools at the same
   SNR), which is why it can exceed the pooled aggregate AUROC — this
   is a legitimate "SNR-conditioned detector" metric, and it should be
   presented as such in the paper (per-SNR operating points + weighted
   average), not mixed with the pooled number.

**Limitation / future work:** routing uses ground-truth SNR (legitimate
for this synthetic benchmark where SNR is a controlled variable).  A
practical system needs an SNR estimator; quantifying the ensemble's
sensitivity to SNR estimation error is the natural next experiment
(perturb the routed SNR by ±3/±6 dB and re-measure).

To regenerate: `python ensemble_analysis.py "results/<glob>_ood_scores.npz"`
(full per-seed tables archived in `results/ensemble_baseline.txt` and
`results/ensemble_lc01.txt`; npz score dumps on the server).

### 2026-08-20 — SNR-estimation noise sensitivity: routing is ROBUST

Follow-up on the limitation above.  `ensemble_analysis.py --snr_noise`
perturbs ONLY the routing decision with a noisy SNR estimate
(est = true + N(0, σ)); the metric still bins by true SNR.  50
Monte-Carlo trials per seed, baseline model, same weighted-avg AUROC.

| SNR-est noise σ | routed AUROC (5-seed mean ± std) | MC spread |
|-----------------:|---------------------------------:|-----------:|
| 0 dB (ideal)     | 0.625 ± 0.031 | — |
| 1 dB             | 0.597 ± 0.031 | 0.003 |
| 3 dB             | 0.593 ± 0.030 | 0.004 |
| 6 dB             | 0.576 ± 0.026 | 0.005 |

Reference points: best single scorer (prototype) = 0.502; oracle = 0.641.

**Verdict: the ensemble degrades gracefully.**  Even with a poor
σ = 6 dB SNR estimator the routed scorer keeps ~80% of its gain over
the best single scorer (0.576 vs 0.502), and the drop from σ = 1 dB to
σ = 3 dB is negligible.  Rationale: mis-routing only affects samples
whose noisy estimate crosses the 0/5 dB decision boundary; bins far
from the boundary are unaffected, and the boundary regions are where
the two scorers differ least.  The ground-truth-SNR limitation is
therefore NOT a blocker for the paper's claim — any reasonable SNR
estimator (σ ≤ 3 dB is standard for these signal classes) suffices.

Full output: `results/ensemble_snr_robustness.txt`.

### 2026-08-21 — Fix #4 tested: embedding-dim ablation (16/32/128) — capacity is NOT the bottleneck

`run_embed_dim_ablation.sh`: embed_dim ∈ {16, 32, 128} × seeds 42–44
(64 = baseline; restricted to seeds 42–44 below for a fair 3-seed
comparison), 100 epochs each, otherwise identical hyperparameters.
9 train+eval runs ≈ 12 h on the RTX 4060.

| embed_dim | closed SI-SDR (dB) | cls_acc | prototype AUROC | **routed AUROC** | oracle |
|----------:|-------------------:|--------:|----------------:|-----------------:|-------:|
| 16        | -0.836 ± 0.020 | 0.422 ± 0.009 | 0.509 ± 0.030 | **0.631 ± 0.034** | 0.656 |
| 32        | -0.864 ± 0.005 | 0.399 ± 0.008 | 0.482 ± 0.024 | 0.588 ± 0.010 | 0.601 |
| 64 (base) | -0.830 ± 0.015 | 0.424 ± 0.003 | 0.498 ± 0.007 | 0.622 ± 0.023 | 0.635 |
| 128       | -0.891 ± 0.069 | 0.410 ± 0.007 | 0.507 ± 0.021 | 0.618 ± 0.021 | 0.627 |

**Verdict: embedding capacity does not matter.**  Closed-set separation
and classification are flat across dims (the backbone, not the head,
limits them), and the routed AUROC varies by ≤ 0.04 with no monotonic
trend — all differences are within seed noise.  A single-seed dim=16
run looked promising (prototype AUROC 0.539, 0.90 at SNR=10) but did
not replicate: its 3-seed routed mean (0.631) is statistically
indistinguishable from dim=64 (0.622).  We keep embed_dim=64 and cite
this table as evidence that OOD performance is determined by the
backbone's feature quality, not head capacity — consistent with the
center-loss negative result.

This closes proposed fix #4.  Remaining untested ideas: #3 (train VOS
on synthetic OOD — design decision forbids it for the main method;
could still run as an ablation) and #5 (bottleneck instead of
mask-weighted features — note: this collapses per-source scoring to
mixture-level, since both heads would see identical features).
### 2026-08-21 — Review-proofing baselines: Mahalanobis added (5 seeds); MSP/ODIN + frequency-gap robustness queued

Venue decision: target changed from ICASSP 2027 / IEEE TSP to
**Physical Communication** (Elsevier, subscription route = no APC).
Pre-submission review-proofing identified three likely reviewer asks:
(i) more standard OOD baselines (Mahalanobis, MSP, ODIN), (ii) less
idealised channel (carrier-frequency separation), (iii) framing of the
0.625 absolute AUROC (handled in the write-up).

**Mahalanobis baseline** (`ood_baselines.py`, local, on the saved npz):
min class-conditional Mahalanobis distance in embedding space, shared
covariance with shrinkage 0.1, fit on the same known pool the prototypes
are computed from (consistent with the existing protocol).

| scorer | pooled AUROC (5 seeds) | weighted-avg per-SNR AUROC |
|--------|-----------------------:|---------------------------:|
| energy         | 0.482 ± 0.018 | 0.487 ± 0.032 |
| prototype      | 0.509 ± 0.013 | 0.502 ± 0.008 |
| vos            | 0.509 ± 0.013 | 0.502 ± 0.008 |
| **mahalanobis**| 0.524 ± 0.046 | 0.526 ± 0.043 |
| **SNR-routed** | — | **0.625 ± 0.031** |
| oracle (incl. maha) | — | 0.672 ± 0.037 |

Mahalanobis is the strongest single scorer but its per-SNR profile is
noisy (e.g. −10 dB: 0.60 ± 0.11 across seeds — the same low-SNR seed
instability seen twice before) and it still trails the routed ensemble
by ≈ 0.10.  Adding it to the oracle pool raises the post-hoc bound from
0.641 to 0.672; the routed rule (fixed a priori, energy ≤ 0 dB /
prototype ≥ 5 dB) is unchanged.

**ODIN baseline** (`odin_dump.py`, gradient-based, T=1000, ε=0.005,
complex extension of the sign perturbation): smoke test on seed 42 gives
pooled AUROC 0.487 with an energy-family per-SNR profile (0.78 @ −5 dB,
0.71 @ 0 dB, 0.20 @ 10 dB) — as expected, logit-based scorers share the
same SNR dependence.  Full 5-seed run COMPLETE (see below).

**Full baseline table** (5 seeds, `run_baseline_dumps.sh` on the server;
npz re-dumped with logits, ODIN via gradient pass):

| scorer | pooled AUROC | weighted-avg per-SNR AUROC |
|--------|-------------:|---------------------------:|
| energy      | 0.482 ± 0.018 | 0.487 ± 0.032 |
| prototype   | 0.509 ± 0.013 | 0.502 ± 0.008 |
| vos         | 0.509 ± 0.013 | 0.502 ± 0.008 |
| mahalanobis | 0.524 ± 0.046 | 0.526 ± 0.043 |
| msp         | 0.433 ± 0.010 | 0.435 ± 0.012 |
| odin        | 0.486 ± 0.016 | 0.495 ± 0.030 |
| **SNR-routed (unchanged rule)** | — | **0.625 ± 0.031** |
| oracle (all 6 scorers, post-hoc) | — | 0.681 ± 0.038 |

Verdict: no standard baseline comes within 0.10 of the routed ensemble.
MSP is the weakest scorer overall; ODIN improves slightly over MSP but
keeps the energy-family SNR profile (works ≤ 0 dB, collapses ≥ 5 dB).
Reviewer question (i) is now answered with six scorers on 5 seeds.

**Frequency-separation robustness** (queued, `run_freqgap.sh`): the
baseline model trained at 5 Hz carrier gap is evaluated at gaps
{10, 50, 100, 500} Hz × 5 seeds via `evaluate.py --carrier_gap`
(npz saved with `_gap<Hz>` suffix).  Claim to verify: the routed
ensemble's gain is not an artefact of the training carrier separation.

**Frequency-separation robustness — COMPLETE (positive).**  Baseline
model (trained at 5 Hz gap) evaluated at gaps {10, 50, 100, 500} Hz ×
5 seeds (`run_freqgap.sh`, `evaluate.py --carrier_gap`; npz with
`_gap<Hz>` suffix archived in `results/`).  Weighted-avg per-SNR AUROC
(energy / prototype / vos scorers, unchanged a-priori routing rule):

| carrier gap | energy | prototype | **routed** | oracle |
|------------:|:------:|:---------:|:----------:|:------:|
| 5 Hz (train, reference) | 0.487 | 0.502 | 0.625 ± 0.031 | 0.641* |
| 10 Hz  | 0.488 | 0.500 | 0.625 ± 0.034 | 0.639 |
| 50 Hz  | 0.486 | 0.509 | 0.631 ± 0.035 | 0.649 |
| 100 Hz | 0.488 | 0.497 | 0.633 ± 0.035 | 0.650 |
| 500 Hz | 0.533 | 0.529 | 0.615 ± 0.043 | 0.665 |

*oracle over 3 scorers; the 6-scorer oracle is 0.681.

Verdict: the routed gain is flat across a 100× range of carrier
separations (0.615–0.633, all within seed noise) — the SNR-routing
conclusion is NOT an artefact of the training carrier gap.  Reviewer
question (ii) answered.  Note: single scorers also stay ≈ 0.5 at all
gaps, so the pooled-metric trap is equally present at every separation.

**Consistency pass (2026-08-21, post-review).**  With the six-scorer
table in place, two earlier framings were recalibrated in the paper:
(i) "best single scorer" is now Mahalanobis (weighted-avg 0.526), not
Prototype (0.502) — the routed ensemble's margin over the best single
scorer is +0.10; (ii) the σ = 6 dB SNR-noise case retains ~half of the
routing gain relative to the strongest baseline (0.576 vs 0.526), not
the "~80%" quoted in the 2026-08-20 entry (that figure was relative to
Prototype; the conclusion — graceful degradation, still above every
single scorer — is unchanged).  `make_figs.py` now includes all six
scorers in `fig_overall_auroc` (oracle bound updated 0.641 → 0.681,
which now also covers Mahalanobis/MSP/ODIN) and its default npz glob is
pinned to the five baseline seeds (a looser glob silently swept in the
emb-ablation dumps, which lack the logits fields).

### 2026-08-21 — Reference-statistics transductivity check: NO effect

Reviewer-proofing follow-up: the main protocol fits reference statistics
(prototypes / VOS outliers / Mahalanobis means+shared covariance) on the
test kk pool.  To rule out any transductive advantage,
`refpool_dump.py` dumps a held-out reference pool (kk protocol, seed
88888 — disjoint from the test seed 99999) per checkpoint, and
`refpool_analysis.py` re-fits all geometry-based scorers on it and
recomputes everything (5 seeds).

| scorer | reference = test kk pool | reference = HELD-OUT pool |
|--------|-------------------------:|--------------------------:|
| energy      | 0.487 ± 0.032 | (unchanged — no reference) |
| prototype   | 0.502 ± 0.008 | 0.502 ± 0.008 |
| vos         | 0.502 ± 0.008 | 0.503 ± 0.008 |
| mahalanobis | 0.526 ± 0.043 | 0.526 ± 0.044 |
| msp / odin  | 0.435 / 0.495 | (unchanged) |
| **routed**  | **0.625 ± 0.031** | **0.625 ± 0.032** |
| oracle      | 0.681 ± 0.038 | 0.681 ± 0.038 |

Every number is identical within ±0.003: in-distribution reference
statistics are stable enough that the transductive fit carries no
measurable advantage.  One defensive sentence added to §3.3 of the
paper; the main tables keep the test-pool-fit numbers (now justified).
ODIN ε stays fixed a priori — no unknown-class data exists at
validation time by construction, so validation-tuning ε is impossible
in an honest open-set protocol.

---

## 2026-08-21 — Pre-submission final check (paper3/ manuscript, no new experiments)

Full read-through of main.tex + page-by-page PDF inspection. Fixes applied:

- §4.8 "Qualitative analysis" was an empty subsection (figure only) —
  added prose describing the PCA panels precisely (incl. the partially
  separated BPSK lobe at low SNR, which the old caption glossed over).
- Float drift: Table 6 and Fig 5 landed after the Conclusion in review
  format. Switched all floats to `[H]` (float.sty; placeins.sty is not
  in the local BasicTeX install) and moved each float after its
  referencing paragraph. Tables/figures now sit inside their own
  subsections; 22 pages.
- Removed the leftover `% TODO` comment at the end of the bibliography.
- Added 3 references, all cited where natural: Hyvärinen & Oja 2000
  (ICA, §2.1), Deng et al. 2024 (co-channel modulation classification
  via BSS, §2.1), Scheirer et al. 2013 (open-set recognition, §2.3).
  Bibliography reordered by first citation (25 refs total).
- Rephrased "detectability flips sign with SNR" (abstract + intro) —
  AUROC does not flip sign; it is the *ranking of scorers* that inverts.
- fig_overall_auroc: value labels moved above the error bars (they were
  struck through by the error-bar lines). Regenerated via make_figs.py.

Verified: 0 overfull boxes, 0 undefined references, citation numbers
monotonic by first appearance, no "??" in the PDF.
- fig_architecture (Fig 1) redrawn: the old layout had the z/f box, the
  SNR router, the geometry-family box and the decision box touching or
  overlapping, and the f_i->logit arrow crossing the head box. New
  layout: router centred below the two scorer families, SNR-estimate
  input underneath, decision box top-right fed by a right-angle elbow
  connector. No overlaps. Regenerated via make_arch_fig.py and synced
  to paper3/figures/; main.pdf rebuilt (22 pages).
- Float whitespace fix (supersedes the all-`[H]` note above): large
  `[H]` figures that missed the remaining page space left big blank
  bands (worst: >50% after Table 2, ~60% on the §4.8 page). Final
  scheme: Fig 1 and Fig 2 stay `[H]` (Fig 2 shrunk to 0.8\columnwidth
  so it fits on the Table 2 page); Figs 3-5 use `[!t]` so following
  text fills the page (Fig 3/Fig 4 at 0.8, Fig 5 at 0.85 width).
  Result: 21 pages (was 22), every page's max internal blank band
  <=15%, no float drifts more than one page from its reference.
- Pre-submission audit round 2: full text re-read + all 21 pages
  visually inspected. Two micro-fixes: cover letter "six scorer
  families" -> "six post-hoc OOD scorers from two families";
  intro roadmap sentence now also names the Conclusion section.
  Verified: 0 overfull, 0 undefined refs, 25 citations monotonic by
  first appearance, highlights <=85 chars, no page with a blank band
  >15%, cover-letter numbers match the manuscript.

---

## 2026-09-20 — Post-rejection revision: LOMO splits, test-free boundary selection, no-SE backbone, R1-6 robustness (LAUNCHED, running on server)

Revision experiments answering four reviewer demands. All generator/eval
changes are OPT-IN (defaults reproduce the legacy behaviour AND RNG stream
bit-for-bit — verified locally AND on the server with the old module loaded
from a pre-deploy backup; 0 mismatches across kk/ku/uu tiny sets + mixture
spot checks. One pre-edit reference array (a single OFDM-containing sample)
showed a 5e-07 cross-process fluctuation that was traced to numerical-library
nondeterminism in the REFERENCE capture itself: the old code re-run in any
fresh process agrees with the new code exactly).

- **R1-5/R2-3 (LOMO + test-free boundary):** `train.py --known_mods` trains
  on a 3-mod subset (labels remapped global->local via a LUT; head output
  dim adapts; checkpoint name tagged `_km<mod>-<mod>-<mod>_` so LOMO runs
  never overwrite baseline ckpts; default = all four = legacy name).
  New `eval_lomo.py`: (a) pseudo-OOD validation (seed 77777, "unknown"
  source = the held-out KNOWN-class mod — genuinely OOD for the model,
  never touches test data) selects the Energy/Prototype routing boundary
  as the profile crossover (fallback 0 dB if no clean crossover); (b) test
  (seed 99999) with 5 unknown classes ({held-out} + 4 original unknowns),
  reporting per-SNR profiles, pooled AUROC, and routed weighted-avg AUROC
  using ONLY the boundary from (a).
- **R1-8 (no-SE backbone):** `train.py --no_se` (pre-existing flag) now gets
  a `_nose_` checkpoint-name tag; standard `evaluate.py` +
  `ensemble_analysis.py` routed-profile analysis per seed.
- **R1-6 (robustness):** `data_generator_extended.py` gains opt-in knobs
  `sir_db` (fixed source power ratio, overrides alpha~U(0.4,0.6)),
  `timing_offset_s2` (source-2 symbol grid circular shift by a uniform
  integer offset in [0, sps)), `carrier_jitter_hz` (default 5 = legacy).
  New `eval_robustness.py` sweeps SIR {-6,-3,0,+3,+6} dB / timing on /
  CFO jitter 25 Hz on the EXISTING baseline checkpoints; per-SNR profiles +
  pooled AUROC + routed weighted-avg AUROC at the paper's fixed 0 dB
  boundary (ground-truth SNR, no re-tuning).
- **R1-7 (minor, docs only):** `ood_scores.py` / `config.py` comments —
  "VOS" is a VOS-inspired, TRAINING-FREE, POST-HOC heuristic (not the
  original Du et al. VOS); removed the false "std-scaled"/"std units"
  claims (no std is computed or used). No behaviour change.
- Shared: `ensemble_analysis.py` gains `analyze_arrays()` (in-memory twin
  of `analyze_file()`; `analyze_file` now wraps it — CLI behaviour
  unchanged).

Server driver: `/data/experiment/paper3_open_set/run_revision_r1.sh`,
launched 2026-09-21 ~18:11 server time via
`nohup bash run_revision_r1.sh >> results/revision_r1_driver.log 2>&1 &`
(driver PID 498413). Sequential phases: (1) robustness evals on baseline
ckpts seeds 42/43/44; (2) LOMO 4 held-out mods x seeds 42/43/44 = 12
train(100 epochs, baseline recipe)+`eval_lomo.py`; (3) no-SE x seeds
42/43/44 = 3 train+`evaluate.py`+`ensemble_analysis.py`. Monitor with
`tail -f /data/experiment/paper3_open_set/results/revision_r1_driver.log`.
Results land in `results/robustness_*.json`, `results/lomo_*.json(+npz)`,
`results/eval_openset_cse_..._nose_...`.

**Phase-1 robustness results (COMPLETE, all 3 seeds; routed weighted-avg
AUROC @ 0 dB boundary):** baseline 0.636/0.590/0.636 (seeds 42/43/44);
SIR -6 dB 0.607/0.508/0.610; -3 dB 0.635/0.535/0.640; 0 dB
0.644/0.605/0.649; +3 dB 0.604/0.654/0.614; +6 dB 0.576/0.665/0.589;
timing offset 0.631/0.594/0.633; CFO 25 Hz 0.630/0.598/0.638. Pooled
AUROCs stay ~0.50 under every condition (the SNR-averaging artifact is
robust); the routed ensemble holds up within ~±0.05 of baseline across
SIR/timing/CFO perturbations. Baseline-condition numbers reproduce the
paper's headline (pooled ~0.50, routed ~0.62-0.64) — sanity anchor passed.

LOMO + no-SE training in progress at log time; full numbers when the
driver finishes (~22-25 h total).

### 2026-09-21 — CRITICAL: per-SNR OOD binning bug (repeat vs tile); headline SNR-routing result is an artifact — full fix + re-analysis

**The bug.** Three dump scripts built per-source SNR labels with
`np.repeat(snr, 2)` (each sample's SNR twice, interleaved) while the
score/embedding/label arrays they annotate are stacked **tile**-wise
(`concatenate([all slot-1 entries, all slot-2 entries])`):

- `evaluate.py:227` — `known_snr = np.repeat(kk['snr'], 2)` (known pool
  built at :222-224 as `concatenate([kk['emb_1'], kk['emb_2']])` etc.).
- `refpool_dump.py:64` — `snr = np.repeat(ref['snr'], 2)` (same stacking).
- `odin_dump.py:169` — `known_snr=np.repeat(pools['kk']['snr'], 2)` (same).

The unknown pool was always exact (its snr array is masked with the same
PIT-swap masks as the scores).  The closed-set path (`evaluate.py`
:188-196) builds masks consistently and is NOT affected; pooled
(all-SNR) OOD AUROCs are NOT affected (they use no snr labels).

**Mechanism — why it manufactured SNR-conditioned structure.** With the
test set ordered in SNR blocks of 192 samples, stored "bin i" of the
known pool actually contains scores from true bins 2i/2i+1 (and, past
the slot boundary, true bins that differ from the label by up to 15 dB),
while the unknown pool in the same stored bin is exactly at the labeled
SNR.  Every "per-SNR" AUROC therefore compared known vs unknown mixtures
at DIFFERENT true SNRs, and since all OOD score scales drift with SNR,
the pools separated by their SNR difference — producing the dramatic
(apparent) "Energy wins ≤ 0 dB / Prototype wins ≥ 5 dB" complementarity
and the 0.625 routed headline.  Under truly SNR-matched bins there is
essentially no complementarity to route on.  Proof: known_mods binned by
the stored (repeat) labels give per-bin class counts 114/96/100/74
(garbage); binned by tile labels they give exactly 96/96/96/96, the
combinatorial counts of the kk protocol.

**How it was found.** WS3 (R1-4 real-SNR-estimator workstream,
`snr_est.py` / `eval_estimated_snr_routing.py`) regenerated the seed-99999
test sets sample-for-sample to estimate SNR from waveforms; its
alignment verification against the saved npz dumps exposed the
repeat-vs-tile mismatch.  Full write-up in
`results/snr_est_routing.{txt,json}` (that analysis intentionally
replicates the buggy convention for comparability and also reports the
corrected-bins sensitivity where the artifact was first quantified).

**Blast radius** — every per-SNR OOD number ever produced: paper Fig 2
(`figures/fig_per_snr_auroc.*`), the overall/noise-robustness figures,
Tables III/V/VI, the center-loss (lc0.1) routed numbers (0.568), the
embed-dim ablation routed numbers (0.631/0.588/...), the 5- and 6-scorer
oracle bounds (0.641/0.681), the freq-gap routed table (0.615-0.633),
the σ = 1/3/6 dB SNR-noise sensitivity table (0.597/0.593/0.576), the
2026-08-21 in-distribution-reference consistency pass, agent-3's
`routed_threshold_metrics.*` sanity anchor, and agent-5's Phase-1
`robustness_*.json` numbers (logged above this entry — superseded).
Also the FIRST LOMO eval (held=BPSK seed 42, dumped 18:25 server time,
3 min before the fix landed at 18:28) — re-dumped, see below.

**Fix (2026-09-21, ~18:28 server time).** All three files changed to
`np.tile(...)` locally and scp'd to the server immediately; the revision
driver runs each step as a fresh python process, so every driver eval
after 18:28 (LOMO seed 43 onward, no-SE phase) uses corrected labels.
Local sanity: applying the correction in-memory to the old seed-42 npz
reproduces the corrected-bins recompute exactly (routed 0.504).

**Re-dump (server, fixed code; contaminated dumps moved — not deleted —
to `results/archive_buggy_snr_labels/`, 95 files).** Driver script
`redump_fixed_labels.sh` (5 nice'd lanes, zero FAILED, log
`results/redump_fixed_labels.log`), replicating the original commands:
  - baseline seeds 42-46: `evaluate.py --checkpoint ..._seedS_best.pt --batch_size 16`
  - center-loss seeds 42-46: `evaluate.py --checkpoint ..._lc0.1_seedS_lc01_best.pt --n_per_snr 200 --n_per_snr_uu 100`
  - embed-dim 16/32/128 seeds 42-44: `evaluate.py --checkpoint ..._seedS_embD_best.pt --n_per_snr 200 --n_per_snr_uu 100`
  - freq-gap {10,50,100,500} Hz × seeds 42-46: `evaluate.py --checkpoint ... --batch_size 16 --carrier_gap G`
  - refpool seeds 42-46: `refpool_dump.py --checkpoint ..._seedS_best.pt --batch_size 16`
  - odin seeds 42-46: `odin_dump.py --checkpoint ..._seedS_best.pt --eps 0.005 --batch_size 16`
  - LOMO held=BPSK seed 42 re-eval: `eval_lomo.py --checkpoint ..._kmQPSK-8PSK-16QAM_seed42_best.pt --n_per_snr 200`
  Spot-check on the fresh dumps (` .check_fixed_labels.py`):
  `known_snr == tile(snr_kk, 2)` and per-bin class multiset 96×4 for
  baseline/gap500/lc01/emb16; tile halves equal for refpool/odin/lomo.
  Robustness re-run: `rerun_robustness.sh` →
  `nice eval_robustness.py --checkpoint ..._seedS_best.pt --n_per_snr 200`
  for seeds 42-44, all 7 conditions (log `results/rerun_robustness.log`,
  zero FAILED; old json archived).
  Re-analyses (local, corrected dumps synced back):
  `ensemble_analysis.py` (baseline / lc01 / 4 gap sets, `--snr_noise 1 3 6`
  on baseline), `ood_baselines.py`, `refpool_analysis.py`,
  `routed_threshold_eval.py`, `make_figs.py` (old figures archived to
  `figures/archive_buggy_snr_labels/`; local analysis outputs of the
  buggy era archived with the dumps).

**CORRECTED HEADLINE NUMBERS** (weighted-avg per-SNR AUROC, 5 seeds
42-46 unless noted; "was" = the artifact value published/logged before):

| experiment | routed (was) | oracle (was) | best single (was) |
|---|---|---|---|
| baseline 5 seeds          | **0.498 ± 0.020** (0.625 ± 0.031) | 0.514 ± 0.013 (0.641) | maha 0.521 / proto 0.506 (maha 0.524-0.526) |
| + σ = 1/3/6 dB SNR noise  | 0.502 / 0.502 / 0.501 (0.597/0.593/0.576) | — | — |
| 6 scorers incl. maha/msp/odin | routed 0.498, oracle **0.549 ± 0.031** (0.681) | | maha 0.521, msp 0.431, odin 0.480 |
| center-loss lc0.1 5 seeds | 0.487 ± 0.012 (0.568 ± 0.021) | 0.502 (0.598) | proto 0.497 |
| freq-gap 10/50/100/500 Hz | 0.500 / 0.503 / 0.496 / 0.503 (0.625/0.631/0.633/0.615) | 0.514/0.518/0.508/0.545 | — |
| held-out refpool reference | **0.498 ± 0.020** (0.625 ± 0.032 "identical") | 0.549 ± 0.032 | — |
| embed-dim 16/32/128 (3 seeds) | 0.508 / 0.477 / 0.500 (0.631/0.588/0.622-64) | 0.530/0.505/0.517 | — |
| embed-dim 64 = baseline    | 0.498 (0.622) | 0.514 | — |
| robustness (fixed, 3 seeds, routed wavg) | baseline 0.486, sir∓6 0.494/0.490, sir∓3 0.490/0.487, sir0 0.490, timing 0.492, cfo25 0.490 — flat ≈ chance under every condition (was 0.58-0.66 "robust") | | |
| LOMO held=BPSK (fixed)     | seed42 0.498 / seed43 0.468 (val boundary 0 dB both) | 0.518 / 0.510 | — |
| single-operating-point threshold metrics (agent-3 script, corrected) | variant B headline: routed Det@5%FRR = 0.058 ± 0.038, OSCR 0.197, JointAcc 0.217 (was computed on contaminated bins) | | |

Honest per-SNR profile (baseline, corrected): energy 0.46-0.51,
prototype 0.48-0.57, all within ~2σ of chance per bin; the regenerated
`figures/fig_per_snr_auroc.*` is essentially flat at 0.5.

**Consequences.** (i) The paper's central claim — pooled AUROC ≈ 0.50 is
an SNR-averaging artifact and SNR-routing lifts it to 0.625 — does NOT
survive: the pooled ≈ 0.50 was the honest number; the per-SNR structure
was the artifact.  (ii) The σ-noise "graceful degradation" table is
flat under correction (nothing to lose).  (iii) Mahalanobis remains the
strongest single scorer (0.521-0.524, pooled AND weighted) but within
seed noise of chance.  (iv) All downstream revision analyses (agent-3
threshold metrics, agent-5 robustness, LOMO/no-SE driver outputs) are
now produced on corrected labels; any manuscript table/figure quoting
per-SNR OOD numbers must be regenerated from the current `results/`
dumps.  (v) WS3's R1-4 estimator study stands, with the added caveat
that its 0.625 reference is the artifact value (documented in
`results/snr_est_routing.txt` "CRITICAL FINDING" section).

### 2026-09-21 (late) — DECISION: paper pivots to cautionary / negative-result study; new manuscript drafted

**User decision (2026-09-21):** rather than salvage a positive claim,
paper 3 is rewritten as a cautionary / negative-result study:
pitfall analysis + corrected deployment-faithful protocol + systematic
negative result.  Work done locally (no new server experiments):

- Old manuscript archived: `paper3/main_v1_rejected.tex`; new
  `paper3/main.tex` — title "Open-Set Single-Channel Blind Source
  Separation: A Per-Source SNR-Labeling Pitfall, a Deployment-Faithful
  Evaluation Protocol, and a Systematic Negative Result".  Sections:
  I Intro, II Related (adds leakage/evaluation-pitfall literature:
  Kapoor & Narayanan 2023, Musgrave et al. 2020, Vaze et al. 2022,
  OpenOOD 2022, Pauluzzi & Beaulieu 2000), III System/scorers
  (VOS renamed "VOS-inspired (training-free, post-hoc)" throughout),
  IV The pitfall (repeat-vs-tile mechanism, artifact catalogue table,
  seductiveness analysis, how caught), V Corrected protocol (P1–P5 +
  validated subspace SNR estimator, M2M4 documented inadequate),
  VI Corrected results (all numbers from the regenerated results/
  dumps — routed 0.498 ± 0.020, oracle3 0.514, oracle6 0.549,
  σ-sim flat, robustness flat 0.487–0.494, center-loss 0.487,
  embed-dim 0.477–0.508, freq-gap 0.496–0.503, Det@5%FRR 0.058,
  OSCR 0.197, JointAcc 0.217), VII Attribution + 7-point protocol
  checklist, VIII Limitations, IX Conclusion.
  Builds clean via `cd paper3 && bash build.sh` (36 pp elsarticle
  review format, 0 overfull).
- New figures (`paper3_open_set/make_figs_revision.py` →
  `paper3/figures/`): fig_pitfall_mechanism (repeat-vs-tile schematic),
  fig_artifact_vs_corrected (TWO-PANEL money figure: same score dumps
  under buggy vs corrected labels), fig_per_snr_all6 (flat profiles),
  fig_snr_estimator (subspace vs M2M4), fig_operating_point (Det@τ /
  FRR@τ per SNR).  Old paper3 figures moved to
  `paper3/figures/archive_buggy_snr_labels/`.
- Supporting files rewritten: `paper3/abstract.txt`, `keywords.txt`,
  `highlights.txt`.
- Repo docs corrected to the new narrative: root `AGENTS.md`
  (Paper 3 bullet), root `README.md` (Paper 3 section),
  `paper3_open_set/README.md` (status block + key finding).

**Placeholders awaiting the running server jobs**
(`run_revision_r1.sh`, ~15 h left at log time): LOMO 4 splits × 3 seeds
(filled so far: held=BPSK seed42 0.498 / seed43 0.468) → manuscript
cells marked `\textbf{[TBD-LOMO]}` with TODO comments; no-SE
cross-backbone × 3 seeds → `\textbf{[TBD-NOSE]}`.  Final-numbers pass:
fill both blocks from `results/lomo_*.json` and the nose checkpoints'
`ensemble_analysis` outputs, rebuild, and re-check table/figure
consistency.  No git commits made.

**LOMO FINAL (2026-09-21, later same day; server driver finished all 12
runs, 0 FAILED).** 4 held-out mods x seeds 42-44, test seed 99999,
unknown pool = held-out mod + 4 real unknowns, routing boundary selected
on pseudo-OOD validation seed 77777 ONLY (the reviewers' legal,
test-free selection).  Routed weighted-avg AUROC per run:

| held-out | seed42 | seed43 | seed44 | mean±std |
|---|---|---|---|---|
| QPSK  | 0dB, 0.516 | 0dB, 0.505 | 20dB, 0.501 | 0.508±0.008 |
| 8PSK  | 0dB, 0.514 | 0dB, 0.506 | 20dB, 0.506 | 0.509±0.005 |
| 16QAM | 0dB, 0.481 | 0dB, 0.493 | 0dB, 0.472  | 0.482±0.011 |
| BPSK  | 0dB, 0.498 | 0dB, 0.468 | 20dB, 0.491 | 0.486±0.016 |

Key observation (now in the manuscript, Table LOMO + discussion): under
LEGAL boundary selection the selected boundary is UNSTABLE — 9/12 runs
pick 0 dB, 3/12 find no crossover and fall back to an all-prototype
route (20 dB) — precisely because there is no real crossover to find;
and the routed AUROC is at chance (0.468-0.516) in every split/seed,
consistent with the corrected baseline 0.498.  `[TBD-LOMO]` cells in
`paper3/main.tex` filled with these numbers.  Remaining placeholder:
no-SE cross-backbone (3 seeds still training) `[TBD-NOSE]`.

### 2026-09-22 — no-SE cross-backbone COMPLETE (final revision evidence)

Driver `run_revision_r1.sh` finished all phases (0 FAILED). no-SE
backbone (60.8K params, `--no_se`), 3 seeds, corrected protocol:
- Closed-set (weaker than SE, as expected): pooled SI-SDR
  -1.07 / -1.19 / -1.20 dB (mean -1.15); cls_acc 0.406 / 0.356 / 0.305
  (mean 0.356).
- Corrected OOD (weighted-avg per-SNR AUROC, global prototypes,
  0 dB route): routed 0.508 +/- 0.011 (0.502/0.502/0.521);
  energy 0.491 +/- 0.031; prototype 0.531 +/- 0.012;
  vos-inspired 0.532 +/- 0.011; oracle(3) 0.536 +/- 0.014.
- Per-SNR: energy flat 0.47-0.50; prototype mild bump at 0/5 dB
  (0.588/0.555) but ~0.51-0.52 elsewhere — NO low/high-SNR
  complementarity, nothing to route. Scorer inversion appears in
  NEITHER backbone under corrected labels; it was a property of the
  mislabeled evaluation, not of any feature representation.

`[TBD-NOSE]` cells in `paper3/main.tex` filled. All revision
experiments are now COMPLETE; npz/json synced back to local
`results/` (30 lomo/nose files). Manuscript rebuilds clean.

### 2026-09-22 — Venue decision: IEEE Signal Processing Letters (letter version)

User decision (2026-09-22): the pivoted paper 3 goes to **IEEE Signal
Processing Letters** as a letter.  Hard constraint: the school cannot
reimburse fees, so the letter must fit **≤ 4 pages final** (SPL: 4 free
pages; 5th page = $220 overlength charge).

- New file `paper3/letter.tex` (IEEEtran journal two-column; vendored
  `paper3/IEEEtran.cls` V1.8b copied from `paper6/`; build via
  `bash build.sh letter` — pdflatex, same pattern as paper5/build.sh).
  **Builds at 3 pages, 0 overfull** — within the free-length budget.
- Title: "A Per-Source SNR-Labeling Pitfall in Open-Set Evaluation of
  Single-Channel Blind Source Separation".  Content is a strict
  condensation of `paper3/main.tex` (every number identical): money
  figure = `fig_artifact_vs_corrected` (figure*); master artifact-vs-
  corrected table (routed 0.625→0.498±0.020, oracle6 0.681→0.549±0.031,
  σ-sim flat, robustness 0.487–0.494, LOMO 0.482–0.509, no-SE
  0.508±0.011, Det@5%FRR 0.058±0.038); 4-point checklist conclusion;
  15 references (dropped OpenOOD/Vaze/embdim-era citations and the
  SC-BSS mid-list; kept OOD foundations + VOS + OSCR + leakage/pitfall
  + Pauluzzi-Beaulieu + PIT/SDR + channel model).
- Cut vs full version (reconsider if a full venue is chosen instead):
  related-work section, system-model equations for the six scorers,
  mechanism schematic (fig_pitfall_mechanism), per-SNR all-6 figure,
  SNR-estimator table/figure, operating-point figure, LOMO table
  (folded into the master table), attribution depth, limitations.
- `paper3/main.tex` (elsarticle, 36 pp review format, no TBD markers)
  kept untouched as the full-paper fallback; `paper3/abstract.txt` and
  `highlights.txt` updated to mention LOMO + cross-backbone evidence.
- NOTE: the server copy of this log was NOT updated this turn (no ssh);
  the no-SE entry above exists only locally — sync manually if wanted.

### 2026-09-23 — Pre-submission peer review (paper3/review1.md) + SPL letter v2 revision

A pre-submission peer review of `paper3/letter.tex` v1 was received
(`paper3/review1.md`, verdict: Major Revision). Core demands: promote
the bug report into a generalizable methodology (formalize the
indexing-permutation failure), define per-source SNR rigorously, full
data-generation protocol, complete six-scorer results with
sample-level CIs, operating-point metrics (FPR95/OSCR/joint acc),
hyperparameter-selection protocol, separation-quality + oracle
clean-source experiments to bound the attribution, softened causal
claims, AMC-literature citations, scope-limited conclusions.

**New analysis (local, CPU): `paper3_open_set/revision2_tables.py`**
(standalone, numpy/scipy only, ~2.5 min; AUROC reuses the Mann-Whitney
convention of `open_set_metrics.auroc`; internal asserts: corrected
routed 0.498, buggy routed 0.6253±0.0313, pi_repeat == archived buggy
per seed, scorer wiring vs stored scores). Output:
`results/revision2_tables.json` (all per-seed/per-bin detail + method
fields). Headline numbers:

- Corrected six-scorer wavg (refpool-fitted, 5 seeds): energy
  0.4755±0.0192, msp 0.4306±0.0090, odin 0.4798±0.0189, mahalanobis
  0.5203±0.0503, prototype 0.5055±0.0103, vos 0.5056±0.0097; routed
  0.4984±0.0205, oracle6 0.5485±0.0319. Buggy wavg: energy 0.4865,
  msp 0.4351, odin 0.4945, mahalanobis 0.5264, prototype 0.5023, vos
  0.5023, routed 0.6253, oracle 0.6812.
- Bootstrap 95% CI (B=2000, test-sample resampling, all-seeds
  concatenated): energy [0.477,0.491], msp [0.429,0.443], odin
  [0.484,0.498], mahalanobis [0.509,0.523] (excludes 0.5 — but only
  0.4σ at seed level: 0.5203±0.0503), prototype [0.494,0.507], vos
  [0.494,0.507]. Paired bootstrap oracle−routed = 0.0531
  [0.0427,0.0641] — significant per-bin max-selection inflation.
- **Permutation ablation (generality proof — the review's key ask)**:
  re-binning corrected scores with synthetic known-label permutations
  π (5-seed mean routed wavg): identity 0.4980, repeat (the bug)
  0.6253, reverse 0.7653, 10 random perms 0.6405±0.0046,
  block-preserving shifts 0.4980–0.4987 (no-ops). Per-bin AUROC up to
  0.85–0.87 under π_reverse. Mechanism: ANY label/SNR decorrelation
  exposes the SNR-slope confound (energy's mismatched AUROC decreases
  with bin, prototype's increases); the a-priori routing rule harvests
  both gradients; single-scorer wavgs stay 0.48–0.51 under every π
  (the average looks innocent while per-bin structure is spurious —
  exactly why the artifact looked like complementarity).
- Per-unknown-class (corrected, wavg): embedding scorers weakly higher
  on far-OOD (mahalanobis MSK 0.5559±0.041, OFDM 0.5511±0.117);
  logit scorers INVERTED on far-OOD (msp OFDM 0.3593±0.027 — the
  multicarrier waveform looks MORE in-distribution than knowns).
- Operating point reformatted from `routed_threshold_metrics.json`:
  variant A Det@τ 0.0575±0.0352 / FRR 0.0516 / FPR95 0.9539 / OSCR
  0.1958 / joint 0.1633; variant B joint 0.2169. Per-SNR Det@τ:
  exactly 0 at ≤0 dB → 0.1434 at 20 dB with FRR tracking (0→0.1388).

**New server experiments (eval-only, 5 seeds, ~2.5 min total; scripts
`eval_sep_quality.py`, `eval_oracle_clean.py`; results
`results/sep_quality.json`, `results/oracle_clean_ood.json`):**

- Separation quality (PIT-aligned SI-SDR on kk/ku/uu, sanity-checked
  bit-identical to the closed-set summaries): known −0.888±0.084 dB
  vs unknown −0.995±0.085 dB pooled (per-bin gap ≤0.21 dB) — the
  separator treats unknown modulations no worse than known ones
  (kills the "class-dependent representation distortion" alternative).
- Oracle clean-source probe (single source + AWGN through the frozen
  network, PIT slot pick; reference set seed-stream 66666, test
  55555): mahalanobis pooled 0.660±0.046 (0.52@−10dB → 0.77@+10dB →
  0.74@+20dB), per-class 64QAM 0.587 / π4-DQPSK 0.685 / MSK 0.547 /
  OFDM 0.821; prototype/vos ≈0.54; energy/msp BELOW chance
  (0.44/0.43) even with no interferer. ⇒ A partial embedding margin
  EXISTS without interference but does not survive the two-source
  separated condition (≈0.50 everywhere): the v1 causal claim
  "the objective shapes no margin" was too strong; revised wording:
  the objectives provide no sufficiently exploitable margin for
  post-hoc scorers ON SEPARATED SOURCES; the bottleneck is the
  mixture/separation condition (and, for logit scorers, the
  confidently-wrong known-class head).

**Manuscript: `paper3/letter.tex` v2** (rebuilt `bash build.sh letter`:
4 pages incl. 17 refs, 0 `[?]`, exit 0 — within SPL's 4-page free
limit). New title per review: "A Per-Source SNR Labeling Pitfall in
Open-Set Detection for Single-Channel Blind Source Separation".
Changes: formal SNR_mix definition + per-source offset (−3 dB) + SIR
range; full datagen parameters (16 kHz, 256 sym × 16 sps, RRC 0.35,
3-tap fading, 2000/2005 Hz ±5 Hz, pool sizes 192/192/96 per bin,
2688/2688 sources); hyperparameter-selection + leakage paragraph
(refpool 88888, pseudo-OOD 77777, test 99999, a-priori defaults);
indexing-permutation formalization (σ_q vs σ_ℓ) + permutation study;
Fig 2 = fig_pitfall_mechanism (column width); Table I = complete
six-scorer artifact/corrected/pooled+bootstrap-CI; Table II =
operating-point A/B; near/far-OOD paragraph; separation-quality +
oracle-clean attribution paragraph; scoped conclusion; new refs
oshea2017intro + li2023expert (TPAMI 45(11):13730–13748) fixing the
"standard practice" citation mismatch; "legally" wording removed;
code-release sentence now promises a pinned tag `paper3-spl`
(USER: create+push the tag at submission time).

**NOT done / pending:** `paper3/main.tex` (36-page elsarticle fallback)
is still v1-narrative — port the v2 experiments there only if the
letter is rejected and a full-length venue is chosen. Reviewer items
consciously descoped: far-OOD beyond the existing 4 unknown classes
(covered instead by the per-class near/far breakdown incl. OFDM),
Figure-1 three-panel redesign (Fig 2 covers the mechanism), real-RF
validation (stated as scope limit).

### 2026-09-30 — Review-2 experiments (paper3/review2.md Major Concerns 2/3/4) + SECOND label-alignment bug found and corrected

All runs on the server (`/data/experiment/paper3_open_set`, venv_bss,
RTX 4060), eval-only, 5 checkpoints (seeds 42–46), refpool-fitted scorers
(seed 88888), ODIN recomputed per run (eps=0.005, T=1000; recompute
verified bit-identical to the stored npz, max|diff| ≤ 3e-8). New scripts
(rsynced to server, smoke-tested with `--smoke` before full runs):
`eval_protocol_split.py`, `eval_sir_sweep.py`, `eval_multiseed_test.py`,
plus follow-ups `eval_sir_sweep_controls.py`, `eval_sir0_isolate.py`,
`eval_truth_anchor.py`, `eval_multiseed_ta.py`. Results pulled to
`results/`: `protocol_split.json`, `sir_sweep.json`,
`multiseed_test.json`, `sir_sweep_controls.json`, `sir0_isolate.json`,
`truth_anchor.json`, `multiseed_test_ta.json` (+ run logs
`review2_runs.log`, `truth_anchor_run.log`, `multiseed_ta_run.log`).
Total wall time ≈ 50 min.

**A. Protocol split (MC3).** `python eval_protocol_split.py` — test seed
99999, exact evaluate.py protocol. Sanity: combined pool reproduces the
revision2 s1 baselines with |dev| = 0.0000 on all six scorers + routed.
Across-seed wavg AUROC (stored labels): K(kk) vs U(kuOOD)
mahalanobis 0.5148/prototype 0.5038/routed 0.4941; K(kk) vs U(uu)
0.5258/0.5072/0.5028; K(ku) vs U(ku) same-mixture 0.5028/0.5046/0.4994
— all ≈ chance. NOTE: the ku-side comparisons inherit the label-swap bug
below (kk↔uu comparison is unaffected).

**B. SIR sweep (MC2).** `python eval_sir_sweep.py` — SNR_mix = 10 dB, SIR
(unknown/known power ratio) ∈ {−20,…,+10} dB + clean(∞), exact gains via
`generate_open_set_mixture(sir_db=±SIR)` (project-native mixing/noise
path), 16 pairs × 16 reps = 256 mixtures per cell, truth-anchored labels
(this script never inherited the swap bug), same-mixture AUROC, scorers
fit on refpool. Across-seed AUROC (mahalanobis/prototype/vos):
SIR −20: 0.472±0.19/0.544±0.25/0.546±0.25 (weak target buried:
SI-SDR_u = −19.2 dB); −10: 0.595/0.418/0.415; 0: 0.587/0.681/0.680;
+10: 0.482/0.556/0.556; clean: 0.624/0.524/0.526. NOT a monotone
0.8→0.5 decay: at extreme negative SIR the scorer sign structure flips
between families, and seed variance is large (±0.2). Controls
(`eval_sir_sweep_controls.py`, `results/sir_sweep_controls.json`):
exact-SIR=0 vs legacy α~U(0.4,0.6) at 10 dB — 0.699 vs 0.636 prototype
(both elevated; exact gain adds ~0.06); clean cell refpool-fit vs
clean-fit — mahalanobis 0.623 vs 0.761 (the 0.77 oracle_clean number is a
clean-fit number; the gap is a reference-fit effect, now documented).

**C. Multi test seeds (MC4).** `python eval_multiseed_test.py` — test
seeds 100001/100002/100003, validation pass on 99999 first: reproduces
baselines with |dev| = 0.0000 (all seven metrics) → pipeline exact.
Per-test-seed routed wavg (stored labels): 99999: 0.4984±0.021,
100001: 0.4928±0.018, 100002: 0.4958±0.022, 100003: 0.5049±0.023; grand
(15 groups) routed 0.4978±0.0214, mahalanobis 0.5205±0.0523 — the
at-chance headline is robust to test realization (stored labels).

**⚠ SECOND EVALUATION BUG (found via B↔A cross-check, fixed-labels
re-run).** `evaluate._collect_predictions` PIT-aligns
waveforms/embeddings/logits to the true sources but ALSO swaps the
per-source labels (`mod1_best = where(swap, mod2, mod1)` etc.). Labels
must be truth-anchored (the src1-aligned slot's label is always src1's).
Swap rate ≈ 0.50 at SIR≈0, so the ku OOD/known pool membership was ~50%
contaminated — attenuating every ku-dependent AUROC toward 0.5 and
depressing closed-set cls_acc (0.428 stored → 0.562 truth-anchored).
kk/uu pools, the refpool, SNR labels (per-mixture) and the 2026-09-21
repeat/tile fix are all unaffected. Detection story: the debug chain was
`.debug_c3_vs_i1.py` (same cell, same emb/scores to 1e-6, different pool
labels) → `eval_sir0_isolate.py` (`results/sir0_isolate.json`: canonical
ku-99999 10-dB bin scores prototype 0.667 when truth-anchored vs 0.506
with swapped labels).

**Truth-anchored corrected headline** (`eval_truth_anchor.py`,
`results/truth_anchor.json`, test 99999, 5 seeds): combined wavg —
energy 0.4314±0.023, msp 0.3936±0.007, odin 0.4354±0.021, mahalanobis
0.5320±0.054, prototype 0.5326±0.011, vos 0.5323±0.010, routed
0.5106±0.020. Protocol split (TA): same-mixture kuK vs kuU prototype
0.6099±0.012 (per-bin 0.47@−5 dB → 0.68@≥15 dB), mahalanobis 0.5491;
kkK vs uuU unchanged 0.507/0.526 (cross-mixture stays ≈ chance);
logit scorers same-mixture strongly INVERTED (energy 0.334). Multi-seed
TA (`eval_multiseed_ta.py`, `results/multiseed_test_ta.json`, validated
on 99999 with |dev| = 0.0000): grand over 15 new-test-seed groups —
routed 0.5086±0.0218, mahalanobis 0.5323±0.0569, prototype 0.5284±0.0173,
energy 0.4362, msp 0.4006, odin 0.4409.

**Revised interpretation for the manuscript:** after both label fixes the
story is NOT "no margin anywhere" — embedding scorers carry a real but
modest SAME-MIXTURE margin (prototype/vos ≈0.61 wavg, 0.66–0.68 at
SNR ≥ 5 dB) that (i) does not survive cross-mixture pooling (kkK vs uuU
≈ 0.51), (ii) is inverted for logit scorers (0.33–0.44), and (iii) is far
below deployment use; the SIR sweep shows the margin collapses when the
unknown source is the weak one (SIR ≤ −10 dB). The cautionary
evaluation-methodology message is STRENGTHENED: two independent
deterministic label misalignments in the same pipeline, one inflating
(0.625 artifact), one attenuating (≈0.50 "null"), both invisible to
multi-seed averaging. Paper-3 letter tables/claims need updating before
submission (stored-label numbers in `revision2_tables.json` s1 inherit
the swap bug for all ku-dependent rows).

**Caveats:** `sep_quality.json` (2026-09-23) used the swapped is_ood flags
— its ku cells are diluted toward the null (kk-vs-uu contrast, which is
unaffected, shows the same ≈0.1 dB gap, so the "class-independent
separator" conclusion stands). `oracle_clean.json` used truth-anchored
single-source labels — unaffected. The precomputed
`*_odin_eps0.005_T1000.npz` unknown pools inherit the swap bug (ku part);
truth-anchored runs recompute ODIN per slot. `.debug_c3_vs_i1.py` is a
throwaway debug script kept for the audit trail.

**Follow-up (same day) — TA standard dumps + TA operating point + 192/96
multiseed.** Scripts: `dump_ta_scores.py` (server, ~4 min), 
`routed_threshold_eval_ta.py` (local CPU, reuses routed_threshold_eval.py
logic on the TA dumps), `eval_multiseed_ta.py` rerun with
`--n_per_snr 192 --n_per_snr_uu 96`. New artifacts (local + server):
`results/openset_..._seed{42..46}_best_ood_scores_ta.npz` (exact same
key/dtype/shape schema as the standard dumps; known pool = kk both slots,
unknown pool = ku true-OOD slot + uu both slots, evaluate.py ordering),
`..._odin_eps0.005_T1000_ta.npz` (ODIN recomputed per slot, same params),
`..._refpool_ta.npz` (seed-88888 reference pool with corrected labels —
37-38% of refpool slot labels had been swapped!), `ta_dumps_sanity.json`,
`routed_threshold_metrics_ta.{json,txt}`, and `multiseed_test_ta.json`
overwritten with the 192/96 run. Sanity: (a) known-side emb/logits/energy
vs the existing corrected dumps bit-exact (max|diff| = 0.0);
prototype/vos known-side arrays reproduce the old dumps bit-exactly when
computed with the old swapped-label prototype fit — the only difference is
the fit labels; (b) pools 2688/2688, 384/384 per bin; (c) per-bin known
modulation multiset exactly 96/96/96/96; (d) combined six-scorer wavg vs
truth_anchor.json max|dev| = 0.00005 (≤ ±0.005 tol; 192/96 ≡ 200/100
datasets since n_per_pair is 12/6/6 in both). TA operating point (variant
A split-half / variant B refpool-refit headline): routed Det@tau
0.0656/0.0638, FRR 0.0524/0.0487, FPR95 0.9467/0.9475, OSCR
0.2719/0.2726 (up from 0.196 stored — correct_k is now truth-anchored),
JointAcc 0.2064/0.2775; TA stored-scores routed wavg AUROC sanity
0.5076±0.0247. Multiseed TA at 192/96: 99999 validation |dev| = 0.0000,
per-group numbers bit-identical to the 200/100 run; grand (15 groups)
routed 0.5086±0.0218. Note for the letter: TA-refpool-fitted combined
headline (ta_dumps_sanity.json) is routed 0.5077±0.0262, prototype
0.5178±0.0219 — the refpool label correction is a small second-order
effect on the headline (vs swapped-refpool 0.5106/0.5326), seed 43 drives
most of the delta.

**Local analyses + letter v3/v3.1 (same day, main session).** New local
scripts: `revision3_tables.py` → `results/revision3_tables.json`
(review2 statistics on the pre-TA corrected dumps: consistent
concatenated-point/CI Table-I pairs, Delta-AUROC with two-sided
label-permutation tests B=1e4, bin-wise correlation-preserving
null-oracle B=2000 + iid chance-scorer simulation, positional
ku/uu protocol split) and `revision4_tables_ta.py` →
`results/revision4_tables_ta.json` (the SAME analyses rerun on the
fully-TA dumps: `revision2_tables.load_run` gained DUMP_SUFFIX/REFPOOL_SUFFIX
hooks; TA dumps + `_refpool_ta.npz` fits). Fully-TA headline (192/96,
test 99999, 5 seeds): wavg — energy 0.4314±0.023, msp 0.3936±0.007,
odin 0.4354±0.021, mahalanobis 0.5291±0.056, prototype 0.5178±0.022,
vos 0.5175±0.022, routed 0.5077±0.026, oracle 0.5496±0.041; pooled
concat Delta (perm p): energy −0.050, msp −0.095, odin −0.040
(all p=1e-4), mahalanobis +0.025 (p=1e-4), prototype/vos +0.009
(p≈0.008/0.01); null-oracle 0.5182 [0.5132, 0.5232] vs observed 0.5496
(P(null≥obs)=5e-4 — selection optimism + genuine embedding margin);
TA permutation ablation (bug-1 isolated on clean pools): pi_identity
0.508, pi_repeat 0.619, pi_reverse 0.746 (per-bin routed up to 0.85),
pi_random 0.632±0.005, block-shifts exact no-ops; per-class TA: MSP
0.609 on pi/4-DQPSK vs 0.216 on OFDM-QPSK (logit inversion),
mahalanobis/prototype 0.54–0.58 on MSK/OFDM (far-OOD) vs 0.46–0.50
near-OOD. `make_figs_revision.py`: fig_pitfall_mechanism annotated
with sigma_q != sigma_l; fig_artifact_vs_corrected panel (b) now reads
the TA dumps (routed profile sanity 0.5076) with retitled panels.
`paper3/letter.tex` v3/v3.1: retitled "A Label–Score Alignment Pitfall
in SNR-Conditioned OOD Detection for SC-BSS" (review2 MC8); keywords
fixed (label misalignment, not "data leakage"); Proposition 1
(conditional generality, MC1); two-bug narrative (PIT label swap
attenuates the genuine same-mixture margin); all tables TA
(Table I/II/III); 0-dB boundary provenance stated honestly
(development-time, frozen before corrected-test evaluation); FPR95
defined; multi-test-seed sentence (TA routed 0.501–0.517 across seeds
100001–100003); SIR-sweep attribution (margin peaks at SIR 0, 0.68;
destabilizes at |SIR|≥10). Builds at 4 pp body + p5 references-only,
0 undefined refs. PENDING (server, WP1/WP2): TA-refpool same-mixture
protocol row refresh, TA reruns of the robustness battery / LOMO /
no-SE backbone — the letter's robustness/LOMO/backbone sentence is
held back (TODO comment) until those land; intro claim (iii) still
lists "two backbones, LOMO splits" and must be reconciled with the TA
rerun results before submission.

**Follow-up #2 (same day) — letter standardizes on the FULLY corrected
pipeline (TA pools + TA-refpool fits); WP1 + WP2 regenerated.** New shared
helper `ta_common.py`; new scripts (all smoke-tested, then run in one
nohup chain on the server, total 14 min: 11:58→12:12):
`eval_protocol_split_ta.py` (WP1), `eval_nose_ta.py` (WP2a),
`eval_lomo_ta.py` (WP2b), `eval_robustness_ta.py` (WP2c). Results pulled:
`protocol_split_ta.json`, `nose_ta.json`, `lomo_ta.json`,
`robustness_ta.json` (+ per-variant `*_refpool_ta.npz` for no-SE /
center-loss / embed-dim checkpoints).

- **WP1 protocol split (192/96, TA-refpool)**: sanity vs
  ta_dumps_sanity TA-refpool headline exact (max|dev| = 0.00000, all 5
  seeds); kk cls acc 0.407 stored -> 0.523 TA (the ≈0.52 target).
  wavg (5 seeds): kkK vs kuU prototype 0.5237±0.019 / mahalanobis
  0.5261±0.051 / routed 0.5063±0.020; kkK vs uuU 0.5118 / 0.5321 /
  0.5090; **same-mixture kuK vs kuU** prototype 0.5433±0.015 /
  mahalanobis 0.5168±0.029 / energy 0.3342 (inverted) / routed
  0.5232±0.006 — per-bin prototype: 0.47@-5 dB -> 0.645@15 dB. Combined:
  routed 0.5077±0.026, prototype 0.5178, mahalanobis 0.5291.
- **WP2a no-SE (3 seeds, TA)**: routed 0.5288±0.021, prototype
  0.5509±0.022, mahalanobis 0.6054±0.034, energy 0.4683; cls acc
  0.356 -> 0.404. (Was: swapped-label routed 0.508±0.011 "no
  complementarity" — TA shows the no-SE backbone keeps a slightly
  STRONGER embedding margin than the C-SE main model.)
- **WP2b LOMO (12 runs, TA; boundary from the TA pseudo-OOD validation
  split 77777 only)**: boundary distribution {0 dB: 6 (fallback, no clean
  crossover), 20 dB: 6 (all-energy route)}; routed wavg 0.4819±0.0316
  (per split: BPSK 0.475, QPSK 0.460, 8PSK 0.489, 16QAM 0.504);
  prototype/vos 0.530±0.033; cls acc 0.505 -> 0.651 TA.
- **WP2c robustness battery (TA)**: sir/timing/cfo (seeds 42-44, kk+ku)
  — baseline routed 0.5007±0.021 (cross-check vs WP1 C1 per seed EXACT,
  |d|=0.00000); all conditions flat ≈0.49-0.50 routed (timing 0.5044,
  cfo25 0.5018, sir -6..+6 → 0.488-0.502). Carrier gap (5 seeds, full
  pools): 10 Hz 0.5132 / 50 Hz 0.5123 / 100 Hz 0.5035 / **500 Hz
  0.4493±0.013 (below chance)**. Center loss (5 ckpts): routed
  0.4654-0.5182 per ckpt. Embed-dim 16/32/128 (3 seeds each): routed
  ranges 0.415-0.543 (seed 43 low everywhere). SNR-noise sim on the TA
  dumps (sigma=1/3/6 dB, 50 MC): 0.5171 / 0.5171 / 0.5145 vs GT 0.5076
  (sigma-robustness carries over to the corrected pipeline).
- Scorer sets: WP2a = six scorers; WP2b = energy/mahalanobis/prototype/
  vos (eval_lomo fidelity) + routed; WP2c = energy/mahalanobis/prototype/
  vos + routed (odin/msp omitted for runtime — six-scorer TA coverage is
  in ta_dumps_sanity.json / protocol_split_ta.json).

**Follow-up #3 (same day) — SIR sweep with TA-refpool fits.**
`eval_sir_sweep.py --ta` (new flag; old `sir_sweep.json` untouched) ->
`results/sir_sweep_ta.json`: identical grid/data (master 77777, 256
mixtures/SIR, SNR_mix=10 dB), prototype/vos/mahalanobis now fitted on the
per-seed `_refpool_ta.npz`. Across-seed (5 seeds) per-SIR same-mixture
AUROC (mahalanobis/prototype/vos): -20: 0.478/0.541/0.541; -15:
0.506/0.544/0.542; -10: 0.609/0.456/0.455; -5: 0.511/0.475/0.476; 0:
0.550/0.628/0.626; +5: 0.550/0.626/0.626; +10: 0.467/0.519/0.520; clean:
0.632/0.533/0.535. Sanity: SIR=0 prototype 0.6284 vs WP1 fully-TA
same-mixture 10-dB bin 0.613 -> within seed noise (OK). Clean
mahalanobis 0.632 matches the refpool-fit expectation (~0.62; the 0.77
oracle-clean value is the clean-fit variant, documented in follow-up #2's
controls). vs the swapped-refpool run: SIR=0 prototype 0.628 vs 0.682 —
the contaminated reference fit had inflated the mid-SIR margin slightly.

**Follow-up #3 + letter v3.1 finalized (same day).** WP1/WP2 landed:
`protocol_split_ta.json` (fully-TA protocol split; the TA-refpool fit
tempers the same-mixture margin to prototype 0.543±0.015 wavg, per-bin
0.596@5 dB → 0.645@15 dB — the contaminated reference fit had inflated
it to 0.610), `nose_ta.json` (no-SE backbone TA: routed 0.5288±0.021,
mahalanobis 0.6054 — margin not backbone-specific), `lomo_ta.json` (12
runs, TA: routed 0.4819±0.032; boundary distribution {0 dB fallback: 6,
20 dB all-energy: 6} — no genuine crossover in any split),
`robustness_ta.json` (SIR/timing/CFO 0.488–0.504; carrier gaps
0.503–0.513 at ≤100 Hz, 0.449 at 500 Hz; center loss ≈0.50; embed dims
0.42–0.54; σ-sim 0.515–0.517), `sir_sweep_ta.json` (TA-refpool SIR
sweep: prototype 0.628@SIR0 / 0.626@+5, ≤0.55 with ±0.2 spread at
|SIR|≥10, clean maha 0.632; qualitative shape unchanged). The letter
(`paper3/letter.tex` v3.1) now carries ONLY fully-TA numbers: Table I
(corrected wavg: energy 0.431, msp 0.394, odin 0.435, maha 0.529, proto
0.518, vos 0.518, routed 0.508±0.026, oracle 0.550±0.041), Table II
(TA operating point: Det@τ 0.064, FRR 0.049, FPR95 0.948, OSCR 0.273,
JointAcc 0.278), Table III (fully-TA protocol split), permutation study
(pi_identity 0.508 / pi_repeat 0.619 / pi_reverse 0.746 / pi_random
0.632), robustness+LOMO+backbone sentence restored with TA values,
SIR attribution with TA-refpool values, acknowledgment moved to the
first-page footnote. Final build: 4 pp body + p5 references-only, 0
undefined references. Remaining known flavor note: the multi-test-seed
per-seed values (0.501–0.517) used the swapped-refpool fit
(documented effect −0.001 on the grand mean; TA-refpool refit deemed
not worth another 15-group rerun).
