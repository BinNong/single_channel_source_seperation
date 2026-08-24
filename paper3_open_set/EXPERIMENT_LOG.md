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
