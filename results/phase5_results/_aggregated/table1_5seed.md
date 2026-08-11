# Table 1 -- per-seed SDR (dB), mean +/- sample std

| Model | N | Seeds | SI-SDR | SDR | SIR | NMSE |
|---|---|---|---|---|---|---|
| Complex CNN + SE (Proposed) | 5 | 42,43,44,45,46 | -0.66+/-0.09 | 2.31+/-0.62 | 11.35+/-7.96 | -2.29+/-0.64 |
| Complex CNN no-SE (matched, H=70) | 5 | 42,43,44,45,46 | -0.98+/-0.06 | 1.81+/-0.48 | 18.01+/-6.79 | -1.78+/-0.49 |
| Real-Valued CNN (matched, H=80,L=12) | 5 | 42,43,44,45,46 | -0.94+/-0.07 | 2.20+/-0.59 | 11.60+/-8.24 | -2.18+/-0.62 |
| Complex Conv-TasNet | 3 | 42,43,44 | 0.58+/-0.92 | 3.23+/-0.40 | 5.88+/-0.70 | -3.23+/-0.40 |
| CNSE (scaled, Hou & Gao 2022) | 3 | 42,43,44 | 1.99+/-0.21 | 2.38+/-0.91 | 16.65+/-8.65 | -2.28+/-0.99 |
| S4-UNET (scaled, Gao et al. 2026) | 3 | 42,43,44 | 0.01+/-0.12 | 3.05+/-0.00 | 5.77+/-0.17 | -3.05+/-0.00 |

## Paired vs C-SE (inner-join by seed label)

| Baseline | Common seeds | n_pairs | Paired Delta SDR (dB) | Unpaired desc. Delta SDR (dB) | Wilcoxon p | n_nonzero | p_min at n_nonzero |
|---|---|---|---|---|---|---|---|
| Complex CNN no-SE (matched, H=70) | 42,43,44,45,46 | 5 | +0.49 | +0.49 (C-SE N=5, base N=5) | 0.1250 | 5 | 0.0312 |
| Real-Valued CNN (matched, H=80,L=12) | 42,43,44,45,46 | 5 | +0.11 | +0.11 (C-SE N=5, base N=5) | 0.0625 | 5 | 0.0312 |
| Complex Conv-TasNet | 42,43,44 | 3 | -0.48 | -0.93 (C-SE N=5, base N=3) | 0.2500 | 3 | 0.1250 |
| CNSE (scaled, Hou & Gao 2022) | 42,43,44 | 3 | +0.38 | -0.07 (C-SE N=5, base N=3) | 0.5000 | 3 | 0.1250 |
| S4-UNET (scaled, Gao et al. 2026) | 42,43,44 | 3 | -0.30 | -0.75 (C-SE N=5, base N=3) | 0.2500 | 3 | 0.1250 |