"""Paper 5 — LaTeX tables for S2 main matrix, S2 ablation, S3 routes.

Inputs (all local, synced from the server):
  results/s2/summary_s2.json            — 5-loss-config main matrix (n=5 seeds;
      pureser n=1) + paired exact permutation tests {delta, ci95, perm_p}
  results/s2/eval_*_sisdr_s42.json,
  results/s2/eval_*_ser_s42.json        — single-seed controls for the ablation
  results/s2_abl/eval_*_ser_{l0.3,l3.0,s0.05,s0.2}_s42.json
      — lambda_ser / sigma^2 dose-response probes (single seed 42)
  results/s3/summary_s3.json            — S3 v2 (learned strided conv), K=2 cell
  results/s3_v1_meanpool/summary_s3.json — S3 v1 (window-mean downsample)

Outputs (written to ../paper5/tables/ and echoed to stdout):
  tab_s2_main.tex      — S2 main matrix (SI-SDRi / SER / BER / count / halluc)
  tab_s2_ablation.tex  — lambda_ser, sigma^2 ablation (no dose-response)
  tab_s3.tex           — S3 K=2 route comparison (joint head ~= chance)

Style: elsarticle booktabs, matching paper2/tables/efficiency.tex.
Precision: SER/BER 4 decimals, SI-SDRi 2 decimals, matching EXPERIMENT_LOG.

All headline results are negative by design: the soft-demod loss has no
statistically significant effect on SER/BER (exact paired permutation,
perm_p >= 0.125), and the S3 joint head is significantly WORSE than the
waveform route. Table captions must state this honestly.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
TABLES = os.path.normpath(os.path.join(HERE, '..', 'paper5', 'tables'))

EVAL_PREFIX = 'eval_slot_h64_l4_k13_bs16_lr0.001_'
# Ablation rows: (row label, results dir, run suffix, lambda_ser, sigma^2).
# lambda_ser / sigma^2 are the training hyperparameters (train.py CLI);
# the main 'ser' control uses the defaults lambda_ser=1.0, ser_sigma2=0.1.
ABL_ROWS = [
    ('sisdr (control)', 's2', 'sisdr_s42', 0.0, '--'),
    ('ser (control)', 's2', 'ser_s42', 1.0, 0.1),
    ('ser, $\\lambda_{\\mathrm{ser}}{=}0.3$', 's2_abl', 'ser_l0.3_s42', 0.3, 0.1),
    ('ser, $\\lambda_{\\mathrm{ser}}{=}3.0$', 's2_abl', 'ser_l3.0_s42', 3.0, 0.1),
    ('ser, $\\sigma^2{=}0.05$', 's2_abl', 'ser_s0.05_s42', 1.0, 0.05),
    ('ser, $\\sigma^2{=}0.2$', 's2_abl', 'ser_s0.2_s42', 1.0, 0.2),
]


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def ms(blk, key, fmt='{:.4f}', n1=False):
    """'mean +- std' cell; n1=True reports the bare mean with a dagger.
    A signed format (e.g. '{:+.2f}') applies the sign to the mean only."""
    m, s = blk[key]['mean'], blk[key]['std']
    std_fmt = fmt.replace('+', '')
    if n1:
        return fmt.format(m) + '$^{\\dagger}$'
    return fmt.format(m) + ' $\\pm$ ' + std_fmt.format(s)


def write(name, tex):
    path = os.path.join(TABLES, name)
    with open(path, 'w') as fh:
        fh.write(tex)
    print(f'--- {name} ' + '-' * max(0, 60 - len(name)))
    print(tex)
    return path


# ---------------------------------------------------------------- table (a)
def tab_s2_main(s2, cells):
    p = s2['paired']
    c = cells
    caps = (
        f"(c) ser vs.\\ (a) SI-SDR, BER: $\\Delta{{=}}{p['ser_vs_sisdr_ber_comp']['delta']:+.4f}$, "
        f"$p{{=}}{p['ser_vs_sisdr_ber_comp']['perm_p']:.4f}$; "
        f"(c) ser vs.\\ (b) MSE, BER: $\\Delta{{=}}{p['ser_vs_mse_ber_comp']['delta']:+.4f}$, "
        f"$p{{=}}{p['ser_vs_mse_ber_comp']['perm_p']:.4f}$; "
        f"(d) ser+mse vs.\\ (a) SI-SDR, SER: $\\Delta{{=}}{p['ser_mse_vs_sisdr_ser_comp']['delta']:+.4f}$, "
        f"$p{{=}}{p['ser_mse_vs_sisdr_ser_comp']['perm_p']:.4f}$"
    )
    cell_caps = (
        f"(c) vs.\\ (a), BER: $\\Delta{{=}}{c['ser_vs_sisdr_ber_comp']['delta']:+.5f}$, "
        f"$p{{=}}{c['ser_vs_sisdr_ber_comp']['mc_p']:.1g}$; "
        f"(c) vs.\\ (b), BER: $\\Delta{{=}}{c['ser_vs_mse_ber_comp']['delta']:+.5f}$, "
        f"$p{{=}}{c['ser_vs_mse_ber_comp']['mc_p']:.1g}$, nominally surviving a "
        f"Bonferroni $\\alpha{{=}}0.05/8{{=}}0.00625$ over the 8 contrasts. "
        f"The effect is concentrated in the high-SNR cells "
        f"($15$--$20$\\,dB, $+0.2$--$0.3$\\,pp in the cell means; 4/5 seeds agree in sign, "
        f"seed-level $p{{=}}0.125$ power-capped) and vanishes at low SNR; "
        f"the fair additive comparison (d) vs.\\ (b) shows nothing "
        f"(SER $p{{=}}{c['ser_mse_vs_mse_ser_comp']['mc_p']:.2f}$, "
        f"BER $p{{=}}{c['ser_mse_vs_mse_ber_comp']['mc_p']:.2f}$), "
        f"nor does any SER contrast"
    )
    rows = []
    for key, lab in (('sisdr', '(a) SI-SDR'),
                     ('mse', '(b) MSE'),
                     ('ser', '(c) soft-SER'),
                     ('ser_mse', '(d) soft-SER + MSE'),
                     ('pureser', '(e) pure soft-SER')):
        b = s2[key]
        n1 = b['n'] == 1
        rows.append(
            f"{lab} & {ms(b, 'si_sdri', '{:+.2f}', n1)} & {ms(b, 'ser', '{:.4f}', n1)}"
            f" & {ms(b, 'ber', '{:.4f}', n1)} & {ms(b, 'count_acc', '{:.3f}', n1)}"
            f" & {ms(b, 'halluc_rate', '{:.3f}', n1)} \\\\")
    base = s2['baseline']
    rows.append(
        f"Mixture (no separation) & -- & {base['ser']:.4f} & {base['ber']:.4f}"
        f" & -- & -- \\\\")
    body = '\n'.join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{S2 main matrix (slot architecture, $K\\in\\{{1,2,3\\}}$, 5 seeds except where noted; mean $\\pm$ population std across seeds). The soft-demodulation loss has at most a practically negligible effect on bit-level metrics. Seed-level exact paired permutation tests ($n{{=}}5$, two-sided; smallest attainable $p{{=}}2/2^5{{=}}0.0625$, power-limited) give {caps}. A cell-level test (7 SNR $\\times$ 5 seeds $=$ 35 paired differences, $10^6$ Monte-Carlo sign-flips) returns small $p$-values for config (c) ({cell_caps}) --- but it treats within-seed cells as exchangeable even though they share one trained model, so it is anti-conservative and is reported as descriptive only; the defensible summary at $n{{=}}5$ is $|\\Delta| \\le 0.17$\\,pp with an unstable sign. Config (c) replaces the MSE anchor with the soft-SER term; the additive design is (d). The soft-SER term does improve source counting (count acc.\\ $0.65\\!\\rightarrow\\!0.85$, occupancy route) and suppresses hallucinated sources. Configs (a)--(e) as defined in Section~\\ref{{sec:configs}}. $^{{\\dagger}}$single seed ($n{{=}}1$), no std.}}
\\label{{tab:s2_main}}
\\setlength{{\\tabcolsep}}{{2pt}}
\\scriptsize
\\begin{{tabular}}{{l r r r r r}}
\\toprule
Config & SI-SDRi (dB) & SER & BER & Count acc. & Halluc. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ---------------------------------------------------------------- table (b)
def tab_s2_ablation():
    rows = []
    for lab, sub, suffix, lam, sig in ABL_ROWS:
        with open(os.path.join(RES, sub, EVAL_PREFIX + suffix + '.json')) as fh:
            d = json.load(fh)
        sep = d['separation']
        occ = d['counting']['occupancy']['overall_acc']
        sig_s = f"{sig:.2f}".rstrip('0').rstrip('.') if sig != '--' else '--'
        lam_s = '--' if lam == 0.0 else f"{lam:g}"
        rows.append(
            f"{lab} & {lam_s} & {sig_s} & {sep['si_sdri']:+.2f}"
            f" & {sep['ser_comp']:.4f} & {sep['ber_comp']:.4f}"
            f" & {occ:.3f} & {sep['halluc_rate']:.3f} \\\\")
    body = '\n'.join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{S2 ablation of the soft-demodulation loss weight $\\lambda_{{\\mathrm{{ser}}}}$ and the soft-decision Gaussian width $\\sigma^2$ (single seed 42, slot architecture). Neither knob shows a dose--response on compensated SER/BER --- all values remain within $\\approx\\!0.005$ of the controls --- while larger $\\lambda_{{\\mathrm{{ser}}}}$ monotonically improves occupancy accuracy and reduces hallucination, at the cost of SI-SDRi. Counting metrics use the occupancy route.}}
\\label{{tab:s2_ablation}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\begin{{tabular}}{{l r r r r r r r}}
\\toprule
Config & $\\lambda_{{\\mathrm{{ser}}}}$ & $\\sigma^2$ & SI-SDRi (dB) & SER & BER & Occ.\\ acc. & Halluc. \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


# ---------------------------------------------------------------- table (c)
def tab_s3(s3_v2, s3_v1):
    pv2, pv1 = s3_v2['paired'], s3_v1['paired']

    def row(label, ser_list, ber_list, base=False):
        if base:
            return f"{label} & {ser_list:.4f} & {ber_list:.4f} \\\\"
        sm, ss = mean_std(ser_list)
        bm, bs = mean_std(ber_list)
        return (f"{label} & {sm:.4f} $\\pm$ {ss:.4f}"
                f" & {bm:.4f} $\\pm$ {bs:.4f} \\\\")

    b = s3_v2['baseline_k2']
    rows = [
        row('Mixture (no separation)', b['ser'], b['ber'], base=True),
        row('S2-ser waveform route', s3_v2['s2_routes_k2']['ser']['ser'],
            s3_v2['s2_routes_k2']['ser']['ber']),
        row('S3 waveform route (v2 model)', s3_v2['s3_waveform_k2']['ser'],
            s3_v2['s3_waveform_k2']['ber']),
        row('S3 joint head v1 (window mean)', s3_v1['s3_joint_k2']['ser'],
            s3_v1['s3_joint_k2']['ber']),
        row('S3 joint head v2 (learned strided conv)',
            s3_v2['s3_joint_k2']['ser'], s3_v2['s3_joint_k2']['ber']),
    ]
    body = '\n'.join(rows)
    d = pv2['joint_vs_s3waveform_SER']
    db = pv2['joint_vs_baseline_SER']
    d1 = pv1['joint_vs_s3waveform_SER']
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{S3 joint LLR head vs.\\ waveform routing, $K{{=}}2$ cell (5 seeds, mean $\\pm$ std; baseline is single-pass). The joint head is significantly WORSE than every reference: paired bootstrap gives $\\Delta$SER ${{=}}{d['delta']:+.4f}$ (95\\% CI $[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]$) for v2 vs.\\ the S3 waveform route and $\\Delta$SER ${{=}}{db['delta']:+.4f}$ ($[{db['ci95'][0]:+.4f}, {db['ci95'][1]:+.4f}]$) vs.\\ the mixture baseline --- all CIs far from 0 (v1: $\\Delta$SER ${{=}}{d1['delta']:+.4f}$ vs.\\ waveform). Both variants sit at the modulation-averaged chance level (SER $\\approx 0.73$, BER $\\approx 0.46$). Candidate failure causes: v1's boxcar averaging cancels the nominal $2000$\\,Hz carrier almost exactly (inferred from the architecture, not verified on features); v2's learned strided convolutions cannot recover burst-level carrier/phase blindly --- a controlled $K{{=}}1$ probe with oracle sync trains the same head to val SER $0.15$ vs.\\ $0.68$ under blind sync, but at $K{{=}}2$ even a frequency-oracle probe (per-source phase alignment is impossible for a mixture) reaches only $0.589$, worse than the mixture baseline ($0.561$): synchronisation alone is not sufficient under phase entanglement.}}
\\label{{tab:s3}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{l r r}}
\\toprule
Route ($K{{=}}2$) & SER & BER (Gray) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_s2_perk(perk):
    """Per-K decomposition of the four pre-declared contrasts
    (results/s2/summary_s2_perk.json, analyze_s2_perk.py).
    Cells: mean paired delta in percentage points + seed sign count."""
    pairs = [(('ser', 'sisdr'), '(c) soft-SER vs (a) SI-SDR'),
             (('ser', 'mse'), '(c) soft-SER vs (b) MSE'),
             (('ser_mse', 'mse'), '(d) soft-SER+MSE vs (b) MSE'),
             (('ser_mse', 'sisdr'), '(d) soft-SER+MSE vs (a) SI-SDR')]
    rows = []
    for (a, b), lab in pairs:
        for metric, key in (('SER', 'ser_comp'), ('BER', 'ber_comp')):
            rec = perk[f'{a}_vs_{b}_{key}']
            cells = []
            for k in (1, 2, 3):
                r = rec[str(k)] if str(k) in rec else rec[k]
                cells.append(f"${r['delta'] * 100:+.2f}$ ({r['n_pos']}/5)")
            rows.append(f"{lab} & {metric} & " + ' & '.join(cells)
                        + ' \\\\')
        rows.append('\\addlinespace[2pt]')
    body = '\n'.join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{S2 per-$K$ decomposition of the four pre-declared contrasts (5 paired seeds per cell; exact two-sided sign-flip permutation, $2^5{{=}}32$). Cells give the mean paired difference in \\textbf{{percentage points}} (positive favours the first-named configuration) and the number of seeds with a positive difference. Exact $p$: $0.0625$ for $5/5$ or $0/5$ (the $n{{=}}5$ floor), $0.125$ for $1/5$, $\\geq 0.125$ for $4/5$ (the exact value depends on the seed magnitudes), $\\geq 0.25$ otherwise. The pooled null of Table~\\ref{{tab:s2_main}} resolves into structure: at $K{{=}}2$ the soft-SER term helps (BER positive, $5/5$ seeds in \\emph{{all four}} contrasts); at $K{{=}}3$ it hurts (SER negative, $0/5$ seeds in \\emph{{all four}} contrasts); at $K{{=}}1$ there is no interference and the contrasts are small and sign-unstable. The two cancel under the 1:2:3 pair-count weighting of the pool.}}
\\label{{tab:s2_perk}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{ll r r r}}
\\toprule
Contrast & Metric & $K{{=}}1$ & $K{{=}}2$ & $K{{=}}3$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def tab_sir_sweep():
    """SIR sweep at fixed K=2 (results/sir_sweep/sir_sweep_summary.json,
    eval_sir_sweep.py; test seed 88888, 100 mixtures per (SIR, SNR) cell).
    Reports config (b) MSE (5 seeds) against the per-cell mixture baseline."""
    with open(os.path.join(RES, 'sir_sweep', 'sir_sweep_summary.json')) as fh:
        sw = json.load(fh)
    rows = []
    for sir in ('0.0', '5.0', '10.0', '20.0'):
        r = sw['configs']['mse']['rows'][sir]
        gap_pp = (r['baseline_ser'] - r['ser_mean']) * 100
        strong = '--' if r['ser_strong'] is None else f"{r['ser_strong']:.3f}"
        weak = '--' if r['ser_weak'] is None else f"{r['ser_weak']:.3f}"
        rows.append(
            f"{int(float(sir))} & {r['baseline_ser']:.4f}"
            f" & {r['ser_mean']:.4f} $\\pm$ {r['ser_std']:.4f}"
            f" & ${gap_pp:+.2f}$ & {r['si_sdri_mean']:+.1f}"
            f" & {strong} / {weak} \\\\")
    body = '\n'.join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{SIR sweep at fixed $K{{=}}2$ (mixing weights set to geometrically symmetric SIR levels about the training mean; independent test seed 88888, full SNR grid; config (b), 5 seeds). $\\Delta$ is baseline $-$ model in percentage points. The pooled SER falls with SIR, but so does the per-cell mixture baseline, and the gap never exceeds one point (crossing zero near $10$\\,dB, clearly negative at $20$\\,dB) while SI-SDRi holds --- the improvement is entirely the strong source's (SER $0.43 \\to 0.19$) as its interferer fades; the weak source degrades ($0.67 \\to 0.74$). The waveform--bit mismatch is not specific to $\\sir \\approx 0$\\,dB. Models are trained at $\\sir \\approx 0$\\,dB, so the $\\sir > 3.5$\\,dB rows are out-of-distribution.}}
\\label{{tab:sir_sweep}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\begin{{tabular}}{{r r r r r r}}
\\toprule
SIR (dB) & Baseline SER & (b) SER & $\\Delta$ (pp) & SI-SDRi (dB) & SER strong / weak \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(RES, 's2', 'summary_s2.json')) as fh:
        s2 = json.load(fh)
    with open(os.path.join(RES, 's3', 'summary_s3.json')) as fh:
        s3_v2 = json.load(fh)
    with open(os.path.join(RES, 's3_v1_meanpool', 'summary_s3.json')) as fh:
        s3_v1 = json.load(fh)
    with open(os.path.join(RES, 's2', 'summary_s2_cells_perm.json')) as fh:
        cells = json.load(fh)
    with open(os.path.join(RES, 's2', 'summary_s2_perk.json')) as fh:
        perk = json.load(fh)

    write('tab_s2_main.tex', tab_s2_main(s2, cells))
    write('tab_s2_ablation.tex', tab_s2_ablation())
    write('tab_s2_perk.tex', tab_s2_perk(perk))
    write('tab_s3.tex', tab_s3(s3_v2, s3_v1))
    write('tab_sir_sweep.tex', tab_sir_sweep())
    print(f'\nAll tables written to {TABLES}')


if __name__ == '__main__':
    main()
