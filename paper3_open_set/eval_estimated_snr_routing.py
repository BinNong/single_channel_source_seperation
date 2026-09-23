"""
Paper 3 — Open-Set SC-BSS: SNR-routed OOD ensemble with a REAL blind SNR
estimator (reviewer R1-4).

Replaces the ground-truth SNR in the SNR-routed ensemble (ensemble_analysis.py)
by the blind waveform-only estimators of snr_est.py, and re-computes the
paper's headline weighted-avg per-SNR AUROC.

Pipeline
--------
  a. Regenerate the EXACT test sets of evaluate.py (seed 99999, protocols
     kk/ku with n_per_snr=200 and uu with 100, SNR grid from config) and verify
     sample-for-sample alignment against the saved per-seed score npz files.
  b. Estimate the SNR of every test mixture from its waveform (M2M4 primary,
     subspace comparison); report bias / std / RMSE per true-SNR bin.
  c. Route per sample: Energy if est <= 0 dB else Prototype (the a-priori rule,
     threshold 0 dB).  Compute the pair-weighted per-SNR AUROC exactly as
     ensemble_analysis.py does (bins by the npz's stored true-SNR arrays), per
     seed 42-46, and compare against GT routing (0.625) and the published
     Gaussian-perturbation simulation (sigma = 1/3/6 dB), which we re-run in
     the same harness.
  d. Fully-deployed variant: bins are defined by the ESTIMATED SNR as well
     (nearest grid point, pair-weighted within estimated bins).

Known alignment limitation (documented, quantified): the npz "unknown" pool
orders its ku-protocol entries by the model's PIT assignment (which slot holds
the OOD source after the SI-SDR swap), which is not recoverable without the
network.  We reconstruct that block with the dataset's pre-swap OOD order.
Within a true-SNR bin this is a bijection between the same 192 mixtures and 192
pool entries, so only the (score, est) pairing inside the bin is permuted; we
quantify the effect with a Monte-Carlo over random within-bin bijections.

Outputs (paper3_open_set/results/):
  snr_est_routing.txt   human-readable tables + protocol documentation
  snr_est_routing.json  machine-readable results
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

import config as C
from data_generator_extended import CommBSSOpenSetTestDataset
from ensemble_analysis import analyze_file, noisy_routed_wavg
from open_set_metrics import auroc
import snr_est

SNR_GRID = [float(s) for s in C.SNR_TEST_POINTS]
SEEDS = (42, 43, 44, 45, 46)
NPZ_TEMPLATE = 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0_seed{}_best_ood_scores.npz'
ROUTING_THRESHOLD = 0.0          # a-priori rule: energy if snr <= 0 else proto
SIGMA_LEVELS = (1.0, 3.0, 6.0)   # published Gaussian-perturbation points
MC_TRIALS_SIGMA = 50             # matches ensemble_analysis.py default
MC_TRIALS_ALIGN = 20             # within-bin bijection sensitivity


# ============================================================================
# a. Test-set regeneration (identical to evaluate.py)
# ============================================================================
def build_test_sets(n_per_snr: int = 200, n_per_snr_uu: int = 100) -> dict:
    """Regenerate the three test datasets exactly as evaluate.py main() does.

    Returns dict protocol -> dict of numpy arrays:
      wave [N, 4096] complex64, snr [N], mod1 [N], mod2 [N], ood1/ood2 [N] bool
    (mod/ood are the dataset's PRE-swap labels).
    """
    common = dict(
        n_per_snr=n_per_snr,
        snr_points=C.SNR_TEST_POINTS,
        signal_length=C.SIGNAL_LENGTH,
        sample_rate=C.SAMPLE_RATE,
        seed=99999,
    )
    out = {}
    for proto in ('kk', 'ku', 'uu'):
        kw = dict(common) if proto != 'uu' else dict(
            n_per_snr=n_per_snr_uu,
            snr_points=C.SNR_TEST_POINTS,
            signal_length=C.SIGNAL_LENGTH,
            sample_rate=C.SAMPLE_RATE,
            seed=99999,
        )
        t0 = time.time()
        ds = CommBSSOpenSetTestDataset(protocol=proto, **kw)
        n = len(ds)
        wave = np.empty((n, C.SIGNAL_LENGTH), dtype=np.complex64)
        snr = np.empty(n, dtype=np.float64)
        mod1 = np.empty(n, dtype=np.int64)
        mod2 = np.empty(n, dtype=np.int64)
        ood1 = np.empty(n, dtype=bool)
        ood2 = np.empty(n, dtype=bool)
        for i, s in enumerate(ds.samples):
            wave[i] = s['mixture'].numpy()[0]
            snr[i] = s['snr']
            mod1[i] = s['mod1_idx']
            mod2[i] = s['mod2_idx']
            ood1[i] = s['mod1_is_ood']
            ood2[i] = s['mod2_is_ood']
        out[proto] = dict(wave=wave, snr=snr, mod1=mod1, mod2=mod2,
                          ood1=ood1, ood2=ood2)
        print(f"  regenerated {proto}: {n} mixtures "
              f"({time.time() - t0:.1f} s)")
        del ds
    return out


def verify_alignment(d: dict, data: dict) -> list[tuple[str, bool, str]]:
    """Check the regenerated test sets against one seed's score npz.

    d    : loaded npz (one seed).
    data : output of build_test_sets().

    Returns a list of (check name, passed, detail).  Exact array checks are
    used wherever the npz content is model-independent; swap-invariant
    multiset checks are used for the per-source modulation labels (the npz
    labels passed through the model-dependent PIT swap, which permutes the
    two slots of each sample but never changes the per-sample multiset).
    """
    kk, ku, uu = data['kk'], data['ku'], data['uu']
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    snr_kk = d['snr_kk'].astype(np.float64)
    add('kk snr sequence exact (npz snr_kk == dataset order)',
        np.array_equal(snr_kk, kk['snr']),
        f'len {len(snr_kk)} vs {len(kk["snr"])}')
    add('known_snr == repeat(snr_kk, 2)  [stored binning convention]',
        np.array_equal(d['known_snr'], np.repeat(snr_kk, 2)),
        'as written by evaluate.py')
    add('pool sizes', len(d['known_snr']) == 2 * len(kk['snr']) and
        len(d['unknown_snr']) == len(ku['snr']) + 2 * len(uu['snr']),
        f"known={len(d['known_snr'])} unknown={len(d['unknown_snr'])}")

    # Per-bin counts of the unknown pool: model-independent (every ku sample
    # contributes exactly one OOD entry; every uu sample two).
    u_snr = d['unknown_snr']
    cnt = {s: int((u_snr == s).sum()) for s in SNR_GRID}
    expect = {s: int((ku['snr'] == s).sum() + 2 * (uu['snr'] == s).sum())
              for s in SNR_GRID}
    add('unknown_snr per-bin counts', cnt == expect, f'{cnt} vs {expect}')
    cnt_k = {s: int((d['known_snr'] == s).sum()) for s in SNR_GRID}
    expect_k = {s: int(2 * (kk['snr'] == s).sum()) for s in SNR_GRID}
    add('known_snr per-bin counts', cnt_k == expect_k,
        f'{cnt_k} vs {expect_k}')

    # Swap-invariant label checks: per-bin multiset of modulation indices.
    def per_bin_multiset(labels, snr_arr):
        return {s: sorted(labels[snr_arr == s].tolist()) for s in SNR_GRID}

    # The npz's stored known_snr follows the repeat(snr_kk, 2) convention
    # while its score/label arrays follow the tile (emb_1-all, emb_2-all)
    # convention, so binning the labels must use the TRUE per-entry bins
    # tile(snr_kk) for the comparison to be meaningful.
    km_expect = per_bin_multiset(
        np.concatenate([kk['mod1'], kk['mod2']]), np.tile(kk['snr'], 2))
    add('known_mods per-bin multiset (swap-invariant, true bins)',
        per_bin_multiset(d['known_mods'], np.tile(snr_kk, 2)) == km_expect,
        'multiset of mod indices per true SNR bin (tile bins)')

    ku_ood_mods = np.where(ku['ood1'], ku['mod1'], ku['mod2'])
    um_expect_labels = np.concatenate(
        [ku_ood_mods, uu['mod1'], uu['mod2']])
    um_expect_snr = np.concatenate(
        [ku['snr'], uu['snr'], uu['snr']])
    add('unknown_mods per-bin multiset (swap-invariant)',
        per_bin_multiset(d['unknown_mods'], d['unknown_snr'])
        == per_bin_multiset(um_expect_labels, um_expect_snr),
        'multiset of OOD-side mod indices per SNR bin')
    return checks


# ============================================================================
# b. SNR estimation
# ============================================================================
def estimate_all(data: dict) -> dict:
    """Run both estimators on every mixture; returns est[proto][method] = [N]."""
    est = {}
    for proto, dd in data.items():
        n = len(dd['snr'])
        out = {m: np.empty(n) for m in ('m2m4', 'subspace')}
        t0 = time.time()
        for i in range(n):
            w = dd['wave'][i]
            out['m2m4'][i] = snr_est.estimate_snr_db(w, method='m2m4')
            out['subspace'][i] = snr_est.estimate_snr_db(w, method='subspace')
        est[proto] = out
        print(f"  estimated {proto}: {n} mixtures ({time.time() - t0:.1f} s)")
    return est


def quality_table(snr_true: np.ndarray, est: np.ndarray) -> dict:
    """bias / std / RMSE (dB) per true-SNR bin."""
    rows = {}
    for s in SNR_GRID:
        m = snr_true == s
        err = est[m] - snr_true[m]
        rows[f'{s:g}'] = dict(
            n=int(m.sum()), mean_est=float(est[m].mean()),
            bias=float(err.mean()), std=float(err.std()),
            rmse=float(np.sqrt(np.mean(err ** 2))),
            p_le0=float((est[m] <= ROUTING_THRESHOLD).mean()),
        )
    err = est - snr_true
    rows['overall'] = dict(
        n=int(len(err)), bias=float(err.mean()), std=float(err.std()),
        rmse=float(np.sqrt(np.mean(err ** 2))))
    return rows


# ============================================================================
# c/d. Routed AUROC machinery (mirrors ensemble_analysis.py)
# ============================================================================
def route(e_score: np.ndarray, p_score: np.ndarray, est_snr: np.ndarray,
          threshold: float = ROUTING_THRESHOLD) -> np.ndarray:
    """Per-sample scorer choice: energy at/below threshold, else prototype."""
    return np.where(est_snr <= threshold, e_score, p_score)


def wavg_auroc(k_routed: np.ndarray, u_routed: np.ndarray,
               k_bins: np.ndarray, u_bins: np.ndarray) -> float:
    """Pair-weighted per-bin AUROC, exactly as ensemble_analysis.py."""
    num, den = 0.0, 0.0
    for s in sorted(set(k_bins.tolist()) & set(u_bins.tolist())):
        mk, mu = k_bins == s, u_bins == s
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        w = int(mk.sum()) * int(mu.sum())
        num += auroc(k_routed[mk], u_routed[mu]) * w
        den += w
    return num / den


def wavg_auroc_oracle(scores: dict, k_bins: np.ndarray,
                      u_bins: np.ndarray) -> float:
    """Post-hoc oracle: best single scorer per bin, pair-weighted.

    scores : {method_name: (scores_known, scores_unknown)}
    """
    num, den = 0.0, 0.0
    for s in sorted(set(k_bins.tolist()) & set(u_bins.tolist())):
        mk, mu = k_bins == s, u_bins == s
        if mk.sum() < 5 or mu.sum() < 5:
            continue
        w = int(mk.sum()) * int(mu.sum())
        num += max(auroc(sk[mk], su[mu]) for sk, su in scores.values()) * w
        den += w
    return num / den


def nearest_grid_bin(est: np.ndarray) -> np.ndarray:
    """Assign each estimate to the nearest SNR grid point (deployed binning)."""
    grid = np.asarray(SNR_GRID)
    idx = np.argmin(np.abs(est[:, None] - grid[None, :]), axis=1)
    return grid[idx]


def build_pool_estimates(est: dict, data: dict) -> dict:
    """Per-pool estimated-SNR arrays aligned to the npz score ordering.

    known pool  : [emb_1 for all kk samples] + [emb_2 for all kk samples]
                  -> tile(est_kk, 2) pairs every entry with ITS mixture (exact,
                     swap-independent).
    unknown pool: [ku emb_1[post-swap mask1]] + [ku emb_2[post-swap mask2]]
                  + [uu emb_1 all] + [uu emb_2 all].  The post-swap masks are
                  model-dependent; we use the dataset's pre-swap OOD order
                  (same 192-mixture <-> 192-entry bijection per bin, arbitrary
                  within-bin pairing; quantified by Monte-Carlo).
    """
    est_kk = est['kk']
    est_ku, est_uu = est['ku'], est['uu']
    m1, m2 = data['ku']['ood1'], data['ku']['ood2']
    return dict(
        known_tile=np.tile(est_kk, 2),
        known_repeat=np.repeat(est_kk, 2),
        unknown=np.concatenate([est_ku[m1], est_ku[m2], est_uu, est_uu]),
    )


def evaluate_seed(path: str, pool_est: dict, data: dict, est: dict,
                  rng: np.random.Generator) -> dict:
    """All routing variants for one seed's npz file."""
    d = np.load(path)
    k_snr, u_snr = d['known_snr'], d['unknown_snr']
    e_k, e_u = d['energy_score_known'], d['energy_score_unknown']
    p_k, p_u = d['prototype_score_known'], d['prototype_score_unknown']
    ref = analyze_file(path, ROUTING_THRESHOLD)['wavg']

    res = {'gt_routed_ensemble_analysis': float(ref['routed']),
           'gt_oracle': float(ref['oracle'])}

    # GT replication through THIS harness (must match ensemble_analysis).
    gt_k = route(e_k, p_k, k_snr)
    gt_u = route(e_u, p_u, u_snr)
    res['gt_routed_this_harness'] = wavg_auroc(gt_k, gt_u, k_snr, u_snr)

    # True per-entry SNR bins (corrects evaluate.py's repeat-vs-tile binning
    # of the known pool; the unknown pool's stored snr is already exact).
    k_bins_true = np.tile(d['snr_kk'].astype(np.float64), 2)

    for method in ('m2m4', 'subspace'):
        est_k_tile = pool_est[method]['known_tile']
        est_k_rep = pool_est[method]['known_repeat']
        est_u = pool_est[method]['unknown']

        # A1: simulation-matched pairing (est of a mixture whose true SNR is
        # the STORED bin value, i.e. the exact analog of noisy_routed_wavg).
        rk = route(e_k, p_k, est_k_rep)
        ru = route(e_u, p_u, est_u)
        res[f'{method}_A1_est_routed_published_bins'] = wavg_auroc(
            rk, ru, k_snr, u_snr)

        # A2: deployed pairing (each score entry routed by ITS OWN mixture's
        # estimate; bins still the published stored arrays).
        rk = route(e_k, p_k, est_k_tile)
        res[f'{method}_A2_est_routed_own_mixture'] = wavg_auroc(
            rk, ru, k_snr, u_snr)

        # B: corrected true-SNR bins, deployed pairing.
        res[f'{method}_B_est_routed_true_bins'] = wavg_auroc(
            rk, ru, k_bins_true, u_snr)

        # D: fully deployed — bins by ESTIMATED snr as well.
        bk = nearest_grid_bin(est_k_tile)
        bu = nearest_grid_bin(est_u)
        res[f'{method}_D_est_routed_est_bins'] = wavg_auroc(rk, ru, bk, bu)

        # MC sensitivity of the ku-block bijection (A1 metric).
        n_ku = len(data['ku']['snr'])
        trials = []
        for _ in range(MC_TRIALS_ALIGN):
            est_u_mc = est_u.copy()
            for s in SNR_GRID:
                idx = np.where(u_snr[:n_ku] == s)[0]
                pool = est['ku'][method][data['ku']['snr'] == s]
                assert len(idx) == len(pool)
                est_u_mc[idx] = rng.permutation(pool)
            ru_mc = route(e_u, p_u, est_u_mc)
            trials.append(wavg_auroc(route(e_k, p_k, est_k_rep), ru_mc,
                                     k_snr, u_snr))
        res[f'{method}_A1_mc_mean'] = float(np.mean(trials))
        res[f'{method}_A1_mc_std'] = float(np.std(trials))

    # GT routing on corrected bins (does the repeat/tile artifact matter?).
    gtk = route(e_k, p_k, k_bins_true)
    gtu = route(e_u, p_u, u_snr)
    res['gt_routed_corrected_bins'] = wavg_auroc(gtk, gtu, k_bins_true, u_snr)

    # Context under corrected (truly SNR-matched) bins: single scorers,
    # post-hoc oracle, and the published-bins single-scorer references.
    v_k = d['vos_score_known']
    v_u = d['vos_score_unknown']
    true_scores = {'energy': (e_k, e_u), 'prototype': (p_k, p_u),
                   'vos': (v_k, v_u)}
    for m_name in true_scores:
        res[f'true_bins_{m_name}'] = wavg_auroc(
            *true_scores[m_name], k_bins_true, u_snr)
    res['true_bins_oracle'] = wavg_auroc_oracle(true_scores, k_bins_true,
                                                u_snr)
    res['published_bins_energy'] = float(ref['energy'])
    res['published_bins_prototype'] = float(ref['prototype'])
    res['published_bins_vos'] = float(ref['vos'])
    return res


def run_sigma_simulation(files: list[str]) -> dict:
    """Re-run the published Gaussian-perturbation simulation in this harness."""
    rng = np.random.default_rng(0)      # mc_seed=0, as ensemble_analysis.main
    out = {}
    for sigma in SIGMA_LEVELS:
        file_means = []
        spread = []
        for f in files:
            d = np.load(f)
            trials = [noisy_routed_wavg(d, ROUTING_THRESHOLD, sigma, rng)
                      for _ in range(MC_TRIALS_SIGMA)]
            file_means.append(float(np.mean(trials)))
            spread.append(float(np.std(trials)))
        out[f'{sigma:g}'] = dict(
            per_seed=file_means,
            mean=float(np.mean(file_means)),
            std=float(np.std(file_means)),
            mc_spread=float(np.mean(spread)),
        )
    return out


def equivalent_sigma(value: float, sigma_curve: dict, gt_value: float):
    """Piecewise-linear inverse of the sigma -> AUROC curve at `value`."""
    xs = [0.0] + [float(s) for s in SIGMA_LEVELS]
    ys = [gt_value] + [sigma_curve[f'{s:g}']['mean'] for s in SIGMA_LEVELS]
    if value >= ys[0]:
        return 0.0
    if value <= ys[-1]:
        return None                      # worse than the sigma=6 point
    for i in range(len(xs) - 1):
        if ys[i + 1] <= value <= ys[i]:
            t = (ys[i] - value) / (ys[i] - ys[i + 1])
            return xs[i] + t * (xs[i + 1] - xs[i])
    return None


# ============================================================================
# main
# ============================================================================
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--n_per_snr_uu', type=int, default=100,
                   help='uu protocol count per SNR (100 = full, as evaluate.py)')
    p.add_argument('--out_dir', type=str, default=C.RESULTS_DIR)
    return p.parse_args()


def main():
    args = get_args()
    t_start = time.time()
    npz_files = [os.path.join(C.RESULTS_DIR, NPZ_TEMPLATE.format(s))
                 for s in SEEDS]
    for f in npz_files:
        if not os.path.exists(f):
            sys.exit(f'missing score file: {f}')

    print('== a. Regenerating test sets (seed 99999, as evaluate.py) ==')
    data = build_test_sets(n_per_snr_uu=args.n_per_snr_uu)

    print('== a2. Verifying alignment against saved npz files ==')
    all_checks = []
    for f in npz_files:
        checks = verify_alignment(np.load(f), data)
        all_checks.append((os.path.basename(f), checks))
    for name, checks in all_checks[:1]:
        print(f'  [{name}]')
        for cname, ok, detail in checks:
            print(f'    {"PASS" if ok else "FAIL"}  {cname}  ({detail})')
    for name, checks in all_checks[1:]:
        ok_all = all(ok for _, ok, _ in checks)
        print(f'  [{name}] all checks pass: {ok_all}')
    hard_fail = [c for _, checks in all_checks for c in checks[:3] if not c[1]]
    if hard_fail:
        sys.exit(f'CRITICAL alignment failure: {hard_fail}')

    print('== b. Blind SNR estimation on all mixtures ==')
    est = estimate_all(data)

    snr_all = np.concatenate([data[p]['snr'] for p in ('kk', 'ku', 'uu')])
    est_all = {m: np.concatenate([est[p][m] for p in ('kk', 'ku', 'uu')])
               for m in ('m2m4', 'subspace')}
    quality = {m: quality_table(snr_all, est_all[m])
               for m in ('m2m4', 'subspace')}
    quality_per_proto = {
        m: {p: quality_table(data[p]['snr'], est[p][m])
            for p in ('kk', 'ku', 'uu')}
        for m in ('m2m4', 'subspace')}

    print('== c/d. Routed AUROC per seed ==')
    rng = np.random.default_rng(20260921)
    pool_est = {m: build_pool_estimates(
        {p: est[p][m] for p in ('kk', 'ku', 'uu')}, data)
        for m in ('m2m4', 'subspace')}
    per_seed = {}
    for seed, f in zip(SEEDS, npz_files):
        per_seed[str(seed)] = evaluate_seed(f, pool_est, data, est, rng)
        r = per_seed[str(seed)]
        print(f'  seed {seed}: GT={r["gt_routed_ensemble_analysis"]:.3f}  '
              f'm2m4(A1)={r["m2m4_A1_est_routed_published_bins"]:.3f}  '
              f'subspace(A1)={r["subspace_A1_est_routed_published_bins"]:.3f}')

    def summarize(key):
        v = np.array([per_seed[str(s)][key] for s in SEEDS])
        return dict(per_seed=v.tolist(), mean=float(v.mean()),
                    std=float(v.std()))

    variant_keys = [k for k in per_seed[str(SEEDS[0])]
                    if not k.startswith(('m2m4_A1_mc', 'subspace_A1_mc'))]
    summary = {k: summarize(k) for k in variant_keys}
    for m in ('m2m4', 'subspace'):
        summary[f'{m}_A1_mc'] = dict(
            mean=float(np.mean([per_seed[str(s)][f'{m}_A1_mc_mean']
                                for s in SEEDS])),
            spread=float(np.mean([per_seed[str(s)][f'{m}_A1_mc_std']
                                  for s in SEEDS])))

    print('== sigma simulation (re-run, same harness) ==')
    sigma_sim = run_sigma_simulation(npz_files)
    for sig, v in sigma_sim.items():
        print(f'  sigma={sig}: {v["mean"]:.3f} ± {v["std"]:.3f}')

    gt_mean = summary['gt_routed_ensemble_analysis']['mean']
    eq_sigma = {}
    for m in ('m2m4', 'subspace'):
        v = summary[f'{m}_A1_est_routed_published_bins']['mean']
        eq = equivalent_sigma(v, sigma_sim, gt_mean)
        eq_sigma[m] = round(eq, 2) if eq is not None else '>6'
    print(f'  equivalent sigma: {eq_sigma}')

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, 'snr_est_routing.json')
    txt_path = os.path.join(args.out_dir, 'snr_est_routing.txt')
    payload = dict(
        meta=dict(
            date=datetime.date.today().isoformat(),
            script=os.path.basename(__file__),
            npz_files=[os.path.basename(f) for f in npz_files],
            test_set=dict(seed=99999, protocols=dict(
                kk=dict(n_per_snr=200), ku=dict(n_per_snr=200),
                uu=dict(n_per_snr=args.n_per_snr_uu)),
                          snr_grid=SNR_GRID),
            routing_rule='energy if snr <= 0 dB else prototype (a priori)',
            estimators=dict(
                m2m4=dict(kappa=snr_est.KAPPA_DEFAULT,
                          note='fourth-moment; kappa calibrated on clean '
                               'two-source mixtures (pair range 1.58-2.01, '
                               'mean 1.71); conservative -> compressed but '
                               'monotone at high SNR'),
                subspace=dict(n_lags=snr_est.N_LAGS_DEFAULT,
                              note='median eigenvalue of the Toeplitz '
                                   'autocorrelation = noise floor')),
            alignment_verification=[(name, [(c, ok, det) for c, ok, det in ch])
                                    for name, ch in all_checks],
            alignment_caveat=('ku-block pool order is model-dependent (PIT '
                              'swap); reconstructed with pre-swap OOD order; '
                              'effect quantified by within-bin Monte-Carlo '
                              f'({MC_TRIALS_ALIGN} trials): see *_A1_mc'),
            runtime_s=round(time.time() - t_start, 1),
        ),
        estimator_quality=quality,
        estimator_quality_per_protocol=quality_per_proto,
        routing_per_seed=per_seed,
        routing_summary=summary,
        sigma_simulation_rerun=sigma_sim,
        equivalent_sigma=eq_sigma,
    )
    with open(json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    write_txt(txt_path, payload)
    print(f'\nSaved {txt_path}\nSaved {json_path}')
    print(f'total runtime {time.time() - t_start:.1f} s')


def _fmt_quality(q: dict) -> list[str]:
    lines = [f"  {'true SNR':>8s} | {'n':>4s} | {'mean est':>8s} | "
             f"{'bias':>6s} | {'std':>5s} | {'RMSE':>5s} | {'P(est<=0)':>9s}"]
    for s in [f'{v:g}' for v in SNR_GRID]:
        r = q[s]
        lines.append(f"  {float(s):+8.0f} | {r['n']:>4d} | {r['mean_est']:+8.2f} | "
                     f"{r['bias']:+6.2f} | {r['std']:>5.2f} | {r['rmse']:>5.2f} | "
                     f"{r['p_le0']:>9.2f}")
    r = q['overall']
    lines.append(f"  {'overall':>8s} | {r['n']:>4d} | {'':>8s} | "
                 f"{r['bias']:+6.2f} | {r['std']:>5.2f} | {r['rmse']:>5.2f} |")
    return lines


def write_txt(path: str, P: dict) -> None:
    S = P['routing_summary']
    lines = []
    w = lines.append
    w('=' * 78)
    w('SNR-ROUTED OOD ENSEMBLE WITH A REAL BLIND SNR ESTIMATOR (reviewer R1-4)')
    w(f"date: {P['meta']['date']}   script: {P['meta']['script']}   "
      f"runtime: {P['meta']['runtime_s']} s")
    w('=' * 78)
    w('')
    w('PROTOCOL')
    w('  Test sets regenerated exactly as evaluate.py: CommBSSOpenSetTestDataset')
    w(f"  seed 99999, kk/ku n_per_snr=200, uu n_per_snr="
      f"{P['meta']['test_set']['protocols']['uu']['n_per_snr']}, "
      f"SNR grid {P['meta']['test_set']['snr_grid']}.")
    w('  Per-seed OOD scores from the saved npz files (no network re-run):')
    for f in P['meta']['npz_files']:
        w(f'    {f}')
    w(f"  Routing rule (a priori, unchanged): {P['meta']['routing_rule']}")
    w('  Weighted-avg per-SNR AUROC: per-bin AUROC (both pools in the same')
    w('  bin) weighted by n_known*n_unknown pairs, as ensemble_analysis.py.')
    w('')
    w('ALIGNMENT VERIFICATION (regenerated data vs npz; shown for seed 42,')
    w('identical result for the other four seeds unless noted)')
    name, checks = P['meta']['alignment_verification'][0]
    w(f'  [{name}]')
    for cname, ok, det in checks:
        w(f'    {"PASS" if ok else "FAIL"}  {cname}  ({det})')
    for name, checks in P['meta']['alignment_verification'][1:]:
        w(f"  [{name}] all checks pass: "
          f"{all(ok for _, ok, _ in checks)}")
    w(f"  Caveat: {P['meta']['alignment_caveat']}")
    w('')
    w('ESTIMATORS (waveform-only, see snr_est.py)')
    w(f"  m2m4    : {P['meta']['estimators']['m2m4']['note']} "
      f"(kappa={P['meta']['estimators']['m2m4']['kappa']})")
    w(f"  subspace: {P['meta']['estimators']['subspace']['note']} "
      f"(L={P['meta']['estimators']['subspace']['n_lags']})")
    w('')
    w('ESTIMATOR QUALITY PER TRUE-SNR BIN (all 3360 mixtures, 480/bin)')
    for m in ('m2m4', 'subspace'):
        w(f'  [{m}]')
        lines.extend(_fmt_quality(P['estimator_quality'][m]))
        w('')
    w('ESTIMATOR QUALITY PER PROTOCOL (RMSE dB per bin; see JSON for full')
    w('bias/std tables)')
    grid = [f'{v:g}' for v in SNR_GRID]
    w(f"  {'est/protocol':>16s} | " + ' | '.join(f'{float(s):+5.0f}' for s in grid))
    for m in ('m2m4', 'subspace'):
        for proto in ('kk', 'ku', 'uu'):
            q = P['estimator_quality_per_protocol'][m][proto]
            w(f"  {m + '/' + proto:>16s} | " +
              ' | '.join(f"{q[s]['rmse']:5.2f}" for s in grid))
    w('')
    w('ROUTED WEIGHTED-AVG AUROC (mean ± std across seeds 42-46)')
    rows = [
        ('GT routing (published bins, = paper headline)',
         'gt_routed_ensemble_analysis'),
        ('GT routing, replicated in this harness', 'gt_routed_this_harness'),
        ('GT routing, corrected true bins', 'gt_routed_corrected_bins'),
        ('M2M4 est, A1 sim-matched pairing', 'm2m4_A1_est_routed_published_bins'),
        ('M2M4 est, A2 own-mixture pairing', 'm2m4_A2_est_routed_own_mixture'),
        ('M2M4 est, B corrected true bins', 'm2m4_B_est_routed_true_bins'),
        ('M2M4 est, D fully deployed (est bins)', 'm2m4_D_est_routed_est_bins'),
        ('subspace est, A1 sim-matched pairing',
         'subspace_A1_est_routed_published_bins'),
        ('subspace est, A2 own-mixture pairing',
         'subspace_A2_est_routed_own_mixture'),
        ('subspace est, B corrected true bins',
         'subspace_B_est_routed_true_bins'),
        ('subspace est, D fully deployed (est bins)',
         'subspace_D_est_routed_est_bins'),
    ]
    for label, key in rows:
        v = S[key]
        w(f"  {label:<46s} {v['mean']:.3f} ± {v['std']:.3f}   "
          f"(per seed: {', '.join(f'{x:.3f}' for x in v['per_seed'])})")
    for m in ('m2m4', 'subspace'):
        v = S[f'{m}_A1_mc']
        w(f"  {m + ' A1 Monte-Carlo over ku-block bijection':<46s} "
          f"mean {v['mean']:.3f}, avg within-seed spread {v['spread']:.4f}")
    w('')
    w('CONTEXT: SINGLE SCORERS AND ORACLE UNDER EACH BINNING')
    for label, key in [
        ('energy, published bins', 'published_bins_energy'),
        ('prototype, published bins', 'published_bins_prototype'),
        ('vos, published bins', 'published_bins_vos'),
        ('energy, corrected true bins', 'true_bins_energy'),
        ('prototype, corrected true bins', 'true_bins_prototype'),
        ('vos, corrected true bins', 'true_bins_vos'),
        ('oracle (post-hoc), corrected true bins', 'true_bins_oracle'),
    ]:
        v = S[key]
        w(f"  {label:<46s} {v['mean']:.3f} ± {v['std']:.3f}")
    w('')
    w('CRITICAL FINDING (binning artifact in the published metric)')
    w('  evaluate.py stores known_snr = repeat(snr_kk, 2) while the known-pool')
    w('  score/label arrays are stacked tile-wise ([slot-1 for all samples,')
    w('  then slot-2]).  ensemble_analysis bins by the stored array, so the')
    w('  known entries of a "per-SNR bin" actually come from OTHER true-SNR')
    w('  bins (verified above: label multisets match only under tile binning).')
    w('  The unknown pool is binned exactly.  Each published bin therefore')
    w('  compares known vs unknown mixtures at DIFFERENT true SNRs, and since')
    w('  all OOD score scales drift with SNR, the apparent "energy wins at')
    w('  low SNR / prototype wins at high SNR" complementarity is produced by')
    w('  the SNR mismatch, not by OOD separability.  Under corrected, truly')
    w('  SNR-matched bins the GT-routed AUROC drops to the single-scorer')
    w('  level (~0.50): the routing gain does not survive fair binning, with')
    w('  GT or with estimated SNR.  On the paper\'s own (artifact-laden)')
    w('  metric, a real blind estimator reproduces the headline (see A1 rows')
    w('  and equivalent-sigma below), but that metric measures SNR-mismatch')
    w('  separability, not per-SNR OOD detection.')
    w('')
    w('GAUSSIAN-PERTURBATION SIMULATION (re-run in this harness, 50 MC trials,')
    w('mc_seed=0; published: sigma=1 -> 0.597, sigma=3 -> 0.593, sigma=6 -> 0.576)')
    for sig, v in P['sigma_simulation_rerun'].items():
        w(f"  sigma={sig:>3s} dB : {v['mean']:.3f} ± {v['std']:.3f} "
          f"(avg MC spread {v['mc_spread']:.3f})")
    w('')
    w('WHERE THE REAL ESTIMATOR LANDS (equivalent Gaussian sigma, piecewise-')
    w('linear interpolation of the re-run sigma curve incl. the GT point;')
    w('">6" = below the sigma=6 dB simulation point)')
    for m, v in P['equivalent_sigma'].items():
        w(f'  {m}: equivalent sigma ~= {v} dB')
    w('')
    w('NOTES')
    w('  - Published-bin variants keep the npz stored binning (known pool bins')
    w('    use evaluate.py\'s repeat(snr_kk,2) array) for exact comparability')
    w('    with the paper\'s 0.625 and the sigma-simulation points.  Variant B')
    w('    shows the corrected-binning sensitivity.')
    w('  - A1 pairs each score entry with the estimate of a mixture whose true')
    w('    SNR equals the stored bin value (the exact analog of the Gaussian')
    w('    simulation); A2 pairs each entry with its own mixture\'s estimate')
    w('    (deployed behaviour).  The two coincide for the unknown pool.')
    w('  - D bins by the estimated SNR itself (nearest grid point, pair-')
    w('    weighted): the fully blind, deployable metric.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
