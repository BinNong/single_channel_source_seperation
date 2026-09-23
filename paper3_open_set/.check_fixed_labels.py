"""Spot-check corrected (tile-ordered) known-SNR labels in re-dumped npz files."""
import numpy as np

R = 'results/'
BASE = R + 'openset_cse_h32_l4_bs16_lr0.001_alpha1.0'


def check_npz(path):
    d = np.load(path)
    ks = d['known_snr']
    n2 = len(ks) // 2
    tile_halves = np.array_equal(ks[:n2], ks[n2:])
    msg = [f"{path.split('/')[-1]}: halves-equal(tile)={tile_halves}"]
    if 'snr_kk' in d:
        msg.append(f"  == tile(snr_kk,2): {np.array_equal(ks, np.tile(d['snr_kk'], 2))}")
    if 'known_mods' in d:
        km = d['known_mods']
        ok = all(np.array_equal(np.bincount(km[ks == s], minlength=4), [96, 96, 96, 96])
                 for s in np.unique(ks))
        msg.append(f"  per-bin known_mods multiset == 96x4: {ok}")
    print('\n'.join(msg))


check_npz(BASE + '_seed42_best_ood_scores.npz')
check_npz(BASE + '_seed46_best_gap500_ood_scores.npz')
check_npz(BASE + '_lc0.1_seed42_lc01_best_ood_scores.npz')
check_npz(BASE + '_seed42_emb16_best_ood_scores.npz')

d = np.load(BASE + '_seed42_best_refpool.npz')
rs = d['ref_snr']
n2 = len(rs) // 2
print(f"refpool seed42: halves-equal(tile)={np.array_equal(rs[:n2], rs[n2:])}")

d = np.load(BASE + '_seed42_best_odin_eps0.005_T1000.npz')
ks = d['known_snr']
n2 = len(ks) // 2
print(f"odin seed42: halves-equal(tile)={np.array_equal(ks[:n2], ks[n2:])}")

d = np.load(R + 'lomo_openset_cse_h32_l4_bs16_lr0.001_alpha1.0_kmQPSK-8PSK-16QAM_seed42_best_scores.npz')
print('lomo keys:', list(d.keys()))
for k in ('test_known_snr', 'val_known_snr'):
    v = d[k]
    n2 = len(v) // 2
    print(f"lomo {k}: halves-equal(tile)={np.array_equal(v[:n2], v[n2:])}")
