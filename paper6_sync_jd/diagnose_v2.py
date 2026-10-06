"""Paper 6 — E4 diagnosis: why does slot-aided ECM (V2) lose to
mixture ECM (V1)?  (plan risk R4 — "when to separate at all")

Working hypothesis: mask-based separation distorts the linear
superposition model r = A [u1 c1; u2 c2] that the ECM assumes, while the
raw mixture preserves it.  Measurements on the same K=2 bursts:

  (a) Converged ECM residual energy on the slot streams vs the mixture
      (both unit-power normalised) — model-fit quality.
  (b) Split-burst gain drift: after convergence, LS-fit A on 4
      sub-blocks of the burst; report the per-column phase/magnitude
      spread across blocks.  The mixture's true gains are constant by
      construction; the separator's mask is time-varying, so slot gains
      drift if the mask modulates within the burst.
  (c) Broken down by modulation-pair class (PSK-only vs
      16QAM-involving).

Local CPU; uses the pulled config-(b) checkpoint (seed 42).
__main__ runs the study.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Vendored paper6 modules (self-contained; no sys.path tricks).
import config as C                                          # noqa: E402
from data_generator import CommBSSVarKTestDataset, MOD_TYPES  # noqa: E402
from joint_detect import constellation_np                    # noqa: E402
from joint_sync_detect import (ecm_joint, mixture_symbols,   # noqa: E402
                               em_converged, _steer, _ls_gains, _res_energy)
from eval_blind_pipeline import build_slot_from_ckpt          # noqa: E402


def gain_drift(R, A, df1, df2, dec1, dec2, const1, const2, n_blocks=4):
    """Split-burst gain drift: LS-fit A per sub-block; return per-column
    phase spread (deg, circular std) and magnitude spread (std/mean)."""
    N = R.shape[1]
    edges = np.linspace(0, N, n_blocks + 1).astype(int)
    cols = {0: [], 1: []}
    for b in range(n_blocks):
        sl = slice(edges[b], edges[b + 1])
        n_idx = np.arange(N)[sl]
        U = np.stack([_steer(df1, n_idx), _steer(df2, n_idx)])
        Ab = _ls_gains(R[:, sl], U, dec1[sl], dec2[sl], const1, const2)
        cols[0].append(Ab[:, 0])
        cols[1].append(Ab[:, 1])
    out = {}
    for k in (0, 1):
        a = np.stack(cols[k])                    # [n_blocks, P]
        # phase drift of the MRC-combined scalar gain (channel 0 for P=1)
        g = a[:, 0]
        ph = np.angle(g)
        ph_spread = float(np.degrees(np.sqrt(-2 * np.log(
            max(abs(np.mean(np.exp(1j * ph))), 1e-12)))))
        mag_spread = float(np.std(np.abs(g)) / (np.mean(np.abs(g)) + 1e-12))
        out[k] = (ph_spread, mag_spread)
    return out


def main():
    device = C.DEVICE
    model, _ = build_slot_from_ckpt(
        os.path.join(C.CHECKPOINT_DIR,
                     'slot_h64_l4_k13_bs16_lr0.001_mse_s42_best.pt'), device)
    ds = CommBSSVarKTestDataset(
        n_per_cell=40, snr_points=[20], seed=C.DataConfig.test_seed,
        return_carriers=True)
    rng = np.random.RandomState(C.SEED)

    recs = []
    done = 0
    for i in rng.permutation(len(ds)):
        s = ds.samples[i]
        if s['k'] != 2:
            continue
        mix_np = s['mixture'].numpy()[0]
        mods_idx = [int(m) for m in s['mods'].numpy()[:2]]
        cls = 'psk' if all(MOD_TYPES[m] != '16QAM' for m in mods_idx) else 'qam'
        c1 = constellation_np(mods_idx[0])
        c2 = constellation_np(mods_idx[1])

        # mixture route
        r_mix = mixture_symbols(mix_np)
        res_m = ecm_joint(r_mix, c1, c2)
        # slot route
        mix_t = torch.from_numpy(mix_np).view(1, 1, -1).to(torch.complex64) \
            .to(device)
        with torch.no_grad():
            slots, occ_logits, _ = model(mix_t)
        slots_np = slots[0].cpu().numpy()
        top2 = np.argsort(-torch.sigmoid(occ_logits[0]).cpu().numpy())[:2]
        R_sl = np.stack([mixture_symbols(slots_np[j]) for j in top2])
        res_s = ecm_joint(R_sl, c1, c2)

        rec = {'snr': float(s['snr']), 'cls': cls,
               'E_mix': res_m['res_energy'], 'E_slot': res_s['res_energy'],
               'df_mix': (res_m['df1'], res_m['df2']),
               'df_slot': (res_s['df1'], res_s['df2'])}
        rec['drift_mix'] = gain_drift(r_mix[None, :], res_m['A'],
                                      res_m['df1'], res_m['df2'],
                                      res_m['dec1'], res_m['dec2'], c1, c2)
        rec['drift_slot'] = gain_drift(R_sl, res_s['A'], res_s['df1'],
                                       res_s['df2'], res_s['dec1'],
                                       res_s['dec2'], c1, c2)
        recs.append(rec)
        done += 1
        if done % 10 == 0:
            print(f"  {done} bursts", flush=True)
        if done >= 40:
            break

    # ---- report ----
    print("\n=== (a) converged ECM residual energy per channel-symbol ===")
    for cls in ('psk', 'qam'):
        rs = [r for r in recs if r['cls'] == cls]
        if not rs:
            continue
        em = np.array([r['E_mix'] / 256 for r in rs])       # P=1
        es = np.array([r['E_slot'] / 512 for r in rs])      # P=2
        print(f"  {cls}: mixture E/(PN) med/mean = {np.median(em):.4f}/"
              f"{em.mean():.4f}   slots E/(PN) med/mean = {np.median(es):.4f}/"
              f"{es.mean():.4f}   (n={len(rs)}; slot/mix median ratio "
              f"{np.median(es/em):.2f})")
    print("\n=== (b) split-burst gain drift (phase spread deg / mag spread) ===")
    for cls in ('psk', 'qam'):
        rs = [r for r in recs if r['cls'] == cls]
        if not rs:
            continue
        for k in (0, 1):
            pm = np.array([r['drift_mix'][k] for r in rs])
            ps = np.array([r['drift_slot'][k] for r in rs])
            print(f"  {cls} col{k}: mixture {np.median(pm[:,0]):.2f}deg/"
                  f"{np.median(pm[:,1]):.3f}   slots "
                  f"{np.median(ps[:,0]):.2f}deg/{np.median(ps[:,1]):.3f}")

    out = {'n_bursts': len(recs),
           'records': [{k: (v if not isinstance(v, (dict, tuple)) else str(v))
                        for k, v in r.items()} for r in recs]}
    import json, os
    path = os.path.join(C.RESULTS_DIR, 'e4_v2_diagnosis.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")


if __name__ == '__main__':
    main()
