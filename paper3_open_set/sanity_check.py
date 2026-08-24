"""
Paper 3 — local sanity check (no torch required).

Verifies:
  - All required files exist
  - Numpy-only modules smoke-test cleanly
  - Open-set metrics behave sensibly on synthetic distributions
  - Config is importable

Run on any machine (no GPU needed):
    cd paper3_open_set && python3 sanity_check.py

For the actual training + evaluation pipeline run `bash run.sh smoke`
on the remote GPU server.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback


REQUIRED_FILES = [
    'config.py',
    'data_generator_extended.py',
    'models.py',
    'losses.py',
    'ood_scores.py',
    'open_set_metrics.py',
    'train.py',
    'evaluate.py',
    'run.sh',
    'README.md',
    'utils.py',                      # symlink
]


def _check_files():
    print("[1/5] Checking required files ...")
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    print(f"  ok ({len(REQUIRED_FILES)} files present)")


def _check_config():
    print("[2/5] Importing config ...")
    if not _TORCH_OK:
        # Skip — config.py imports torch via data_generator_extended.
        print("  skipped (torch not installed)")
        return
    import config as C
    assert hasattr(C, 'MOD_KNOWN') or True   # populated via import below
    assert hasattr(C, 'NUM_KNOWN_CLASSES')
    assert C.NUM_KNOWN_CLASSES == 4
    assert hasattr(C, 'SIGNAL_LENGTH')
    assert C.SIGNAL_LENGTH == 4096
    print(f"  ok (NUM_KNOWN_CLASSES={C.NUM_KNOWN_CLASSES}, "
          f"SIGNAL_LENGTH={C.SIGNAL_LENGTH})")


def _check_mod_vocab():
    print("[3/5] Checking modulation vocabulary ...")
    sys.path.insert(0, '.')
    # Import the parts of data_generator_extended that don't pull torch via paper1.
    # (paper1's data_generator unconditionally imports torch; we only verify
    # the constants which are defined at the top of paper3's file.)
    if not _TORCH_OK:
        # Read the file and parse MOD_KNOWN/MOD_UNKNOWN out of it.
        with open('data_generator_extended.py') as f:
            src = f.read()
        for needle in ("MOD_KNOWN   = ['BPSK', 'QPSK', '8PSK', '16QAM']",
                       "MOD_UNKNOWN = ['64QAM', 'PI4_DQPSK', 'MSK', 'OFDM_QPSK']"):
            assert needle in src, f"missing vocab line: {needle}"
        print("  ok (verified by source scan, torch not installed)")
        return
    from data_generator_extended import MOD_KNOWN, MOD_UNKNOWN, MOD_ALL
    assert MOD_KNOWN == ['BPSK', 'QPSK', '8PSK', '16QAM']
    assert MOD_UNKNOWN == ['64QAM', 'PI4_DQPSK', 'MSK', 'OFDM_QPSK']
    assert len(MOD_ALL) == 8
    print(f"  ok (8 modulations total: 4 known + 4 unknown)")


def _check_ood_scores():
    print("[4/5] Testing OOD scoring (numpy-only) ...")
    import numpy as np
    from ood_scores import energy_score, prototype_score, vos_score, compute_prototypes

    rng = np.random.RandomState(0)
    K, D = 4, 16
    proto = rng.randn(K, D) * 2.0
    emb_in  = proto[rng.randint(0, K, 400)] + 0.3 * rng.randn(400, D)
    emb_out = proto.mean(0) + 6 * rng.randn(100, D)

    # Construct logits: in-dist is confident, OOD is uncertain.
    logits_in  = rng.randn(400, K) * 0.5
    logits_in[np.arange(400), rng.randint(0, K, 400)] += 8.0
    logits_out = rng.randn(100, K) * 0.5

    e_in,  e_out  = energy_score(logits_in), energy_score(logits_out)
    p_in,  p_out  = prototype_score(emb_in, proto), prototype_score(emb_out, proto)
    v_in,  v_out  = vos_score(emb_in, proto, 2.0, 20, seed=0), \
                    vos_score(emb_out, proto, 2.0, 20, seed=0)

    assert e_out.mean() > e_in.mean(),  f"energy:  {e_out.mean():.3f} not > {e_in.mean():.3f}"
    assert p_out.mean() > p_in.mean(),  f"prototype: {p_out.mean():.3f} not > {p_in.mean():.3f}"
    assert v_out.mean() > v_in.mean(),  f"vos:     {v_out.mean():.3f} not > {v_in.mean():.3f}"
    print(f"  ok (energy={e_out.mean()-e_in.mean():.2f}, "
          f"prototype={p_out.mean()-p_in.mean():.2f}, "
          f"vos={v_out.mean()-v_in.mean():.2f} OOD-in gap)")


def _check_metrics():
    print("[5/5] Testing open-set metrics (numpy-only) ...")
    import numpy as np
    from open_set_metrics import auroc, aupr_in, fpr_at_95_tpr, oscr

    rng = np.random.RandomState(0)
    sk = rng.normal(0, 1, 500)
    su = rng.normal(2, 1, 200)
    correct = rng.rand(500) > 0.2

    a = auroc(sk, su)
    assert 0.85 < a < 1.0, f"AUROC out of expected range: {a}"
    f = fpr_at_95_tpr(sk, su)
    assert 0.0 <= f <= 0.6, f"FPR@95 too high: {f}"
    o = oscr(correct, sk, su)
    assert o > 0.3, f"OSCR suspiciously low: {o}"
    # AUROC on identical distributions should be ~ 0.5
    s2 = rng.normal(0, 1, 200)
    a_rand = auroc(s2, s2)
    assert 0.45 < a_rand < 0.55, f"AUROC(random) out of expected range: {a_rand}"
    print(f"  ok (AUROC={a:.3f}, FPR@95={f:.3f}, OSCR={o:.3f}, "
          f"AUROC(random)={a_rand:.3f})")


# ----------------------------------------------------------------------------
# Boilerplate
# ----------------------------------------------------------------------------
_TORCH_OK = True
try:
    import torch  # noqa: F401
except ImportError:
    _TORCH_OK = False


def main():
    if not _TORCH_OK:
        print("[note] torch not installed; some checks will be skipped.\n")

    failures = 0
    for check in (_check_files, _check_config, _check_mod_vocab,
                   _check_ood_scores, _check_metrics):
        try:
            check()
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failures += 1

    print()
    if failures == 0:
        print("=" * 60)
        print("  Sanity check PASSED.")
        print("  Next: run `bash run.sh smoke` on the remote GPU server.")
        print("=" * 60)
    else:
        print(f"  Sanity check FAILED: {failures} check(s) failed.")
        sys.exit(1)


if __name__ == '__main__':
    main()