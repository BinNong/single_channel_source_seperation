"""Aggregate per-seed evaluation results and produce:
  - Table 1 with mean ± std across seeds (sample std, ddof=1)
  - Paired ΔSDR computed on common seeds (inner-join by seed label, NOT positional slicing)
  - Unpaired descriptive ΔSDR (table mean vs table mean) -- separately labeled
  - Wilcoxon signed-rank tests on common-seed pairs (exploratory, N<=5)
  - JSON / Markdown output for paper update

Honesty rules:
  - Paired ΔSDR and paired p-values ALWAYS use the same set of seed IDs (inner join).
  - Table-mean descriptive ΔSDR is reported ONLY when the paired ΔSDR is also reported.
  - We never silently pair s42 of one model with s43 of another.
"""
from __future__ import annotations

import glob
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

RESULTS_ROOT = Path("/data/experiment/paper1_cnn_se/results/phase5_results")
OUT_DIR = Path("/data/experiment/paper1_cnn_se/results/phase5_results/_aggregated")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_overall(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def parse_seed_id(dir_name: str, tag_prefix: str) -> Optional[int]:
    """Extract seed integer from `tag_prefix_s42`, `tag_prefix_s43`, ... Returns int or None."""
    m = re.match(rf"^{re.escape(tag_prefix)}_s(\d+)$", dir_name)
    return int(m.group(1)) if m else None


def per_seed_by_label(model_tag: str) -> dict[int, dict]:
    """Collect per-seed metrics keyed by seed label (int).

    Directories follow the pattern `${RESULTS_ROOT}/${tag}_s${seed}/overall.json`.
    Keys are seed labels so the caller can inner-join across models.
    """
    out: dict[int, dict] = {}
    if not RESULTS_ROOT.exists():
        return out
    for seed_dir in sorted(RESULTS_ROOT.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed_id = parse_seed_id(seed_dir.name, model_tag)
        if seed_id is None:
            continue
        d = load_overall(seed_dir / "overall.json")
        if d is None:
            for f in seed_dir.glob("*overall.json"):
                d = load_overall(f)
                if d:
                    break
        if d is not None:
            out[seed_id] = {"seed": seed_id, **d}
    return out


def summarise(metric: str, per_seed: list[dict]) -> tuple[float, float, int]:
    vals = [d[metric] for d in per_seed if metric in d]
    if not vals:
        return (float("nan"), float("nan"), 0)
    n = len(vals)
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if n >= 2 else 0.0
    return (m, s, n)


def inner_join(a: dict[int, dict], b: dict[int, dict]) -> list[tuple[dict, dict]]:
    """Inner join by seed label. Returns list of (a_seed_dict, b_seed_dict) pairs,
    in sorted seed order. THIS is the proper way to form paired data."""
    common = sorted(set(a.keys()) & set(b.keys()))
    return [(a[s], b[s]) for s in common]


def paired_delta(a: list[float], b: list[float]) -> Optional[float]:
    if not a or len(a) != len(b):
        return None
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return None
    return float(np.mean(diffs))


def wilcoxon_paired(a: list[float], b: list[float]) -> Optional[tuple[float, int]]:
    """Paired Wilcoxon signed-rank (two-sided, exact when possible).
    Returns (p_value, n_nonzero_pairs) or None if N < 2 or all differences are zero.

    IMPORTANT: For very small N, scipy switches between exact and asymptotic.
    We force `method='exact'` when N is small enough (< 50) to be safe.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return (1.0, 0)
    nonzero = sum(1 for d in diffs if d != 0)
    if nonzero < 1:
        return (1.0, 0)
    try:
        method = "exact" if nonzero < 50 else "auto"
        stat = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method=method)
        return (float(stat.pvalue), nonzero)
    except ValueError:
        return None


MODELS = [
    ("cse",       "Complex CNN + SE (Proposed)",            "complex_cnn_se"),
    ("pm_no_se",  "Complex CNN no-SE (matched, H=70)",     "complex_cnn_no_se"),
    ("pm_real",   "Real-Valued CNN (matched, H=80,L=12)", "real_baseline"),
    ("ctasnet",   "Complex Conv-TasNet",                    "conv_tasnet"),
    ("cnse",      "CNSE (scaled, Hou & Gao 2022)",          "cnse"),
    ("s4unet",    "S4-UNET (scaled, Gao et al. 2026)",      "s4unet"),
]

POOLING_VARIANTS = [
    ("power",      "pool_power"),
    ("magnitude",  "pool_magnitude"),
    ("mean_power", "pool_mean_power"),
]

METRICS = ["SI-SDR", "SDR", "SIR", "NMSE"]

# Load per-seed by seed label (NOT positional slicing).
all_per_seed: dict[str, dict[int, dict]] = {}
for tag, _, _ in MODELS:
    all_per_seed[tag] = per_seed_by_label(tag)

all_pooling: dict[str, dict[int, dict]] = {}
for variant, tag in POOLING_VARIANTS:
    all_pooling[variant] = per_seed_by_label(tag)


# -----------------------------------------------------------------------
# Table 1: per-model mean ± std
# -----------------------------------------------------------------------
print("=" * 78)
print("Table 1 -- Overall (mean +/- sample std across seeds)")
print("=" * 78)
table1_rows = []
for tag, name, _ in MODELS:
    per_seed_list = list(all_per_seed[tag].values())
    if not per_seed_list:
        continue
    cells = []
    for metric in METRICS:
        m, s, n = summarise(metric, per_seed_list)
        cells.append((metric, m, s, n))
    table1_rows.append((tag, name, per_seed_list, cells))

header = f"{'Model':<40} {'N':>3}  {'SI-SDR':>11}  {'SDR':>11}  {'SIR':>11}  {'NMSE':>11}"
print(header)
print("-" * len(header))
for tag, name, per_seed_list, cells in table1_rows:
    n = max(c[3] for c in cells)
    row = f"{name:<40} {n:>3}  "
    for metric, m, s, nn in cells:
        if math.isnan(m):
            row += f"{'n/a':>11}  "
        else:
            row += f"{m:>6.2f}+/-{s:<4.2f}  "
    print(row)


# -----------------------------------------------------------------------
# Paired ΔSDR via inner join on seed label
# -----------------------------------------------------------------------
print()
print("=" * 78)
print("Paired analysis: C-SE vs baselines (inner-join by seed label)")
print("=" * 78)

cse_data = all_per_seed["cse"]
if not cse_data:
    print("C-SE data missing; skipping paired tests.")
else:
    for tag, name, _ in MODELS:
        if tag == "cse":
            continue
        baseline_data = all_per_seed[tag]
        pairs = inner_join(cse_data, baseline_data)
        n_pairs = len(pairs)
        if n_pairs < 2:
            print(f"  {name:<40} n/a (only {n_pairs} common seed)")
            continue

        cse_sdr = [p[0]["SDR"] for p in pairs]
        base_sdr = [p[1]["SDR"] for p in pairs]
        seeds = sorted(set(p[0]["seed"] for p in pairs))

        # Paired ΔSDR (mean of per-pair differences on common seeds)
        paired_d = paired_delta(cse_sdr, base_sdr)

        # Unpaired descriptive ΔSDR (table mean of all C-SE seeds minus table mean of all baseline seeds)
        cse_all_sdr = [d["SDR"] for d in cse_data.values() if "SDR" in d]
        base_all_sdr = [d["SDR"] for d in baseline_data.values() if "SDR" in d]
        if cse_all_sdr and base_all_sdr:
            unpaired_d = float(np.mean(cse_all_sdr) - np.mean(base_all_sdr))
            n_cse = len(cse_all_sdr)
            n_base = len(base_all_sdr)
        else:
            unpaired_d = None
            n_cse = n_base = 0

        # Paired Wilcoxon (exact when feasible)
        wp = wilcoxon_paired(cse_sdr, base_sdr)
        if wp is None:
            p_str = "n/a"
            n_nz = 0
        else:
            p, n_nz = wp
            p_str = f"{p:.4f}" if p >= 0.0001 else "< 0.0001"

        print(f"  {name}")
        print(f"    Common seeds:    {seeds}  (n_pairs={n_pairs})")
        print(f"    Paired Delta SDR (mean of per-pair diffs on common seeds): "
              f"{paired_d:+.2f} dB")
        if unpaired_d is not None:
            print(f"    Unpaired descriptive Delta SDR (table mean {n_cse}-seed C-SE "
                  f"vs {n_base}-seed baseline): {unpaired_d:+.2f} dB")
        print(f"    Paired Wilcoxon signed-rank (exact, two-sided): "
              f"p={p_str}  (n_nonzero={n_nz})")
        print()


# -----------------------------------------------------------------------
# Pooling ablation (unchanged logic, just print seeds too)
# -----------------------------------------------------------------------
print()
print("=" * 78)
print("Pooling ablation")
print("=" * 78)
print(f"{'Variant':<14} {'N':>3}  {'SDR (dB)':>11}  {'SIR (dB)':>11}  {'NMSE':>11}  {'Seeds'}")
for variant, _ in POOLING_VARIANTS:
    per_seed_dict = all_pooling[variant]
    per_seed_list = list(per_seed_dict.values())
    if not per_seed_list:
        continue
    n = len(per_seed_list)
    sdr_m, sdr_s, _ = summarise("SDR", per_seed_list)
    sir_m, sir_s, _ = summarise("SIR", per_seed_list)
    nmse_m, nmse_s, _ = summarise("NMSE", per_seed_list)
    seeds = sorted(per_seed_dict.keys())
    print(f"{variant:<14} {n:>3}  "
          f"{sdr_m:>6.2f}+/-{sdr_s:<4.2f}  "
          f"{sir_m:>6.2f}+/-{sir_s:<4.2f}  "
          f"{nmse_m:>6.2f}+/-{nmse_s:<4.2f}  {seeds}")


# -----------------------------------------------------------------------
# JSON output
# -----------------------------------------------------------------------
def serialise_per_seed(per_seed_dict: dict[int, dict]) -> list[dict]:
    return [
        {k: v for k, v in d.items() if k in {"seed", "SI-SDR", "SDR", "SIR", "NMSE"}}
        for d in sorted(per_seed_dict.values(), key=lambda x: x["seed"])
    ]


# Build the JSON structure with paired/unpaired clearly separated.
paired_block: dict[str, dict] = {}
cse_data = all_per_seed["cse"]
if cse_data:
    for tag, name, _ in MODELS:
        if tag == "cse":
            continue
        baseline_data = all_per_seed[tag]
        pairs = inner_join(cse_data, baseline_data)
        if len(pairs) < 2:
            continue
        cse_sdr = [p[0]["SDR"] for p in pairs]
        base_sdr = [p[1]["SDR"] for p in pairs]
        seeds = sorted(p[0]["seed"] for p in pairs)
        paired_d = paired_delta(cse_sdr, base_sdr)
        wp = wilcoxon_paired(cse_sdr, base_sdr)
        p_val, n_nz = wp if wp is not None else (None, 0)

        # Unpaired descriptive
        cse_all = [d["SDR"] for d in cse_data.values() if "SDR" in d]
        base_all = [d["SDR"] for d in baseline_data.values() if "SDR" in d]
        unpaired_d = (float(np.mean(cse_all) - np.mean(base_all))
                      if cse_all and base_all else None)
        n_cse_all = len(cse_all)
        n_base_all = len(base_all)

        paired_block[tag] = {
            "baseline_name": name,
            "common_seeds": seeds,
            "n_pairs": len(pairs),
            "paired_delta_sdr_db": paired_d,
            "paired_wilcoxon_p": p_val,
            "paired_wilcoxon_n_nonzero": n_nz,
            "unpaired_descriptive_delta_sdr_db": unpaired_d,
            "n_cse_seeds": n_cse_all,
            "n_baseline_seeds": n_base_all,
            # Discrete Wilcoxon floor at this N for context
            "wilcoxon_p_min_at_n": 1.0 / (2 ** n_nz) if n_nz > 0 else None,
        }

out = {
    "table1": [
        {
            "tag": tag,
            "name": name,
            "n_seeds": max((c[3] for c in cells), default=0),
            "seeds": sorted(all_per_seed[tag].keys()),
            "metrics": {metric: {"mean": m, "std": s, "n": n}
                        for metric, m, s, n in cells},
        }
        for tag, name, _, cells in table1_rows
    ],
    "paired_vs_cse": paired_block,
    "pooling": {
        variant: {
            "n": len(all_pooling[variant]),
            "seeds": sorted(all_pooling[variant].keys()),
            "metrics": {
                "SDR":  {"mean": summarise("SDR", list(all_pooling[variant].values()))[0],
                          "std":  summarise("SDR", list(all_pooling[variant].values()))[1]},
                "SIR":  {"mean": summarise("SIR", list(all_pooling[variant].values()))[0],
                          "std":  summarise("SIR", list(all_pooling[variant].values()))[1]},
                "NMSE": {"mean": summarise("NMSE", list(all_pooling[variant].values()))[0],
                          "std":  summarise("NMSE", list(all_pooling[variant].values()))[1]},
            },
        }
        for variant in all_pooling if all_pooling[variant]
    },
    "per_seed_raw": {
        tag: serialise_per_seed(all_per_seed[tag])
        for tag in all_per_seed
    },
    "per_seed_pooling": {
        variant: serialise_per_seed(all_pooling[variant])
        for variant in all_pooling if all_pooling[variant]
    },
}

json_path = OUT_DIR / "table1_5seed.json"
with open(json_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {json_path}")


# -----------------------------------------------------------------------
# Markdown
# -----------------------------------------------------------------------
md = ["# Table 1 -- per-seed SDR (dB), mean +/- sample std\n",
      "| Model | N | Seeds | SI-SDR | SDR | SIR | NMSE |",
      "|---|---|---|---|---|---|---|"]
for row in out["table1"]:
    name = row["name"]
    n = row["n_seeds"]
    seeds = ",".join(str(s) for s in row["seeds"])
    cells = []
    for metric in METRICS:
        d = row["metrics"][metric]
        cells.append(f"{d['mean']:.2f}+/-{d['std']:.2f}" if d["n"] else "n/a")
    md.append(f"| {name} | {n} | {seeds} | " + " | ".join(cells) + " |")

md.append("")
md.append("## Paired vs C-SE (inner-join by seed label)")
md.append("")
md.append("| Baseline | Common seeds | n_pairs | Paired Delta SDR (dB) | Unpaired desc. Delta SDR (dB) | Wilcoxon p | n_nonzero | p_min at n_nonzero |")
md.append("|---|---|---|---|---|---|---|---|")
for tag, info in paired_block.items():
    p_min = info["wilcoxon_p_min_at_n"]
    p_min_s = f"{p_min:.4f}" if p_min is not None else "n/a"
    p_s = f"{info['paired_wilcoxon_p']:.4f}" if info['paired_wilcoxon_p'] is not None else "n/a"
    unpaired_d = info["unpaired_descriptive_delta_sdr_db"]
    unpaired_s = (f"{unpaired_d:+.2f} (C-SE N={info['n_cse_seeds']}, base N={info['n_baseline_seeds']})"
                  if unpaired_d is not None else "n/a")
    md.append(f"| {info['baseline_name']} | "
              f"{','.join(str(s) for s in info['common_seeds'])} | "
              f"{info['n_pairs']} | "
              f"{info['paired_delta_sdr_db']:+.2f} | {unpaired_s} | "
              f"{p_s} | {info['paired_wilcoxon_n_nonzero']} | {p_min_s} |")

md_path = OUT_DIR / "table1_5seed.md"
with open(md_path, "w") as f:
    f.write("\n".join(md))
print(f"Wrote {md_path}")

print()
print("Done.")