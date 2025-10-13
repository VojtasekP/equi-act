# pip install wandb pandas numpy
from collections import defaultdict
import os
import math

import wandb
import pandas as pd
import numpy as np

# ========= USER INPUTS =========
ENTITY     = "vojtasek-petr"      # e.g. "vojtasek-petr"
PROJECT    = "mnist_rot_first_try"     # e.g. "equi-act"
SWEEP_ID   = "547ku2ui"        # the short id in the sweep URL
SEED_PARAM = "seed"             # your seed key in config
# Metrics you want to aggregate (must exist in run.summary)
METRICS    = ["test_acc", "test_loss", "test_invar_error"]  # edit to your names
OUT_CSV    = "wandb_agg_by_config.csv"
# =================================

api = wandb.Api(timeout=60)

# Grab sweep runs
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
runs = sweep.runs

rows = []
for r in runs:
    # Keep only finished runs with a summary dict
    if r.state != "finished":
        continue
    summ = dict(r.summary or {})

    # Skip runs where none of the requested metrics exist
    if not any(m in summ and pd.notna(summ[m]) for m in METRICS):
        continue

    cfg = dict(r.config or {})
    # Flatten lists to tuples for hashing/comparison
    norm_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, list):
            norm_cfg[k] = tuple(v)
        else:
            norm_cfg[k] = v

    row = {
        "_run_id": r.id,
        "_name": r.name,
        "_state": r.state,
    }
    # Attach metrics
    for m in METRICS:
        row[m] = summ.get(m, np.nan)

    # Attach config
    row.update({f"cfg.{k}": v for k, v in norm_cfg.items()})
    rows.append(row)

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No finished runs with requested metrics. Fix your inputs.")

# Identify config columns and the seed column
cfg_cols = [c for c in df.columns if c.startswith("cfg.")]
seed_col = f"cfg.{SEED_PARAM}"

# Drop truly useless config junk if W&B injected anything odd
drop_like = ["_wandb", "job_type", "program", "save_code", "notes"]
cfg_cols = [c for c in cfg_cols if not any(x in c for x in drop_like)]

# Some people log seed as int, some as str. Normalize.
if seed_col in df.columns:
    df[seed_col] = pd.to_numeric(df[seed_col], errors="ignore")
else:
    # No explicit seed key found, fine, we’ll aggregate as-is
    seed_col = None

# Decide group keys = all config except the seed
group_keys = [c for c in cfg_cols if c != seed_col]

# Helper for CI with t-critical if SciPy available; otherwise normal approx
try:
    from scipy.stats import t as student_t
    def ci_halfwidth(mean, std, n, alpha=0.05):
        if n <= 1 or not np.isfinite(std):
            return np.nan
        tcrit = student_t.ppf(1 - alpha/2, df=n - 1)
        return tcrit * std / math.sqrt(n)
except Exception:
    def ci_halfwidth(mean, std, n, alpha=0.05):
        if n <= 1 or not np.isfinite(std):
            return np.nan
        # z=1.96 fallback; with small n this underestimates slightly. Your problem, you skipped SciPy.
        return 1.96 * std / math.sqrt(n)

# Aggregate per metric
agg_frames = []
for m in METRICS:
    if m not in df.columns:
        continue
    sub = df[group_keys + ([seed_col] if seed_col else []) + ["_run_id", m]].copy()
    # Drop rows where metric is NaN
    sub = sub[pd.notna(sub[m])]
    if sub.empty:
        continue

    grp = sub.groupby(group_keys, dropna=False)

    stats = grp[m].agg(["count", "mean", "std", "min", "max"]).reset_index()
    stats.rename(columns={
        "count": f"{m}__n",
        "mean":  f"{m}__mean",
        "std":   f"{m}__std",
        "min":   f"{m}__min",
        "max":   f"{m}__max",
    }, inplace=True)

    # CI halfwidth
    n = stats[f"{m}__n"].to_numpy()
    s = stats[f"{m}__std"].to_numpy()
    mu = stats[f"{m}__mean"].to_numpy()
    ci = [ci_halfwidth(mu[i], s[i], int(n[i])) for i in range(len(stats))]
    stats[f"{m}__ci95"] = ci

    # Best seed/run for that metric per group
    idx = grp[m].idxmin() if "loss" in m.lower() else grp[m].idxmax()
    best_rows = sub.loc[idx].reset_index(drop=True)
    best_cols = group_keys + [m, "_run_id"] + ([seed_col] if seed_col else [])
    best_rows = best_rows[best_cols]
    best_rows.rename(columns={
        m: f"{m}__best",
        "_run_id": f"{m}__best_run",
        (seed_col or ""): f"{m}__best_seed"
    }, inplace=True, errors="ignore")

    # Merge best back
    stats = stats.merge(best_rows, on=group_keys, how="left")
    agg_frames.append(stats)

# Merge all metric aggregates side-by-side
from functools import reduce
agg = reduce(lambda left, right: pd.merge(left, right, on=group_keys, how="outer"), agg_frames)

# Nice ordering: configs first, then metric blocks
metric_col_order = []
for m in METRICS:
    base = [f"{m}__n", f"{m}__mean", f"{m}__std", f"{m}__ci95", f"{m}__min", f"{m}__max", f"{m}__best", f"{m}__best_run"]
    if seed_col:
        base.append(f"{m}__best_seed")
    metric_col_order.extend([c for c in base if c in agg.columns])

agg = agg[group_keys + metric_col_order]

# Sort sensibly: by dataset if present, then by metric descending/ascending
sort_cols = [c for c in group_keys if c.endswith("dataset")] + [c for c in group_keys if c.endswith("activation_type")]
primary = METRICS[0]
if f"{primary}__mean" in agg.columns:
    asc = True if "loss" in primary.lower() else False
    agg = agg.sort_values(sort_cols + [f"{primary}__mean"], ascending=([True]*len(sort_cols)) + [asc])

# Save
agg.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print(agg.head(20).to_string(index=False))
