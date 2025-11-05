# pip install wandb pandas numpy
import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# ========= USER INPUTS =========
ENTITY      = "vojtasek-petr"
PROJECT     = "mnist_rot"
SWEEP_ID    = "7xs0xcu8"
ACT_KEY     = "activation_type"   # config key name
BN_KEY      = "bn"                # config key name
OUT_CSV     = "else2.csv"
# =================================

api = wandb.Api(timeout=60)
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
runs = sweep.runs

rows = []
for r in runs:
    if r.state != "finished":
        continue
    summ = dict(r.summary or {})
    cfg  = dict(r.config or {})

    # pull only what we need; missing values become None
    rows.append({
        "activation": cfg.get(ACT_KEY),
        "bn":         cfg.get(BN_KEY),
        "test_acc":   summ.get("test_acc", np.nan),
        "test_invar_error": summ.get("test_invar_error", np.nan),
    })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No finished runs found with the requested fields.")

# Group STRICTLY by activation and bn. This collapses seeds and any other varying junk.
grp = df.groupby(["activation", "bn"], dropna=False)

stats = grp.agg({
    "test_acc": ["mean", "std"],
    "test_invar_error": ["mean", "std"],
}).reset_index()

# flatten columns
stats.columns = [
    "activation", "bn",
    "test_acc_mean", "test_error_std",
    "test_invar_error_mean", "test_invar_error_std",
]

# optional: sort for readability
stats = stats.sort_values(["activation", "bn"], kind="mergesort")

Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
stats.to_csv(OUT_CSV, index=False)

print(f"Saved {OUT_CSV}")
print(stats.head(20).to_string(index=False))
