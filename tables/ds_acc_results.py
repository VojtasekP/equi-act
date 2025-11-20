import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

# ========= USER INPUTS =========
ENTITY      = "vojtasek-petr"
# List of projects you want to compare
PROJECTS    = ["Eurosat_results", "Mnistrot_results", "Colorectal_hist_results"] 

ACT_KEY     = "activation_type"   # Common config key
BN_KEY      = "bn"                # Common config key
only_noflip = True
OUT_CSV     = "Results.csv"
# =================================

api = wandb.Api(timeout=60)

def get_project_stats(project_name):
    """Fetches runs, filters them, and returns grouped stats."""
    print(f"Fetching runs from: {project_name}...")
    
    # Using filters is FASTER than iterating and checking 'if' in Python
    filters = {"state": "finished"}
    runs = api.runs(path=f"{ENTITY}/{project_name}", filters=filters)
    
    rows = []
    for r in runs:
        summ = r.summary._json_dict
        cfg  = r.config
        if cfg.get(ACT_KEY) == "fourier_elu_32":
            continue
        if cfg.get("flip"):
            continue
        rows.append({
            "activation": cfg.get(ACT_KEY),
            "bn":         cfg.get(BN_KEY),
            # Rename metric to include project name immediately to avoid collision later
            f"{project_name}_mean": summ.get("test_acc", np.nan),
            # Temporary column for aggregation, renamed later
            "test_acc_raw": summ.get("test_acc", np.nan) 
        })

    df = pd.DataFrame(rows)
    
    if df.empty:
        print(f"Warning: No runs found for {project_name}")
        return None

    # Group strictly by hyperparameters
    grp = df.groupby(["activation", "bn"], dropna=False)

    # Calculate mean and std
    stats = grp.agg({
        "test_acc_raw": ["mean", "std"]
    }).reset_index()

    # Flatten MultiIndex columns
    stats.columns = ["activation", "bn", f"{project_name}_mean", f"{project_name}_std"]
    
    return stats

# 1. Collect DataFrames for all projects
dfs = []
for proj in PROJECTS:
    df_stats = get_project_stats(proj)
    if df_stats is not None:
        dfs.append(df_stats)

if not dfs:
    raise SystemExit("No data found in any project.")

# 2. Merge all DataFrames on the common hyperparams
# This ensures that rows align correctly even if one project is missing a specific combo
final_df = reduce(lambda left, right: pd.merge(left, right, on=["activation", "bn"], how="outer"), dfs)

# 3. Clean up formatting (Sort and Round)
final_df = final_df.sort_values(["activation", "bn"])
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
final_df[numeric_cols] = final_df[numeric_cols].round(4)

# 4. Save
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved combined results to {OUT_CSV}")
print("-" * 60)
print(final_df.head(20).to_string(index=False))