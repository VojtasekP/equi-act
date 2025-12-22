import wandb
import pandas as pd
import numpy as np
from pathlib import Path

# ========= USER INPUTS =========
ENTITY       = "vojtasek-petr"
PROJECT      = "Results"  # Unified equivariant + baseline runs
DATASETS     = ["mnist_rot", "eurosat", "colorectal_hist"]

DATASET_KEY  = "dataset"          # Config key that identifies the dataset
ACT_KEY      = "activation_type"  # Common config key
BN_KEY       = "bn"               # Common config key
AUG_KEY      = "aug"              # Indicates whether augmentation was used
MODEL_KEY    = "model_type"
only_noflip  = True
OUT_CSV      = "Results.csv"
TRAIN_TIME_KEYS = ["train_time", "_runtime"]  # Try both explicit metric and wandb runtime (seconds)
SEED_KEYS    = ["seed"]
# =================================

api = wandb.Api(timeout=60)

def _extract_train_time_minutes(summary_dict):
    """Pick the first available training time key and convert seconds -> minutes."""
    for key in TRAIN_TIME_KEYS:
        if key in summary_dict and summary_dict.get(key) is not None:
            try:
                return float(summary_dict.get(key)) / 60.0
            except Exception:
                return np.nan
    return np.nan

def _extract_seed(cfg):
    """Return the first available seed value from the config."""
    for key in SEED_KEYS:
        if key in cfg:
            return cfg.get(key)
    return None

def fetch_runs():
    """Pull runs from the unified project once to avoid repeated API calls."""
    filters = {"state": "finished"}
    print(f"Fetching runs from: {PROJECT}...")
    project_runs = list(api.runs(path=f"{ENTITY}/{PROJECT}", filters=filters))
    print(f"  found {len(project_runs)} runs")
    return project_runs

def get_dataset_rows(dataset_name, runs):
    """Filters runs for a dataset and returns per-run rows (no aggregation)."""
    rows = []
    for r in runs:
        cfg = r.config
        if cfg.get(DATASET_KEY) != dataset_name:
            continue
        if cfg.get(ACT_KEY) == "fourier_elu_32":
            continue
        if only_noflip and cfg.get("flip"):
            continue

        summ = r.summary._json_dict
        model_type = cfg.get(MODEL_KEY, "equivariant")
        activation = cfg.get(ACT_KEY) or (model_type if model_type != "equivariant" else "none")
        bn = cfg.get(BN_KEY) or "none"
        aug = cfg.get(AUG_KEY)

        rows.append({
            DATASET_KEY: dataset_name,
            MODEL_KEY:    model_type,
            "activation": activation,
            "bn":         bn,
            "aug":        aug,
            "seed":       _extract_seed(cfg),
            "test_acc":   summ.get("test_acc", np.nan),
            "train_time_minutes": _extract_train_time_minutes(summ),  # already in minutes
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"Warning: No runs found for dataset '{dataset_name}'")
        return None

    return df

# 1. Fetch runs once
runs = fetch_runs()

# 2. Collect DataFrames for all datasets
dfs = []
for dataset in DATASETS:
    df_dataset = get_dataset_rows(dataset, runs)
    if df_dataset is not None:
        dfs.append(df_dataset)

if not dfs:
    raise SystemExit("No data found for any dataset.")

# 3. Combine all per-dataset DataFrames
final_df = pd.concat(dfs, ignore_index=True)

# 4. Clean up formatting (Sort and Round)
sort_cols = [DATASET_KEY, MODEL_KEY, "activation", BN_KEY, AUG_KEY, "seed"]
final_df = final_df.sort_values(sort_cols, na_position="last")
numeric_cols = list(final_df.select_dtypes(include=[np.number]).columns)
time_cols = [c for c in numeric_cols if "train_time" in c]
acc_cols = [c for c in numeric_cols if c not in time_cols]
if acc_cols:
    final_df[acc_cols] = final_df[acc_cols].round(4)
if time_cols:
    final_df[time_cols] = final_df[time_cols].round(2)  # minutes rounded to 2 decimals

# 5. Save
Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved combined results to {OUT_CSV}")
print("-" * 60)
print(final_df.head(20).to_string(index=False))
