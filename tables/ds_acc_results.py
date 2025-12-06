import wandb
import pandas as pd
import numpy as np
from pathlib import Path
from functools import reduce

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
GROUP_COLS   = [MODEL_KEY, "activation", BN_KEY, AUG_KEY]
TRAIN_TIME_KEYS = ["train_time", "_runtime"]  # Try both explicit metric and wandb runtime (seconds)
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

def fetch_runs():
    """Pull runs from the unified project once to avoid repeated API calls."""
    filters = {"state": "finished"}
    print(f"Fetching runs from: {PROJECT}...")
    project_runs = list(api.runs(path=f"{ENTITY}/{PROJECT}", filters=filters))
    print(f"  found {len(project_runs)} runs")
    return project_runs

def get_dataset_stats(dataset_name, runs):
    """Filters runs for a dataset and returns grouped stats."""
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
            MODEL_KEY:    model_type,
            "activation": activation,
            "bn":         bn,
            "aug":        aug,
            # Temporary columns for aggregation, renamed later
            "test_acc_raw": summ.get("test_acc", np.nan),
            "train_time_minutes": _extract_train_time_minutes(summ),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"Warning: No runs found for dataset '{dataset_name}'")
        return None

    def _is_no_aug(val):
        if pd.isna(val):
            return True
        if isinstance(val, bool):
            return not val
        v = str(val).strip().lower()
        return v in {"0", "false", "f", "no", "n", "no aug", ""}

    # Accuracy grouped by augmentation
    grp_acc = df.groupby(GROUP_COLS, dropna=False)
    stats_acc = grp_acc.agg(
        **{
            f"{dataset_name}_mean": ("test_acc_raw", "mean"),
            f"{dataset_name}_std": ("test_acc_raw", "std"),
        }
    ).reset_index()

    # Time aggregated across aug/no-aug (one value per activation/bn/model)
    grp_time = df.groupby([MODEL_KEY, "activation", BN_KEY], dropna=False)
    stats_time = grp_time.agg(
        **{
            f"{dataset_name}_train_time_mean_min": ("train_time_minutes", "mean"),
            f"{dataset_name}_train_time_std_min": ("train_time_minutes", "std"),
        }
    ).reset_index()

    # Merge time aggregates onto accuracy stats and keep time only on no-aug rows
    stats = stats_acc.merge(stats_time, on=[MODEL_KEY, "activation", BN_KEY], how="left")
    time_mean_col = f"{dataset_name}_train_time_mean_min"
    time_std_col = f"{dataset_name}_train_time_std_min"
    mask_no_aug = stats[AUG_KEY].apply(_is_no_aug)
    stats.loc[~mask_no_aug, [time_mean_col, time_std_col]] = np.nan
    return stats

# 1. Fetch runs once
runs = fetch_runs()

# 2. Collect DataFrames for all datasets
dfs = []
for dataset in DATASETS:
    df_stats = get_dataset_stats(dataset, runs)
    if df_stats is not None:
        dfs.append(df_stats)

if not dfs:
    raise SystemExit("No data found for any dataset.")

# 3. Merge all DataFrames on the common hyperparams
final_df = reduce(lambda left, right: pd.merge(left, right, on=GROUP_COLS, how="outer"), dfs)

# 4. Clean up formatting (Sort and Round)
final_df = final_df.sort_values(GROUP_COLS)
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
