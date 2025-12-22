import pandas as pd
from pathlib import Path
from typing import Optional
import glob

CSV_DIR = Path("tables/csv_outputs/invariance")
OUT_TEX = Path("tables/invariance.tex")

# Formatting
DECIMALS       = 2    # decimals for invariance value
AUG_COL        = "aug"
DATASET_ORDER  = ["mnist_rot", "eurosat", "colorectal_hist"]
BOLD_LOWEST    = True  # bold the lowest value per dataset within each augmentation slice
DATASET_LABELS = {
    "eurosat": "Eurosat",
    "mnist_rot": "Mnistrot",
    "colorectal_hist": "Colorectal Hist",
}

def tex_escape(s):
    if pd.isna(s):
        return ""
    s = str(s)
    repl = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': '_',
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def pretty_activation(s):
    if pd.isna(s):
        return ""
    s = str(s)
    if s.lower() in {"resnet", "resnet18"}:
        return "ResNet18"
    s = s.replace(r"\_", "_")
    s = s.replace("_", " ")
    s = s.replace("\\", "")
    s = " ".join(s.split())
    return tex_escape(s)

def pretty_dataset(name: str) -> str:
    label = DATASET_LABELS.get(name, name)
    return tex_escape(label)

def pretty_aug(val) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, bool):
        return "aug" if val else "no aug"
    v = str(val).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "aug"}:
        return "aug"
    if v in {"0", "false", "f", "no", "n", "no aug"}:
        return "no aug"
    return tex_escape(str(val))

def fmt_val(mean, std, dec=DECIMALS):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"

def _is_resnet(row):
    return str(row.get("model_type", "")).strip().lower() == "resnet18"

def _resolve_activation(row):
    if _is_resnet(row):
        return "resnet18"
    return row.get("activation", "")

# 1. Load and aggregate data
all_files = list(CSV_DIR.glob("*.csv"))
if not all_files:
    raise SystemExit(f"No CSV files found in {CSV_DIR}")

df_list = []
for f in all_files:
    df_list.append(pd.read_csv(f))

raw_df = pd.concat(df_list, ignore_index=True)

# Group by keys and calculate mean/std
group_cols = ["dataset", "model_type", "activation", "bn", "aug"]
agg_df = raw_df.groupby(group_cols)["invariance"].agg(["mean", "std"]).reset_index()

# Pivot to get dataset columns
# We want columns like: dataset1_mean, dataset1_std, dataset2_mean, ...
# The index should be: model_type, activation, bn, aug

pivot_df = agg_df.pivot(
    index=["model_type", "activation", "bn", "aug"],
    columns="dataset",
    values=["mean", "std"]
)

# Flatten columns
pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
pivot_df = pivot_df.reset_index()

df = pivot_df

# Identify dataset names present in the dataframe
dataset_names = []
seen = set()
for col in df.columns:
    if col.endswith("_mean"):
        key = col[:-5]
        if key not in seen:
            dataset_names.append(key)
            seen.add(key)

# Order datasets
ordered_names = []
for name in DATASET_ORDER:
    if name in dataset_names:
        ordered_names.append(name)
    elif f"{name}_mean" not in df.columns:
        # Add missing columns if we want to show them as empty
        df[f"{name}_mean"] = pd.NA
        df[f"{name}_std"] = pd.NA
        ordered_names.append(name)

# Append any extras
ordered_names += [n for n in dataset_names if n not in ordered_names]
dataset_names = ordered_names

# Prepare bold masks (lowest value is best)
bold_masks = {}
if BOLD_LOWEST:
    for name in dataset_names:
        mean_col = f"{name}_mean"
        if mean_col in df.columns:
            vals = df[mean_col]
            if vals.notna().any():
                min_val = vals.min()
                bold_masks[name] = vals == min_val

# Build LaTeX table
lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
col_spec = "ll" + "r" * len(dataset_names)
lines.append(r"\begin{tabular}"+f"{{{col_spec}}}")
lines.append(r"\toprule")
header_cells = ["Activation", "Aug"] + [f"{pretty_dataset(name)}" for name in dataset_names]
lines.append(" & ".join(header_cells) + r" \\")
lines.append(r"\midrule")

resnet_mask = df["model_type"].fillna("").str.lower() == "resnet18"
resnet_rows = list(df[resnet_mask].index)
other_sorted = df[~resnet_mask].sort_values(["activation", AUG_COL] if AUG_COL in df.columns else ["activation"])
row_order = resnet_rows + list(other_sorted.index)

current_act = None
for pos, idx in enumerate(row_order):
    r = df.loc[idx]
    act_raw = _resolve_activation(r)
    act_cell = pretty_activation(act_raw) if act_raw != current_act else ""
    current_act = act_raw
    aug = pretty_aug(r[AUG_COL]) if AUG_COL in df.columns else ""

    dataset_cells = []
    for name in dataset_names:
        mean_col = f"{name}_mean"
        std_col = f"{name}_std"
        
        mean_val = r[mean_col] * 100
        std_val = r[std_col] * 100
        
        cell = fmt_val(mean_val, std_val, DECIMALS)
        if BOLD_LOWEST and name in bold_masks and bold_masks[name].get(idx, False):
            cell = r"\textbf{" + cell + "}"
        dataset_cells.append(cell)

    lines.append(" & ".join([act_cell, aug] + dataset_cells) + r" \\")

    if pos + 1 < len(row_order):
        next_act = _resolve_activation(df.loc[row_order[pos + 1]])
        if next_act != current_act:
            lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Invariance (mean $\pm$ std, $\times 10^{-2}$) per dataset, activation, and augmentation.}")
lines.append(r"\label{tab:invariance}")
lines.append(r"\end{table}")

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_TEX}")
