# pip install pandas numpy
import pandas as pd
from pathlib import Path
from typing import Optional

CSV_PATH = "Results.csv"   # from your previous export
OUT_TEX  = Path("tables/methods_simple.tex")

# Formatting
ACC_DEC        = 2    # decimals for percentage
TIME_DEC       = 2    # decimals for minutes
AUG_COL        = "aug"
DATASET_ORDER  = ["mnist_rot", "eurosat", "colorectal_hist"]
BOLD_LOWEST    = True  # bold the lowest error per dataset within each augmentation slice
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

# Pretty-print activation names for LaTeX: turn underscores/backslashes into spaces,
# then escape remaining special characters.
def pretty_activation(s):
    if pd.isna(s):
        return ""
    s = str(s)
    if s.lower() in {"resnet", "resnet18"}:
        return "ResNet18"
    # Normalize any LaTeX-escaped underscore to a plain underscore first
    s = s.replace(r"\_", "_")
    # Replace underscores with spaces
    s = s.replace("_", " ")
    # Drop any remaining backslashes (e.g., from pre-escaped inputs)
    s = s.replace("\\", "")
    # Collapse repeated whitespace
    s = " ".join(s.split())
    return tex_escape(s)

df = pd.read_csv(CSV_PATH)

# Identify dataset column pairs dynamically: <dataset>_mean / <dataset>_std
def _dataset_key(col: str) -> Optional[str]:
    if col.endswith("_mean"):
        return col[: -len("_mean")]
    if col.endswith("_std"):
        return col[: -len("_std")]
    return None

def _time_key(col: str) -> Optional[str]:
    for suffix in ("_train_time_mean_min", "_train_time_std_min", "_train_time_mean_h", "_train_time_std_h"):
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return None

def _time_in_minutes(row, name: str):
    """Return (mean, std) time in minutes for a dataset name from a row, handling hour/minute columns."""
    mean_col_min = f"{name}_train_time_mean_min"
    std_col_min = f"{name}_train_time_std_min"
    mean_col_h = f"{name}_train_time_mean_h"
    std_col_h = f"{name}_train_time_std_h"

    mean_val = pd.NA
    std_val = pd.NA

    if mean_col_min in row.index:
        mean_val = row[mean_col_min]
    elif mean_col_h in row.index:
        mean_val = row[mean_col_h] * 60 if pd.notna(row[mean_col_h]) else pd.NA

    if std_col_min in row.index:
        std_val = row[std_col_min]
    elif std_col_h in row.index:
        std_val = row[std_col_h] * 60 if pd.notna(row[std_col_h]) else pd.NA

    return mean_val, std_val

dataset_names = []
seen = set()
for col in df.columns:
    if col in {"activation", "bn", AUG_COL, "model_type"}:
        continue
    key = _dataset_key(col)
    if key and key not in seen:
        mean_col = f"{key}_mean"
        std_col = f"{key}_std"
        if mean_col in df.columns and std_col in df.columns:
            dataset_names.append(key)
            seen.add(key)

time_names = []
seen_time = set()
for col in df.columns:
    key = _time_key(col)
    if key and key not in seen_time:
        mean_col_min = f"{key}_train_time_mean_min"
        std_col_min = f"{key}_train_time_std_min"
        mean_col_h = f"{key}_train_time_mean_h"
        std_col_h = f"{key}_train_time_std_h"
        has_min_pair = mean_col_min in df.columns and std_col_min in df.columns
        has_h_pair = mean_col_h in df.columns and std_col_h in df.columns
        if has_min_pair or has_h_pair:
            time_names.append(key)
            seen_time.add(key)

# Respect preferred ordering for time columns, but only keep those that exist
ordered_time_names = [n for n in DATASET_ORDER if n in time_names]
ordered_time_names += [n for n in time_names if n not in ordered_time_names]
time_names = ordered_time_names

# Precompute time aggregates per activation across augmentations (average over aug/no-aug)

if not dataset_names:
    raise SystemExit("No dataset columns found (need *_mean and *_std pairs).")

# Respect a preferred ordering and create placeholders for missing datasets
ordered_names = []
for name in DATASET_ORDER:
    if name not in df.columns and f"{name}_mean" not in df.columns:
        df[f"{name}_mean"] = pd.NA
        df[f"{name}_std"] = pd.NA
    ordered_names.append(name)
# Append any extras not in the preferred order
ordered_names += [n for n in dataset_names if n not in DATASET_ORDER]
dataset_names = ordered_names

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

# Convert dataset accuracies into percentage test errors (mean ± std)
dataset_errors = {}
for name in dataset_names:
    mean_col = f"{name}_mean"
    std_col = f"{name}_std"

    acc_max = df[mean_col].max()
    if acc_max <= 1.5:
        err_mean_pct = (1.0 - df[mean_col]) * 100.0
        err_std_pct = df[std_col] * 100.0
    else:
        err_mean_pct = 100.0 - df[mean_col]
        err_std_pct = df[std_col]

    dataset_errors[name] = (err_mean_pct, err_std_pct)

# Format cells
def fmt_pct(mean, std, dec=2):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"

def _is_resnet(row):
    return str(row.get("model_type", "")).strip().lower() == "resnet18"

def _resolve_activation(row):
    """Force resnet18 activation for resnet model_type; otherwise use the provided activation."""
    if _is_resnet(row):
        return "resnet18"
    return row.get("activation", "")

def build_table(df_slice, aug_label: Optional[str]):
    pass  # unused in unified-table version

# Unified table with Aug column
bold_masks = {}
if BOLD_LOWEST:
    for name in dataset_names:
        err_mean_pct, _ = dataset_errors[name]
        errs = err_mean_pct
        if errs.notna().any():
            min_val = errs.min()
            bold_masks[name] = errs == min_val

lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
col_spec = "ll" + "r" * len(dataset_names)
lines.append(r"\begin{tabular}"+f"{{{col_spec}}}")
lines.append(r"\toprule")
header_cells = ["Activation", "Aug"] + [f"{pretty_dataset(name)} [\\%]" for name in dataset_names]
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
        err_mean_pct, err_std_pct = dataset_errors[name]
        cell = fmt_pct(err_mean_pct.loc[idx], err_std_pct.loc[idx], ACC_DEC)
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
lines.append(r"\caption{Test error (mean $\pm$ std, in percent) per dataset, activation, and augmentation.}")
lines.append(r"\label{tab:methods-simple}")
lines.append(r"\end{table}")

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_TEX}")

# Optional: training time table (minutes; one value shown on no-aug row per activation)
if time_names:
    lines_time = []
    lines_time.append(r"\begin{table}[ht]")
    lines_time.append(r"\centering")
    lines_time.append(r"\small")
    col_spec_time = "l" + "r" * len(time_names)
    lines_time.append(r"\begin{tabular}"+f"{{{col_spec_time}}}")
    lines_time.append(r"\toprule")
    header_time = ["Activation"] + [f"{pretty_dataset(name)} train [min]" for name in time_names]
    lines_time.append(" & ".join(header_time) + r" \\")
    lines_time.append(r"\midrule")

    resnet_rows_time = list(df[resnet_mask].index)
    other_sorted_time = df[~resnet_mask].sort_values(["activation", AUG_COL] if AUG_COL in df.columns else ["activation"])
    row_order_time = resnet_rows_time + list(other_sorted_time.index)

    # Unique activation order preserving the existing sort
    act_order = []
    seen_act = set()
    for idx in row_order_time:
        act_raw = _resolve_activation(df.loc[idx])
        if act_raw not in seen_act:
            act_order.append(act_raw)
            seen_act.add(act_raw)

    def _row_for_activation(act_raw):
        """Pick a representative row for an activation (prefer one with time values)."""
        candidate_idx = None
        for idx in row_order_time:
            if _resolve_activation(df.loc[idx]) != act_raw:
                continue
            mean_vals = [_time_in_minutes(df.loc[idx], name)[0] for name in time_names]
            if any(pd.notna(v) for v in mean_vals):
                return idx
            if candidate_idx is None:
                candidate_idx = idx
        return candidate_idx

    for pos, act_raw in enumerate(act_order):
        idx = _row_for_activation(act_raw)
        row = df.loc[idx]
        act_cell = pretty_activation(act_raw)

        time_cells = []
        for name in time_names:
            mean_val, std_val = _time_in_minutes(row, name)
            time_cells.append(fmt_pct(mean_val, std_val, TIME_DEC))

        lines_time.append(" & ".join([act_cell] + time_cells) + r" \\")

        if pos + 1 < len(act_order):
            lines_time.append(r"\midrule")

    lines_time.append(r"\bottomrule")
    lines_time.append(r"\end{tabular}")
    lines_time.append(r"\caption{Training time (minutes, mean $\pm$ std) per dataset, model, activation, and augmentation.}")
    lines_time.append(r"\label{tab:train-time}")
    lines_time.append(r"\end{table}")

    time_tex = OUT_TEX.with_name(OUT_TEX.stem + "_time.tex")
    time_tex.write_text("\n".join(lines_time), encoding="utf-8")
    print(f"Wrote {time_tex}")
