# pip install pandas numpy
import pandas as pd
from pathlib import Path

CSV_PATH = "Results.csv"   # from your previous export
OUT_TEX  = Path("tables/methods_simple.tex")

# Formatting
ACC_DEC  = 2    # decimals for percentage

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
def _dataset_key(col: str) -> str | None:
    if col.endswith("_mean"):
        return col[: -len("_mean")]
    if col.endswith("_std"):
        return col[: -len("_std")]
    return None

dataset_names = []
seen = set()
for col in df.columns:
    if col in {"activation", "bn"}:
        continue
    key = _dataset_key(col)
    if key and key not in seen:
        mean_col = f"{key}_mean"
        std_col = f"{key}_std"
        if mean_col in df.columns and std_col in df.columns:
            dataset_names.append(key)
            seen.add(key)

if not dataset_names:
    raise SystemExit("No dataset columns found (need *_mean and *_std pairs).")

def pretty_dataset(name: str) -> str:
    label = name.replace("_results", "")
    label = label.replace("_", " ")
    label = label.strip()
    if not label:
        label = name
    return tex_escape(label.title())

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

table_rows = []
for idx, r in df.iterrows():
    act = pretty_activation(r["activation"])
    bn  = tex_escape(r["bn"])
    dataset_cells = []
    for name in dataset_names:
        err_mean_pct, err_std_pct = dataset_errors[name]
        dataset_cells.append(fmt_pct(err_mean_pct.loc[idx], err_std_pct.loc[idx], ACC_DEC))
    table_rows.append((act, bn, dataset_cells))

# Build LaTeX
lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
col_spec = "ll" + "r" * len(dataset_names)
lines.append(r"\begin{tabular}"+f"{{{col_spec}}}")
lines.append(r"\toprule")
header_cells = ["Activation", "BN"] + [f"{pretty_dataset(name)} [\\%]" for name in dataset_names]
lines.append(" & ".join(header_cells) + r" \\")
lines.append(r"\midrule")
for act, bn, dataset_cells in table_rows:
    row_cells = [act, bn] + dataset_cells
    lines.append(" & ".join(row_cells) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Test error (mean $\pm$ std, in percent) per dataset, activation, and normalization.}")
lines.append(r"\label{tab:methods-simple}")
lines.append(r"\end{table}")

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_TEX}")
