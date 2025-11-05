# pip install pandas numpy
import pandas as pd
from pathlib import Path

CSV_PATH = "el.csv"   # from your previous export
OUT_TEX  = Path("tables/methods_simple.tex")

# Formatting
ACC_DEC  = 2    # decimals for percentage
ERR_DEC  = 3    # decimals for invariant error

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

# Expected columns from your previous step:
# activation, bn, test_acc_mean, test_acc_std, test_invar_error_mean, test_invar_error_std
need = [
    "activation", "bn",
    "test_acc_mean", "test_acc_std",
    "test_invar_error_mean", "test_invar_error_std",
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in CSV: {missing}")

# Detect whether accuracy is 0-1 or 0-100
acc_max = df["test_acc_mean"].max()
if acc_max <= 1.5:
    # stored as fraction; convert to percentage
    err_mean_pct = (1.0 - df["test_acc_mean"]) * 100.0
    err_std_pct  = df["test_acc_std"] * 100.0
else:
    # stored as percent already
    err_mean_pct = 100.0 - df["test_acc_mean"]
    err_std_pct  = df["test_acc_std"]

# Format cells
def fmt_pct(mean, std, dec=2):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"

def fmt_err(mean, std, dec=3):
    if pd.isna(mean):
        return "-"
    if pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"

table_rows = []
for _, r in df.iterrows():
    act = pretty_activation(r["activation"])  # nicer label without backslashes/underscores
    bn  = tex_escape(r["bn"])
    test_err_cell = fmt_pct(err_mean_pct.loc[_], err_std_pct.loc[_], ACC_DEC)
    invar_cell    = fmt_err(r["test_invar_error_mean"], r["test_invar_error_std"], ERR_DEC)
    table_rows.append((act, bn, test_err_cell, invar_cell))

# Build LaTeX
lines = []
lines.append(r"\begin{table}[ht]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llrr}")
lines.append(r"\toprule")
lines.append(r"Activation & BN & Test error [\%] & Invar error \\")
lines.append(r"\midrule")
for act, bn, terr, ierr in table_rows:
    lines.append(f"{act} & {bn} & {terr} & {ierr} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Method summary: test error in percent (mean ± std) and invariant error (mean ± std).}")
lines.append(r"\label{tab:methods-simple}")
lines.append(r"\end{table}")

OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT_TEX}")
