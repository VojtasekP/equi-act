import glob
import json
from collections import defaultdict
from pathlib import Path

import yaml

# Input sweep and output path
SWEEP_ID = "7xs0xcu8"
SWEEP_DIR = Path("wandb") / f"sweep-{SWEEP_ID}"
OUT_TEX = Path("tables/methods_specs.tex")


def tex_escape(value: str) -> str:
    """Escape LaTeX-sensitive characters."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    out = str(value)
    for src, tgt in replacements:
        out = out.replace(src, tgt)
    return out


def fmt_list(values) -> str:
    """Join list-like hyperparameters with forward slashes."""
    return "/".join(str(v) for v in values)


def fmt_float(value) -> str:
    """Format floats compactly for LaTeX."""
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    if value >= 0.01:
        return f"{value:.3g}"
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e")


def load_configs():
    if not SWEEP_DIR.exists():
        raise SystemExit(f"Sweep directory not found: {SWEEP_DIR}")

    grouped = defaultdict(list)
    for path_str in glob.glob(str(SWEEP_DIR / "config-*.yaml")):
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        flattened = {
            key: val["value"]
            for key, val in data.items()
            if isinstance(val, dict) and "value" in val
        }
        act = flattened.get("activation_type")
        bn = flattened.get("bn")
        if act is None or bn is None:
            raise SystemExit(f"Missing activation or bn in {path}")
        grouped[(act, bn)].append(flattened)

    # Collapse seeds: keep first config per (activation, bn), but check consistency.
    collapsed = {}
    for key, configs in grouped.items():
        base = configs[0]
        for other in configs[1:]:
            diff = {
                k
                for k in base
                if base.get(k) != other.get(k) and k not in {"seed", "project"}
            }
            if diff:
                raise SystemExit(
                    f"Inconsistent hyperparameters for {key}: {json.dumps(diff)}"
                )
        collapsed[key] = base
    return collapsed


def extract_shared(configs):
    """Return hyperparameters that are identical across all configs."""
    if not configs:
        raise SystemExit("No configurations found.")

    varying = {"activation_type", "bn", "seed", "project"}
    sample_config = next(iter(configs.values()))
    shared = {}
    for key, first_value in sample_config.items():
        if key in varying:
            continue
        if all(cfg.get(key) == first_value for cfg in configs.values()):
            shared[key] = first_value
    return shared


CATEGORY_MAP = {
    "dataset": ("Data", "Dataset"),
    "img_size": ("Data", "Image size"),
    "channels_per_block": ("Architecture", "Channels per block"),
    "kernels_per_block": ("Architecture", "Kernels per block"),
    "paddings_per_block": ("Architecture", "Paddings per block"),
    "invariant_channels": ("Architecture", "Invariant channels"),
    "max_rot_order": ("Architecture", "Max rotation order"),
    "conv_sigma": ("Architecture", "Convolution sigma"),
    "invar_type": ("Architecture", "Invariant type"),
    "pool_type": ("Pooling", "Pooling type"),
    "pool_size": ("Pooling", "Pool size"),
    "pool_sigma": ("Pooling", "Pool sigma"),
    "pool_after_every_n_blocks": ("Pooling", "Pool cadence"),
    "batch_size": ("Training", "Batch size"),
    "lr": ("Training", "Learning rate"),
    "weight_decay": ("Training", "Weight decay"),
    "epochs": ("Training", "Epochs"),
    "patience": ("Training", "Patience"),
    "burin_in_period": ("Training", "Burn-in period"),
    "exp_dump": ("Training", "Exp dump"),
}

CATEGORY_ORDER = ["Data", "Architecture", "Pooling", "Training", "Other"]


def format_value(value):
    if isinstance(value, list):
        return tex_escape(fmt_list(value))
    if isinstance(value, float):
        return tex_escape(fmt_float(value))
    return tex_escape(value)


def build_rows(configs):
    shared = extract_shared(configs)
    rows = []
    for key, value in shared.items():
        category, label = CATEGORY_MAP.get(key, ("Other", key))
        rows.append((category, label, format_value(value)))

    def sort_key(item):
        category, label, _ = item
        category_index = CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER)
        return (category_index, label)

    rows.sort(key=sort_key)
    return rows


def write_table(rows):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Category & Hyperparameter & Value \\",
        r"\midrule",
    ]
    last_category = None
    for category, label, value in rows:
        if last_category and category != last_category:
            lines.append(r"\addlinespace")
        lines.append(f"{category} & {tex_escape(label)} & {value} \\\\")
        last_category = category
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Hyperparameters shared across all activation and batch-norm configurations.}",
            r"\label{tab:methods-specs}",
            r"\end{table}",
        ]
    )

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_TEX}")


def main():
    configs = load_configs()
    rows = build_rows(configs)
    write_table(rows)


if __name__ == "__main__":
    main()
