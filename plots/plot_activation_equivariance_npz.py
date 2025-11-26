"""
Plot activation-level equivariance curves saved by `activation_equivariance_check.py`.

Each `.npz` file contains:
    - theta: sampled angles (radians)
    - mean: mean equivariance error per angle
    - std: optional stddev (may be empty)
    - metadata: dict with dataset, activation, normalization, flip, position
The script groups files by (dataset, activation, normalization, flip) and plots one curve
per activation position (first/middle/last) with optional ±1 std shading.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot activation equivariance curves from .npz files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="plots/activation_equivariance_data",
        help="Directory containing .npz files produced by activation_equivariance_check.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/activation_equivariance_plots",
        help="Directory where plots will be saved.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional filter: only plot these datasets.",
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="*",
        default=None,
        help="Optional filter: only plot these activation names.",
    )
    parser.add_argument(
        "--normalizations",
        type=str,
        nargs="*",
        default=None,
        help="Optional filter: only plot these normalization types.",
    )
    parser.add_argument(
        "--include-flip",
        action="store_true",
        help="When set, include flip=True groups; otherwise both flip/no-flip are plotted if present.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional fixed upper bound for y-axis (lower bound is 0). If omitted, scales per plot.",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict]:
    data = np.load(path, allow_pickle=True)
    theta = data["theta"]
    mean = data["mean"]
    std = data["std"]
    if std.size == 0:
        std = None
    metadata = {}
    if "metadata" in data:
        raw = data["metadata"]
        if isinstance(raw, np.ndarray):
            metadata = raw.item()
        elif isinstance(raw, dict):
            metadata = raw
    return theta, mean, std, metadata


def _should_keep(meta: Dict, args: argparse.Namespace) -> bool:
    if args.datasets and meta.get("dataset") not in args.datasets:
        return False
    if args.activations and meta.get("activation") not in args.activations:
        return False
    if args.normalizations and meta.get("normalization") not in args.normalizations:
        return False
    if args.include_flip is False and meta.get("flip", False) is True:
        return False
    return True


def _group_files(input_dir: Path, args: argparse.Namespace):
    grouped: Dict[Tuple, List[Tuple[str, Path]]] = defaultdict(list)
    for path in sorted(input_dir.glob("*.npz")):
        try:
            theta, mean, std, meta = _load_npz(path)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if not meta:
            print(f"Skipping {path.name}: missing metadata")
            continue
        if not _should_keep(meta, args):
            continue
        key = (
            meta.get("dataset"),
            meta.get("activation"),
            meta.get("normalization"),
            bool(meta.get("flip", False)),
        )
        grouped[key].append((meta.get("position", "unknown"), path))
    return grouped


def _plot_group(
    key: Tuple,
    entries: List[Tuple[str, Path]],
    output_dir: Path,
    *,
    y_max: float | None,
) -> Path | None:
    dataset, activation, normalization, flip = key
    colors = {"first": "C0", "middle": "C1", "last": "C2"}
    linestyles = {"first": "-", "middle": "--", "last": "-."}

    plt.figure(figsize=(8, 4))
    theta_ref = None
    label_seen = set()

    def _wrap(theta: np.ndarray, mean: np.ndarray, std: np.ndarray | None):
        if theta.size == 0 or mean.size == 0:
            return theta, mean, std
        needs_wrap = not np.isclose(theta[-1], theta[0]) and not np.isclose(theta[-1] - theta[0], 2 * np.pi)
        if not needs_wrap:
            return theta, mean, std
        theta_w = np.concatenate([theta, theta[:1] + 2 * np.pi])
        mean_w = np.concatenate([mean, mean[:1]])
        std_w = None
        if std is not None and std.size != 0:
            std_w = np.concatenate([std, std[:1]])
        return theta_w, mean_w, std_w

    for position, path in sorted(entries, key=lambda t: t[0]):
        theta, mean, std, meta = _load_npz(path)
        theta, mean, std = _wrap(theta, mean, std)
        theta_pi = theta / np.pi
        theta_ref = theta_ref if theta_ref is not None else theta_pi
        color = colors.get(position, None)
        ls = linestyles.get(position, "-")
        label = position if position not in label_seen else None
        plt.plot(theta_pi, mean, color=color, linestyle=ls, label=label)
        label_seen.add(position)
        if std is not None and std.shape == mean.shape:
            plt.fill_between(theta_pi, mean - std, mean + std, color=color, alpha=0.2)

    if theta_ref is None:
        plt.close()
        return None

    plt.xlabel(r"Rotation angle ($\pi$ units)")
    plt.ylabel("Relative equivariance error")
    plt.ylim(bottom=0.0, top=y_max if y_max is not None else None)
    flip_suffix = "flip" if flip else "noflip"
    plt.title(f"{dataset} | {activation} / {normalization} | {flip_suffix}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{dataset}_{activation}_{normalization}_{flip_suffix}.pdf".replace("/", "-")
    out_path = output_dir / safe_name
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir '{input_dir}' does not exist")

    grouped = _group_files(input_dir, args)
    if not grouped:
        print("No matching .npz files found.")
        return

    saved: List[Path] = []
    for key, entries in grouped.items():
        out = _plot_group(key, entries, output_dir, y_max=args.y_max)
        if out:
            saved.append(out)

    print(f"Saved {len(saved)} plots to {output_dir}")


if __name__ == "__main__":
    main()
