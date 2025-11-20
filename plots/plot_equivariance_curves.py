"""Aggregate equivariance curves per activation + normalization combination.

Point this script at one `saved_models/<subfolder>/` directory that contains
`.ckpt` files emitted by `train.py` (named like
`dataset_activation_batchnorm_seedX.ckpt`). For every activation/batchnorm pair
found in that directory, the script loads every seed checkpoint, measures the
equivariance curve, averages the results, and stores one plot named
`<activation>_<batchnorm>.png` under the requested output directory.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets_utils.data_classes import (
    ColorectalHistDataModule,
    EuroSATDataModule,
    MnistRotDataModule,
    Resisc45DataModule,
)
import nets.equivariance_metric as em
from train import LitHnn


_DATASET_PREFIXES = [
    "mnist_rot",
    "resisc45",
    "colorectal_hist",
    "eurosat",
]

_BN_NAMES = {
    "IIDbn",
    "Normbn",
    "FieldNorm",
    "GNormBatchNorm",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot averaged equivariance curves per activation/batchnorm"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing *.ckpt files (one sweep subfolder).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/equivariance",
        help="Directory where the generated plots will be written.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=32,
        help="Number of rotation angles for the equivariance metric.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Chunk size forwarded to the equivariance metric helper.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
        help="Batch size used when sampling evaluation batches.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=-1,
        help="How many batches per checkpoint to average; set to -1 to use the full loader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the models on (auto prefers CUDA when available).",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=str,
        default="./src/datasets_utils/mnist_rotation_new",
        help="Path to MNIST-rot dataset (used only for mnist_rot checkpoints).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="If > 0, evaluate equivariance at the specified layer index instead of the full model.",
    )
    return parser.parse_args()


def _select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_hparam(hparams, key: str, default=None):
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


def _parse_checkpoint_filename(stem: str) -> Dict:
    if "_seed" not in stem:
        raise ValueError("missing '_seed' segment")

    base, seed_str = stem.rsplit("_seed", 1)
    seed = int(seed_str)

    dataset = None
    rest = base
    for candidate in sorted(_DATASET_PREFIXES, key=len, reverse=True):
        if rest.startswith(candidate):
            dataset = candidate
            rest = rest[len(candidate) :]
            break
    if dataset is None:
        raise ValueError("dataset prefix not recognized")

    rest = rest.lstrip("_")
    if not rest:
        raise ValueError("activation/bn part missing")

    flip_hint = False
    if rest.endswith("_flip"):
        flip_hint = True
        rest = rest[: -len("_flip")]
        rest = rest.rstrip("_")

    normalization = None
    activation = None
    for bn in _BN_NAMES:
        suffix = f"_{bn}"
        if rest.endswith(suffix):
            normalization = bn
            activation = rest[: -len(suffix)]
            break
        if rest == bn:
            normalization = bn
            activation = ""
            break

    if normalization is None:
        raise ValueError("unable to infer batchnorm from filename")
    activation = activation.rstrip("_")
    if not activation:
        raise ValueError("activation missing in filename")

    return {
        "dataset": dataset,
        "activation_hint": activation,
        "normalization_hint": normalization,
        "flip_hint": flip_hint,
        "seed": seed,
    }


def _collect_checkpoints(models_dir: Path) -> List[Dict]:
    entries: List[Dict] = []
    for ckpt_path in sorted(models_dir.glob("*.ckpt")):
        try:
            parsed = _parse_checkpoint_filename(ckpt_path.stem)
        except ValueError as exc:
            print(f"[WARN] Skipping {ckpt_path.name}: {exc}")
            continue
        parsed["path"] = ckpt_path
        entries.append(parsed)
    return entries


def _build_datamodule(dataset: str, hparams, args: SimpleNamespace, seed_override: int) -> torch.nn.Module:
    img_size = int(_get_hparam(hparams, "img_size", 64))
    seed = int(seed_override)
    batch_size = args.eval_batch_size

    if dataset == "mnist_rot":
        return MnistRotDataModule(
            batch_size=batch_size, data_dir=args.mnist_data_dir, img_size=img_size, seed=seed
        )
    if dataset == "resisc45":
        return Resisc45DataModule(batch_size=batch_size, img_size=img_size, seed=seed)
    if dataset == "colorectal_hist":
        return ColorectalHistDataModule(batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0)
    if dataset == "eurosat":
        return EuroSATDataModule(batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0)

    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _compute_curve(
    lit_model: LitHnn,
    datamodule,
    device: torch.device,
    num_angles: int,
    chunk_size: int,
    max_batches: int,
    *,
    layer: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(datamodule, "prepare_data"):
        try:
            datamodule.prepare_data()
        except Exception:
            pass
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    curves: List[np.ndarray] = []
    theta_template: np.ndarray | None = None

    lit_model.eval()
    lit_model.model.eval()
    lit_model.to(device)

    max_batches = int(max_batches)

    for batch_idx, batch in enumerate(loader):
        x, _ = batch
        x = x.to(device)
        with torch.no_grad():
            thetas, curve = em.check_equivariance_batch_r2(
                x,
                lit_model.model,
                num_samples=num_angles,
                chunk_size=chunk_size,
                layer=layer,
            )
        if theta_template is None:
            theta_template = np.array(thetas, dtype=np.float64)
        curves.append(np.array(curve, dtype=np.float64))
        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break
    print(f"Processed {len(curves)} batches for equivariance curve.")
    if not curves or theta_template is None:
        raise RuntimeError("No batches were processed; increase --max-eval-batches?")

    return theta_template, np.stack(curves, axis=0).mean(axis=0)


def _wrap_segment(theta_segment: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if theta_segment.size == 0 or values.size == 0:
        return theta_segment, values
    if np.isclose(theta_segment[0], theta_segment[-1]):
        return theta_segment, values
    theta_wrapped = np.concatenate([theta_segment, theta_segment[:1] + 2 * np.pi])
    values_wrapped = np.concatenate([values, values[:1]])
    return theta_wrapped, values_wrapped


def _build_plot_segments(
    theta_rad: np.ndarray,
    values: np.ndarray,
    flip: bool,
    base_angles: int,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    if not flip:
        segments = [("rot", theta_rad, values, 0.0)]
    else:
        base = max(1, int(base_angles))
        total_needed = base * 2
        if theta_rad.shape[0] < total_needed or values.shape[0] < total_needed:
            raise ValueError("Not enough samples for flip/grouped segments.")
        segments = [
            ("rot", theta_rad[:base], values[:base], 0.0),
            ("flip", theta_rad[base:base * 2], values[base:base * 2], 2.0),
        ]

    plot_segments: List[Tuple[np.ndarray, np.ndarray, str]] = []
    for seg_type, theta_seg, val_seg, offset in segments:
        theta_wrap, val_wrap = _wrap_segment(theta_seg, val_seg)
        theta_plot = theta_wrap / np.pi + offset
        plot_segments.append((theta_plot, val_wrap, seg_type))
    return plot_segments


def _save_plot(
    theta_rad: np.ndarray,
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    output_path: Path,
    dataset: str,
    activation: str,
    normalization: str,
    layer_label: str,
    flip: bool,
    base_angles: int,
) -> None:
    mean_segments = _build_plot_segments(theta_rad, mean_curve, flip, base_angles)
    std_segments = (
        _build_plot_segments(theta_rad, std_curve, flip, base_angles) if std_curve is not None else None
    )

    plt.figure(figsize=(8, 4))
    color_map = {"rot": "C0", "flip": "C1"}
    label_map = {"rot": "rotations", "flip": "flips"}
    legend_used = {"rot_mean": False, "flip_mean": False, "rot_std": False, "flip_std": False}

    for idx, (theta_vals, mean_vals, seg_type) in enumerate(mean_segments):
        color = color_map.get(seg_type, "C0")
        mean_label_key = f"{seg_type}_mean"
        label = f"{label_map.get(seg_type, seg_type)} mean" if not legend_used[mean_label_key] else None
        plt.plot(theta_vals, mean_vals, color=color, label=label)
        legend_used[mean_label_key] = True

        if std_segments is not None:
            theta_std, std_vals, _ = std_segments[idx]
            if theta_std.shape != theta_vals.shape:
                raise ValueError("Theta/std segments have mismatched shapes.")
            std_label_key = f"{seg_type}_std"
            std_label = (
                f"{label_map.get(seg_type, seg_type)} ±1 std"
                if not legend_used[std_label_key]
                else None
            )
            plt.fill_between(
                theta_vals,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=0.2,
                label=std_label,
            )
            legend_used[std_label_key] = True

    plt.xlabel(r"Rotation angle ($\pi$ units)")
    plt.ylabel("Relative equivariance error")
    plt.title(f"Equivariance ({layer_label}): {dataset} | {activation} / {normalization}")
    span = 2 if not flip else 4
    tick_vals = np.linspace(0, span, num=9)
    tick_labels = []
    for val in tick_vals:
        if np.isclose(val, 0):
            tick_labels.append("0")
        elif np.isclose(val, 2) and not flip:
            tick_labels.append("2")
        elif np.isclose(val, 2) and flip:
            tick_labels.append("2 | flip")
        elif flip and np.isclose(val, 4):
            tick_labels.append("4")
        elif np.isclose(val, 1):
            tick_labels.append("1")
        else:
            tick_labels.append(f"{val:.1f}")
    plt.xticks(tick_vals, tick_labels)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


class _CurveAccumulator:
    def __init__(
        self,
        dataset: str,
        activation: str,
        normalization: str,
        flip: bool,
        layer_label: str,
        base_angles: int,
        theta_rad: np.ndarray,
        curve: np.ndarray,
    ):
        self.dataset = dataset
        self.activation = activation
        self.normalization = normalization
        self.flip = bool(flip)
        self.layer_label = layer_label
        self.base_angles = max(1, int(base_angles))
        self.theta = theta_rad
        self.sum = curve.copy()
        self.sumsq = curve ** 2
        self.count = 1

    def add(self, theta_rad: np.ndarray, curve: np.ndarray):
        if theta_rad.shape != self.theta.shape or not np.allclose(theta_rad, self.theta):
            raise RuntimeError("Theta grids do not match within the same group.")
        self.sum += curve
        self.sumsq += curve ** 2
        self.count += 1

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, bool, str, int]:
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean ** 2
        var = np.clip(var, a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return (
            self.theta,
            mean,
            std,
            self.dataset,
            self.activation,
            self.normalization,
            self.flip,
            self.layer_label,
            self.base_angles,
        )


def _finalize_and_plot(accumulator: _CurveAccumulator, output_dir: Path) -> Path | None:
    if accumulator is None:
        return None
    theta, mean, std, dataset, activation, normalization, flip, layer_label, base_angles = (
        accumulator.finalize()
    )
    flip_suffix = "flip" if flip else "noflip"
    layer_suffix = layer_label.replace(" ", "")
    safe_name = f"{dataset}_{activation}_{normalization}_{flip_suffix}_{layer_suffix}".replace("/", "-")
    out_path = output_dir / f"{safe_name}.png"
    _save_plot(
        theta,
        mean,
        std,
        out_path,
        dataset,
        activation,
        normalization,
        layer_label,
        flip,
        base_angles,
    )
    print(f"Saved averaged plot to {out_path}")
    return out_path


def generate_equivariance_plots(
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    num_angles: int = 32,
    chunk_size: int = 4,
    eval_batch_size: int = 8,
    max_eval_batches: int = -1,
    device: str = "auto",
    mnist_data_dir: str = "./src/datasets_utils/mnist_rotation_new",
    layer: int = 0,
) -> List[Path]:
    device_obj = _select_device(device)
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir_path}' does not exist.")

    entries = _collect_checkpoints(models_dir_path)
    if not entries:
        print(f"No checkpoint files found in {models_dir_path}; nothing to plot.")
        return []

    entries.sort(
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
            e["seed"],
        )
    )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    saved_plots: List[Path] = []
    args_ns = SimpleNamespace(eval_batch_size=eval_batch_size, mnist_data_dir=mnist_data_dir)
    layer_index = layer if layer > 0 else None
    layer_label = f"layer {layer_index}" if layer_index else "full model"

    for (dataset, act_hint, norm_hint, flip_hint), group_iter in itertools.groupby(
        entries,
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
        ),
    ):
        group_entries = list(group_iter)
        first_entry = group_entries[0]
        print(
            f"Processing group: dataset={dataset}, activation={act_hint}, bn={norm_hint}, flip={flip_hint}"
        )

        lit_model = LitHnn.load_from_checkpoint(first_entry["path"], map_location=device_obj)
        activation = _get_hparam(lit_model.hparams, "activation_type", act_hint)
        normalization = _get_hparam(lit_model.hparams, "bn", norm_hint)
        flip_flag = bool(_get_hparam(lit_model.hparams, "flip", flip_hint))

        datamodule = _build_datamodule(dataset, lit_model.hparams, args_ns, first_entry["seed"])
        base_angles = max(1, int(num_angles))
        samples_per_group = base_angles * (2 if flip_flag else 1)

        theta_rad, curve = _compute_curve(
            lit_model,
            datamodule,
            device=device_obj,
            num_angles=samples_per_group,
            chunk_size=chunk_size,
            max_batches=max_eval_batches,
            layer=layer_index,
        )
        accumulator = _CurveAccumulator(
            dataset,
            activation,
            normalization,
            flip_flag,
            layer_label,
            base_angles,
            theta_rad,
            curve,
        )

        for entry in group_entries[1:]:
            checkpoint = torch.load(entry["path"], map_location=device_obj)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise KeyError(f"Checkpoint {entry['path']} missing 'state_dict'")
            lit_model.load_state_dict(state_dict)
            theta_rad, curve = _compute_curve(
                lit_model,
                datamodule,
                device=device_obj,
                num_angles=samples_per_group,
                chunk_size=chunk_size,
                max_batches=max_eval_batches,
                layer=layer_index,
            )
            accumulator.add(theta_rad, curve)

        saved = _finalize_and_plot(accumulator, output_dir_path)
        if saved:
            saved_plots.append(saved)

    return saved_plots


def main() -> None:
    args = _parse_args()
    generate_equivariance_plots(
        args.models_dir,
        args.output_dir,
        num_angles=args.num_angles,
        chunk_size=args.chunk_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
        mnist_data_dir=args.mnist_data_dir,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()
