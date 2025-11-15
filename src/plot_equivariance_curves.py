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
from pathlib import Path
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


def _build_datamodule(dataset: str, hparams, args: argparse.Namespace, seed_override: int) -> torch.nn.Module:
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
            )
        if theta_template is None:
            theta_template = np.array(thetas, dtype=np.float64)
        curves.append(np.array(curve, dtype=np.float64))
        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    if not curves or theta_template is None:
        raise RuntimeError("No batches were processed; increase --max-eval-batches?")

    return theta_template, np.stack(curves, axis=0).mean(axis=0)


def _save_plot(
    theta_rad: np.ndarray,
    mean_curve: np.ndarray,
    std_curve: np.ndarray,
    output_path: Path,
    dataset: str,
    activation: str,
    normalization: str,
) -> None:
    theta_deg = np.rad2deg(theta_rad)

    plt.figure(figsize=(7, 4))
    plt.plot(theta_deg, mean_curve, label="mean error", color="C0")
    if std_curve is not None and np.any(std_curve > 0):
        plt.fill_between(
            theta_deg,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="C0",
            alpha=0.2,
            label="±1 std",
        )
    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Relative equivariance error")
    plt.title(f"Equivariance: {dataset} | {activation} / {normalization}")
    plt.xticks(np.linspace(0, 360, num=9))
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
        theta_rad: np.ndarray,
        curve: np.ndarray,
    ):
        self.dataset = dataset
        self.activation = activation
        self.normalization = normalization
        self.flip = bool(flip)
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

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, bool]:
        mean = self.sum / self.count
        var = self.sumsq / self.count - mean ** 2
        var = np.clip(var, a_min=0.0, a_max=None)
        std = np.sqrt(var)
        return self.theta, mean, std, self.dataset, self.activation, self.normalization, self.flip


def _finalize_and_plot(accumulator: _CurveAccumulator, output_dir: Path):
    if accumulator is None:
        return
    theta, mean, std, dataset, activation, normalization, flip = accumulator.finalize()
    flip_suffix = "flip" if flip else "noflip"
    safe_name = f"{dataset}_{activation}_{normalization}_{flip_suffix}".replace("/", "-")
    out_path = output_dir / f"{safe_name}.png"
    _save_plot(theta, mean, std, out_path, dataset, activation, normalization)
    print(f"Saved averaged plot to {out_path}")


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory '{models_dir}' does not exist.")

    entries = _collect_checkpoints(models_dir)
    if not entries:
        print(f"No checkpoint files found in {models_dir}; nothing to plot.")
        return

    entries.sort(
        key=lambda e: (e["dataset"], e["activation_hint"], e["normalization_hint"], e["seed"])
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accumulator: _CurveAccumulator | None = None
    current_key: Tuple[str, str, str] | None = None

    for entry in entries:
        ckpt_path = entry["path"]
        dataset = entry["dataset"]
        print(f"Processing {ckpt_path.name} (dataset={dataset})")

        lit_model = LitHnn.load_from_checkpoint(ckpt_path, map_location=device)
        activation = _get_hparam(
            lit_model.hparams, "activation_type", entry["activation_hint"]
        )
        normalization = _get_hparam(
            lit_model.hparams, "bn", entry["normalization_hint"]
        )
        flip_flag = bool(_get_hparam(lit_model.hparams, "flip", False))

        datamodule = _build_datamodule(dataset, lit_model.hparams, args, entry["seed"])
        theta_rad, curve = _compute_curve(
            lit_model,
            datamodule,
            device=device,
            num_angles=args.num_angles,
            chunk_size=args.chunk_size,
            max_batches=args.max_eval_batches,
        )

        key = (dataset, activation, normalization, flip_flag)

        if current_key is None:
            accumulator = _CurveAccumulator(dataset, activation, normalization, flip_flag, theta_rad, curve)
            current_key = key
        elif key != current_key:
            _finalize_and_plot(accumulator, output_dir)
            accumulator = _CurveAccumulator(dataset, activation, normalization, flip_flag, theta_rad, curve)
            current_key = key
        else:
            accumulator.add(theta_rad, curve)

    _finalize_and_plot(accumulator, output_dir)


if __name__ == "__main__":
    main()
