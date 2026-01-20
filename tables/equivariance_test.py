"""Aggregate equivariance metrics per activation + normalization combination.

Point this script at one `saved_models/<subfolder>/` directory that contains
`.ckpt` files emitted by `train.py` (named like
`dataset_activation_batchnorm_seedX.ckpt`). For every activation/batchnorm pair
found in that directory, the script loads every seed checkpoint, measures the
equivariance error across all angles/batches/seeds (for the full network and
layers 1–5 when available), averages the results, and stores one summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
}

_METRIC_COLUMNS: Tuple[Tuple[str, int | None], ...] = (
    ("error_metric_net", None),
    ("error_metric_1layer", 1),
    ("error_metric_2layer", 2),
    ("error_metric_3layer", 3),
    ("error_metric_4layer", 4),
    ("error_metric_5layer", 5),
    ("error_metric_6layer", 6),
    ("error_metric_7layer", 7),
    ("error_metric_8layer", 8)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate equivariance metrics per activation/batchnorm"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing *.ckpt files (one sweep subfolder).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tables/equivariance_results.csv",
        help="Path where the aggregated CSV file will be written.",
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
        default=32,
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
        "--show-progress",
        action="store_true",
        help="Display tqdm progress bars while evaluating batches.",
    )
    parser.add_argument(
        "--stratified-samples",
        type=int,
        default=None,
        help="Use stratified sampling with this many total samples from test set (distributed equally across classes). If not specified, uses all test data.",
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

    # Handle format: dataset_seedX_noaug_activation_bn or dataset_activation_bn_seedX
    parts = stem.split("_seed")
    if len(parts) != 2:
        raise ValueError("unexpected _seed format")

    before_seed = parts[0]
    after_seed = parts[1]

    # Extract seed number (first digits after _seed)
    seed_match = after_seed.split("_")[0]
    seed = int(seed_match)

    # Remove seed number and get remaining parts
    remaining = after_seed[len(seed_match):].lstrip("_")

    # Handle noaug prefix if present
    noaug_hint = False
    if remaining.startswith("noaug_"):
        noaug_hint = True
        remaining = remaining[len("noaug_"):]

    # Combine parts (before seed might be just dataset, remaining has activation_bn)
    # or (before seed has dataset_activation_bn, remaining is empty/flip)
    if remaining:
        # Format: dataset_seedX_noaug_activation_bn
        full_string = before_seed + "_" + remaining
    else:
        # Format: dataset_activation_bn_seedX
        full_string = before_seed

    # Extract dataset
    dataset = None
    rest = full_string
    for candidate in sorted(_DATASET_PREFIXES, key=len, reverse=True):
        if candidate in rest:
            # Handle both "equivariant_dataset" and "dataset" prefixes
            if rest.startswith("equivariant_" + candidate):
                dataset = candidate
                rest = rest[len("equivariant_" + candidate):]
                break
            elif rest.startswith(candidate):
                dataset = candidate
                rest = rest[len(candidate):]
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


def _get_stratified_indices(dataset, max_samples: int, seed: int = 42) -> List[int]:
    """Get stratified sample indices ensuring equal representation per class."""
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_to_indices[label].append(idx)

    # Calculate samples per class
    num_classes = len(class_to_indices)
    samples_per_class = max_samples // num_classes

    # Sample from each class
    rng = np.random.RandomState(seed)
    stratified_indices = []
    for class_label in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_label]
        if len(indices) <= samples_per_class:
            stratified_indices.extend(indices)
        else:
            sampled = rng.choice(indices, size=samples_per_class, replace=False)
            stratified_indices.extend(sampled.tolist())

    # Shuffle the final indices
    rng.shuffle(stratified_indices)
    return stratified_indices


def _compute_curve(
    lit_model: LitHnn,
    datamodule,
    device: torch.device,
    num_angles: int,
    chunk_size: int,
    max_batches: int,
    *,
    layer: int | None = None,
    show_progress: bool = False,
    stratified_samples: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(datamodule, "prepare_data"):
        try:
            datamodule.prepare_data()
        except Exception:
            pass
    datamodule.setup("test")

    # Get test dataset and optionally stratify
    test_dataset = datamodule.test_dataloader().dataset
    if stratified_samples is not None and stratified_samples > 0:
        stratified_indices = _get_stratified_indices(test_dataset, stratified_samples, seed=seed)
        test_dataset = Subset(test_dataset, stratified_indices)
        print(f"Using stratified sample of {len(stratified_indices)} examples from test set")

    # Create new dataloader with potentially stratified dataset
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=datamodule.test_dataloader().batch_size,
        shuffle=False,
        num_workers=0,
    )

    curves: List[np.ndarray] = []
    theta_template: np.ndarray | None = None

    lit_model.eval()
    lit_model.model.eval()
    lit_model.to(device)

    max_batches = int(max_batches)

    iterator = enumerate(loader)
    if show_progress:
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        desc = f"Equivariance batches (layer {layer if layer else 'net'})"
        iterator = tqdm(iterator, total=total_batches, desc=desc, leave=False)

    for batch_idx, batch in iterator:
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


class _CurveAccumulator:
    def __init__(self):
        self.curves: List[np.ndarray] = []

    def add(self, curve: np.ndarray):
        self.curves.append(curve)

    def mean(self) -> np.ndarray | None:
        if not self.curves:
            return None
        return np.stack(self.curves, axis=0).mean(axis=0)


def _evaluate_layers(
    lit_model: LitHnn,
    datamodule,
    device: torch.device,
    *,
    num_angles: int,
    chunk_size: int,
    max_eval_batches: int,
    show_progress: bool,
    dataset: str,
    stratified_samples: int | None = None,
    seed: int = 42,
) -> Dict[str, np.ndarray | None]:
    """Compute equivariance curves for each requested layer for a single checkpoint."""
    metrics: Dict[str, np.ndarray | None] = {}
    total_layers = len(getattr(lit_model.model, "eq_layers", []))
    thetas = None

    for column_name, layer_idx in _METRIC_COLUMNS:
        if layer_idx is not None and layer_idx > total_layers:
            metrics[column_name] = None
            continue
        # Skip layers 7 and 8 for non-colorectal datasets
        if layer_idx is not None and layer_idx >= 7 and dataset != "colorectal_hist":
            metrics[column_name] = None
            continue
        theta_vals, curve = _compute_curve(
            lit_model,
            datamodule,
            device=device,
            num_angles=num_angles,
            chunk_size=chunk_size,
            max_batches=max_eval_batches,
            layer=layer_idx,
            show_progress=show_progress,
            stratified_samples=stratified_samples,
            seed=seed,
        )
        if thetas is None:
            thetas = theta_vals
        metrics[column_name] = curve
    return metrics, thetas


def aggregate_equivariance_metrics(
    models_dir: str | Path,
    output_csv: str | Path,
    *,
    num_angles: int = 32,
    chunk_size: int = 4,
    eval_batch_size: int = 8,
    max_eval_batches: int = -1,
    device: str = "auto",
    mnist_data_dir: str = "./src/datasets_utils/mnist_rotation_new",
    show_progress: bool = False,
    stratified_samples: int | None = None,
):
    device_obj = _select_device(device)
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir_path}' does not exist.")

    entries = _collect_checkpoints(models_dir_path)
    if not entries:
        print(f"No checkpoint files found in {models_dir_path}; nothing to aggregate.")
        return None

    entries.sort(
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
            e["seed"],
        )
    )

    aggregated_rows: List[Dict[str, float | str]] = []
    args_ns = SimpleNamespace(eval_batch_size=eval_batch_size, mnist_data_dir=mnist_data_dir)

    grouped_entries: List[Tuple[Tuple[str, str, str, bool], List[Dict]]] = []
    for key, group_iter in itertools.groupby(
        entries,
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
        ),
    ):
        grouped_entries.append((key, list(group_iter)))

    for (dataset, act_hint, norm_hint, flip_hint), group_entries in tqdm(
        grouped_entries, desc="Activation/BN groups"
    ):
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

        accumulators = {name: _CurveAccumulator() for name, _ in _METRIC_COLUMNS}
        theta_values = None

        def _accumulate_current_model():
            nonlocal theta_values
            layer_metrics, thetas = _evaluate_layers(
                lit_model,
                datamodule,
                device=device_obj,
                num_angles=samples_per_group,
                chunk_size=chunk_size,
                max_eval_batches=max_eval_batches,
                show_progress=show_progress,
                dataset=dataset,
                stratified_samples=stratified_samples,
                seed=first_entry["seed"],
            )
            if theta_values is None:
                theta_values = thetas
            for name, curve in layer_metrics.items():
                if curve is None:
                    continue
                accumulators[name].add(curve)

        _accumulate_current_model()

        for entry in group_entries[1:]:
            checkpoint = torch.load(entry["path"], map_location=device_obj)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise KeyError(f"Checkpoint {entry['path']} missing 'state_dict'")
            lit_model.load_state_dict(state_dict)
            _accumulate_current_model()

        row: Dict[str, np.ndarray | str] = {"activation": activation, "bn": normalization, "dataset": dataset}
        for name, _ in _METRIC_COLUMNS:
            mean_curve = accumulators[name].mean()
            row[name] = mean_curve
        row["theta_values"] = theta_values
        aggregated_rows.append(row)

    if not aggregated_rows:
        print("No model groups produced metrics; nothing to save.")
        return None

    # Save as NPZ file to preserve array data
    npz_path = Path(output_csv).with_suffix('.npz')
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for NPZ format
    save_dict = {}
    for idx, row in enumerate(aggregated_rows):
        prefix = f"row{idx}_"
        save_dict[f"{prefix}activation"] = row["activation"]
        save_dict[f"{prefix}bn"] = row["bn"]
        save_dict[f"{prefix}dataset"] = row["dataset"]
        save_dict[f"{prefix}theta_values"] = row["theta_values"]
        for name, _ in _METRIC_COLUMNS:
            curve = row.get(name)
            if curve is not None:
                save_dict[f"{prefix}{name}"] = curve

    np.savez(npz_path, **save_dict, num_rows=len(aggregated_rows))
    print(f"Saved equivariance curves to {npz_path}")

    # Also save a human-readable CSV with mean values for quick inspection
    csv_path = Path(output_csv)
    fieldnames = ["dataset", "activation", "bn"] + [name for name, _ in _METRIC_COLUMNS]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            csv_row = {
                "dataset": row["dataset"],
                "activation": row["activation"],
                "bn": row["bn"]
            }
            for name, _ in _METRIC_COLUMNS:
                curve = row.get(name)
                if curve is not None:
                    csv_row[name] = float(curve.mean())
                else:
                    csv_row[name] = None
            writer.writerow(csv_row)
    print(f"Saved mean metrics to {csv_path}")
    return npz_path


def main() -> None:
    args = _parse_args()
    aggregate_equivariance_metrics(
        args.models_dir,
        args.output_csv,
        num_angles=args.num_angles,
        chunk_size=args.chunk_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
        mnist_data_dir=args.mnist_data_dir,
        show_progress=args.show_progress,
        stratified_samples=args.stratified_samples,
    )


if __name__ == "__main__":
    main()
