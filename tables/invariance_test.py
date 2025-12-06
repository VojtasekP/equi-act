"""Aggregate invariance metrics per activation + normalization (aug/noaug) group.

Point this script at one `saved_models/<subfolder>/` directory that contains
`.ckpt` files emitted by `train.py` (named like
`dataset_activation_bn_seedX.ckpt`). For every activation/bn/aug group found in
that directory, the script loads every seed checkpoint, measures the invariance
error on a stratified subset of the test set (up to 128 examples), averages the
results across seeds, and stores one summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
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


_DATASET_PREFIXES = ["mnist_rot", "resisc45", "colorectal_hist", "eurosat"]
_BN_NAMES = {"IIDbn", "Normbn"}
_ACT_WITH_BUILTIN_BN = ("normbn", "normbnvec", "fourierbn")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate invariance metrics per activation/batchnorm"
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
        default="tables/invariance_results.csv",
        help="Path where the aggregated CSV file will be written.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=16,
        help="Number of rotation angles for the invariance metric.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Chunk size forwarded to the invariance metric helper.",
    )
    parser.add_argument(
        "--max-samples", 
        type=int,
        default=128,
        help="Target number of stratified test samples to evaluate (per checkpoint).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="Batch size used when running the metric on the sampled subset.",
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
    return parser.parse_args()


def _select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _get_hparam(hparams, key: str, default=None):
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


def _parse_checkpoint_filename(stem: str) -> Dict:
    """
    Expected pattern: modeltype_dataset_seedX_[aug|noaug]_activation[_BN]
    Examples:
      equivariant_mnist_rot_seed0_aug_fourierbn_elu_16_Normbn
      equivariant_mnist_rot_seed1_noaug_normbnvec_relu
      resnet18_eurosat_seed0_aug
    """
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError("filename too short to parse")

    # model type, dataset, seedX, ...
    model_type = parts[0]
    dataset = parts[1]
    if dataset in "mnist_rot":
        dataset = "mnist_rot"
    if dataset in "colorectal_hist":
        dataset = "colorectal_hist"

    seed_part = next((p for p in parts if p.startswith("seed")), None)
    if seed_part is None:
        raise ValueError("missing seed segment")
    try:
        seed = int(seed_part.replace("seed", ""))
    except ValueError:
        raise ValueError("seed segment malformed")

    seed_idx = parts.index(seed_part)
    aug_flag = None
    if seed_idx + 1 < len(parts) and parts[seed_idx + 1] in {"aug", "noaug"}:
        aug_flag = parts[seed_idx + 1] == "aug"
        tail = parts[seed_idx + 2 :]
    else:
        tail = parts[seed_idx + 1 :]

    activation = None
    normalization = None
    flip_hint = False

    if model_type == "resnet18":
        activation = "resnet18"
        normalization = None
    else:
        if tail:
            if len(tail) > 1:
                normalization = tail[-1] if tail[-1] in _BN_NAMES else None
                activation = "_".join(tail[:-1]) if normalization else "_".join(tail)
            else:
                activation = "_".join(tail)
        if activation is None or activation == "":
            raise ValueError("activation missing in filename")
        if activation.startswith(_ACT_WITH_BUILTIN_BN):
            normalization = None

    return {
        "model_type": model_type,
        "dataset": dataset,
        "activation_hint": activation,
        "normalization_hint": normalization,
        "flip_hint": flip_hint,
        "seed": seed,
        "aug_hint": aug_flag,
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
        parsed["aug_hint"] = parsed.get("aug_hint", None)
        entries.append(parsed)
    return entries


def _build_datamodule(dataset: str, hparams, args: SimpleNamespace, seed_override: int, *, aug: bool):
    img_size = int(_get_hparam(hparams, "img_size", 64))
    batch_size = args.eval_batch_size
    if dataset == "mnist_rot":
        return MnistRotDataModule(
            batch_size=batch_size,
            data_dir=args.mnist_data_dir,
            img_size=img_size,
            seed=seed_override,
            aug=aug,
            normalize=True,
        )
    if dataset == "resisc45":
        return Resisc45DataModule(batch_size=batch_size, img_size=img_size, seed=seed_override, aug=aug, normalize=True)
    if dataset == "colorectal_hist":
        return ColorectalHistDataModule(batch_size=batch_size, img_size=img_size, seed=seed_override, aug=aug, normalize=True)
    if dataset == "eurosat":
        return EuroSATDataModule(batch_size=batch_size, img_size=img_size, seed=seed_override, aug=aug, normalize=True)
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _sample_stratified(dataset, num_classes: int, max_samples: int):
    targets = [max_samples // num_classes] * num_classes
    for i in range(max_samples % num_classes):
        targets[i] += 1

    buckets: List[List[torch.Tensor]] = [[] for _ in range(num_classes)]
    collected = 0
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        cls = int(y)
        if cls < 0 or cls >= num_classes:
            continue
        if len(buckets[cls]) >= targets[cls]:
            continue
        buckets[cls].append(x)
        collected += 1
        if collected >= max_samples:
            break

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    for cls, bucket in enumerate(buckets):
        xs.extend(bucket)
        ys.extend([cls] * len(bucket))
    if not xs:
        raise RuntimeError("No samples collected for invariance evaluation.")
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def _prepare_eval_batch(datamodule, max_samples: int) -> torch.Tensor:
    if hasattr(datamodule, "prepare_data"):
        try:
            datamodule.prepare_data()
        except Exception:
            pass
    datamodule.setup("test")
    test_ds = getattr(datamodule, "test_ds", None)
    if test_ds is None:
        loader = datamodule.test_dataloader()
        test_ds = getattr(loader, "dataset", None)
    if test_ds is None:
        raise RuntimeError("Unable to access test dataset for stratified sampling.")
    x, _ = _sample_stratified(test_ds, getattr(datamodule, "num_classes", 1), max_samples)
    return x


def _compute_invariance(
    lit_model: LitHnn,
    batch: torch.Tensor,
    device: torch.device,
    num_angles: int,
    chunk_size: int,
) -> float:
    lit_model.eval()
    lit_model.model.eval()
    batch = batch.to(device)
    lit_model.to(device)
    _, errs = em.chech_invariance_batch_r2(
        batch,
        lit_model.model,
        num_samples=num_angles,
        chunk_size=chunk_size,
    )
    return float(np.mean(errs))


class _ListAccumulator:
    def __init__(self):
        self.values: List[float] = []

    def add(self, value: float):
        self.values.append(float(value))

    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    def std(self) -> float:
        return float(np.std(self.values)) if self.values else float("nan")


def aggregate_invariance_metrics(
    models_dir: str | Path,
    output_csv: str | Path,
    *,
    num_angles: int = 16,
    chunk_size: int = 4,
    max_samples: int = 128,
    eval_batch_size: int = 32,
    device: str = "auto",
    mnist_data_dir: str = "./src/datasets_utils/mnist_rotation_new",
    show_progress: bool = False,
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

    grouped_entries: List[Tuple[Tuple[str, str, str, str | None, bool, bool], List[Dict]]] = []
    for key, group_iter in itertools.groupby(
        entries,
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
            bool(
                _get_hparam(
                    torch.load(e["path"], map_location="cpu").get("hyper_parameters", {}),
                    "aug",
                    e.get("aug_hint", False),
                )
            ),
            e["model_type"],
        ),
    ):
        grouped_entries.append((key, list(group_iter)))

    for (dataset, act_hint, norm_hint, flip_hint, aug_flag, model_type), group_entries in tqdm(
        grouped_entries, desc="Activation/BN groups"
    ):
        first_entry = group_entries[0]
        print(
            f"Processing group: dataset={dataset}, model={model_type}, activation={act_hint}, bn={norm_hint}, flip={flip_hint}, aug={aug_flag}"
        )

        lit_model = LitHnn.load_from_checkpoint(first_entry["path"], map_location=device_obj)
        activation = _get_hparam(lit_model.hparams, "activation_type", act_hint)
        normalization = _get_hparam(lit_model.hparams, "bn", norm_hint)
        flip_flag = bool(_get_hparam(lit_model.hparams, "flip", flip_hint))
        aug_used = bool(_get_hparam(lit_model.hparams, "aug", aug_flag))

        if not hasattr(lit_model.model, "forward_features"):
            print(f"[WARN] Skipping {first_entry['path'].name}: model lacks equivariant features.")
            continue

        datamodule = _build_datamodule(dataset, lit_model.hparams, args_ns, first_entry["seed"], aug=aug_used)
        eval_batch = _prepare_eval_batch(datamodule, max_samples)

        accumulator = _ListAccumulator()

        def _accumulate_current_model():
            inv_err = _compute_invariance(
                lit_model,
                eval_batch,
                device=device_obj,
                num_angles=num_angles,
                chunk_size=chunk_size,
            )
            accumulator.add(inv_err)

        _accumulate_current_model()

        for entry in group_entries[1:]:
            checkpoint = torch.load(entry["path"], map_location=device_obj)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise KeyError(f"Checkpoint {entry['path']} missing 'state_dict'")
            lit_model.load_state_dict(state_dict)
            _accumulate_current_model()

        row: Dict[str, float | str] = {
            "model_type": model_type,
            "dataset": dataset,
            "activation": activation,
            "bn": normalization,
            "flip": flip_flag,
            "aug": aug_used,
            "invariance_mean": accumulator.mean(),
            "invariance_std": accumulator.std(),
        }
        aggregated_rows.append(row)

    if not aggregated_rows:
        print("No model groups produced metrics; nothing to save.")
        return None

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model_type", "dataset", "activation", "bn", "flip", "aug", "invariance_mean", "invariance_std"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)
    print(f"Saved aggregated metrics to {csv_path}")
    return csv_path


def main() -> None:
    args = _parse_args()
    aggregate_invariance_metrics(
        args.models_dir,
        args.output_csv,
        num_angles=args.num_angles,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        mnist_data_dir=args.mnist_data_dir,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
