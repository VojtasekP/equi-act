"""
Measure activation-level equivariance errors for saved checkpoints.

For each activation/nonlinearity layer (first, middle, last), the script compares
`activation(f.transform(g))` against `activation(f).transform(g)` across a grid of
group elements. Results are averaged over batches and seeds and written as:
  - one summary CSV with mean errors per position
  - optional .npz curve dumps per activation position for plotting later
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from escnn import nn

from datasets_utils.data_classes import (
    ColorectalHistDataModule,
    EuroSATDataModule,
    MnistRotDataModule,
    Resisc45DataModule,
)
from nets.equivariance_metric import make_group_elements, rel_err
from nets.new_layers import NormNonlinearityWithBN
from nets.RnNet import NonEquivariantTorchOp
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

_DEFAULT_POSITIONS = ("first", "middle", "last")


# ---------- arg parsing ----------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activation-level equivariance check (first/middle/last activation)."
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
        default="plots/activation_equivariance_data",
        help="Directory where per-position .npz curves will be written.",
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="tables/activation_equivariance_summary.csv",
        help="CSV file collecting mean errors per activation position.",
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
        default=32,
        help="Chunk size forwarded to the activation equivariance helper.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size used when sampling evaluation batches.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=1,
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
        "--positions",
        type=str,
        nargs="*",
        choices=_DEFAULT_POSITIONS,
        default=None,
        help="Activation positions to evaluate; defaults to all (first/middle/last).",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display tqdm-style progress while evaluating batches.",
    )
    return parser.parse_args()


# ---------- helpers ----------

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
    if "_seed" not in stem:
        raise ValueError("missing '_seed' segment in filename")

    base, seed_str = stem.rsplit("_seed", 1)
    seed = int(seed_str)

    dataset = None
    rest = base
    for candidate in sorted(_DATASET_PREFIXES, key=len, reverse=True):
        if rest.startswith(candidate):
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
        except Exception as exc:
            print(f"Skipping {ckpt_path.name}: {exc}")
            continue
        parsed["path"] = ckpt_path
        entries.append(parsed)
    return entries


def _build_datamodule(dataset: str, hparams, args: SimpleNamespace, seed_override: int):
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
        return ColorectalHistDataModule(
            batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0
        )
    if dataset == "eurosat":
        return EuroSATDataModule(batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0)
    raise ValueError(f"Unsupported dataset '{dataset}'.")


def _is_activation_layer(layer: torch.nn.Module) -> bool:
    activation_types = (
        nn.GatedNonLinearity1,
        nn.NormNonLinearity,
        nn.FourierPointwise,
        NormNonlinearityWithBN,
        NonEquivariantTorchOp,
    )
    if isinstance(layer, activation_types):
        return True

    def _child_modules(mod):
        try:
            for sub in mod.modules():
                if sub is mod:
                    continue
                yield sub
        except TypeError:
            return

    if isinstance(layer, (nn.SequentialModule, nn.MultipleModule)):
        return any(_is_activation_layer(sub) for sub in _child_modules(layer))
    return False


def _find_activation_indices(layers: List[torch.nn.Module]) -> List[int]:
    return [i for i, layer in enumerate(layers) if _is_activation_layer(layer)]


def _resolve_activation_targets(
    layers: List[torch.nn.Module],
    requested_positions: Sequence[str] | None,
) -> Dict[str, int]:
    positions = list(requested_positions) if requested_positions else list(_DEFAULT_POSITIONS)
    positions = [p for p in positions if p in _DEFAULT_POSITIONS]
    activation_indices = _find_activation_indices(layers)
    if not activation_indices:
        raise ValueError("No activation modules found in model.eq_layers.")

    mapping: Dict[str, int] = {}
    if "first" in positions:
        mapping["first"] = activation_indices[0]
    if "middle" in positions:
        mapping["middle"] = activation_indices[len(activation_indices) // 2]
    if "last" in positions:
        mapping["last"] = activation_indices[-1]
    return mapping


def _expand_theta(theta: np.ndarray, curve_len: int) -> np.ndarray:
    """Match theta sampling to the curve length (handles duplicated flip samples)."""
    if curve_len == len(theta):
        return theta
    if curve_len == len(theta) * 2:
        return np.concatenate([theta, theta])
    raise ValueError(f"Curve length {curve_len} incompatible with theta length {len(theta)}")


@torch.inference_mode()
def _activation_equivariance_from_feature(
    feature: nn.GeometricTensor,
    activation_module: torch.nn.Module,
    elems,
    chunk_size: int,
) -> np.ndarray:
    """Compute activation equivariance errors for one feature map and activation."""
    if not isinstance(feature, nn.GeometricTensor):
        feature = nn.GeometricTensor(feature, activation_module.in_type)  # type: ignore[arg-type]
    base = activation_module(feature)
    
    if not isinstance(base, nn.GeometricTensor):
        raise TypeError("Activation module must return a GeometricTensor.")

    errs: List[torch.Tensor] = []
    batch = feature.tensor.shape[0]
    chunk = max(1, int(chunk_size))

    for start in range(0, len(elems), chunk):
        chunk_elems = elems[start:start + chunk]
        f_rot = [feature.transform(g).tensor for g in chunk_elems]
        f_rot_batch = torch.cat(f_rot, dim=0)
        a_rot = activation_module(nn.GeometricTensor(f_rot_batch, feature.type))
        if not isinstance(a_rot, nn.GeometricTensor):
            raise TypeError("Activation module must return a GeometricTensor.")

        base_rot = [base.transform(g).tensor for g in chunk_elems]
        base_rot_batch = torch.cat(base_rot, dim=0)

        for a1, a2 in zip(torch.split(a_rot.tensor, batch, dim=0), torch.split(base_rot_batch, batch, dim=0)):
            errs.append(rel_err(a1, a2).mean())

    return torch.stack(errs).detach().cpu().numpy()


@torch.inference_mode()
def _activation_curves_from_model(
    x: torch.Tensor,
    model,
    target_indices: Sequence[int],
    elems,
    chunk_size: int,
) -> Dict[int, np.ndarray]:
    """Collect activation equivariance curves for the requested eq_layers indices."""
    x_geo = nn.GeometricTensor(x, model.input_type)
    layers = list(getattr(model, "eq_layers", []))
    targets = set(target_indices)
    pre_features: Dict[int, nn.GeometricTensor] = {}

    cur = x_geo
    for idx, layer in enumerate(layers):
        if idx in targets:
            pre_features[idx] = cur
        cur = layer(cur)
        if len(pre_features) == len(targets):
            break

    missing = targets - set(pre_features.keys())
    if missing:
        raise ValueError(f"Could not capture features for activation indices: {sorted(missing)}")

    curves: Dict[int, np.ndarray] = {}
    for idx in target_indices:
        print(pre_features[idx].type)
        print(pre_features[idx].shape)
    for idx in target_indices:
        curves[idx] = _activation_equivariance_from_feature(
            pre_features[idx], layers[idx], elems, chunk_size
        )
    return curves


def _compute_activation_curves(
    lit_model: LitHnn,
    datamodule,
    device: torch.device,
    *,
    label_to_index: Dict[str, int],
    num_angles: int,
    chunk_size: int,
    max_eval_batches: int,
    show_progress: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Average activation equivariance curves across a few batches."""
    if hasattr(datamodule, "prepare_data"):
        try:
            datamodule.prepare_data()
        except Exception:
            pass
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    lit_model.eval()
    lit_model.model.eval()
    lit_model.to(device)

    thetas = np.linspace(0.0, 2 * np.pi, num_angles, endpoint=False)
    elems = make_group_elements(lit_model.model.r2_act, thetas)  # type: ignore[attr-defined]

    per_label: Dict[str, List[np.ndarray]] = {label: [] for label in label_to_index}
    max_batches = int(max_eval_batches)

    iterator = enumerate(loader)
    if show_progress:
        try:
            from tqdm import tqdm
            total_batches = len(loader) if hasattr(loader, "__len__") else None
            if max_batches > 0:
                total_batches = min(total_batches, max_batches) if total_batches is not None else max_batches
            iterator = tqdm(iterator, total=total_batches, desc="Activation batches", leave=False)
        except Exception:
            pass

    unique_indices = sorted(set(label_to_index.values()))
    theta_template: np.ndarray | None = None

    for batch_idx, batch in iterator:
        x, _ = batch
        x = x.to(device, non_blocking=True)
        curves = _activation_curves_from_model(
            x, lit_model.model, unique_indices, elems, chunk_size
        )
        curve_len = next(iter(curves.values())).shape[0]
        for label, idx in label_to_index.items():
            per_label[label].append(np.array(curves[idx], dtype=np.float64))
        if theta_template is None:
            theta_template = _expand_theta(np.array(thetas, dtype=np.float64), curve_len)
        if max_batches > 0 and (batch_idx + 1) >= max_batches:
            break

    if theta_template is None:
        raise RuntimeError("No batches were processed; increase --max-eval-batches?")

    mean_curves: Dict[str, np.ndarray] = {}
    for label, curve_list in per_label.items():
        if not curve_list:
            raise RuntimeError(f"No curves recorded for label '{label}'.")
        mean_curves[label] = np.stack(curve_list, axis=0).mean(axis=0)
    return theta_template, mean_curves


class _ActivationAccumulator:
    def __init__(self, theta: np.ndarray):
        self.theta = np.array(theta, dtype=np.float64)
        self.storage: Dict[str, List[np.ndarray]] = {}

    def add(self, curves: Dict[str, np.ndarray]):
        for label, curve in curves.items():
            self.storage.setdefault(label, []).append(np.array(curve, dtype=np.float64))

    def stats(self) -> Dict[str, Tuple[np.ndarray, np.ndarray | None, int]]:
        stats: Dict[str, Tuple[np.ndarray, np.ndarray | None, int]] = {}
        for label, curves in self.storage.items():
            stacked = np.stack(curves, axis=0)
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0) if stacked.shape[0] > 1 else None
            stats[label] = (mean, std, stacked.shape[0])
        return stats


def _save_curves(
    output_dir: Path,
    *,
    dataset: str,
    activation: str,
    normalization: str,
    flip: bool,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray | None, int]],
    theta: np.ndarray,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    flip_suffix = "flip" if flip else "noflip"
    saved: List[Path] = []
    for label, (mean_curve, std_curve, _) in stats.items():
        name = f"{dataset}_{activation}_{normalization}_{flip_suffix}_{label}.npz".replace("/", "-")
        path = output_dir / name
        np.savez(
            path,
            theta=theta,
            mean=mean_curve,
            std=std_curve if std_curve is not None else np.array([]),
            metadata=dict(
                dataset=dataset,
                activation=activation,
                normalization=normalization,
                flip=flip,
                position=label,
            ),
        )
        saved.append(path)
    return saved


def _write_summary(
    summary_path: Path,
    rows: List[Dict[str, str | float | int | bool]],
):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "activation",
        "normalization",
        "flip",
        "position",
        "mean_error",
        "std_error",
        "num_seeds",
        "num_angles",
    ]
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def generate_activation_equivariance_data(
    models_dir: str,
    output_dir: str,
    *,
    summary_csv: str,
    num_angles: int,
    chunk_size: int,
    eval_batch_size: int,
    max_eval_batches: int,
    device: str,
    mnist_data_dir: str,
    positions: Sequence[str] | None,
    show_progress: bool,
) -> List[Path]:
    device_obj = _select_device(device)
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir_path}' does not exist.")

    entries = _collect_checkpoints(models_dir_path)
    if not entries:
        print(f"No checkpoint files found in {models_dir_path}; nothing to compute.")
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
    summary_path = Path(summary_csv)
    saved_files: List[Path] = []
    args_ns = SimpleNamespace(eval_batch_size=eval_batch_size, mnist_data_dir=mnist_data_dir)

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
        print(first_entry)
        activation = _get_hparam(lit_model.hparams, "activation_type", act_hint)
        normalization = _get_hparam(lit_model.hparams, "bn", norm_hint)
        flip_flag = bool(_get_hparam(lit_model.hparams, "flip", flip_hint))

        layer_map = _resolve_activation_targets(list(lit_model.model.eq_layers), positions)
        datamodule = _build_datamodule(dataset, lit_model.hparams, args_ns, first_entry["seed"])

        theta_rad, curves = _compute_activation_curves(
            lit_model,
            datamodule,
            device=device_obj,
            label_to_index=layer_map,
            num_angles=num_angles,
            chunk_size=chunk_size,
            max_eval_batches=max_eval_batches,
            show_progress=show_progress,
        )
        accumulator = _ActivationAccumulator(theta_rad)
        accumulator.add(curves)

        for entry in group_entries[1:]:
            checkpoint = torch.load(entry["path"], map_location=device_obj)
            state_dict = checkpoint.get("state_dict")
            if state_dict is None:
                raise KeyError(f"Checkpoint {entry['path']} missing 'state_dict'")
            lit_model.load_state_dict(state_dict)
            theta_rad, curves = _compute_activation_curves(
                lit_model,
                datamodule,
                device=device_obj,
                label_to_index=layer_map,
                num_angles=num_angles,
                chunk_size=chunk_size,
                max_eval_batches=max_eval_batches,
                show_progress=show_progress,
            )
            accumulator.add(curves)

        stats = accumulator.stats()
        saved_files.extend(
            _save_curves(
                output_dir_path,
                dataset=dataset,
                activation=activation,
                normalization=normalization,
                flip=flip_flag,
                stats=stats,
                theta=accumulator.theta,
            )
        )

        summary_rows: List[Dict[str, str | float | int | bool]] = []
        for label, (mean_curve, std_curve, n) in stats.items():
            summary_rows.append(
                {
                    "dataset": dataset,
                    "activation": activation,
                    "normalization": normalization,
                    "flip": flip_flag,
                    "position": label,
                    "mean_error": float(mean_curve.mean()),
                    "std_error": float(std_curve.mean()) if std_curve is not None else 0.0,
                    "num_seeds": int(n),
                    "num_angles": int(len(accumulator.theta)),
                }
            )
        _write_summary(summary_path, summary_rows)

    return saved_files


def main() -> None:
    args = _parse_args()
    generate_activation_equivariance_data(
        args.models_dir,
        args.output_dir,
        summary_csv=args.summary_csv,
        num_angles=args.num_angles,
        chunk_size=args.chunk_size,
        eval_batch_size=args.eval_batch_size,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
        mnist_data_dir=args.mnist_data_dir,
        positions=args.positions,
        show_progress=args.show_progress,
    )


if __name__ == "__main__":
    main()
