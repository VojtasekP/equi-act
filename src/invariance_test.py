"""Compute invariance error for every checkpoint in a directory.

The script loads the target dataset once, then iterates over all `.ckpt` files
in the given folder, computes the invariance metric, and saves one CSV row per
checkpoint (no averaging across seeds).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute invariance per checkpoint.")
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing *.ckpt files to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tables/invariance_results.csv",
        help="Path where the per-checkpoint CSV will be written.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="CUDA device index to use for evaluation (default: 0).",
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
        default=1,
        help="Chunk size forwarded to the invariance metric helper.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=128,
        help="Target number of stratified test samples to evaluate per checkpoint.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Batch size used when running the metric on the sampled subset.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display tqdm progress bars while evaluating batches.",
    )
    return parser.parse_args()



def _get_hparam(hparams, key: str, default=None):
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


def _parse_checkpoint_filename(stem: str) -> Dict:
    """
    Best-effort parser for names like:
      equivariant_mnist_rot_seed0_aug_fourierbn_elu_16_Normbn
      equivariant_mnist_rot_seed1_noaug_normbnvec_relu
      resnet18_eurosat_seed0_aug
    """
    parts = stem.split("_")
    model_type = parts[0] if parts else "unknown"

    dataset = next((p for p in _DATASET_PREFIXES if p in parts), None)
    seed_part = next((p for p in parts if p.startswith("seed")), None)
    seed = int(seed_part.replace("seed", "")) if seed_part else None

    aug_flag: Optional[bool] = None
    if "aug" in parts:
        aug_flag = True
    if "noaug" in parts:
        aug_flag = False

    activation = None
    normalization = None
    if dataset:
        dataset_idx = parts.index(dataset)
        tail = [p for p in parts[dataset_idx + 1 :] if p not in {"aug", "noaug"} and not p.startswith("seed")]
        if tail:
            if tail[-1] in _BN_NAMES:
                normalization = tail[-1]
                activation_tail = tail[:-1]
            else:
                activation_tail = tail
            activation = "_".join(activation_tail) if activation_tail else None

    return {
        "model_type": model_type,
        "dataset": dataset,
        "activation_hint": activation,
        "normalization_hint": normalization,
        "seed": seed,
        "aug_hint": aug_flag,
        "path_stem": stem,
    }


def _collect_checkpoints(models_dir: Path) -> List[Dict]:
    entries: List[Dict] = []
    for ckpt_path in sorted(models_dir.glob("*.ckpt")):
        parsed = _parse_checkpoint_filename(ckpt_path.stem)
        parsed["path"] = ckpt_path
        entries.append(parsed)
    return entries


def _extract_metadata(entry: Dict) -> Dict:
    ckpt = torch.load(entry["path"], map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    dataset = _get_hparam(hparams, "dataset", entry.get("dataset"))
    return {
        "path": entry["path"],
        "checkpoint_name": entry["path"].name,
        "model_type": _get_hparam(hparams, "model_type", entry.get("model_type")),
        "dataset": dataset,
        "seed": _get_hparam(hparams, "seed", entry.get("seed")),
        "activation": _get_hparam(hparams, "activation_type", entry.get("activation_hint")),
        "bn": _get_hparam(hparams, "bn", entry.get("normalization_hint")),
        "aug": bool(_get_hparam(hparams, "aug", entry.get("aug_hint", False))),
    }


def _build_datamodule(dataset: str, batch_size):
    if dataset == "mnist_rot":
        return MnistRotDataModule(
            batch_size=batch_size,
            data_dir="./src/datasets_utils/mnist_rotation_new",
            img_size=29,
        )
    if dataset == "colorectal_hist":
        return ColorectalHistDataModule(batch_size=batch_size, img_size=150, aug=False, normalize=True)
    if dataset == "eurosat":
        return EuroSATDataModule(batch_size=batch_size, img_size=64, aug=False, normalize=True)
    if dataset == "resisc45":
        return Resisc45DataModule(batch_size=batch_size, img_size=256, aug=False, normalize=True)
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
    with torch.no_grad():
        batch_on_device = batch.to(device, non_blocking=True)
        lit_model.to(device)
        _, errs = em.chech_invariance_batch_r2(
            batch_on_device,
            lit_model.model,
            num_samples=num_angles,
            chunk_size=chunk_size,
        )
    return float(np.mean(errs))


def evaluate_checkpoints(
    models_dir: str | Path,
    output_csv: str | Path,
    *,
    num_angles: int = 16,
    chunk_size: int = 32,
    max_samples: int = 128,
    eval_batch_size: int = 64,
    cuda: int = 0,
):
    device_obj = torch.device(f"cuda:{cuda}") if torch.cuda.is_available() else torch.device("cpu")
    models_dir_path = Path(models_dir)
    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir_path}' does not exist.")

    entries = _collect_checkpoints(models_dir_path)
    if not entries:
        print(f"No checkpoint files found in {models_dir_path}; nothing to evaluate.")
        return None

    meta_entries = [_extract_metadata(e) for e in entries]
    dataset_name = meta_entries[0]["dataset"]
    inconsistent = [m["checkpoint_name"] for m in meta_entries if m["dataset"] != dataset_name]
    if inconsistent:
        print(f"[WARN] {len(inconsistent)} checkpoints report a different dataset than '{dataset_name}'; they will be skipped.")
        meta_entries = [m for m in meta_entries if m["dataset"] == dataset_name]

    if not meta_entries:
        print("No checkpoints left after filtering by dataset; nothing to evaluate.")
        return None

    datamodule = _build_datamodule(dataset_name, eval_batch_size)
    eval_batch = _prepare_eval_batch(datamodule, max_samples)

    rows: List[Dict[str, float | str | bool | int]] = []
    # Group checkpoints by architecture (dataset/model_type/activation/bn) so we
    # instantiate each model once, then just swap weights across seeds/aug.
    grouped: Dict[tuple, List[Dict]] = {}
    for meta in meta_entries:
        signature = (meta["dataset"], meta["model_type"], meta["activation"], meta["bn"])
        grouped.setdefault(signature, []).append(meta)

    group_iter = tqdm(grouped.items(), desc="Architectures")

    for signature, metas in group_iter:
        # Initialize once from the first checkpoint in the group.
        base_meta = metas[0]
        lit_model = LitHnn.load_from_checkpoint(base_meta["path"], map_location="cpu")
        if not hasattr(lit_model.model, "forward_features"):
            print(f"[WARN] Skipping group {signature}: model lacks equivariant features.")
            continue

        # Evaluate the first checkpoint (already loaded).
        inv_err = _compute_invariance(
            lit_model,
            eval_batch,
            device=device_obj,
            num_angles=num_angles,
            chunk_size=chunk_size,
        )
        rows.append(
            {
                "checkpoint": base_meta["checkpoint_name"],
                "model_type": base_meta["model_type"],
                "dataset": dataset_name,
                "seed": base_meta["seed"],
                "activation": base_meta["activation"],
                "bn": base_meta["bn"],
                "aug": base_meta["aug"],
                "invariance": inv_err,
            }
        )
        # Move off GPU before reloading weights for the next checkpoint.
        lit_model.to("cpu")
        torch.cuda.empty_cache()

        # Reuse the same model and just swap weights for the remaining checkpoints.
        for meta in metas[1:]:
            ckpt = torch.load(meta["path"], map_location="cpu")
            state_dict = ckpt.get("state_dict")
            if state_dict is None:
                print(f"[WARN] Skipping {meta['checkpoint_name']}: missing state_dict.")
                continue
            try:
                lit_model.load_state_dict(state_dict, strict=True)
            except Exception as exc:
                print(f"[WARN] Reloading model for {meta['checkpoint_name']} due to state_dict mismatch: {exc}")
                lit_model = LitHnn.load_from_checkpoint(meta["path"], map_location="cpu")

            inv_err = _compute_invariance(
                lit_model,
                eval_batch,
                device=device_obj,
                num_angles=num_angles,
                chunk_size=chunk_size,
                )
            rows.append(
                    {
                        "checkpoint": meta["checkpoint_name"],
                        "model_type": meta["model_type"],
                    "dataset": dataset_name,
                    "seed": meta["seed"],
                    "activation": meta["activation"],
                    "bn": meta["bn"],
                    "aug": meta["aug"],
                        "invariance": inv_err,
                    }
                )

            # Free GPU memory before the next checkpoint.
            lit_model.to("cpu")
            torch.cuda.empty_cache()

        # Drop reference and clear cache when this architecture group is done.
        del lit_model
        torch.cuda.empty_cache()

    if not rows:
        print("No invariance metrics produced; nothing to save.")
        return None

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "model_type", "dataset", "seed", "activation", "bn", "aug", "invariance"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved invariance metrics to {csv_path}")
    return csv_path


def main() -> None:
    args = _parse_args()
    evaluate_checkpoints(
        args.models_dir,
        args.output_csv,
        num_angles=args.num_angles,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        eval_batch_size=args.eval_batch_size,
        cuda=args.cuda,
    )


if __name__ == "__main__":
    main()
