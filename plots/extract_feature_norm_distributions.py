#!/usr/bin/env python3
# extract_feature_norm_distributions.py
#
# Extract feature norm distributions from all layers across all checkpoints
# and save as histogram data (bins + counts) for later plotting.

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

from escnn import nn

# Add src to path
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
from train import LitHnn
from nets.new_layers import FourierPointwiseInnerBn, NormNonlinearityWithBN


_DATASET_PREFIXES = [
    "mnist_rot",
    "resisc45",
    "colorectal_hist",
    "eurosat",
]

_BN_NAMES = {
    "IIDbn",
    "Normbn"
}


# ---------- arg parsing ----------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract feature norm distributions from all layers and save as numpy arrays"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing *.ckpt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/norm_distributions",
        help="Directory where numpy arrays will be saved.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of stratified test images to use.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Use density normalization instead of raw counts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run models on.",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=str,
        default="./src/datasets_utils/mnist_rotation_new",
        help="Path to MNIST-rot dataset.",
    )
    return parser.parse_args()


# ---------- helpers from plot_activation_responses.py ----------

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
    """
    Parse checkpoint filename.

    Expected patterns:
    - equivariant_{dataset}_seed{N}_{aug|noaug}_{activation}_{bn}
    - {dataset}_{activation}_{bn}_seed{N}
    - {dataset}_{activation}_{bn}_seed{N}_flip
    """
    if "_seed" not in stem:
        raise ValueError("missing '_seed' segment in filename")

    # Split on _seed to get base and seed part
    base, seed_part = stem.rsplit("_seed", 1)

    # Extract seed number (first element after seed)
    seed_tokens = seed_part.split("_")
    try:
        seed = int(seed_tokens[0])
    except ValueError:
        raise ValueError(f"Cannot parse seed from: {seed_tokens[0]}")

    # Check for aug/noaug/flip indicators
    flip_hint = False
    activation_tokens = []

    for token in seed_tokens[1:]:
        if token == "aug":
            flip_hint = True
        elif token == "noaug":
            flip_hint = False
        elif token == "flip":
            flip_hint = True
        else:
            activation_tokens.append(token)

    # Find dataset prefix in base
    dataset = None
    for candidate in sorted(_DATASET_PREFIXES, key=len, reverse=True):
        if candidate in base:
            dataset = candidate
            break
    if dataset is None:
        raise ValueError("dataset prefix not recognized")

    # Extract activation and BN from tokens after seed
    if activation_tokens:
        # Pattern: seed{N}_{aug|noaug}_{activation}_{bn}
        full_rest = "_".join(activation_tokens)
    else:
        # Pattern: {dataset}_{activation}_{bn}_seed{N}
        # Extract part after dataset from base
        idx = base.index(dataset) + len(dataset)
        rest = base[idx:].lstrip("_")
        full_rest = rest

    # Remove trailing _flip if present
    if full_rest.endswith("_flip"):
        full_rest = full_rest[:-5]
        flip_hint = True

    # Extract batch norm and activation
    normalization = None
    activation = None
    for bn in _BN_NAMES:
        if full_rest.endswith(f"_{bn}"):
            normalization = bn
            activation = full_rest[:-(len(bn) + 1)]
            break
        elif full_rest.endswith(bn):
            normalization = bn
            activation = full_rest[:-len(bn)].rstrip("_")
            break

    if normalization is None:
        raise ValueError(f"unable to infer batchnorm from: {full_rest}")

    activation = activation.rstrip("_")
    if not activation:
        raise ValueError(f"activation missing in: {full_rest}")

    return {
        "dataset": dataset,
        "activation_hint": activation,
        "normalization_hint": normalization,
        "flip_hint": flip_hint,
        "seed": seed,
    }


def _collect_checkpoints(models_dir: Path) -> List[Dict]:
    entries: List[Dict] = []
    for ckpt_path in sorted(models_dir.rglob("*.ckpt")):
        try:
            parsed = _parse_checkpoint_filename(ckpt_path.stem)
        except ValueError as exc:
            print(f"[WARN] Skipping {ckpt_path.name}: {exc}")
            continue
        parsed["path"] = ckpt_path
        entries.append(parsed)
    return entries


def _build_datamodule(dataset: str, hparams, args: SimpleNamespace):
    img_size = int(_get_hparam(hparams, "img_size", 64))
    batch_size = 16  # Fixed for extraction

    if dataset == "mnist_rot":
        return MnistRotDataModule(
            batch_size=batch_size, data_dir=args.mnist_data_dir, img_size=img_size
        )
    if dataset == "resisc45":
        return Resisc45DataModule(batch_size=batch_size, img_size=img_size)
    if dataset == "colorectal_hist":
        return ColorectalHistDataModule(
            batch_size=batch_size, img_size=img_size, train_fraction=1.0
        )
    if dataset == "eurosat":
        return EuroSATDataModule(
            batch_size=batch_size, img_size=img_size, train_fraction=1.0
        )

    raise ValueError(f"Unsupported dataset '{dataset}'.")


_ACTIVATION_LAYER_TYPES = (
    nn.FourierPointwise,
    nn.NormNonLinearity,
    nn.GatedNonLinearity1,
    nn.GatedNonLinearity2,
    FourierPointwiseInnerBn,
    NormNonlinearityWithBN,
)


def _is_activation_layer(layer: torch.nn.Module) -> bool:
    if isinstance(layer, _ACTIVATION_LAYER_TYPES):
        return True
    if isinstance(layer, nn.MultipleModule):
        for sub in layer.modules():
            if sub is layer:
                continue
            if isinstance(sub, _ACTIVATION_LAYER_TYPES):
                return True
    # Also check SequentialModule for nested activations
    if isinstance(layer, nn.SequentialModule):
        for sub in layer.modules():
            if sub is layer:
                continue
            if isinstance(sub, _ACTIVATION_LAYER_TYPES):
                return True
    return False


def _find_activation_layer(layers: List[torch.nn.Module], desired_index: int) -> int:
    activation_indices = [i for i, layer in enumerate(layers) if _is_activation_layer(layer)]
    if not activation_indices:
        raise ValueError("No activation modules found in model.eq_layers.")

    if desired_index == 0:
        raise ValueError("layer_index uses 1-based indexing; zero is not allowed.")

    if desired_index > 0:
        resolved = desired_index - 1
    else:
        resolved = len(activation_indices) + desired_index

    if resolved < 0 or resolved >= len(activation_indices):
        raise IndexError(
            f"layer_index {desired_index} out of range for {len(activation_indices)} activations."
        )
    return activation_indices[resolved]


def _forward_to_layer(model, x: torch.Tensor, layer_index: int):
    """Extract GeometricTensor from specific activation layer."""
    if not isinstance(x, nn.GeometricTensor):
        x = nn.GeometricTensor(x, model.input_type)

    if hasattr(model, "mask"):
        x = model.mask(x)

    if not hasattr(model, "eq_layers"):
        raise AttributeError("Model has no attribute 'eq_layers'.")

    layers = list(model.eq_layers)
    target_idx = _find_activation_layer(layers, layer_index)

    for i, layer in enumerate(layers):
        x = layer(x)
        if i == target_idx:
            break
    return x


# ---------- stratified sampling ----------

def _stratified_sample_indices(
    dataset: torch.utils.data.Dataset,
    num_samples: int,
    seed: int = 42
) -> List[int]:
    """Sample indices with equal representation from each class."""
    # Extract all labels
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    elif hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        # Fallback: iterate dataset
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label if isinstance(label, int) else label.item())
        labels = np.array(labels)

    # Clamp num_samples to dataset size
    if num_samples > len(labels):
        print(f"[WARN] num_samples ({num_samples}) > dataset size ({len(labels)}), using all data")
        num_samples = len(labels)

    # Group indices by class
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    indices_by_class = {c: np.where(labels == c)[0] for c in unique_classes}

    # Stratified sampling
    rng = np.random.default_rng(seed)
    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    selected = []
    for cls in unique_classes:
        cls_indices = indices_by_class[cls]
        n_samples = min(samples_per_class, len(cls_indices))
        selected.extend(rng.choice(cls_indices, size=n_samples, replace=False))

    # Handle remainder
    if remainder > 0 and len(selected) < num_samples:
        all_indices = np.arange(len(labels))
        remaining = np.setdiff1d(all_indices, selected)
        if len(remaining) > 0:
            extra = rng.choice(remaining, size=min(remainder, len(remaining)), replace=False)
            selected.extend(extra)

    return list(selected)


# ---------- irrep frequency identification ----------

def _infer_irrep_frequency(rep, rep_index: int, gspace) -> int:
    """
    Determine irrep frequency from representation object.

    Strategy:
    1. Check if trivial (size==1)
    2. Parse from rep.name (e.g., "irrep_1" -> 1)
    3. Match against gspace.irreps
    4. Fallback: position-based heuristic
    """
    # Check if trivial
    if rep.size == 1:
        try:
            if rep.is_trivial():
                return 0
        except:
            return 0

    # Try parsing name
    if hasattr(rep, 'name'):
        name = rep.name
        if 'irrep_' in name:
            try:
                freq = int(name.split('irrep_')[1].split('_')[0])
                return freq
            except (ValueError, IndexError):
                pass

    # Fallback: match against gspace.irreps
    if hasattr(gspace, 'irreps'):
        for m, irrep in enumerate(gspace.irreps):
            if rep == irrep:
                return m

    # Last resort
    print(f"[WARN] Could not determine frequency for rep {rep_index}, using fallback")
    return 0 if rep.size == 1 else (rep_index % 4)


# ---------- norm computation ----------

def _compute_norms_by_irrep(
    features: nn.GeometricTensor,
    model
) -> Dict[int, np.ndarray]:
    """
    Compute norms for each field, grouped by irrep frequency.

    Returns dict mapping frequency -> array of all norm values.
    """
    field_type = features.type
    tensor = features.tensor  # [B, C, H, W]

    if not hasattr(model, 'r2_act'):
        raise AttributeError("Model missing r2_act attribute")

    gspace = model.r2_act
    norms_by_freq = {}

    channel_offset = 0
    for rep_idx, rep in enumerate(field_type.representations):
        size = rep.size
        freq = _infer_irrep_frequency(rep, rep_idx, gspace)

        # Extract channels for this field
        channels = tensor[:, channel_offset:channel_offset+size, :, :]  # [B, size, H, W]

        # Compute norms
        if size == 1:
            # Scalar: absolute value
            norms = channels.abs().flatten().cpu().numpy()
        elif size == 2:
            # Vector: L2 norm
            ch1 = channels[:, 0, :, :]
            ch2 = channels[:, 1, :, :]
            norms = torch.sqrt(ch1**2 + ch2**2).flatten().cpu().numpy()
        else:
            # General case
            norms = torch.linalg.vector_norm(channels, dim=1).flatten().cpu().numpy()

        # Accumulate by frequency
        if freq not in norms_by_freq:
            norms_by_freq[freq] = []
        norms_by_freq[freq].append(norms)

        channel_offset += size

    # Concatenate all norms for each frequency
    norms_by_freq = {freq: np.concatenate(vals) for freq, vals in norms_by_freq.items()}
    return norms_by_freq


# ---------- histogram computation ----------

def _compute_histogram(
    norms: np.ndarray,
    num_bins: int,
    density: bool,
    bin_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram for norm values.

    Returns (bin_edges, counts)
    """
    if len(norms) == 0:
        return np.array([]), np.array([])

    counts, bin_edges = np.histogram(norms, bins=num_bins, range=bin_range, density=density)
    return bin_edges, counts


# ---------- main extraction logic ----------

def _extract_all_layers_with_loader(
    lit_model: LitHnn,
    sampled_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_bins: int,
    density: bool,
) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Extract norm distributions from all activation layers using pre-made dataloader.

    Returns:
        {layer_idx: {freq: (bin_edges, counts)}}
    """
    lit_model.eval()
    lit_model.model.eval()

    # Find all activation layers
    if not hasattr(lit_model.model, 'eq_layers'):
        print(f"  [ERROR] Model has no 'eq_layers' attribute!")
        return {}

    layers = list(lit_model.model.eq_layers)
    activation_indices = [i for i, layer in enumerate(layers) if _is_activation_layer(layer)]
    num_activations = len(activation_indices)

    print(f"  Found {num_activations} activation layers")

    results = {}

    # Process each activation layer
    for act_idx in range(1, num_activations + 1):
        print(f"  Processing layer {act_idx}/{num_activations}...")

        # Extract features
        features_list = []
        with torch.no_grad():
            for batch in sampled_loader:
                x, _ = batch
                x = x.to(device)
                feat = _forward_to_layer(lit_model.model, x, act_idx)
                features_list.append(feat)

        # Concatenate
        tensors = [f.tensor for f in features_list]
        combined_tensor = torch.cat(tensors, dim=0)
        combined_geo = nn.GeometricTensor(combined_tensor, features_list[0].type)

        # Compute norms by irrep
        norms_by_freq = _compute_norms_by_irrep(combined_geo, lit_model.model)

        # Compute histograms
        layer_histograms = {}
        for freq, norms in norms_by_freq.items():
            bin_edges, counts = _compute_histogram(norms, num_bins, density)
            layer_histograms[freq] = (bin_edges, counts)

        results[act_idx] = layer_histograms

    return results


def _aggregate_across_seeds(
    results_by_seed: List[Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]]
) -> Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]:
    """
    Average histogram counts across seeds.

    Assumes all seeds have same bin edges (computed on same data range).

    Returns:
        {layer_idx: {freq: (bin_edges, avg_counts)}}
    """
    if len(results_by_seed) == 0:
        return {}

    if len(results_by_seed) == 1:
        return results_by_seed[0]

    # Use first seed as template
    template = results_by_seed[0]
    aggregated = {}

    for layer_idx in template.keys():
        aggregated[layer_idx] = {}
        for freq in template[layer_idx].keys():
            # Collect counts from all seeds
            bin_edges = template[layer_idx][freq][0]
            counts_list = []

            for seed_results in results_by_seed:
                if layer_idx in seed_results and freq in seed_results[layer_idx]:
                    _, counts = seed_results[layer_idx][freq]
                    counts_list.append(counts)

            if counts_list:
                avg_counts = np.mean(counts_list, axis=0)
                aggregated[layer_idx][freq] = (bin_edges, avg_counts)

    return aggregated


def _save_results(
    results: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
    metadata: Dict
):
    """
    Save results to .npz file.

    File structure:
        metadata: JSON string with config info
        layer_{i}_freq_{f}_bins: bin edges
        layer_{i}_freq_{f}_counts: histogram counts
    """
    save_dict = {
        'metadata': str(metadata)
    }

    for layer_idx, layer_data in results.items():
        for freq, (bin_edges, counts) in layer_data.items():
            save_dict[f'layer_{layer_idx}_freq_{freq}_bins'] = bin_edges
            save_dict[f'layer_{layer_idx}_freq_{freq}_counts'] = counts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_dict)
    print(f"  Saved to {output_path}")


def main():
    args = _parse_args()

    device_obj = _select_device(args.device)
    models_dir_path = Path(args.models_dir)

    if not models_dir_path.exists():
        raise FileNotFoundError(f"Models directory '{models_dir_path}' does not exist.")

    entries = _collect_checkpoints(models_dir_path)
    if not entries:
        print(f"No checkpoint files found in {models_dir_path}")
        return

    # Group by dataset first to load data once per dataset
    entries.sort(
        key=lambda e: (
            e["dataset"],
            e["activation_hint"],
            e["normalization_hint"],
            bool(e.get("flip_hint", False)),
            e["seed"],
        )
    )

    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    args_ns = SimpleNamespace(
        eval_batch_size=16,
        mnist_data_dir=args.mnist_data_dir
    )

    # Group by dataset to load data once
    for dataset, dataset_entries in itertools.groupby(entries, key=lambda e: e["dataset"]):
        dataset_entries = list(dataset_entries)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Load data ONCE for this dataset
        print(f"Loading test data for {dataset}...")
        first_entry = dataset_entries[0]
        temp_lit_model = LitHnn.load_from_checkpoint(first_entry["path"], map_location="cpu")
        datamodule = _build_datamodule(dataset, temp_lit_model.hparams, args_ns)

        # Prepare data
        if hasattr(datamodule, "prepare_data"):
            try:
                datamodule.prepare_data()
            except Exception:
                pass
        datamodule.setup("test")
        test_dataset = datamodule.test_dataloader().dataset

        # Handle Subset wrapper
        if isinstance(test_dataset, torch.utils.data.Subset):
            base_dataset = test_dataset.dataset
        else:
            base_dataset = test_dataset

        # Stratified sampling - SAME indices for all models in this dataset
        print(f"Sampling {args.num_samples} stratified test images...")
        sampled_indices = _stratified_sample_indices(base_dataset, args.num_samples, seed=42)

        # Create subset dataloader - REUSED for all models
        subset = torch.utils.data.Subset(base_dataset, sampled_indices)
        sampled_loader = torch.utils.data.DataLoader(
            subset,
            batch_size=16,
            shuffle=False,
            num_workers=0
        )

        print(f"Data loaded: {len(sampled_indices)} samples\n")
        del temp_lit_model  # Free memory

        # Now process each configuration within this dataset
        for (dataset_name, act_hint, norm_hint, flip_hint), group_iter in itertools.groupby(
            dataset_entries,
            key=lambda e: (
                e["dataset"],
                e["activation_hint"],
                e["normalization_hint"],
                bool(e.get("flip_hint", False)),
            ),
        ):
            group_entries = list(group_iter)
            print(f"\nProcessing: {dataset_name} | {act_hint} | {norm_hint} | flip={flip_hint}")
            print(f"  Seeds: {[e['seed'] for e in group_entries]}")

            # Load model architecture once from first checkpoint
            first_entry = group_entries[0]
            print(f"  Loading model architecture from seed {first_entry['seed']}...")
            lit_model = LitHnn.load_from_checkpoint(first_entry["path"], map_location="cpu")
            activation = _get_hparam(lit_model.hparams, "activation_type", act_hint)
            normalization = _get_hparam(lit_model.hparams, "bn", norm_hint)

            # Extract for each seed (load architecture once, swap weights)
            results_by_seed = []
            for idx, entry in enumerate(group_entries):
                print(f"  Processing seed {entry['seed']} ({idx+1}/{len(group_entries)})...")

                # For subsequent seeds, swap weights instead of reloading
                if idx > 0:
                    # Move to CPU before loading state dict
                    lit_model.to("cpu")
                    torch.cuda.empty_cache()

                    print(f"    Loading weights from checkpoint...")
                    ckpt = torch.load(entry["path"], map_location="cpu")
                    state_dict = ckpt.get("state_dict")
                    if state_dict is None:
                        print(f"    [WARN] Missing state_dict, skipping seed {entry['seed']}")
                        continue
                    try:
                        lit_model.load_state_dict(state_dict, strict=True)
                    except Exception as exc:
                        print(f"    [WARN] State dict mismatch, reloading full model: {exc}")
                        lit_model = LitHnn.load_from_checkpoint(entry["path"], map_location="cpu")

                # Move to device and set eval mode
                lit_model.eval()
                lit_model.to(device_obj)

                # Extract features using the SHARED dataloader
                seed_results = _extract_all_layers_with_loader(
                    lit_model,
                    sampled_loader,
                    device_obj,
                    args.num_bins,
                    args.density,
                )

                # Debug: check what was extracted
                if seed_results:
                    n_layers = len(seed_results)
                    n_freqs = sum(len(layer_data) for layer_data in seed_results.values())
                    print(f"    Extracted: {n_layers} layers, {n_freqs} freq entries")
                else:
                    print(f"    [WARN] No data extracted for seed {entry['seed']}")

                results_by_seed.append(seed_results)

                # Move back to CPU and clear cache
                lit_model.to("cpu")
                torch.cuda.empty_cache()

            # Aggregate across seeds for this configuration
            print(f"  Aggregating across seeds... ({len(results_by_seed)} seeds processed)")

            if len(results_by_seed) == 0:
                print("  [ERROR] No results collected from any seed! Skipping this configuration.")
                continue

            aggregated = _aggregate_across_seeds(results_by_seed)

            # Debug: check if aggregated has data
            total_entries = sum(len(layer_data) for layer_data in aggregated.values())
            print(f"  Aggregated data: {len(aggregated)} layers, {total_entries} total freq entries")

            if total_entries == 0:
                print("  [ERROR] Aggregation produced no data! Skipping save.")
                continue

            # Save - filename does NOT include flip (we don't apply augmentation during extraction)
            filename = f"{dataset_name}_{act_hint}_{norm_hint}.npz"
            output_path = output_dir_path / filename

            metadata = {
                'dataset': dataset_name,
                'activation': act_hint,
                'normalization': norm_hint,
                'num_samples': args.num_samples,
                'num_bins': args.num_bins,
                'density': args.density,
                'seeds': [e['seed'] for e in group_entries]
            }

            _save_results(aggregated, output_path, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
