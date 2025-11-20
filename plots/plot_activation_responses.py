# plot_activation_responses.py
#
# Plot angular response of activations:
# for a few channels at a chosen equivariant layer, show how the
# activation magnitude changes as we rotate the input.

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from escnn import nn

from datasets_utils.data_classes import (
    ColorectalHistDataModule,
    EuroSATDataModule,
    MnistRotDataModule,
    Resisc45DataModule,
)
from train import LitHnn


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
        description="Plot angular activation responses per activation/batchnorm"
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
        default="plots/activation_responses",
        help="Directory where the generated plots will be written.",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=32,
        help="Number of rotation angles over [0, 2pi).",
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
        default=1,
        help="How many batches per checkpoint to average (keep this small).",
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
        "--layer-index",
        type=int,
        default=3,
        help=(
            "1-based index of the activation module to probe (-1 = last activation, "
            "-2 = second-to-last, etc.)."
        ),
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Channel indices to plot in the probed layer. "
            "When --field-index is provided these are relative to that field."
        ),
    )
    parser.add_argument(
        "--field-index",
        type=int,
        default=None,
        help=(
            "Optional 0-based field/irrep index in the probed layer. "
            "All plotted channels must belong to this field."
        ),
    )
    parser.add_argument(
        "--channel-irrep-labels",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional list assigning an irrep label to every channel in the probed layer. "
            "Provide one integer per channel (e.g., 0-3) corresponding to your irrep IDs."
        ),
    )
    parser.add_argument(
        "--irrep-filter",
        type=int,
        default=None,
        help=(
            "When set, restricts plotting to channels whose irrep label equals this value. "
            "Requires --channel-irrep-labels."
        ),
    )
    parser.add_argument(
        "--auto-channel-count",
        type=int,
        default=4,
        help=(
            "When auto-selecting channels (no --channels specified), pick up to this many "
            "from the filtered set."
        ),
    )
    parser.add_argument(
        "--spatial-aggregation",
        type=str,
        default="mean",
        choices=["center", "mean"],
        help="How to aggregate over spatial dims: 'center' pixel or spatial mean.",
    )
    return parser.parse_args()


# ---------- helpers ported from your equivariance script ----------

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
        return ColorectalHistDataModule(
            batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0
        )
    if dataset == "eurosat":
        return EuroSATDataModule(
            batch_size=batch_size, img_size=img_size, seed=seed, train_fraction=1.0
        )

    raise ValueError(f"Unsupported dataset '{dataset}'.")


# ---------- NEW: activation response machinery ----------

def _rotate_batch(x: torch.Tensor, model, theta: float) -> torch.Tensor:
    """
    Rotate input batch by angle theta using the model's escnn gspace.

    x: [B, C, H, W] plain tensor
    returns: rotated tensor with same shape
    """
    if not hasattr(model, "r2_act") or not hasattr(model, "input_type"):
        raise AttributeError("Model is missing r2_act or input_type; cannot use escnn transforms.")

    gspace = model.r2_act
    # For pure rotations, gspace.element(theta) usually works.
    # If you're using flip groups, you can extend this to also test flips.
    if hasattr(gspace, "element"):
        g = gspace.element(theta)
    elif hasattr(gspace, "fibergroup"):
        # fallback: assume first component is rotation subgroup
        g = gspace.fibergroup.element(theta)
    else:
        raise RuntimeError("Unexpected gspace structure; adjust _rotate_batch manually.")

    geo = nn.GeometricTensor(x, model.input_type)
    geo_rot = geo.transform(g)
    return geo_rot.tensor

_ACTIVATION_LAYER_TYPES = (
    nn.FourierPointwise,
    nn.NormNonLinearity,
    nn.GatedNonLinearity1,
    nn.GatedNonLinearity2,
)


def _is_activation_layer(layer: torch.nn.Module) -> bool:
    if isinstance(layer, _ACTIVATION_LAYER_TYPES):
        return True
    if isinstance(layer, nn.MultipleModule):
        # MultipleModule wraps several submodules; check if any is an activation.
        for sub in layer.modules():
            if sub is layer:
                continue
            if isinstance(sub, _ACTIVATION_LAYER_TYPES):
                return True
    return False


def _find_activation_layer(layers: List[torch.nn.Module], desired_index: int) -> int:
    """
    Resolve a user-provided activation index (1-based, negatives allowed) into an eq_layers index.
    """
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
    """
    Run equivariant part up to the requested activation module and return the GeometricTensor.
    """
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


def _aggregate_activation(
    geo: nn.GeometricTensor,
    channel_idx: int,
    spatial_aggregation: str = "center",
) -> float:
    """
    Take one GeometricTensor, pick one channel, and aggregate over batch+space.

    Returns a scalar (float) = average |activation|.
    """
    t = geo.tensor  # [B, C, H, W] or [B, C, H, W, ...] – assume 2D here
    if t.dim() != 4:
        raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {tuple(t.shape)}")

    if channel_idx < 0 or channel_idx >= t.shape[1]:
        raise IndexError(f"channel_idx {channel_idx} out of range for C={t.shape[1]}")

    if spatial_aggregation == "center":
        h = t.shape[2] // 2
        w = t.shape[3] // 2
        vals = t[:, channel_idx, h, w]  # [B]
    elif spatial_aggregation == "mean":
        vals = t[:, channel_idx].mean(dim=(-2, -1))  # [B]
    else:
        raise ValueError(f"Unknown spatial_aggregation: {spatial_aggregation}")

    return vals.abs().mean().item()


def _enumerate_field_slices(field_type: nn.FieldType) -> List[Dict]:

    slices: List[Dict] = []
    offset = 0
    for idx, rep in enumerate(field_type.representations):
        size = getattr(rep, "size", None)
        if size is None:
            raise AttributeError("Representation missing 'size' attribute.")
        name = getattr(rep, "name", f"field_{idx}")
        slices.append(
            {
                "index": idx,
                "start": offset,
                "end": offset + size,
                "size": size,
                "name": name,
            }
        )
        offset += size
    return slices


def _resolve_channels_for_field(
    field_type: nn.FieldType,
    requested_channels: List[int],
    field_index: int | None,
) -> Tuple[List[int], Dict | None]:
    """
    Map requested channels to absolute indices, enforcing a specific field when asked.
    """
    if field_index is None:
        return list(requested_channels), None

    slices = _enumerate_field_slices(field_type)
    if field_index < 0 or field_index >= len(slices):
        raise ValueError(
            f"field_index {field_index} out of range for probed layer ({len(slices)} fields)."
        )

    field_meta = slices[field_index]
    field_size = field_meta["size"]
    if not requested_channels:
        requested_channels = list(range(field_size))

    invalid = [c for c in requested_channels if c < 0 or c >= field_size]
    if invalid:
        raise ValueError(
            f"Channels {invalid} fall outside field '{field_meta['name']}' of size {field_size}."
        )

    absolute = [field_meta["start"] + rel_idx for rel_idx in requested_channels]
    return absolute, field_meta


def _select_channels_for_plot(
    feat: nn.GeometricTensor,
    requested_channels: List[int],
    *,
    field_index: int | None,
    channel_irrep_labels: List[int] | None,
    irrep_filter: int | None,
    auto_select: bool,
    auto_channel_count: int,
) -> Tuple[List[int], Dict | None]:
    """
    Combine field filtering, optional irrep labels, and auto selection into a final channel list.
    """
    total_channels = feat.tensor.shape[1]
    base_request = list(requested_channels)

    if field_index is not None:
        base_channels, field_meta = _resolve_channels_for_field(
            feat.type, base_request, field_index
        )
    else:
        if base_request:
            base_channels = base_request
        else:
            base_channels = list(range(total_channels))
        field_meta = None

    if channel_irrep_labels is not None:
        if len(channel_irrep_labels) != total_channels:
            raise ValueError(
                f"Expected {total_channels} irrep labels, got {len(channel_irrep_labels)}."
            )
        if irrep_filter is None:
            raise ValueError("--irrep-filter must be provided when using --channel-irrep-labels.")
        filtered = [ch for ch in base_channels if channel_irrep_labels[ch] == irrep_filter]
    else:
        if irrep_filter is not None:
            raise ValueError("--irrep-filter requires --channel-irrep-labels.")
        filtered = base_channels

    if not filtered:
        raise ValueError("No channels remain after applying field/irrep filters.")

    if auto_select:
        if auto_channel_count <= 0:
            raise ValueError("--auto-channel-count must be positive.")
        filtered = filtered[: min(len(filtered), auto_channel_count)]

    return filtered, field_meta


def _compute_activation_responses(
    lit_model: LitHnn,
    datamodule,
    device: torch.device,
    num_angles: int,
    layer_index: int,
    channels: List[int],
    *,
    channel_irrep_labels: List[int] | None,
    irrep_filter: int | None,
    auto_select: bool,
    auto_channel_count: int,
    field_index: int | None,
    max_batches: int,
    spatial_agg: str,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    For given model, return:

      thetas: shape [num_angles] in radians
      responses: shape [len(resolved_channels), num_angles]
        responses[i, j] = mean |activation(channel_i)| at angle_j,
        averaged over max_batches batches.
      resolved_channels: absolute indices actually used for plotting.
    """
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

    max_batches = int(max_batches)
    if max_batches <= 0:
        max_batches = 1  # be conservative

    # torch.linspace does not support endpoint=False in older versions, so
    # generate one extra sample (2pi inclusive) and drop it to mimic [0, 2pi).
    thetas = torch.linspace(0.0, 2.0 * np.pi, steps=num_angles + 1)[:-1]
    all_responses = []  # list of [len(resolved_channels), num_angles]
    resolved_channels: List[int] | None = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, _ = batch
            x = x.to(device)

            batch_resp = None

            for ang_idx, theta in enumerate(thetas):
                x_rot = _rotate_batch(x, lit_model.model, float(theta))
                feat = _forward_to_layer(lit_model.model, x_rot, layer_index)

                if resolved_channels is None:
                    resolved_channels, _ = _select_channels_for_plot(
                        feat,
                        channels,
                        field_index=field_index,
                        channel_irrep_labels=channel_irrep_labels,
                        irrep_filter=irrep_filter,
                        auto_select=auto_select,
                        auto_channel_count=auto_channel_count,
                    )
                    if not resolved_channels:
                        raise ValueError("No channels selected for plotting.")

                if batch_resp is None:
                    batch_resp = np.zeros((len(resolved_channels), num_angles), dtype=np.float64)

                for ch_i, ch in enumerate(resolved_channels):
                    val = _aggregate_activation(feat, ch, spatial_aggregation=spatial_agg)
                    batch_resp[ch_i, ang_idx] = val

            all_responses.append(batch_resp)

            if (batch_idx + 1) >= max_batches:
                break

    if resolved_channels is None:
        raise RuntimeError("Failed to resolve channel indices; no batches were processed.")

    responses = np.stack(all_responses, axis=0).mean(axis=0)  # [len(resolved_channels), num_angles]
    return thetas.cpu().numpy(), responses, resolved_channels


# ---------- plotting & aggregation ----------

def _save_activation_plot(
    theta_rad: np.ndarray,
    responses: np.ndarray,  # [num_channels, num_angles]
    channels: List[int],
    output_path: Path,
    dataset: str,
    activation: str,
    normalization: str,
    layer_label: str,
) -> None:
    plt.figure(figsize=(7, 4))

    x = theta_rad / np.pi  # plot in units of pi
    for ch_i, ch in enumerate(channels):
        plt.plot(x, responses[ch_i], label=f"ch {ch}")

    plt.xlabel(r"Rotation angle ($\pi$ units)")
    plt.ylabel(r"mean $|a_{c}(x_\theta)|$")
    plt.title(f"Activation response ({layer_label}): {dataset} | {activation} / {normalization}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved activation response plot to {output_path}")


def generate_activation_response_plots(
    models_dir: str | Path,
    output_dir: str | Path,
    *,
    num_angles: int = 32,
    eval_batch_size: int = 8,
    max_eval_batches: int = 1,
    device: str = "auto",
    mnist_data_dir: str = "./src/datasets_utils/mnist_rotation_new",
    layer_index: int = -1,
    channels: List[int] | None = None,
    field_index: int | None = None,
    channel_irrep_labels: List[int] | None = None,
    irrep_filter: int | None = None,
    auto_channel_count: int = 4,
    spatial_aggregation: str = "center",
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

    if channels is None:
        if field_index is not None or channel_irrep_labels is not None:
            channels = []
        else:
            channels = [0, 1, 2, 3]
    auto_select_channels = len(channels) == 0

    saved_plots: List[Path] = []
    args_ns = SimpleNamespace(eval_batch_size=eval_batch_size, mnist_data_dir=mnist_data_dir)
    layer_label = f"eq_layers[{layer_index}]" if layer_index >= 0 else "last equivariant layer"

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
            f"Processing group: dataset={dataset}, activation={act_hint}, "
            f"bn={norm_hint}, flip={flip_hint}"
        )

        lit_model = LitHnn.load_from_checkpoint(first_entry["path"], map_location=device_obj)
        activation = _get_hparam(lit_model.hparams, "activation_type", act_hint)
        normalization = _get_hparam(lit_model.hparams, "bn", norm_hint)

        datamodule = _build_datamodule(dataset, lit_model.hparams, args_ns, first_entry["seed"])

        theta_rad, responses, resolved_channels = _compute_activation_responses(
            lit_model,
            datamodule,
            device=device_obj,
            num_angles=num_angles,
            layer_index=layer_index,
            channels=channels,
            channel_irrep_labels=channel_irrep_labels,
            irrep_filter=irrep_filter,
            auto_select=auto_select_channels,
            auto_channel_count=auto_channel_count,
            field_index=field_index,
            max_batches=max_eval_batches,
            spatial_agg=spatial_aggregation,
        )
        if field_index is not None:
            print(
                f"  -> field_index={field_index}, relative channels={channels}, "
                f"absolute channels={resolved_channels}"
            )

        flip_suffix = "flip" if flip_hint else "noflip"
        layer_suffix = layer_label.replace(" ", "").replace("[", "_").replace("]", "_")
        field_suffix = f"field{field_index}_" if field_index is not None else ""
        chan_suffix = f"ch{'-'.join(str(c) for c in resolved_channels)}"
        safe_name = (
            f"{dataset}_{activation}_{normalization}_{flip_suffix}_"
            f"{field_suffix}{layer_suffix}_{chan_suffix}"
            .replace("/", "-")
        )
        out_path = output_dir_path / f"{safe_name}.png"

        _save_activation_plot(
            theta_rad,
            responses,
            resolved_channels,
            out_path,
            dataset,
            activation,
            normalization,
            layer_label,
        )
        saved_plots.append(out_path)

    return saved_plots


def main() -> None:
    args = _parse_args()
    generate_activation_response_plots(
        args.models_dir,
        args.output_dir,
        num_angles=args.num_angles,
        eval_batch_size=args.eval_batch_size,
        max_eval_batches=args.max_eval_batches,
        device=args.device,
        mnist_data_dir=args.mnist_data_dir,
        layer_index=args.layer_index,
        channels=args.channels,
        channel_irrep_labels=args.channel_irrep_labels,
        irrep_filter=args.irrep_filter,
        auto_channel_count=args.auto_channel_count,
        field_index=args.field_index,
        spatial_aggregation=args.spatial_aggregation,
    )


if __name__ == "__main__":
    main()
