"""
Plot a small grid of class examples for the supported datasets.

- Rotated MNIST: 10 classes (digits 0-9)
- EuroSAT RGB:   10 classes
- Colorectal Histology: 8 classes

Figures are saved to the output directory (defaults to ./plots).
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import ClassLabel, load_dataset

from datasets_utils.data_classes import MnistRotDataset
from datasets_utils.mnist_download import download_mnist_rotation


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "plots"


def to_numpy_image(img) -> np.ndarray:
    """Convert PIL or torch image to HWC numpy array for plotting."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # C,H,W -> H,W,C
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(img)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def plot_examples_grid(
    examples: list[tuple[str, object]],
    title: str,
    output_path: Path,
) -> None:
    """Save a 2-row grid of labeled images."""
    n = len(examples)
    n_rows = 2
    n_cols = math.ceil(n / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.2 * n_rows))
    axes = axes.flatten()

    for ax, (label_name, img) in zip(axes, examples):
        arr = to_numpy_image(img)
        cmap = "gray" if arr.ndim == 2 else None
        ax.imshow(arr, cmap=cmap)
        ax.set_title(label_name, fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def sample_one_per_class(
    ds: Iterable,
    num_classes: int,
    label_names: list[str] | None,
    label_key: str = "label",
    image_key: str = "image",
    seed: int | None = None,
) -> list[tuple[str, object]]:
    """
    Iterate the dataset once and collect the first example for each class.
    Order follows label id 0..num_classes-1.
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    found: dict[int, object] = {}
    for idx in indices:
        ex = ds[idx]
        label_id = int(ex[label_key]) if isinstance(ex, dict) else int(ex[1])
        img = ex[image_key] if isinstance(ex, dict) else ex[0]
        if label_id not in found:
            found[label_id] = img
            if len(found) == num_classes:
                break

    missing = [k for k in range(num_classes) if k not in found]
    if missing:
        raise RuntimeError(f"Did not find samples for classes: {missing}")

    names = label_names or [str(i) for i in range(num_classes)]
    return [(names[i], found[i]) for i in range(num_classes)]


def plot_mnist_examples(output_dir: Path, seed: int | None, data_dir: Path | None = None) -> Path:
    download_mnist_rotation(data_dir)
    mnist_ds = MnistRotDataset(mode="train", transform=None, data_dir=data_dir)
    examples = sample_one_per_class(mnist_ds, num_classes=10, label_names=None, seed=seed)
    out_path = output_dir / "mnist_rot_examples.png"
    plot_examples_grid(examples, "Rotated MNIST examples (one per class)", out_path)
    return out_path


def plot_eurosat_examples(output_dir: Path, seed: int | None) -> Path:
    ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    label_names = None
    if isinstance(ds.features.get("label"), ClassLabel):
        label_names = list(ds.features["label"].names)
    examples = sample_one_per_class(ds, num_classes=10, label_names=label_names, seed=seed)
    out_path = output_dir / "eurosat_examples.png"
    plot_examples_grid(examples, "EuroSAT RGB examples (one per class)", out_path)
    return out_path


def plot_colorectal_examples(output_dir: Path, seed: int | None) -> Path:
    ds = load_dataset("dpdl-benchmark/colorectal_histology", split="train")
    # Ensure labels are 0..7
    label_feature = ds.features.get("label")
    if not isinstance(label_feature, ClassLabel):
        ds = ds.cast_column("label", ClassLabel(num_classes=8))
        label_feature = ds.features["label"]
    label_names = [
        "tumour epithelium",
        "simple stroma",
        "complex stroma",
        "immune cell conglomerates",
        "debris and mucus",
        "mucosal glands",
        "adipose tissue",
        "background",
    ]

    examples = sample_one_per_class(ds, num_classes=8, label_names=label_names, seed=seed)
    out_path = output_dir / "colorectal_hist_examples.png"
    plot_examples_grid(examples, "Colorectal Histology examples (one per class)", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a grid of example images from each dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures will be saved (default: ./plots).",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=Path,
        default=None,
        help="Override download directory for rotated MNIST (optional).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling examples (omit for fresh random samples each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir

    paths = []
    paths.append(plot_mnist_examples(out_dir, args.seed, data_dir=args.mnist_data_dir))
    paths.append(plot_eurosat_examples(out_dir, args.seed))
    paths.append(plot_colorectal_examples(out_dir, args.seed))

    print("Saved figures:")
    for p in paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
