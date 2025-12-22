"""
Quickly print train/val/test split sizes for supported datasets.

Usage:
    python src/inspect_dataset_splits.py           # all datasets
    python src/inspect_dataset_splits.py --datasets mnist_rot eurosat
    python src/inspect_dataset_splits.py --train-fraction 0.25
"""
from __future__ import annotations

import argparse
import collections
from typing import Dict, Tuple, Any

from torch.utils.data import Subset
from datasets_utils.data_classes import (
    MnistRotDataModule,
    EuroSATDataModule,
    ColorectalHistDataModule,
    Resisc45DataModule,
    HFImageTorchDataset,
)


def get_distribution(dataset) -> Dict[int, int]:
    labels = None
    if isinstance(dataset, Subset):
        # For MnistRotDataModule train/val
        if hasattr(dataset.dataset, 'labels'):
             labels = dataset.dataset.labels[dataset.indices]
    elif hasattr(dataset, 'labels'):
        # For MnistRotDataModule test
        labels = dataset.labels
    elif isinstance(dataset, HFImageTorchDataset):
        # For HF datasets
        labels = dataset.ds['label']
    
    if labels is not None:
        return dict(sorted(collections.Counter(labels).items()))
    return {}


def mnist_counts(train_fraction: float) -> Dict[str, Any]:
    dm = MnistRotDataModule(train_fraction=train_fraction)
    dm.prepare_data()
    dm.setup()
    return {
        "train": {"count": len(dm.mnist_train), "dist": get_distribution(dm.mnist_train)},
        "val": {"count": len(dm.mnist_val), "dist": get_distribution(dm.mnist_val)},
        "test": {"count": len(dm.mnist_test), "dist": get_distribution(dm.mnist_test)},
    }


def eurosat_counts(train_fraction: float) -> Dict[str, Any]:
    dm = EuroSATDataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": {"count": len(dm.train_ds), "dist": get_distribution(dm.train_ds)},
        "val": {"count": len(dm.val_ds), "dist": get_distribution(dm.val_ds)},
        "test": {"count": len(dm.test_ds), "dist": get_distribution(dm.test_ds)},
    }


def colorectal_counts(train_fraction: float) -> Dict[str, Any]:
    dm = ColorectalHistDataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": {"count": len(dm.train_ds), "dist": get_distribution(dm.train_ds)},
        "val": {"count": len(dm.val_ds), "dist": get_distribution(dm.val_ds)},
        "test": {"count": len(dm.test_ds), "dist": get_distribution(dm.test_ds)},
    }


def resisc_counts(train_fraction: float) -> Dict[str, Any]:
    dm = Resisc45DataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": {"count": len(dm.train_ds), "dist": get_distribution(dm.train_ds)},
        "val": {"count": len(dm.val_ds), "dist": get_distribution(dm.val_ds)},
        "test": {"count": len(dm.test_ds), "dist": get_distribution(dm.test_ds)},
    }


COUNT_FNS: Dict[str, callable] = {
    "mnist_rot": mnist_counts,
    "eurosat": eurosat_counts,
    "colorectal_hist": colorectal_counts,
    "resisc45": resisc_counts,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print dataset split sizes.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(COUNT_FNS.keys()),
        default=list(COUNT_FNS.keys()),
        help="Datasets to inspect (default: all).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Train subset fraction to match your training configuration (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in args.datasets:
        print(f"--- {name} ---")
        counts = COUNT_FNS[name](args.train_fraction)
        
        global_dist = collections.Counter()
        total = 0
        for info in counts.values():
            global_dist.update(info['dist'])
            total += info['count']

        for split, info in counts.items():
            print(f"  {split}: count={info['count']}, dist={info['dist']}")
        print(f"  (total={total})")

        print("  Global Distribution:")
        for label in sorted(global_dist.keys()):
            count = global_dist[label]
            pct = (count / total) * 100 if total > 0 else 0
            print(f"    Class {label}: {count} ({pct:.2f}%)")
        print()


if __name__ == "__main__":
    main()
