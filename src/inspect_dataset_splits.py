"""
Quickly print train/val/test split sizes for supported datasets.

Usage:
    python src/inspect_dataset_splits.py           # all datasets
    python src/inspect_dataset_splits.py --datasets mnist_rot eurosat
    python src/inspect_dataset_splits.py --train-fraction 0.25
"""
from __future__ import annotations

import argparse
from typing import Dict, Tuple

from datasets_utils.data_classes import (
    MnistRotDataModule,
    EuroSATDataModule,
    ColorectalHistDataModule,
    Resisc45DataModule,
)


def mnist_counts(train_fraction: float) -> Dict[str, int]:
    dm = MnistRotDataModule(train_fraction=train_fraction)
    dm.prepare_data()
    dm.setup()
    return {
        "train": len(dm.mnist_train),
        "val": len(dm.mnist_val),
        "test": len(dm.mnist_test),
    }


def eurosat_counts(train_fraction: float) -> Dict[str, int]:
    dm = EuroSATDataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": len(dm.train_ds),
        "val": len(dm.val_ds),
        "test": len(dm.test_ds),
    }


def colorectal_counts(train_fraction: float) -> Dict[str, int]:
    dm = ColorectalHistDataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": len(dm.train_ds),
        "val": len(dm.val_ds),
        "test": len(dm.test_ds),
    }


def resisc_counts(train_fraction: float) -> Dict[str, int]:
    dm = Resisc45DataModule(train_fraction=train_fraction)
    dm.setup()
    return {
        "train": len(dm.train_ds),
        "val": len(dm.val_ds),
        "test": len(dm.test_ds),
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
        counts = COUNT_FNS[name](args.train_fraction)
        total = sum(counts.values())
        print(f"{name}: train={counts['train']}, val={counts['val']}, test={counts['test']} (total={total})")


if __name__ == "__main__":
    main()
