"""Load and plot equivariance curves from NPZ file.

Usage:
    python tables/plot_equivariance_curves.py \
        --npz-file tables/csv_outputs/equivariance_results.npz \
        --output-dir plots/equivariance_curves/
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_equivariance_data(npz_path: Path) -> List[Dict]:
    """Load equivariance curve data from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    num_rows = int(data["num_rows"])

    rows = []
    for idx in range(num_rows):
        prefix = f"row{idx}_"
        row = {
            "activation": str(data[f"{prefix}activation"]),
            "bn": str(data[f"{prefix}bn"]),
            "dataset": str(data[f"{prefix}dataset"]),
            "theta_values": data[f"{prefix}theta_values"],
        }

        # Load metric curves
        for key in data.keys():
            if key.startswith(prefix) and key.endswith("layer") or key == f"{prefix}error_metric_net":
                metric_name = key[len(prefix):]
                row[metric_name] = data[key]

        rows.append(row)

    return rows


def plot_curves_by_layer(rows: List[Dict], output_dir: Path):
    """Plot equivariance curves grouped by layer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    datasets = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(row)

    for dataset, dataset_rows in datasets.items():
        # Find all available layers
        layer_metrics = set()
        for row in dataset_rows:
            for key in row.keys():
                if key.startswith("error_metric_"):
                    layer_metrics.add(key)

        layer_metrics = sorted(layer_metrics)

        for metric_name in layer_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            for row in dataset_rows:
                if metric_name not in row or row[metric_name] is None:
                    continue

                curve = row[metric_name]
                theta_values = row["theta_values"]
                label = f"{row['activation']}_{row['bn']}"

                ax.plot(np.degrees(theta_values), curve, label=label, marker='o', markersize=3)

            ax.set_xlabel("Rotation Angle (degrees)")
            ax.set_ylabel("Equivariance Error")
            ax.set_title(f"{dataset} - {metric_name}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = output_dir / f"{dataset}_{metric_name}.pdf"
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")


def plot_all_layers_comparison(rows: List[Dict], output_dir: Path):
    """Plot all layers for each activation/bn combination."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        dataset = row["dataset"]
        activation = row["activation"]
        bn = row["bn"]
        theta_values = row["theta_values"]

        fig, ax = plt.subplots(figsize=(10, 6))

        for key, curve in row.items():
            if not key.startswith("error_metric_") or curve is None:
                continue

            label = key.replace("error_metric_", "").replace("layer", "Layer ")
            if label == "net":
                label = "Full Network"

            ax.plot(np.degrees(theta_values), curve, label=label, marker='o', markersize=3)

        ax.set_xlabel("Rotation Angle (degrees)")
        ax.set_ylabel("Equivariance Error")
        ax.set_title(f"{dataset} - {activation} - {bn}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{dataset}_{activation}_{bn}_all_layers.pdf"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot equivariance curves from NPZ file")
    parser.add_argument("--npz-file", type=str, required=True, help="Path to NPZ file")
    parser.add_argument("--output-dir", type=str, default="plots/equivariance_curves/",
                        help="Output directory for plots")
    parser.add_argument("--plot-type", type=str, default="both",
                        choices=["by_layer", "all_layers", "both"],
                        help="Type of plots to generate")
    args = parser.parse_args()

    npz_path = Path(args.npz_file)
    output_dir = Path(args.output_dir)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    print(f"Loading data from {npz_path}...")
    rows = load_equivariance_data(npz_path)
    print(f"Loaded {len(rows)} configurations")

    if args.plot_type in ["by_layer", "both"]:
        print("\nGenerating layer-wise comparison plots...")
        plot_curves_by_layer(rows, output_dir / "by_layer")

    if args.plot_type in ["all_layers", "both"]:
        print("\nGenerating all-layers plots...")
        plot_all_layers_comparison(rows, output_dir / "all_layers")

    print("\nDone!")


if __name__ == "__main__":
    main()
