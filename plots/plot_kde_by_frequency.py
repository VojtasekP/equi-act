#!/usr/bin/env python3
"""
plot_kde_by_frequency.py

Plot KDE curves for all layers, with one plot per frequency.
Each plot shows all layers as different curves.
Uses scikit-learn's KernelDensity with Epanechnikov kernel and Scott's rule for bandwidth selection.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot KDE curves for all layers by frequency"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to .npz file with histogram data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/kde_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=None,
        help="Specific frequency to plot (default: plot all frequencies)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1000,
        help="Number of points for KDE evaluation (default: 1000)",
    )
    return parser.parse_args()


def _load_histogram_data(npz_path: Path) -> Tuple[Dict, Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Load histogram data from .npz file.

    Returns:
        (metadata_dict, {layer_idx: {freq: (bins, counts)}})
    """
    data = np.load(npz_path)

    # Parse metadata
    metadata_str = str(data['metadata'])
    metadata = eval(metadata_str)

    # Parse histogram data
    histograms = {}
    for key in data.files:
        if key == 'metadata':
            continue

        # Parse key: layer_{i}_freq_{f}_bins or layer_{i}_freq_{f}_counts
        parts = key.split('_')
        if len(parts) < 5:
            continue

        layer_idx = int(parts[1])
        freq = int(parts[3])
        dtype = parts[4]  # 'bins' or 'counts'

        if layer_idx not in histograms:
            histograms[layer_idx] = {}
        if freq not in histograms[layer_idx]:
            histograms[layer_idx][freq] = [None, None]

        if dtype == 'bins':
            histograms[layer_idx][freq][0] = data[key]
        elif dtype == 'counts':
            histograms[layer_idx][freq][1] = data[key]

    # Convert to tuples
    for layer_idx in histograms:
        for freq in histograms[layer_idx]:
            histograms[layer_idx][freq] = tuple(histograms[layer_idx][freq])

    return metadata, histograms


def _histogram_to_samples(bins: np.ndarray, counts: np.ndarray, num_samples: int = 10000) -> np.ndarray:
    """
    Convert histogram (bins, counts) to sample points for KDE.

    Sample proportionally from each bin based on counts.
    """
    if len(bins) == 0 or len(counts) == 0:
        return np.array([])

    # Bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)

    # Normalize counts to probabilities
    total = np.sum(counts)
    if total == 0:
        return np.array([])

    probs = counts / total

    # Sample from bins proportionally
    samples = []
    for i, (center, width, prob) in enumerate(zip(bin_centers, bin_widths, probs)):
        n_samples = int(num_samples * prob)
        if n_samples > 0:
            # Sample uniformly within the bin
            bin_samples = np.random.uniform(center - width/2, center + width/2, n_samples)
            samples.extend(bin_samples)

    return np.array(samples)


def _compute_kde(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """
    Compute KDE using scikit-learn's KernelDensity with Epanechnikov kernel.

    Uses Epanechnikov kernel with Scott's rule for automatic bandwidth selection.
    Scott's rule: h = n^(-1/5) * σ

    Args:
        samples: Data samples
        x_grid: Points to evaluate KDE

    Returns:
        KDE values at x_grid points
    """
    # Calculate bandwidth using Scott's rule
    n = len(samples)
    sigma = np.std(samples, ddof=1)
    bandwidth = n ** (-1/5) * sigma

    # Create KDE object with Epanechnikov kernel
    kde = KernelDensity(kernel='epanechnikov', bandwidth=bandwidth)
    kde.fit(samples.reshape(-1, 1))

    # Evaluate KDE on the grid
    log_density = kde.score_samples(x_grid.reshape(-1, 1))
    kde_values = np.exp(log_density)

    return kde_values


def _plot_kde_single_frequency(
    ax,
    freq: int,
    layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    colors,
    num_points: int = 1000,
    show_legend: bool = True,
    density_threshold: float = 0.01
):
    """
    Plot KDE curves for all layers for a single frequency on given axis.

    Args:
        density_threshold: Trim x-axis where max KDE value < threshold * max(all KDE values)
    """
    layers = sorted(layer_data.keys())
    n_layers = len(layers)

    if n_layers == 0:
        print(f"[WARN] No layers found for frequency {freq}")
        return

    # Determine global x range across all layers
    all_x_min, all_x_max = float('inf'), float('-inf')

    for layer_idx in layers:
        bins, counts = layer_data[layer_idx]
        if len(bins) > 0:
            all_x_min = min(all_x_min, bins[0])
            all_x_max = max(all_x_max, bins[-1])

    if all_x_min == float('inf'):
        print(f"[WARN] No valid data for frequency {freq}")
        return

    # Add small margin
    x_range = all_x_max - all_x_min
    all_x_min -= 0.05 * x_range
    all_x_max += 0.05 * x_range

    # First pass: compute all KDEs to find trimming range
    x_grid = np.linspace(all_x_min, all_x_max, num_points)
    all_kde_values = []
    kde_data = []  # Store for second pass

    for idx, layer_idx in enumerate(layers):
        bins, counts = layer_data[layer_idx]

        if len(bins) == 0 or len(counts) == 0:
            continue

        # Convert histogram to samples
        samples = _histogram_to_samples(bins, counts, num_samples=10000)

        if len(samples) < 2:
            continue

        try:
            # Compute KDE using scipy
            kde_values = _compute_kde(samples, x_grid)
            all_kde_values.append(kde_values)
            kde_data.append((layer_idx, idx, kde_values))
        except Exception as e:
            print(f"[WARN] KDE failed for layer {layer_idx}, freq {freq}: {e}")
            continue

    if len(all_kde_values) == 0:
        print(f"[WARN] No valid KDE data for frequency {freq}")
        return

    # Find x-axis trimming range based on KDE values
    all_kde_values = np.array(all_kde_values)
    max_kde = np.max(all_kde_values)
    threshold = density_threshold * max_kde

    # For each x position, check if ANY layer has KDE above threshold
    significant_mask = np.any(all_kde_values > threshold, axis=0)

    if np.any(significant_mask):
        # Find first and last significant positions
        significant_indices = np.where(significant_mask)[0]
        first_idx = significant_indices[0]
        last_idx = significant_indices[-1]

        # Add small margin (5% of range)
        margin_idx = max(1, int(0.05 * (last_idx - first_idx)))
        first_idx = max(0, first_idx - margin_idx)
        last_idx = min(len(x_grid) - 1, last_idx + margin_idx)

        trimmed_x_min = x_grid[first_idx]
        trimmed_x_max = x_grid[last_idx]
    else:
        # No trimming if no significant values found
        trimmed_x_min = all_x_min
        trimmed_x_max = all_x_max

    # Second pass: plot with trimmed data
    for layer_idx, idx, kde_values in kde_data:
        # Plot
        ax.plot(x_grid, kde_values, label=f'Layer {layer_idx}',
               color=colors[idx], linewidth=2, alpha=0.8)

    # Set trimmed x-axis limits
    ax.set_xlim(trimmed_x_min, trimmed_x_max)

    ax.set_xlabel('Norm magnitude', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Frequency {freq}', fontsize=11, fontweight='bold')
    if show_legend:
        ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_all_frequencies_combined(
    histograms: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    metadata: Dict,
    output_path: Path,
    num_points: int = 1000
):
    """
    Plot all frequencies in one figure with subplots.
    """
    # Collect all frequencies
    all_freqs = set()
    for layer_idx in histograms:
        all_freqs.update(histograms[layer_idx].keys())

    freqs = sorted(all_freqs)
    n_freqs = len(freqs)

    if n_freqs == 0:
        print("[WARN] No frequencies to plot")
        return

    # Determine layout
    if n_freqs <= 2:
        nrows, ncols = 1, n_freqs
        figsize = (8 * n_freqs, 6)
    elif n_freqs <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    else:
        ncols = 3
        nrows = (n_freqs + ncols - 1) // ncols
        figsize = (15, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes for easier indexing
    if n_freqs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Get number of layers for color palette
    all_layers = set()
    for layer_idx in histograms:
        all_layers.add(layer_idx)
    n_layers = len(all_layers)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_layers))

    # Plot each frequency
    for idx, freq in enumerate(freqs):
        # Collect data for this frequency across all layers
        layer_data = {}
        for layer_idx in sorted(histograms.keys()):
            if freq in histograms[layer_idx]:
                layer_data[layer_idx] = histograms[layer_idx][freq]

        if not layer_data:
            print(f"[WARN] No data found for frequency {freq}")
            continue

        # Show legend only on first subplot
        _plot_kde_single_frequency(
            axes[idx], freq, layer_data, colors, num_points, show_legend=(idx == 0)
        )

    # Hide unused subplots
    for idx in range(n_freqs, len(axes)):
        axes[idx].axis('off')

    # Overall title
    dataset = metadata.get('dataset', 'unknown')
    activation = metadata.get('activation', 'unknown')
    normalization = metadata.get('normalization', 'unknown')

    fig.suptitle(
        f"{dataset} | {activation} | {normalization}\n"
        f"Samples: {metadata.get('num_samples', 'N/A')}, Seeds: {len(metadata.get('seeds', []))}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    args = _parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading histogram data from: {input_path}")
    metadata, histograms = _load_histogram_data(input_path)

    # Print summary
    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\nAvailable data:")
    for layer_idx in sorted(histograms.keys()):
        freqs = sorted(histograms[layer_idx].keys())
        print(f"  Layer {layer_idx}: frequencies {freqs}")

    output_dir = Path(args.output_dir)
    base_name = input_path.stem

    if args.freq is not None:
        # Plot single frequency
        print(f"\nGenerating KDE plot for frequency {args.freq}")
        layer_data = {}
        for layer_idx in sorted(histograms.keys()):
            if args.freq in histograms[layer_idx]:
                layer_data[layer_idx] = histograms[layer_idx][args.freq]

        if not layer_data:
            print(f"[WARN] No data found for frequency {args.freq}")
            return

        # Create single figure with one subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_layers = len(layer_data)
        colors = plt.cm.viridis(np.linspace(0, 0.9, n_layers))
        _plot_kde_single_frequency(ax, args.freq, layer_data, colors, args.num_points, show_legend=True)

        dataset = metadata.get('dataset', 'unknown')
        activation = metadata.get('activation', 'unknown')
        normalization = metadata.get('normalization', 'unknown')
        fig.suptitle(
            f"{dataset} | {activation} | {normalization}\n"
            f"Samples: {metadata.get('num_samples', 'N/A')}, Seeds: {len(metadata.get('seeds', []))}",
            fontsize=12
        )
        plt.tight_layout()

        output_path = output_dir / f"{base_name}_kde_freq{args.freq}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        # Plot all frequencies in one combined figure
        print(f"\nGenerating combined KDE plot with all frequencies")
        output_path = output_dir / f"{base_name}_kde_all.png"
        _plot_all_frequencies_combined(histograms, metadata, output_path, args.num_points)

    print("\nDone!")


if __name__ == "__main__":
    main()
