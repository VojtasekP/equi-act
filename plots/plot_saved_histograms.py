#!/usr/bin/env python3
"""
plot_saved_histograms.py

Load histogram data from .npz files and create visualizations.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histograms from saved .npz files"
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
        default="plots/histogram_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to plot (default: plot all layers)",
    )
    parser.add_argument(
        "--freq",
        type=int,
        default=None,
        help="Specific frequency to plot (default: plot all frequencies)",
    )
    parser.add_argument(
        "--separate-layers",
        action="store_true",
        help="Save separate plot for each layer (default: one plot per frequency across layers)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Create one combined figure with all frequencies (4 subplots, all layers per frequency)",
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
    metadata = eval(metadata_str)  # Safe here since we created the file

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


def _plot_single_layer_all_freqs(
    layer_idx: int,
    freq_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    metadata: Dict,
    output_path: Path
):
    """
    Plot all frequencies for a single layer in one figure.
    """
    freqs = sorted(freq_data.keys())
    n_freqs = len(freqs)

    if n_freqs == 0:
        print(f"[WARN] No frequencies found for layer {layer_idx}")
        return

    # Determine layout
    if n_freqs <= 3:
        fig, axes = plt.subplots(1, n_freqs, figsize=(5*n_freqs, 4))
    elif n_freqs == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    else:
        ncols = min(3, n_freqs)
        nrows = (n_freqs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    # Flatten axes for easier indexing
    if n_freqs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    # Plot each frequency
    for idx, freq in enumerate(freqs):
        ax = axes[idx]
        bins, counts = freq_data[freq]

        if len(bins) == 0 or len(counts) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'Irrep freq {freq}')
            continue

        # Bar plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)

        ax.bar(bin_centers, counts, width=bin_widths, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Norm magnitude')
        ax.set_ylabel('Density' if metadata.get('density', False) else 'Count')
        ax.set_title(f'Irrep freq {freq}')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_freqs, len(axes)):
        axes[idx].axis('off')

    # Overall title
    dataset = metadata.get('dataset', 'unknown')
    activation = metadata.get('activation', 'unknown')
    normalization = metadata.get('normalization', 'unknown')

    fig.suptitle(
        f"Layer {layer_idx} - {dataset} | {activation} | {normalization}\n"
        f"Samples: {metadata.get('num_samples', 'N/A')}, Seeds: {len(metadata.get('seeds', []))}",
        fontsize=12
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _plot_single_freq_all_layers(
    freq: int,
    layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    metadata: Dict,
    output_path: Path
):
    """
    Plot a single frequency across all layers.
    """
    layers = sorted(layer_data.keys())
    n_layers = len(layers)

    if n_layers == 0:
        print(f"[WARN] No layers found for frequency {freq}")
        return

    # Determine layout
    if n_layers <= 3:
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    elif n_layers == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    else:
        ncols = min(3, n_layers)
        nrows = (n_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    # Flatten axes for easier indexing
    if n_layers == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    # Plot each layer
    for idx, layer_idx in enumerate(layers):
        ax = axes[idx]
        bins, counts = layer_data[layer_idx]

        if len(bins) == 0 or len(counts) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'Layer {layer_idx}')
            continue

        # Bar plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)

        ax.bar(bin_centers, counts, width=bin_widths, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Norm magnitude')
        ax.set_ylabel('Density' if metadata.get('density', False) else 'Count')
        ax.set_title(f'Layer {layer_idx}')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')

    # Overall title
    dataset = metadata.get('dataset', 'unknown')
    activation = metadata.get('activation', 'unknown')
    normalization = metadata.get('normalization', 'unknown')

    fig.suptitle(
        f"Irrep Frequency {freq} - {dataset} | {activation} | {normalization}\n"
        f"Samples: {metadata.get('num_samples', 'N/A')}, Seeds: {len(metadata.get('seeds', []))}",
        fontsize=12
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def _plot_hist_single_frequency(
    ax,
    freq: int,
    layer_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    colors,
    show_legend: bool = True
):
    """
    Plot histograms for all layers for a single frequency on given axis.
    Each layer is shown as a separate histogram with different color.
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

    # Plot each layer's histogram
    for idx, layer_idx in enumerate(layers):
        bins, counts = layer_data[layer_idx]

        if len(bins) == 0 or len(counts) == 0:
            continue

        # Bar plot
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = np.diff(bins)

        ax.bar(bin_centers, counts, width=bin_widths, alpha=0.5,
               edgecolor=colors[idx], linewidth=1.5,
               color=colors[idx], label=f'Layer {layer_idx}')

    ax.set_xlim(all_x_min, all_x_max)
    ax.set_xlabel('Norm magnitude', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'Frequency {freq}', fontsize=11, fontweight='bold')
    if show_legend:
        ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_all_frequencies_combined(
    histograms: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    metadata: Dict,
    output_path: Path
):
    """
    Plot all frequencies in one figure with subplots (2x2 grid).
    Each subplot shows all layers for one frequency.
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
        _plot_hist_single_frequency(
            axes[idx], freq, layer_data, colors, show_legend=(idx == 0)
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
    base_name = input_path.stem  # filename without .npz

    # Filter by layer/freq if specified
    layers_to_plot = [args.layer] if args.layer is not None else sorted(histograms.keys())

    if args.combined:
        # Plot all frequencies in one combined figure (4 subplots)
        print("\nGenerating combined histogram plot with all frequencies")
        output_path = output_dir / f"{base_name}_hist_all.png"
        _plot_all_frequencies_combined(histograms, metadata, output_path)

    elif args.separate_layers:
        # One plot per layer (all frequencies in that layer)
        print("\nGenerating plots (one per layer)...")
        for layer_idx in layers_to_plot:
            if layer_idx not in histograms:
                print(f"[WARN] Layer {layer_idx} not found in data")
                continue

            freq_data = histograms[layer_idx]

            # Filter by frequency if specified
            if args.freq is not None:
                if args.freq not in freq_data:
                    print(f"[WARN] Frequency {args.freq} not found in layer {layer_idx}")
                    continue
                freq_data = {args.freq: freq_data[args.freq]}

            output_path = output_dir / f"{base_name}_layer{layer_idx}.png"
            _plot_single_layer_all_freqs(layer_idx, freq_data, metadata, output_path)

    else:
        # One plot per frequency (all layers for that frequency)
        print("\nGenerating plots (one per frequency)...")

        # Collect all frequencies
        all_freqs = set()
        for layer_idx in layers_to_plot:
            if layer_idx in histograms:
                all_freqs.update(histograms[layer_idx].keys())

        # Filter by frequency if specified
        freqs_to_plot = [args.freq] if args.freq is not None else sorted(all_freqs)

        for freq in freqs_to_plot:
            # Collect data for this frequency across layers
            layer_data = {}
            for layer_idx in layers_to_plot:
                if layer_idx in histograms and freq in histograms[layer_idx]:
                    layer_data[layer_idx] = histograms[layer_idx][freq]

            if not layer_data:
                print(f"[WARN] No data found for frequency {freq}")
                continue

            output_path = output_dir / f"{base_name}_freq{freq}.png"
            _plot_single_freq_all_layers(freq, layer_data, metadata, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
