#!/usr/bin/env python3
"""
compare_histograms.py

Compare histogram distributions between two models (e.g., fourier vs fourierbn).
Creates side-by-side or overlaid plots showing differences clearly.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare histogram distributions between two models"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first .npz file (e.g., fourier_relu_16)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second .npz file (e.g., fourierbn_relu_16)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/comparison_plots",
        help="Directory to save comparison plots",
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
        "--plot-type",
        choices=['sidebyside', 'overlay', 'both'],
        default='both',
        help="Type of comparison plot (default: both)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Custom output filename (without extension)",
    )
    parser.add_argument(
        "--format",
        choices=['png', 'pdf', 'both'],
        default='both',
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output (default: 300)",
    )
    return parser.parse_args()


def _save_figure(output_path: Path, format_choice: str, dpi: int):
    """
    Save figure in requested format(s).

    Args:
        output_path: Base output path (with or without extension)
        format_choice: 'png', 'pdf', or 'both'
        dpi: DPI for output
    """
    # Remove extension if present
    base_path = output_path.with_suffix('')

    formats_to_save = []
    if format_choice == 'both':
        formats_to_save = ['png', 'pdf']
    else:
        formats_to_save = [format_choice]

    for fmt in formats_to_save:
        save_path = base_path.with_suffix(f'.{fmt}')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format=fmt)
        print(f"Saved: {save_path}")


def _normalize_histogram(bins: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize histogram so that the total area equals 1.

    This ensures fair comparison between models that may have different
    numbers of feature channels per frequency.

    Args:
        bins: Bin edges (length n+1)
        counts: Histogram counts or density values (length n)

    Returns:
        (bins, normalized_counts) where integral of normalized_counts = 1
    """
    if len(bins) == 0 or len(counts) == 0:
        return bins, counts

    # Calculate bin widths
    bin_widths = np.diff(bins)

    # Calculate total area
    total_area = np.sum(counts * bin_widths)

    if total_area == 0:
        return bins, counts

    # Normalize so area = 1
    normalized_counts = counts / total_area

    return bins, normalized_counts


def _get_trimmed_xlim(bins1: np.ndarray, counts1: np.ndarray,
                      bins2: np.ndarray, counts2: np.ndarray,
                      threshold: float = 0.01) -> Tuple[float, float]:
    """
    Calculate appropriate x-axis limits by trimming empty space.

    Finds the range where either histogram has values above threshold * max_value.

    Args:
        bins1, counts1: First histogram
        bins2, counts2: Second histogram
        threshold: Fraction of max value to use as cutoff (default: 1%)

    Returns:
        (x_min, x_max) for setting axis limits
    """
    if len(bins1) == 0 and len(bins2) == 0:
        return 0, 1

    # Get bin centers for both histograms
    all_bin_centers = []
    all_counts = []

    if len(bins1) > 0 and len(counts1) > 0:
        centers1 = (bins1[:-1] + bins1[1:]) / 2
        all_bin_centers.append(centers1)
        all_counts.append(counts1)

    if len(bins2) > 0 and len(counts2) > 0:
        centers2 = (bins2[:-1] + bins2[1:]) / 2
        all_bin_centers.append(centers2)
        all_counts.append(counts2)

    if len(all_bin_centers) == 0:
        return 0, 1

    # Combine all data
    all_centers = np.concatenate(all_bin_centers)
    all_vals = np.concatenate(all_counts)

    # Find threshold value
    max_val = np.max(all_vals)
    cutoff = threshold * max_val

    # Find where values exceed threshold
    significant_mask = all_vals > cutoff

    if not np.any(significant_mask):
        # If no values above threshold, use full range
        return np.min(all_centers), np.max(all_centers)

    significant_centers = all_centers[significant_mask]
    x_min = np.min(significant_centers)
    x_max = np.max(significant_centers)

    # Add 10% margin
    x_range = x_max - x_min
    margin = 0.1 * x_range
    x_min = max(0, x_min - margin)  # Don't go below 0 for norms
    x_max = x_max + margin

    return x_min, x_max


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

    # Convert to tuples and normalize
    for layer_idx in histograms:
        for freq in histograms[layer_idx]:
            bins, counts = histograms[layer_idx][freq]
            if bins is not None and counts is not None:
                bins, counts = _normalize_histogram(bins, counts)
            histograms[layer_idx][freq] = (bins, counts)

    return metadata, histograms


def _plot_sidebyside_comparison(
    layer_idx: int,
    freq: int,
    bins1: np.ndarray,
    counts1: np.ndarray,
    bins2: np.ndarray,
    counts2: np.ndarray,
    metadata1: Dict,
    metadata2: Dict,
    output_path: Path,
    format_choice: str = 'both',
    dpi: int = 300
):
    """
    Create side-by-side comparison plot for a single layer and frequency.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Model 1
    if len(bins1) > 0 and len(counts1) > 0:
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        bin_widths1 = np.diff(bins1)
        ax1.bar(bin_centers1, counts1, width=bin_widths1, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5)

    model1_name = f"{metadata1.get('activation', 'unknown')} + {metadata1.get('normalization', 'unknown')}"
    ax1.set_xlabel('Norm magnitude', fontsize=11)
    ax1.set_ylabel('Normalized Density', fontsize=11)
    ax1.set_title(f'{model1_name}', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Model 2
    if len(bins2) > 0 and len(counts2) > 0:
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
        bin_widths2 = np.diff(bins2)
        ax2.bar(bin_centers2, counts2, width=bin_widths2, alpha=0.7,
                color='coral', edgecolor='black', linewidth=0.5)

    model2_name = f"{metadata2.get('activation', 'unknown')} + {metadata2.get('normalization', 'unknown')}"
    ax2.set_xlabel('Norm magnitude', fontsize=11)
    ax2.set_ylabel('Normalized Density', fontsize=11)
    ax2.set_title(f'{model2_name}', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Match x-axis ranges with trimming
    if len(bins1) > 0 and len(bins2) > 0:
        x_min, x_max = _get_trimmed_xlim(bins1, counts1, bins2, counts2)
        ax1.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)

    # Match y-axis ranges for easier comparison
    if len(counts1) > 0 and len(counts2) > 0:
        y_max = max(np.max(counts1), np.max(counts2))
        ax1.set_ylim(0, y_max * 1.05)
        ax2.set_ylim(0, y_max * 1.05)

    # Overall title
    dataset = metadata1.get('dataset', 'unknown')
    fig.suptitle(
        f"Layer {layer_idx}, Irrep {freq} - {dataset}\n"
        f"Comparison: {metadata1.get('activation', 'M1')} vs {metadata2.get('activation', 'M2')}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    _save_figure(output_path, format_choice, dpi)
    plt.close()


def _plot_overlay_comparison(
    layer_idx: int,
    freq: int,
    bins1: np.ndarray,
    counts1: np.ndarray,
    bins2: np.ndarray,
    counts2: np.ndarray,
    metadata1: Dict,
    metadata2: Dict,
    output_path: Path,
    format_choice: str = 'both',
    dpi: int = 300
):
    """
    Create overlay comparison plot for a single layer and frequency.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Model 1
    model1_label = metadata1.get('activation', 'Model 1')
    if len(bins1) > 0 and len(counts1) > 0:
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        bin_widths1 = np.diff(bins1)
        ax.bar(bin_centers1, counts1, width=bin_widths1, alpha=0.5,
               color='steelblue', edgecolor='darkblue', linewidth=1.5,
               label=model1_label)

    # Model 2
    model2_label = metadata2.get('activation', 'Model 2')
    if len(bins2) > 0 and len(counts2) > 0:
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
        bin_widths2 = np.diff(bins2)
        ax.bar(bin_centers2, counts2, width=bin_widths2, alpha=0.5,
               color='coral', edgecolor='darkred', linewidth=1.5,
               label=model2_label)

    # Trim x-axis
    if len(bins1) > 0 and len(bins2) > 0:
        x_min, x_max = _get_trimmed_xlim(bins1, counts1, bins2, counts2)
        ax.set_xlim(x_min, x_max)

    ax.set_xlabel('Norm magnitude', fontsize=12)
    ax.set_ylabel('Normalized Density', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add text labels at bottom of plot
    y_min, y_max = ax.get_ylim()
    text_y = -0.15 * y_max  # Below the plot
    ax.text(0.25, text_y, model1_label,
            transform=ax.transAxes, fontsize=11, color='steelblue',
            ha='center', weight='bold')
    ax.text(0.75, text_y, model2_label,
            transform=ax.transAxes, fontsize=11, color='coral',
            ha='center', weight='bold')

    # Overall title
    dataset = metadata1.get('dataset', 'unknown')
    ax.set_title(
        f"Layer {layer_idx}, Irrep {freq} - {dataset}\n"
        f"Comparison: {metadata1.get('activation', 'M1')} vs {metadata2.get('activation', 'M2')}",
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    _save_figure(output_path, format_choice, dpi)
    plt.close()


def _plot_grid_comparison(
    histograms1: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    histograms2: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    metadata1: Dict,
    metadata2: Dict,
    output_path: Path,
    plot_type: str = 'sidebyside',
    format_choice: str = 'both',
    dpi: int = 300
):
    """
    Create grid comparison showing all layer/frequency combinations.
    Each subplot shows comparison for one layer-frequency pair.
    """
    # Get all layer-frequency combinations
    all_combos = set()
    for layer_idx in histograms1:
        for freq in histograms1[layer_idx]:
            all_combos.add((layer_idx, freq))
    for layer_idx in histograms2:
        for freq in histograms2[layer_idx]:
            all_combos.add((layer_idx, freq))

    combos = sorted(all_combos)
    n_combos = len(combos)

    if n_combos == 0:
        print("[WARN] No matching layer-frequency combinations found")
        return

    # Determine layout
    if n_combos <= 3:
        nrows, ncols = 1, n_combos
    elif n_combos <= 4:
        nrows, ncols = 2, 2
    elif n_combos <= 6:
        nrows, ncols = 2, 3
    elif n_combos <= 9:
        nrows, ncols = 3, 3
    else:
        ncols = 4
        nrows = (n_combos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))

    # Flatten axes
    if n_combos == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    # Get model names for labels
    model1_name = metadata1.get('activation', 'M1')
    model2_name = metadata2.get('activation', 'M2')

    # Plot each combination
    for idx, (layer_idx, freq) in enumerate(combos):
        ax = axes[idx]

        # Get data
        bins1, counts1 = histograms1.get(layer_idx, {}).get(freq, (np.array([]), np.array([])))
        bins2, counts2 = histograms2.get(layer_idx, {}).get(freq, (np.array([]), np.array([])))

        # Plot (overlay mode always for grid)
        if len(bins1) > 0 and len(counts1) > 0:
            bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
            bin_widths1 = np.diff(bins1)
            ax.bar(bin_centers1, counts1, width=bin_widths1, alpha=0.5,
                   color='steelblue', edgecolor='darkblue', linewidth=1)

        if len(bins2) > 0 and len(counts2) > 0:
            bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
            bin_widths2 = np.diff(bins2)
            ax.bar(bin_centers2, counts2, width=bin_widths2, alpha=0.5,
                   color='coral', edgecolor='darkred', linewidth=1)

        # Trim x-axis for this subplot
        if len(bins1) > 0 and len(bins2) > 0:
            x_min, x_max = _get_trimmed_xlim(bins1, counts1, bins2, counts2)
            ax.set_xlim(x_min, x_max)

        ax.set_title(f'L{layer_idx}, I{freq}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Norm', fontsize=9)
        ax.set_ylabel('Normalized Density', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_combos, len(axes)):
        axes[idx].axis('off')

    # Overall title
    dataset = metadata1.get('dataset', 'unknown')
    fig.suptitle(
        f"{dataset} - Comparison: {model1_name} vs {model2_name}\n"
        f"All Layers and Irreps",
        fontsize=15, fontweight='bold'
    )

    # Add legend at the bottom of the figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='darkblue', alpha=0.5, label=model1_name),
        Patch(facecolor='coral', edgecolor='darkred', alpha=0.5, label=model2_name)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Leave space for legend and title
    _save_figure(output_path, format_choice, dpi)
    plt.close()


def main():
    args = _parse_args()

    # Load data
    model1_path = Path(args.model1)
    model2_path = Path(args.model2)

    if not model1_path.exists():
        raise FileNotFoundError(f"Model 1 file not found: {model1_path}")
    if not model2_path.exists():
        raise FileNotFoundError(f"Model 2 file not found: {model2_path}")

    print(f"Loading Model 1: {model1_path}")
    metadata1, histograms1 = _load_histogram_data(model1_path)

    print(f"Loading Model 2: {model2_path}")
    metadata2, histograms2 = _load_histogram_data(model2_path)

    # Print summaries
    print("\nModel 1:")
    print(f"  Dataset: {metadata1.get('dataset', 'unknown')}")
    print(f"  Activation: {metadata1.get('activation', 'unknown')}")
    print(f"  Layers: {sorted(histograms1.keys())}")

    print("\nModel 2:")
    print(f"  Dataset: {metadata2.get('dataset', 'unknown')}")
    print(f"  Activation: {metadata2.get('activation', 'unknown')}")
    print(f"  Layers: {sorted(histograms2.keys())}")

    output_dir = Path(args.output_dir)

    # Generate output filename
    if args.output_name:
        base_name = args.output_name
    else:
        act1 = metadata1.get('activation', 'model1')
        act2 = metadata2.get('activation', 'model2')
        dataset = metadata1.get('dataset', 'unknown')
        base_name = f"{dataset}_{act1}_vs_{act2}"

    # Specific layer and frequency
    if args.layer is not None and args.freq is not None:
        print(f"\nGenerating comparison for Layer {args.layer}, Frequency {args.freq}...")

        bins1, counts1 = histograms1.get(args.layer, {}).get(args.freq, (np.array([]), np.array([])))
        bins2, counts2 = histograms2.get(args.layer, {}).get(args.freq, (np.array([]), np.array([])))

        if len(bins1) == 0 or len(bins2) == 0:
            print(f"[WARN] No data for Layer {args.layer}, Frequency {args.freq} in one or both models")
            return

        if args.plot_type in ['sidebyside', 'both']:
            output_path = output_dir / f"{base_name}_L{args.layer}_F{args.freq}_sidebyside"
            _plot_sidebyside_comparison(args.layer, args.freq, bins1, counts1, bins2, counts2,
                                       metadata1, metadata2, output_path, args.format, args.dpi)

        if args.plot_type in ['overlay', 'both']:
            output_path = output_dir / f"{base_name}_L{args.layer}_F{args.freq}_overlay"
            _plot_overlay_comparison(args.layer, args.freq, bins1, counts1, bins2, counts2,
                                    metadata1, metadata2, output_path, args.format, args.dpi)

    else:
        # Generate grid for all combinations
        print(f"\nGenerating grid comparison for all layer-frequency combinations...")
        output_path = output_dir / f"{base_name}_grid_{args.plot_type}"
        _plot_grid_comparison(histograms1, histograms2, metadata1, metadata2,
                             output_path, args.plot_type, args.format, args.dpi)

    print("\nDone!")


if __name__ == "__main__":
    main()
