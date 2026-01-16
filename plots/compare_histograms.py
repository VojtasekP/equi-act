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

# Set Times New Roman as the default font (with fallbacks)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Times']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts are similar to Times


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


def _format_dataset_name(dataset: str) -> str:
    """
    Format dataset name into publication-ready format.

    Examples:
        mnist_rot -> MNIST-rot
        eurosat -> EuroSAT
        colorectal_hist -> Colorectal-hist
        resisc45 -> RESISC45
    """
    if dataset == 'mnist_rot':
        return 'MNIST-rot'
    elif dataset == 'eurosat':
        return 'EuroSAT'
    elif dataset == 'colorectal_hist':
        return 'Colorectal-hist'
    elif dataset == 'resisc45':
        return 'RESISC45'
    else:
        return dataset.replace('_', '-').title()


def _format_model_name(activation: str) -> str:
    """
    Format activation name into publication-ready format.

    Examples:
        fourier_relu_16 -> Fourier-ReLU (N=16)
        fourierbn_relu_16 -> Fourier-BN-ReLU (N=16)
        gated_sigmoid -> Gated-Sigmoid
        norm_relu -> Norm-ReLU
        normbn_relu -> Norm-BN-ReLU
    """
    # Extract N value if present
    n_value = None
    parts = activation.split('_')
    if parts[-1].isdigit():
        n_value = parts[-1]
        parts = parts[:-1]  # Remove N from parts

    # Format each part
    formatted_parts = []
    for part in parts:
        if part == 'fourier':
            formatted_parts.append('Fourier')
        elif part == 'fourierbn':
            formatted_parts.append('Fourier-BN')
        elif part == 'relu':
            formatted_parts.append('ReLU')
        elif part == 'elu':
            formatted_parts.append('ELU')
        elif part == 'sigmoid':
            formatted_parts.append('Sigmoid')
        elif part == 'gated':
            formatted_parts.append('Gated')
        elif part == 'shared':
            formatted_parts.append('Shared')
        elif part == 'norm':
            formatted_parts.append('Norm')
        elif part == 'normbn':
            formatted_parts.append('Norm-BN')
        elif part == 'normbnvec':
            formatted_parts.append('Norm-BN-Vec')
        elif part == 'squash':
            formatted_parts.append('Squash')
        else:
            formatted_parts.append(part.capitalize())

    # Join with hyphens
    name = '-'.join(formatted_parts)

    # Add N value if present
    if n_value:
        name += f' (N={n_value})'

    return name


def _compute_histogram_statistics(bins: np.ndarray, counts: np.ndarray) -> Dict:
    """
    Compute statistics from histogram data.

    Returns:
        Dict with 'mass_at_zero', 'mean', 'median', 'total_mass'
    """
    if len(bins) == 0 or len(counts) == 0:
        return {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan, 'total_mass': 0}

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_widths = np.diff(bins)

    # Total mass (integral of histogram)
    total_mass = np.sum(counts * bin_widths)

    if total_mass == 0:
        return {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan, 'total_mass': 0}

    # Mass at zero: find the bin containing 0 and compute its contribution
    zero_bin_idx = np.searchsorted(bins, 0, side='right') - 1
    if zero_bin_idx < 0:
        zero_bin_idx = 0
    if zero_bin_idx >= len(counts):
        zero_bin_idx = len(counts) - 1

    # Mass at zero bin as percentage
    zero_bin_mass = counts[zero_bin_idx] * bin_widths[zero_bin_idx]
    mass_at_zero_pct = (zero_bin_mass / total_mass) * 100

    # Mean: weighted average of bin centers
    mean = np.sum(bin_centers * counts * bin_widths) / total_mass

    # Median: find the value where cumulative mass reaches 50%
    cumulative_mass = np.cumsum(counts * bin_widths)
    median_idx = np.searchsorted(cumulative_mass, total_mass / 2)
    if median_idx >= len(bin_centers):
        median_idx = len(bin_centers) - 1
    median = bin_centers[median_idx]

    return {
        'mass_at_zero': mass_at_zero_pct,
        'mean': mean,
        'median': median,
        'total_mass': total_mass
    }


def _print_comparison_statistics_table(
    histograms1: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    histograms2: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    metadata1: Dict,
    metadata2: Dict
):
    """
    Print a comparison table of statistics for both models.
    """
    model1_name = _format_model_name(metadata1.get('activation', 'Model1'))
    model2_name = _format_model_name(metadata2.get('activation', 'Model2'))

    print("\n" + "=" * 100)
    print("DISTRIBUTION STATISTICS COMPARISON")
    print("=" * 100)

    # Collect all frequencies from both models
    all_freqs = set()
    for layer_idx in histograms1:
        all_freqs.update(histograms1[layer_idx].keys())
    for layer_idx in histograms2:
        all_freqs.update(histograms2[layer_idx].keys())
    freqs = sorted(all_freqs)

    # All layers from both models
    all_layers = sorted(set(histograms1.keys()) | set(histograms2.keys()))

    # Print header
    print(f"\n{'Layer':<6} {'Irrep':<6} | "
          f"{'Mass@0(%)':>10} {'Mean':>10} {'Median':>10} | "
          f"{'Mass@0(%)':>10} {'Mean':>10} {'Median':>10}")
    print(f"{'':6} {'':6} | "
          f"{'--- ' + model1_name[:20] + ' ---':^32} | "
          f"{'--- ' + model2_name[:20] + ' ---':^32}")
    print("-" * 100)

    # Print statistics for each layer and frequency
    for layer_idx in all_layers:
        for freq in freqs:
            # Get stats for model 1
            if layer_idx in histograms1 and freq in histograms1[layer_idx]:
                bins1, counts1 = histograms1[layer_idx][freq]
                stats1 = _compute_histogram_statistics(bins1, counts1)
            else:
                stats1 = {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan}

            # Get stats for model 2
            if layer_idx in histograms2 and freq in histograms2[layer_idx]:
                bins2, counts2 = histograms2[layer_idx][freq]
                stats2 = _compute_histogram_statistics(bins2, counts2)
            else:
                stats2 = {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan}

            # Skip if both are nan
            if np.isnan(stats1['mass_at_zero']) and np.isnan(stats2['mass_at_zero']):
                continue

            # Format values (handle nan)
            def fmt_mass(v):
                return f"{v:10.2f}" if not np.isnan(v) else f"{'N/A':>10}"

            def fmt_val(v):
                return f"{v:10.3f}" if not np.isnan(v) else f"{'N/A':>10}"

            print(f"{layer_idx:<6} {freq:<6} | "
                  f"{fmt_mass(stats1['mass_at_zero'])} {fmt_val(stats1['mean'])} {fmt_val(stats1['median'])} | "
                  f"{fmt_mass(stats2['mass_at_zero'])} {fmt_val(stats2['mean'])} {fmt_val(stats2['median'])}")

    print("-" * 100)

    # Summary by layer (averaged over irreps)
    print(f"\nSummary by Layer (averaged over irreps):")
    print(f"{'Layer':<6} | "
          f"{'Mass@0(%)':>10} {'Mean':>10} {'Median':>10} | "
          f"{'Mass@0(%)':>10} {'Mean':>10} {'Median':>10}")
    print("-" * 80)

    for layer_idx in all_layers:
        masses1, means1, medians1 = [], [], []
        masses2, means2, medians2 = [], [], []

        for freq in freqs:
            if layer_idx in histograms1 and freq in histograms1[layer_idx]:
                bins1, counts1 = histograms1[layer_idx][freq]
                stats1 = _compute_histogram_statistics(bins1, counts1)
                if not np.isnan(stats1['mass_at_zero']):
                    masses1.append(stats1['mass_at_zero'])
                    means1.append(stats1['mean'])
                    medians1.append(stats1['median'])

            if layer_idx in histograms2 and freq in histograms2[layer_idx]:
                bins2, counts2 = histograms2[layer_idx][freq]
                stats2 = _compute_histogram_statistics(bins2, counts2)
                if not np.isnan(stats2['mass_at_zero']):
                    masses2.append(stats2['mass_at_zero'])
                    means2.append(stats2['mean'])
                    medians2.append(stats2['median'])

        def fmt_avg_mass(lst):
            return f"{np.mean(lst):10.2f}" if lst else f"{'N/A':>10}"

        def fmt_avg_val(lst):
            return f"{np.mean(lst):10.3f}" if lst else f"{'N/A':>10}"

        print(f"{layer_idx:<6} | "
              f"{fmt_avg_mass(masses1)} {fmt_avg_val(means1)} {fmt_avg_val(medians1)} | "
              f"{fmt_avg_mass(masses2)} {fmt_avg_val(means2)} {fmt_avg_val(medians2)}")

    print("=" * 100 + "\n")


def _save_comparison_statistics_latex(
    histograms1: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    histograms2: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    metadata1: Dict,
    metadata2: Dict,
    output_path: Path
):
    """
    Save comparison statistics as a LaTeX table.
    """
    model1_name = _format_model_name(metadata1.get('activation', 'Model1'))
    model2_name = _format_model_name(metadata2.get('activation', 'Model2'))

    # Collect all frequencies from both models
    all_freqs = set()
    for layer_idx in histograms1:
        all_freqs.update(histograms1[layer_idx].keys())
    for layer_idx in histograms2:
        all_freqs.update(histograms2[layer_idx].keys())
    freqs = sorted(all_freqs)

    # All layers from both models
    all_layers = sorted(set(histograms1.keys()) | set(histograms2.keys()))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("% Distribution Statistics Comparison\n")
        f.write(f"% {model1_name} vs {model2_name}\n\n")

        # Main table
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Distribution statistics comparison: " +
                f"{model1_name} vs {model2_name}}}\n")
        f.write("\\label{tab:dist_stats_comparison}\n")
        f.write("\\begin{tabular}{cc|ccc|ccc}\n")
        f.write("\\toprule\n")
        f.write(f" & & \\multicolumn{{3}}{{c|}}{{{model1_name}}} & "
                f"\\multicolumn{{3}}{{c}}{{{model2_name}}} \\\\\n")
        f.write("Layer & Irrep & Mass@0 (\\%) & Mean & Median & "
                "Mass@0 (\\%) & Mean & Median \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for layer_idx in all_layers:
            for freq in freqs:
                # Get stats for model 1
                if layer_idx in histograms1 and freq in histograms1[layer_idx]:
                    bins1, counts1 = histograms1[layer_idx][freq]
                    stats1 = _compute_histogram_statistics(bins1, counts1)
                else:
                    stats1 = {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan}

                # Get stats for model 2
                if layer_idx in histograms2 and freq in histograms2[layer_idx]:
                    bins2, counts2 = histograms2[layer_idx][freq]
                    stats2 = _compute_histogram_statistics(bins2, counts2)
                else:
                    stats2 = {'mass_at_zero': np.nan, 'mean': np.nan, 'median': np.nan}

                # Skip if both are nan
                if np.isnan(stats1['mass_at_zero']) and np.isnan(stats2['mass_at_zero']):
                    continue

                def fmt_mass(v):
                    return f"{v:.2f}" if not np.isnan(v) else "N/A"

                def fmt_val(v):
                    return f"{v:.3f}" if not np.isnan(v) else "N/A"

                f.write(f"{layer_idx} & {freq} & "
                        f"{fmt_mass(stats1['mass_at_zero'])} & "
                        f"{fmt_val(stats1['mean'])} & "
                        f"{fmt_val(stats1['median'])} & "
                        f"{fmt_mass(stats2['mass_at_zero'])} & "
                        f"{fmt_val(stats2['mean'])} & "
                        f"{fmt_val(stats2['median'])} \\\\\n")

        f.write("\\midrule\n")

        # Summary row (averaged over irreps per layer)
        f.write("\\multicolumn{2}{c|}{\\textbf{Average}} & "
                "\\multicolumn{3}{c|}{} & \\multicolumn{3}{c}{} \\\\\n")

        for layer_idx in all_layers:
            masses1, means1, medians1 = [], [], []
            masses2, means2, medians2 = [], [], []

            for freq in freqs:
                if layer_idx in histograms1 and freq in histograms1[layer_idx]:
                    bins1, counts1 = histograms1[layer_idx][freq]
                    stats1 = _compute_histogram_statistics(bins1, counts1)
                    if not np.isnan(stats1['mass_at_zero']):
                        masses1.append(stats1['mass_at_zero'])
                        means1.append(stats1['mean'])
                        medians1.append(stats1['median'])

                if layer_idx in histograms2 and freq in histograms2[layer_idx]:
                    bins2, counts2 = histograms2[layer_idx][freq]
                    stats2 = _compute_histogram_statistics(bins2, counts2)
                    if not np.isnan(stats2['mass_at_zero']):
                        masses2.append(stats2['mass_at_zero'])
                        means2.append(stats2['mean'])
                        medians2.append(stats2['median'])

            def fmt_avg_mass(lst):
                return f"{np.mean(lst):.2f}" if lst else "N/A"

            def fmt_avg_val(lst):
                return f"{np.mean(lst):.3f}" if lst else "N/A"

            f.write(f"{layer_idx} & -- & "
                    f"{fmt_avg_mass(masses1)} & "
                    f"{fmt_avg_val(means1)} & "
                    f"{fmt_avg_val(medians1)} & "
                    f"{fmt_avg_mass(masses2)} & "
                    f"{fmt_avg_val(means2)} & "
                    f"{fmt_avg_val(medians2)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved LaTeX table: {output_path}")


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

    model1_name = _format_model_name(metadata1.get('activation', 'unknown'))
    ax1.set_xlabel('Norm magnitude', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title(f'{model1_name}', fontsize=18, fontweight='bold')
    ax1.tick_params(labelsize=14)
    ax1.grid(True, alpha=0.3)

    # Add stats for model 1
    stats1 = _compute_histogram_statistics(bins1, counts1)
    stats1_text = (
        f"Mass@0: {stats1['mass_at_zero']:.2f}%\n"
        f"Mean: {stats1['mean']:.3f}\n"
        f"Median: {stats1['median']:.3f}"
    )
    ax1.text(0.97, 0.97, stats1_text, transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Model 2
    if len(bins2) > 0 and len(counts2) > 0:
        bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
        bin_widths2 = np.diff(bins2)
        ax2.bar(bin_centers2, counts2, width=bin_widths2, alpha=0.7,
                color='coral', edgecolor='black', linewidth=0.5)

    model2_name = _format_model_name(metadata2.get('activation', 'unknown'))
    ax2.set_xlabel('Norm magnitude', fontsize=16)
    ax2.set_ylabel('Density', fontsize=16)
    ax2.set_title(f'{model2_name}', fontsize=18, fontweight='bold')
    ax2.tick_params(labelsize=14)
    ax2.grid(True, alpha=0.3)

    # Add stats for model 2
    stats2 = _compute_histogram_statistics(bins2, counts2)
    stats2_text = (
        f"Mass@0: {stats2['mass_at_zero']:.2f}%\n"
        f"Mean: {stats2['mean']:.3f}\n"
        f"Median: {stats2['median']:.3f}"
    )
    ax2.text(0.97, 0.97, stats2_text, transform=ax2.transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

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
    dataset = _format_dataset_name(metadata1.get('dataset', 'unknown'))
    fig.suptitle(
        f"Layer {layer_idx}, Irrep {freq} - {dataset}\n"
        f"Comparison: {model1_name} vs {model2_name}",
        fontsize=20, fontweight='bold'
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
    model1_label = _format_model_name(metadata1.get('activation', 'Model 1'))
    if len(bins1) > 0 and len(counts1) > 0:
        bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
        bin_widths1 = np.diff(bins1)
        ax.bar(bin_centers1, counts1, width=bin_widths1, alpha=0.5,
               color='steelblue', edgecolor='darkblue', linewidth=1.5,
               label=model1_label)

    # Model 2
    model2_label = _format_model_name(metadata2.get('activation', 'Model 2'))
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

    ax.set_xlabel('Norm magnitude', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.legend(loc='upper right', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)

    # Add statistics in top left corner (both models)
    stats1 = _compute_histogram_statistics(bins1, counts1)
    stats2 = _compute_histogram_statistics(bins2, counts2)

    stats_text = (
        f"Mass@0: {stats1['mass_at_zero']:.2f}% / {stats2['mass_at_zero']:.2f}%\n"
        f"Mean: {stats1['mean']:.3f} / {stats2['mean']:.3f}\n"
        f"Median: {stats1['median']:.3f} / {stats2['median']:.3f}"
    )
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Overall title
    dataset = _format_dataset_name(metadata1.get('dataset', 'unknown'))
    ax.set_title(
        f"Layer {layer_idx}, Irrep {freq} - {dataset}\n"
        f"Comparison: {model1_label} vs {model2_label}",
        fontsize=18, fontweight='bold'
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
    model1_name = _format_model_name(metadata1.get('activation', 'M1'))
    model2_name = _format_model_name(metadata2.get('activation', 'M2'))

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

        ax.set_title(f'L{layer_idx}, I{freq}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Norm', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

        # Add statistics in top right corner
        stats1 = _compute_histogram_statistics(bins1, counts1)
        stats2 = _compute_histogram_statistics(bins2, counts2)

        # Draw statistics with colored values (blue for model1, orange for model2)
        from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea

        lines_data = [
            ("m@0: ", f"{stats1['mass_at_zero']:.1f}%", " / ", f"{stats2['mass_at_zero']:.1f}%"),
            ("Mean: ", f"{stats1['mean']:.2f}", " / ", f"{stats2['mean']:.2f}"),
            ("Med: ", f"{stats1['median']:.2f}", " / ", f"{stats2['median']:.2f}"),
        ]

        line_boxes = []
        fs = 15
        for label, val1, sep, val2 in lines_data:
            txt_label = TextArea(label, textprops=dict(fontsize=fs, color='black'))
            txt_val1 = TextArea(val1, textprops=dict(fontsize=fs, color='steelblue', fontweight='bold'))
            txt_sep = TextArea(sep, textprops=dict(fontsize=fs, color='black'))
            txt_val2 = TextArea(val2, textprops=dict(fontsize=fs, color='coral', fontweight='bold'))
            line_box = HPacker(children=[txt_label, txt_val1, txt_sep, txt_val2], align='baseline', pad=0, sep=0)
            line_boxes.append(line_box)

        vbox = VPacker(children=line_boxes, align='right', pad=0, sep=2)
        anchored_box = AnchoredOffsetbox(loc='upper right', child=vbox, pad=0.3,
                                          frameon=True, bbox_to_anchor=(1, 1),
                                          bbox_transform=ax.transAxes, borderpad=0.3)
        anchored_box.patch.set_boxstyle("round,pad=0.3")
        anchored_box.patch.set_facecolor('white')
        anchored_box.patch.set_alpha(0.8)
        ax.add_artist(anchored_box)

    # Hide unused subplots
    for idx in range(n_combos, len(axes)):
        axes[idx].axis('off')

    # Overall title
    dataset = _format_dataset_name(metadata1.get('dataset', 'unknown'))
    # fig.suptitle(
    #     f"{dataset} - Comparison: {model1_name} vs {model2_name}\n"
    #     f"All Layers and Irreps",
    #     fontsize=20, fontweight='bold'
    # )

    # Add legend at the bottom of the figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='darkblue', alpha=0.5, label=model1_name),
        Patch(facecolor='coral', edgecolor='darkred', alpha=0.5, label=model2_name)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=16, frameon=True, bbox_to_anchor=(0.5, -0.02))

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

    # Print comparison statistics table
    _print_comparison_statistics_table(histograms1, histograms2, metadata1, metadata2)

    # Save LaTeX table
    tex_output_dir = Path("plots/tex_outputs")
    act1 = metadata1.get('activation', 'model1')
    act2 = metadata2.get('activation', 'model2')
    dataset = metadata1.get('dataset', 'unknown')
    tex_path = tex_output_dir / f"dist_stats_{dataset}_{act1}_vs_{act2}.tex"
    _save_comparison_statistics_latex(histograms1, histograms2, metadata1, metadata2, tex_path)

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
