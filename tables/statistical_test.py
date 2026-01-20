#!/usr/bin/env python3
"""
Statistical significance testing for comparing model accuracies.

Performs paired or independent t-tests to compare accuracies between two model variants
across all datasets and augmentation settings, producing a comprehensive table.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict
import argparse


def load_results(csv_path: str = "csv_outputs/Results_with_invariance.csv") -> pd.DataFrame:
    """Load the combined results CSV."""
    return pd.read_csv(csv_path)


def get_model_accuracies(
    df: pd.DataFrame,
    activation: str,
    bn: str,
    dataset: str,
    aug: bool
) -> pd.Series:
    """
    Extract accuracies for a specific model configuration, indexed by seed.
    Only considers equivariant models (not baseline ResNet18).

    Args:
        df: Results dataframe
        activation: Activation type (e.g., 'fourier_relu_16', 'gated_sigmoid')
        bn: Batch normalization type (e.g., 'Normbn', 'IIDbn')
        dataset: Dataset name (e.g., 'mnist_rot', 'colorectal_hist', 'eurosat')
        aug: Augmentation setting (True or False)

    Returns:
        Series of test accuracies indexed by seed
    """
    mask = (
        (df['model_type'] == 'equivariant') &
        (df['activation'] == activation) &
        (df['bn'] == bn) &
        (df['dataset'] == dataset) &
        (df['aug'] == aug)
    )

    subset = df[mask]

    # Handle duplicate seeds by averaging their accuracies
    if subset['seed'].duplicated().any():
        subset = subset.groupby('seed')['test_acc'].mean()
    else:
        subset = subset.set_index('seed')['test_acc']

    return subset.sort_index()


def perform_paired_ttest(
    accuracies1: pd.Series,
    accuracies2: pd.Series
) -> Dict:
    """
    Perform paired samples t-test.

    Args:
        accuracies1: Series of accuracies for model 1 (indexed by seed)
        accuracies2: Series of accuracies for model 2 (indexed by seed)

    Returns:
        Dictionary with test results
    """
    # Get common seeds
    common_seeds = accuracies1.index.intersection(accuracies2.index)

    if len(common_seeds) == 0:
        return {'error': 'No common seeds'}

    acc1 = accuracies1.loc[common_seeds].values
    acc2 = accuracies2.loc[common_seeds].values

    # Calculate statistics
    mean1 = np.mean(acc1)
    mean2 = np.mean(acc2)
    std1 = np.std(acc1, ddof=1)
    std2 = np.std(acc2, ddof=1)
    n = len(acc1)

    # Perform paired t-test (two-tailed)
    t_stat, p_value = stats.ttest_rel(acc1, acc2)

    # Calculate effect size (Cohen's d for paired samples)
    differences = acc1 - acc2
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    return {
        'mean1': mean1,
        'std1': std1,
        'mean2': mean2,
        'std2': std2,
        'n': n,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean_diff
    }


def perform_independent_ttest(
    accuracies1: pd.Series,
    accuracies2: pd.Series
) -> Dict:
    """
    Perform independent samples t-test.

    Args:
        accuracies1: Series of accuracies for model 1
        accuracies2: Series of accuracies for model 2

    Returns:
        Dictionary with test results
    """
    if len(accuracies1) == 0 or len(accuracies2) == 0:
        return {'error': 'One or both models have no data'}

    acc1 = accuracies1.values
    acc2 = accuracies2.values

    # Calculate statistics
    mean1 = np.mean(acc1)
    mean2 = np.mean(acc2)
    std1 = np.std(acc1, ddof=1)
    std2 = np.std(acc2, ddof=1)
    n1 = len(acc1)
    n2 = len(acc2)

    # Perform t-test (independent samples, two-tailed)
    t_stat, p_value = stats.ttest_ind(acc1, acc2)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    return {
        'mean1': mean1,
        'std1': std1,
        'mean2': mean2,
        'std2': std2,
        'n': n1,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean1 - mean2
    }


def format_model_name(activation: str, bn: str) -> str:
    """Format model name for display."""
    return f"{activation} + {bn}"


def create_comparison_table(
    df: pd.DataFrame,
    activation1: str,
    bn1: str,
    activation2: str,
    bn2: str,
    test_type: str = 'paired'
) -> pd.DataFrame:
    """
    Create a comprehensive comparison table across all datasets and augmentation settings.

    Args:
        df: Results dataframe
        activation1: First model's activation type
        bn1: First model's BN type
        activation2: Second model's activation type
        bn2: Second model's BN type
        test_type: 'paired' or 'independent'

    Returns:
        DataFrame with test results for all dataset/augmentation combinations
    """
    datasets = df['dataset'].unique()
    aug_settings = [False, True]

    results = []

    for dataset in datasets:
        for aug in aug_settings:
            acc1 = get_model_accuracies(df, activation1, bn1, dataset, aug)
            acc2 = get_model_accuracies(df, activation2, bn2, dataset, aug)

            if len(acc1) == 0 or len(acc2) == 0:
                continue

            # Perform appropriate t-test
            if test_type == 'paired':
                test_result = perform_paired_ttest(acc1, acc2)
            else:
                test_result = perform_independent_ttest(acc1, acc2)

            if 'error' in test_result:
                continue

            # Format significance
            p_val = test_result['p_value']
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            results.append({
                'Dataset': dataset,
                'Augmentation': 'Yes' if aug else 'No',
                f'{format_model_name(activation1, bn1)} Mean': test_result['mean1'],
                f'{format_model_name(activation1, bn1)} Std': test_result['std1'],
                f'{format_model_name(activation2, bn2)} Mean': test_result['mean2'],
                f'{format_model_name(activation2, bn2)} Std': test_result['std2'],
                'Mean Diff': test_result['mean_diff'],
                't-stat': test_result['t_statistic'],
                'p-value': test_result['p_value'],
                "Cohen's d": test_result['cohens_d'],
                'Significance': sig,
                'N': test_result['n']
            })

    return pd.DataFrame(results)


def print_comparison_table(
    results_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    test_type: str
):
    """Print the comparison table in a readable format."""
    print("\n" + "="*120)
    print(f"STATISTICAL COMPARISON: {model1_name} vs {model2_name}")
    print(f"Test Type: {test_type.capitalize()} t-test")
    print("="*120)
    print()

    if len(results_df) == 0:
        print("No results to display. Check that both models have data.")
        return

    # Print table with formatted values
    print(f"{'Dataset':<20} {'Aug':<6} {'Model 1 Mean':<14} {'Model 2 Mean':<14} "
          f"{'Diff':<10} {'p-value':<12} {'Sig':<6} {'N':<4}")
    print("-" * 120)

    for _, row in results_df.iterrows():
        print(f"{row['Dataset']:<20} {row['Augmentation']:<6} "
              f"{row[f'{model1_name} Mean']*100:>6.2f}% ± {row[f'{model1_name} Std']*100:>4.2f}  "
              f"{row[f'{model2_name} Mean']*100:>6.2f}% ± {row[f'{model2_name} Std']*100:>4.2f}  "
              f"{row['Mean Diff']*100:>+7.2f}%  "
              f"{row['p-value']:>10.6f}  "
              f"{row['Significance']:>4}  "
              f"{int(row['N']):<4}")

    print("="*120)
    print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print()


def save_tables(
    results_df: pd.DataFrame,
    model1_name: str,
    model2_name: str,
    test_type: str,
    output_dir: str = "tables/tables"
):
    """Save results as CSV and LaTeX tables."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create safe filename
    safe_model1 = model1_name.replace(" ", "_").replace("+", "")
    safe_model2 = model2_name.replace(" ", "_").replace("+", "")
    base_name = f"ttest_{safe_model1}_vs_{safe_model2}_{test_type}"

    # Save CSV
    csv_path = Path(output_dir) / f"{base_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # Save LaTeX table
    latex_path = Path(output_dir) / f"{base_name}.tex"

    # Format for LaTeX
    latex_df = results_df.copy()
    for col in latex_df.columns:
        if 'Mean' in col:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x*100:.2f}")
        elif 'Std' in col:
            latex_df[col] = latex_df[col].apply(lambda x: f"{x*100:.2f}")
        elif col == 'Mean Diff':
            latex_df[col] = latex_df[col].apply(lambda x: f"{x*100:+.2f}")
        elif col == 'p-value':
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.4f}")
        elif col == "Cohen's d":
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.3f}")
        elif col == 't-stat':
            latex_df[col] = latex_df[col].apply(lambda x: f"{x:.3f}")

    # Build LaTeX table manually for desired format
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lllll}")
    lines.append(r"\toprule")
    lines.append(r"Dataset (Augmentation) & Mean Diff & p-value & Cohen's d & Significance \\")
    lines.append(r"\midrule")

    # Dataset name mapping for nicer display
    dataset_names = {
        'mnist_rot': 'MNIST-rot',
        'colorectal_hist': 'Colorectal Histology',
        'eurosat': 'EuroSAT',
        'resisc45': 'RESISC45'
    }

    for _, row in results_df.iterrows():
        dataset_display = dataset_names.get(row['Dataset'], row['Dataset'])
        aug_display = "Yes" if row['Augmentation'] == 'Yes' else "No"

        mean_diff = row['Mean Diff'] * 100
        p_val = row['p-value']
        cohens_d = row["Cohen's d"]
        sig = row['Significance']

        # Bold p-value if significant
        if p_val < 0.05:
            p_str = f"\\textbf{{{p_val:.4f}}}"
        else:
            p_str = f"{p_val:.4f}"

        lines.append(f"{dataset_display} ({aug_display}) & {mean_diff:+.2f} & {p_str} & {cohens_d:.3f} & {sig} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{Statistical comparison: {model1_name} vs {model2_name} (two-sided {test_type} t-test)}}")
    lines.append(f"\\label{{tab:ttest_{safe_model1}_vs_{safe_model2}}}")
    lines.append(r"\end{table}")

    latex_table = "\n".join(lines)

    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform t-tests comparing two model variants across all datasets and augmentation settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paired t-test comparing Fourier variants (default)
  python statistical_test.py \\
    --model1 fourier_relu_16 Normbn \\
    --model2 fourier_elu_16 Normbn

  # Independent t-test comparing gated models
  python statistical_test.py \\
    --model1 gated_sigmoid Normbn \\
    --model2 gated_shared_sigmoid Normbn \\
    --test-type independent

  # Compare with custom CSV path
  python statistical_test.py \\
    --model1 normbnvec_relu Normbn \\
    --model2 norm_relu Normbn \\
    --csv path/to/results.csv
        """
    )

    parser.add_argument(
        '--model1',
        nargs=2,
        metavar=('ACTIVATION', 'BN'),
        required=True,
        help='First model: activation type and BN type (e.g., fourier_relu_16 Normbn)'
    )

    parser.add_argument(
        '--model2',
        nargs=2,
        metavar=('ACTIVATION', 'BN'),
        required=True,
        help='Second model: activation type and BN type'
    )

    parser.add_argument(
        '--test-type',
        choices=['paired', 'independent'],
        default='paired',
        help='Type of t-test to perform (default: paired)'
    )

    parser.add_argument(
        '--csv',
        default='tables/csv_outputs/Results_with_invariance.csv',
        help='Path to results CSV file (default: tables/csv_outputs/Results_with_invariance.csv)'
    )

    parser.add_argument(
        '--output-dir',
        default='tables/tables',
        help='Directory to save output tables (default: tables/tables)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save CSV and LaTeX tables to disk'
    )

    args = parser.parse_args()

    # Extract model configurations
    activation1, bn1 = args.model1
    activation2, bn2 = args.model2

    model1_name = format_model_name(activation1, bn1)
    model2_name = format_model_name(activation2, bn2)

    # Load data
    print(f"Loading results from {args.csv}...")
    df = load_results(args.csv)

    print(f"\nComparing models:")
    print(f"  Model 1: {model1_name}")
    print(f"  Model 2: {model2_name}")
    print(f"  Test type: {args.test_type}")

    # Create comparison table
    results_df = create_comparison_table(
        df, activation1, bn1, activation2, bn2, args.test_type
    )

    # Print results
    print_comparison_table(results_df, model1_name, model2_name, args.test_type)

    # Save tables
    if not args.no_save:
        save_tables(results_df, model1_name, model2_name, args.test_type, args.output_dir)


if __name__ == "__main__":
    main()
