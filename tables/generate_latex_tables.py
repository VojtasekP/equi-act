"""Generate LaTeX tables for accuracy, invariance, train time, and num params.

This script reads the combined results CSV and produces 4 separate LaTeX tables:
1. Test accuracy table
2. Invariance error table
3. Training time table
4. Number of parameters table

Each table has models as rows and (dataset, augmentation) combinations as columns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ========= FORMATTING CONFIGURATION =========
TEST_ERROR_DECIMALS = 2  # Number of decimal places for test error mean and std
# ============================================


def load_results(csv_path: str = "tables/csv_outputs/Results_with_invariance_new.csv") -> pd.DataFrame:
    """Load the combined results CSV."""
    return pd.read_csv(csv_path)


def create_pivot_table(df: pd.DataFrame, metric: str, agg_func: str = 'mean', include_aug: bool = True) -> pd.DataFrame:
    """Create a pivot table for a specific metric.

    Args:
        df: Results dataframe
        metric: Column name to aggregate ('test_acc', 'invariance_error', 'train_time_minutes', 'num_params')
        agg_func: Aggregation function ('mean', 'std', 'first')
        include_aug: If True, include augmentation in columns; if False, only dataset

    Returns:
        Pivot table with models as rows, (dataset, aug) or just dataset as columns
    """
    # Create model identifier combining model_type, activation, and bn
    df = df.copy()

    # For equivariant models: use activation + bn
    # For ResNet18: just use "ResNet18"
    def make_model_name(row):
        if row['model_type'] == 'resnet18':
            return 'ResNet18'
        else:
            # Equivariant: combine activation and bn
            activation = row['activation'] if pd.notna(row['activation']) else 'unknown'
            bn = row['bn'] if pd.notna(row['bn']) else 'none'
            return f"{activation}_{bn}"

    df['model_name'] = df.apply(make_model_name, axis=1)

    # Create column identifier
    if include_aug:
        # Create column identifier (dataset, aug)
        df['dataset_aug'] = df.apply(
            lambda row: f"{row['dataset']}_{'aug' if row['aug'] else 'noaug'}",
            axis=1
        )
        group_cols = ['model_name', 'dataset_aug']
    else:
        # Only use dataset (no augmentation distinction)
        df['dataset_aug'] = df['dataset']
        group_cols = ['model_name', 'dataset_aug']

    # Aggregate by model and dataset_aug
    if agg_func == 'first':
        # For num_params, just take the first value (they're all the same for a given model)
        pivot = df.groupby(group_cols)[metric].first().unstack()
    else:
        # For other metrics, compute mean or std across seeds
        pivot = df.groupby(group_cols)[metric].agg(agg_func).unstack()

    # Sort columns
    if include_aug:
        # Sort by dataset then aug
        columns_sorted = sorted(pivot.columns, key=lambda x: (x.split('_')[0], 'noaug' in x))
    else:
        # Just sort by dataset name
        columns_sorted = sorted(pivot.columns)

    pivot = pivot[columns_sorted]

    return pivot


def format_column_name(col: str, include_aug: bool = True) -> str:
    """Format column name for LaTeX table header.

    Args:
        col: Column name (e.g., 'colorectal_aug' or 'colorectal')
        include_aug: If True, expect aug/noaug in name; if False, just dataset
    """
    # Clean up dataset names
    dataset_map = {
        'colorectal': 'Colorectal',
        'eurosat': 'EuroSAT',
        'mnist': 'MNIST'
    }

    if not include_aug:
        # Just format dataset name
        dataset = col
        for key, value in dataset_map.items():
            if key in dataset:
                return value
        return col.title()

    # Include augmentation in name
    parts = col.split('_')
    dataset = parts[0]

    for key, value in dataset_map.items():
        if key in dataset:
            dataset = value
            break

    # Aug/NoAug
    aug = 'Aug' if 'aug' in col and 'noaug' not in col else 'NoAug'

    return f"{dataset} {aug}"


def get_model_category(model_name: str) -> str:
    """Determine the category for a model based on activation type (before _Normbn/_IIDbn suffix)."""
    if model_name == 'ResNet18':
        return 'Baseline'

    # Extract activation part (before _Normbn or _IIDbn suffix)
    activation = model_name.split('_Normbn')[0].split('_IIDbn')[0]

    # Categorize based on activation type
    if 'fourierbn' in activation.lower():
        return 'FourierBN'
    elif 'fourier' in activation.lower():
        return 'Fourier'
    elif 'normbnvec' in activation.lower():
        return 'Split Norm-BN'
    elif 'normbn' in activation.lower():
        return 'Unified Norm-BN'
    elif 'gated' in activation.lower():
        return 'Gated'
    elif 'norm' in activation.lower():
        return 'Split Norm'
    else:
        return 'Other'


def format_model_name(model: str) -> str:
    """Format model name for LaTeX table row with improved readability."""
    if model == 'ResNet18':
        return 'ResNet18'

    original = model
    # Remove BN suffix for parsing
    model_clean = model.split('_Normbn')[0].split('_IIDbn')[0]

    # Handle FourierBN models
    if 'fourierbn' in model_clean.lower():
        parts = model_clean.split('_')
        activation = parts[1] if len(parts) > 1 else 'relu'
        n_value = parts[2] if len(parts) > 2 else '16'
        return f"FourierBN-{activation.title()} (N={n_value})"

    # Handle Fourier models (without BN)
    if 'fourier' in model_clean.lower() and 'fourierbn' not in model_clean.lower():
        parts = model_clean.split('_')
        activation = parts[1] if len(parts) > 1 else 'relu'
        n_value = parts[2] if len(parts) > 2 else '16'
        return f"Fourier-{activation.title()} (N={n_value})"

    # Handle normbnvec (Split Norm-BN)
    if 'normbnvec' in model_clean.lower():
        parts = model_clean.split('_')
        activation = parts[1] if len(parts) > 1 else 'relu'
        return f"Split Norm-BN-{activation.title()} + ELU"

    # Handle normbn (Unified Norm-BN)
    if 'normbn' in model_clean.lower() and 'normbnvec' not in model_clean.lower():
        parts = model_clean.split('_')
        activation = parts[1] if len(parts) > 1 else 'relu'
        return f"Unified Norm-BN-{activation.title()}"

    # Handle gated models
    if 'gated' in model_clean.lower():
        if 'shared' in model_clean.lower():
            return "Split Gated-Sigmoid-shared"
        else:
            return "Split Gated-Sigmoid"

    # Handle norm models (without BN)
    if 'norm' in model_clean.lower():
        parts = model_clean.split('_')
        activation = parts[1] if len(parts) > 1 else 'relu'
        if activation == 'relu':
            return "Split Norm-ReLU + ELU"
        elif activation == 'squash':
            return "Split Norm-Squash + ELU"
        elif activation == 'elu':
            return "Split Norm-Elu + ELU"
        elif activation == 'sigmoid':
            return "Split Norm-Sigmoid + ELU"
        else:
            return f"Split Norm-{activation.title()} + ELU"

    # Fallback: just clean up underscores
    return original.replace('_', ' ').title()


def to_latex_table(df: pd.DataFrame, metric_name: str, caption: str, label: str, include_aug: bool = True, use_resizebox: bool = False) -> str:
    """Convert DataFrame to LaTeX table format.

    Args:
        df: Pivot table with models as rows, datasets as columns
        metric_name: Name of the metric for formatting
        caption: Table caption
        label: Table label for referencing
        include_aug: If True, columns include aug/noaug; if False, only dataset
        use_resizebox: If True, wrap tabular in \resizebox{\textwidth}{!}{}

    Returns:
        LaTeX table string
    """
    # Format column names
    df_formatted = df.copy()
    df_formatted.columns = [format_column_name(col, include_aug) for col in df_formatted.columns]

    # Create category mapping before formatting row names
    categories = {idx: get_model_category(idx) for idx in df_formatted.index}

    # Format row names (index)
    df_formatted.index = [format_model_name(idx) for idx in df_formatted.index]

    # Format values based on metric type
    if metric_name == 'test_acc':
        # Convert to percentages with 2 decimal places
        df_formatted = df_formatted * 100
        format_str = '{:.2f}'
    elif metric_name == 'test_err':
        # Already converted to error percentage, just format
        format_str = '{:.2f}'
    elif metric_name == 'invariance_error':
        # 4 decimal places for invariance error
        format_str = '{:.4f}'
    elif metric_name == 'train_time':
        # 1 decimal place for time
        format_str = '{:.1f}'
    elif metric_name == 'num_params':
        # Format as K or M for readability
        def format_params(x):
            if pd.isna(x) or x == 0:
                return '---'
            elif x >= 1_000_000:
                return f'{x/1_000_000:.2f}M'
            else:
                return f'{x/1_000:.0f}K'
        df_formatted = df_formatted.applymap(format_params)
        format_str = None
    else:
        format_str = '{:.4f}'

    # Apply formatting if needed
    if format_str:
        df_formatted = df_formatted.applymap(lambda x: format_str.format(x) if pd.notna(x) else '---')

    # Replace NaN with ---
    df_formatted = df_formatted.fillna('---')

    # Generate LaTeX table
    n_cols = len(df_formatted.columns)
    col_format = 'l' + 'c' * n_cols

    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"

    # Add resizebox if requested
    if use_resizebox:
        latex_str += "\\resizebox{\\textwidth}{!}{\n"

    latex_str += "\\begin{tabular}{" + col_format + "}\n"
    latex_str += "\\toprule\n"

    # Header
    header = "Model & " + " & ".join(df_formatted.columns) + " \\\\\n"
    latex_str += header
    latex_str += "\\midrule\n"

    # Data rows with category grouping
    # Group rows by category
    category_order = ['Baseline', 'Fourier', 'FourierBN', 'Gated', 'Split Norm', 'Unified Norm-BN', 'Split Norm-BN', 'Other']

    # Get original index names (before formatting) to access categories
    original_indices = list(df.index)
    formatted_indices = list(df_formatted.index)

    # Create a mapping from formatted to original
    idx_map = dict(zip(formatted_indices, original_indices))

    # Group by category
    rows_by_category = {}
    for formatted_idx, row in df_formatted.iterrows():
        orig_idx = idx_map.get(formatted_idx, formatted_idx)
        cat = categories.get(orig_idx, 'Other')
        if cat not in rows_by_category:
            rows_by_category[cat] = []
        rows_by_category[cat].append((formatted_idx, row))

    # Write rows grouped by category
    first_category = True
    for category in category_order:
        if category not in rows_by_category:
            continue

        # Add category header (except for first category)
        if not first_category:
            latex_str += "\\midrule\n"
        first_category = False

        # Add rows in this category
        for formatted_idx, row in rows_by_category[category]:
            row_str = str(formatted_idx) + " & " + " & ".join([str(val) for val in row]) + " \\\\\n"
            latex_str += row_str

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"

    # Close resizebox if it was opened
    if use_resizebox:
        latex_str += "}\n"

    latex_str += "\\end{table}\n"

    return latex_str


def save_latex_table(latex_str: str, filename: str, output_dir: str = "tables/tex_outputs"):
    """Save LaTeX table to file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, 'w') as f:
        f.write(latex_str)

    print(f"Saved: {filepath}")


def main():
    # Load data
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} rows\n")

    # 1. Test Error Table (100 - accuracy)
    print("=== Generating Test Error LaTeX Table ===")
    acc_table = create_pivot_table(df, 'test_acc', agg_func='mean')
    # Convert accuracy to error: 100 - (accuracy * 100)
    err_table = 100 - (acc_table * 100)
    latex_err = to_latex_table(
        err_table,
        'test_err',
        caption="Test Error (\\%) across different models and datasets. Lower is better. Results averaged over 5 seeds.",
        label="tab:test_error",
        use_resizebox=True
    )
    save_latex_table(latex_err, 'test_error.tex')

    # 2. Invariance Error Table
    print("=== Generating Invariance Error LaTeX Table ===")
    inv_table = create_pivot_table(df, 'invariance_error', agg_func='mean')
    latex_inv = to_latex_table(
        inv_table,
        'invariance_error',
        caption="Invariance Error across different models and datasets. Lower values indicate better rotation invariance. Results averaged over 5 seeds.",
        label="tab:invariance_error",
        use_resizebox=True
    )
    save_latex_table(latex_inv, 'invariance_error.tex')

    # 3. Training Time Table (without augmentation) - convert to sec/epoch
    print("=== Generating Training Time LaTeX Table ===")
    time_table = create_pivot_table(df, 'train_time_minutes', agg_func='mean', include_aug=False)

    # Convert from total minutes to seconds/epoch
    # ResNet models: divide by 200 epochs
    # Equivariant MNIST: divide by 30 epochs
    # Equivariant Colorectal/EuroSAT: divide by 60 epochs
    time_table_sec_per_epoch = time_table.copy()
    for idx in time_table_sec_per_epoch.index:
        for col in time_table_sec_per_epoch.columns:
            if pd.notna(time_table_sec_per_epoch.loc[idx, col]):
                minutes = time_table_sec_per_epoch.loc[idx, col]

                # Determine divisor based on model and dataset
                if idx == 'ResNet18':
                    # ResNet: 200 epochs
                    divisor = 200
                else:
                    # Equivariant models: depends on dataset
                    dataset = col.lower()
                    if 'mnist' in dataset:
                        divisor = 30
                    elif 'colorectal' in dataset or 'eurosat' in dataset:
                        divisor = 60
                    else:
                        divisor = 30  # default

                # Convert minutes to seconds and divide by epochs
                time_table_sec_per_epoch.loc[idx, col] = (minutes * 60) / divisor

    latex_time = to_latex_table(
        time_table_sec_per_epoch,
        'train_time',
        caption="Training Time (seconds/epoch) across different models and datasets. Results averaged over 5 seeds and augmentation settings.",
        label="tab:train_time",
        include_aug=False
    )
    save_latex_table(latex_time, 'train_time.tex')

    # 4. Number of Parameters Table (without augmentation)
    print("=== Generating Number of Parameters LaTeX Table ===")
    params_table = create_pivot_table(df, 'num_params', agg_func='first', include_aug=False)
    latex_params = to_latex_table(
        params_table,
        'num_params',
        caption="Number of Trainable Parameters across different models and datasets.",
        label="tab:num_params",
        include_aug=False
    )
    save_latex_table(latex_params, 'num_params.tex')

    # 5. Combined table with mean ± std for error (100 - accuracy)
    print("\n=== Generating Combined Error Table (mean ± std) ===")
    acc_mean = create_pivot_table(df, 'test_acc', agg_func='mean')
    acc_std = create_pivot_table(df, 'test_acc', agg_func='std')

    # Convert to error: 100 - (accuracy * 100)
    err_mean = 100 - (acc_mean * 100)
    err_std = acc_std * 100  # convert std to percentage as well

    # Combine mean and std
    err_combined = err_mean.copy()
    for idx in err_combined.index:
        for col in err_combined.columns:
            mean_val = err_mean.loc[idx, col]
            std_val = err_std.loc[idx, col]
            if pd.notna(mean_val) and pd.notna(std_val):
                err_combined.loc[idx, col] = f"{mean_val:.{TEST_ERROR_DECIMALS}f} " + r"{\small$\pm$ " + f"{std_val:.{TEST_ERROR_DECIMALS}f}" + "}"
            elif pd.notna(mean_val):
                err_combined.loc[idx, col] = f"{mean_val:.{TEST_ERROR_DECIMALS}f}"
            else:
                err_combined.loc[idx, col] = '---'

    # Create category mapping before formatting
    categories_combined = {idx: get_model_category(idx) for idx in err_combined.index}
    original_indices_combined = list(err_combined.index)

    # Format for LaTeX
    err_combined.columns = [format_column_name(col) for col in err_combined.columns]
    err_combined.index = [format_model_name(idx) for idx in err_combined.index]

    # Create mapping from formatted to original
    formatted_indices_combined = list(err_combined.index)
    idx_map_combined = dict(zip(formatted_indices_combined, original_indices_combined))

    # Generate LaTeX with category grouping
    n_cols = len(err_combined.columns)
    col_format = 'l' + 'c' * n_cols

    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Test Error (\\%) with standard deviation across different models and datasets. Lower is better.}\n"
    latex_str += "\\label{tab:test_error_with_std}\n"
    latex_str += "\\resizebox{\\textwidth}{!}{\n"
    latex_str += "\\begin{tabular}{" + col_format + "}\n"
    latex_str += "\\toprule\n"
    latex_str += "Model & " + " & ".join(err_combined.columns) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Group by category
    category_order = ['Baseline', 'Fourier', 'FourierBN', 'Gated', 'Split Norm', 'Unified Norm-BN', 'Split Norm-BN', 'Other']
    rows_by_category = {}
    for formatted_idx, row in err_combined.iterrows():
        orig_idx = idx_map_combined.get(formatted_idx, formatted_idx)
        cat = categories_combined.get(orig_idx, 'Other')
        if cat not in rows_by_category:
            rows_by_category[cat] = []
        rows_by_category[cat].append((formatted_idx, row))

    # Write rows grouped by category
    first_category = True
    for category in category_order:
        if category not in rows_by_category:
            continue
        if not first_category:
            latex_str += "\\midrule\n"
        first_category = False
        for formatted_idx, row in rows_by_category[category]:
            row_str = str(formatted_idx) + " & " + " & ".join([str(val) for val in row]) + " \\\\\n"
            latex_str += row_str

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "}\n"
    latex_str += "\\end{table}\n"

    save_latex_table(latex_str, 'test_error_with_std.tex')

    print("\n" + "="*60)
    print("All LaTeX tables generated successfully!")
    print("Output directory: tex_outputs/")
    print("="*60)

    # Print sample of one table
    print("\nSample output (Test Error table):")
    print(latex_err[:500] + "...")


if __name__ == "__main__":
    main()
