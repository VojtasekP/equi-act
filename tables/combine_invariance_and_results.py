import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "csv_outputs" / "Results.csv"
df_acc = pd.read_csv(csv_path)
print(df_acc.head())
new = "_new"
# Load equivariant model invariance files
df_invariance_mnist_equi = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_mnist{new}.csv")
df_invariance_eurosat_equi = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_eurosat{new}.csv")
df_invariance_hist_equi = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_colorectal{new}.csv")

# Load baseline (ResNet18) invariance files
df_invariance_mnist_baseline = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_mnist_baseline.csv")
df_invariance_eurosat_baseline = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_eurosat_baseline.csv")
df_invariance_hist_baseline = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / f"invariance_colorectal_baseline.csv")

# Combine equivariant and baseline invariance data for each dataset
df_invariance_mnist = pd.concat([df_invariance_mnist_equi, df_invariance_mnist_baseline], ignore_index=True)
df_invariance_eurosat = pd.concat([df_invariance_eurosat_equi, df_invariance_eurosat_baseline], ignore_index=True)
df_invariance_hist = pd.concat([df_invariance_hist_equi, df_invariance_hist_baseline], ignore_index=True)

for row in df_acc.itertuples():
    dataset = row.dataset
    model = row.model_type
    activation = row.activation
    seed = row.seed
    aug = row.aug

    # Select appropriate invariance dataframe based on dataset
    if dataset == "mnist_rot":
        df_invariance = df_invariance_mnist
    elif dataset == "eurosat":
        df_invariance = df_invariance_eurosat
    elif dataset == "colorectal_hist":
        df_invariance = df_invariance_hist
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Different matching logic for resnet18 vs equivariant models
    if model == "resnet18":
        # For ResNet18: match only on model_type, seed, aug (ignore activation)
        # since the activation field might differ between Results.csv and invariance files
        invariance_value = df_invariance[
            (df_invariance['model_type'] == model) &
            (df_invariance['seed'] == seed) &
            (df_invariance['aug'] == aug)
        ]['invariance'].values
    else:
        # For equivariant models: match on all fields including activation
        invariance_value = df_invariance[
            (df_invariance['model_type'] == model) &
            (df_invariance['seed'] == seed) &
            (df_invariance['aug'] == aug) &
            (df_invariance['activation'] == activation)
        ]['invariance'].values

    if len(invariance_value) > 0:
        df_acc.at[row.Index, 'invariance_error'] = invariance_value[0]
    else:
        df_acc.at[row.Index, 'invariance_error'] = None


print(df_acc.head())
save_path = Path(__file__).parent / "csv_outputs" / f"Results_with_invariance{new}.csv"
df_acc.to_csv(save_path, index=False)