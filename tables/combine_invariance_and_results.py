import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "csv_outputs" / "Results.csv"  
df_acc = pd.read_csv(csv_path)
print(df_acc.head())

df_invariance_mnist = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / "invariance_mnist.csv")
df_invariance_eurosat = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / "invariance_eurosat.csv")
df_invariance_hist = pd.read_csv(Path(__file__).parent / "csv_outputs" / "invariance" / "invariance_colorectal.csv")

for row in df_acc.itertuples():
    dataset = row.dataset
    model = row.model_type
    activation = row.activation
    seed = row.seed
    aug = row.aug
    if dataset == "mnist_rot":
        df_invariance = df_invariance_mnist
    elif dataset == "eurosat":
        df_invariance = df_invariance_eurosat
    elif dataset == "colorectal_hist":
        df_invariance = df_invariance_hist
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
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
save_path = Path(__file__).parent / "csv_outputs" / "Results_with_invariance.csv"
df_acc.to_csv(save_path, index=False)