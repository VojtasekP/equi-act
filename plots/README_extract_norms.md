# Extract Feature Norm Distributions

This script extracts feature norm distributions from all activation layers across all checkpoints and saves them as histogram data for later plotting.

## Usage

```bash
python plots/extract_feature_norm_distributions.py \
  --models-dir saved_models/mnist_rot_final \
  --output-dir plots/norm_distributions \
  --num-samples 128 \
  --num-bins 50 \
  --density \
  --device auto
```

## Arguments

- `--models-dir` (required): Directory containing .ckpt files
- `--output-dir` (default: "plots/norm_distributions"): Output directory for .npz files
- `--num-samples` (default: 128): Number of stratified test images to use
- `--num-bins` (default: 50): Number of histogram bins
- `--density`: Use density normalization instead of raw counts
- `--device` (default: "auto"): Device (auto/cuda/cpu)
- `--mnist-data-dir` (default: "./src/datasets_utils/mnist_rotation_new"): MNIST data path

## Output

One `.npz` file per configuration (dataset/activation/bn/flip), averaged across seeds:
- `mnist_rot_fourierbn_elu_16_Normbn_noaug.npz`
- `eurosat_fourierbn_relu_16_Normbn_aug.npz`

Each file contains:
- `metadata`: Configuration info (JSON string)
- `layer_{i}_freq_{f}_bins`: Histogram bin edges for layer i, frequency f
- `layer_{i}_freq_{f}_counts`: Histogram counts (averaged across seeds)

## Loading Data

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn_noaug.npz')

# List all keys
print(data.files)

# Load specific layer and frequency
bins = data['layer_3_freq_1_bins']
counts = data['layer_3_freq_1_counts']

# Plot
plt.figure(figsize=(8, 5))
plt.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, edgecolor='black')
plt.xlabel('Norm magnitude')
plt.ylabel('Density' if args.density else 'Count')
plt.title('Layer 3, Frequency 1')
plt.show()
```

## Example: Process all datasets

```bash
# MNIST
python plots/extract_feature_norm_distributions.py --models-dir saved_models/mnist_rot_final --num-samples 128

# EuroSAT
python plots/extract_feature_norm_distributions.py --models-dir saved_models/eurosat_final --num-samples 128

# Colorectal
python plots/extract_feature_norm_distributions.py --models-dir saved_models/colorectal_hist_final --num-samples 128

# Resisc45
python plots/extract_feature_norm_distributions.py --models-dir saved_models/resisc45_final --num-samples 128
```
