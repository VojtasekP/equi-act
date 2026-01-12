# Plot Saved Histograms

Script to visualize histogram data extracted by `extract_feature_norm_distributions.py`.

## Usage

### Plot all frequencies across all layers (one plot per frequency)

```bash
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn.npz \
  --output-dir plots/histogram_plots
```

**Output**: One plot per frequency showing all layers
- `mnist_rot_fourierbn_elu_16_Normbn_freq0.png`
- `mnist_rot_fourierbn_elu_16_Normbn_freq1.png`
- `mnist_rot_fourierbn_elu_16_Normbn_freq2.png`
- `mnist_rot_fourierbn_elu_16_Normbn_freq3.png`

### Plot all layers separately (one plot per layer)

```bash
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn.npz \
  --output-dir plots/histogram_plots \
  --separate-layers
```

**Output**: One plot per layer showing all frequencies
- `mnist_rot_fourierbn_elu_16_Normbn_layer1.png`
- `mnist_rot_fourierbn_elu_16_Normbn_layer2.png`
- `mnist_rot_fourierbn_elu_16_Normbn_layer3.png`
- etc.

### Plot specific layer and/or frequency

```bash
# Only layer 3
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn.npz \
  --layer 3 \
  --separate-layers

# Only frequency 1
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn.npz \
  --freq 1

# Layer 3, frequency 1 only
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_fourierbn_elu_16_Normbn.npz \
  --layer 3 \
  --freq 1 \
  --separate-layers
```

## Arguments

- `--input-file` (required): Path to `.npz` file with histogram data
- `--output-dir` (default: "plots/histogram_plots"): Directory to save plots
- `--layer` (optional): Specific layer to plot (1-based index)
- `--freq` (optional): Specific frequency to plot (0, 1, 2, 3, ...)
- `--separate-layers`: Create one plot per layer instead of one per frequency

## Examples

### Compare frequencies across layers

Show how irrep frequency 1 evolves through the network:

```bash
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_gated_sigmoid_Normbn.npz \
  --freq 1
```

This creates a single plot with subplots for each layer, all showing frequency 1.

### Compare frequencies within a layer

Show all irrep frequencies at layer 3:

```bash
python plots/plot_saved_histograms.py \
  --input-file plots/norm_distributions/mnist_rot_gated_sigmoid_Normbn.npz \
  --layer 3 \
  --separate-layers
```

This creates a single plot with subplots for each frequency at layer 3.

### Batch processing

Plot histograms for all saved files:

```bash
for file in plots/norm_distributions/*.npz; do
    echo "Processing $file..."
    python plots/plot_saved_histograms.py --input-file "$file" --output-dir plots/histogram_plots
done
```

## Plot Layout

The script automatically adjusts the subplot layout based on the number of items:
- **1-3 items**: Horizontal row (1×n)
- **4 items**: 2×2 grid
- **5+ items**: Dynamic grid (3 columns)

Each histogram shows:
- X-axis: Norm magnitude
- Y-axis: Density (if `--density` was used) or Count
- Title: Layer or frequency identifier
- Grid lines for readability

The overall figure title includes:
- Dataset name
- Activation type
- Normalization type
- Number of samples used
- Number of seeds averaged
