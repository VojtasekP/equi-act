# Comparing Histogram Distributions Between Models

This guide explains how to use `compare_histograms.py` to visualize differences between two model architectures (e.g., Fourier vs FourierBN).

## Purpose

The `compare_histograms.py` script creates comparison visualizations that show histogram distributions side-by-side or overlaid, making it easy to see how different activation functions affect feature norm distributions across layers and frequencies.

**Key Feature**: All histograms are automatically normalized to have unit area (integral = 1), ensuring fair comparison even when models have different numbers of feature channels per frequency.

## Prerequisites

1. You must first extract feature norm distributions using `extract_feature_norm_distributions.py`
2. The extracted `.npz` files should be in `plots/norm_distributions/`

## Usage

### Basic Syntax

```bash
python plots/compare_histograms.py \
  --model1 <path_to_npz1> \
  --model2 <path_to_npz2> \
  [--plot-type overlay|sidebyside|both] \
  [--layer N] [--freq F] \
  [--output-dir DIR]
```

### Plot Types

- **`overlay`**: Both histograms overlaid in same plot (best for seeing differences)
- **`sidebyside`**: Two separate subplots (best for detailed comparison)
- **`both`**: Generate both overlay and side-by-side versions

## Examples

### Example 1: Grid Comparison (All Layers and Frequencies)

Compare Fourier vs FourierBN across all layers and frequencies:

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --plot-type overlay
```

**Output**: `plots/comparison_plots/mnist_rot_fourier_relu_16_vs_fourierbn_relu_16_grid_overlay.png`

This creates a multi-panel figure with one subplot per (layer, frequency) combination, showing both models overlaid.

### Example 2: Single Layer/Frequency Comparison

Focus on a specific layer and frequency:

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --layer 3 --freq 1 \
  --plot-type both
```

**Output**:
- `mnist_rot_fourier_relu_16_vs_fourierbn_relu_16_L3_F1_overlay.png`
- `mnist_rot_fourier_relu_16_vs_fourierbn_relu_16_L3_F1_sidebyside.png`

### Example 3: Compare Gated Activations

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_gated_sigmoid_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_gated_shared_sigmoid_Normbn.npz \
  --plot-type overlay
```

### Example 4: Compare Norm-based Activations

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_norm_relu_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_normbn_relu_Normbn.npz \
  --plot-type sidebyside
```

### Example 5: High-DPI PDF for Publication

Generate publication-quality PDF at 300 DPI:

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --plot-type overlay \
  --format pdf \
  --dpi 300
```

**Output**: `plots/comparison_plots/mnist_rot_fourier_relu_16_vs_fourierbn_relu_16_grid_overlay.pdf`

Or generate both PNG and PDF:

```bash
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --plot-type overlay \
  --format both \
  --dpi 300
```

**Output**: Both `.png` and `.pdf` files

## Parameters

### Required

- `--model1 PATH`: Path to first model's `.npz` file
- `--model2 PATH`: Path to second model's `.npz` file

### Optional

- `--plot-type {overlay,sidebyside,both}`: Type of comparison (default: `both`)
- `--layer N`: Specific layer index to plot (1-based)
- `--freq F`: Specific frequency to plot (0, 1, 2, 3...)
- `--output-dir DIR`: Output directory (default: `plots/comparison_plots`)
- `--output-name NAME`: Custom output filename prefix
- `--format {png,pdf,both}`: Output format (default: `both`)
- `--dpi N`: DPI for output images (default: `300`)

## Understanding the Output

### Grid Plots

When you don't specify `--layer` and `--freq`, the script creates a grid showing all combinations:

- **Rows & Columns**: Organized by layer and frequency
- **Each subplot**: Shows distribution for one (layer, frequency) pair
- **Colors**:
  - Blue: Model 1 (e.g., `fourier_relu_16`)
  - Red/Coral: Model 2 (e.g., `fourierbn_relu_16`)

### Overlay Plots

- Both histograms shown with transparency (alpha=0.5)
- **Normalization**: Both histograms have equal total area (1.0), so peak heights are directly comparable
- **Key to look for**:
  - Overlapping regions: Models behave similarly
  - Separated peaks: Models produce different norm distributions
  - Width differences: Different variance in feature magnitudes
  - Peak height differences: Different concentration of values (now meaningful after normalization!)

### Side-by-Side Plots

- Two separate panels with matched axes
- Easier to see exact histogram shapes
- Same x and y ranges for direct comparison

## Interpreting Differences

### Important: All Histograms Are Normalized

**All histograms are normalized to have unit area (integral = 1)**. This means:
- Peak heights are directly comparable between models
- Higher peaks = more concentrated distribution (values cluster around that norm)
- Lower, wider distributions = more spread out values
- The total "volume" under both curves is identical, so differences in shape are meaningful

### What to Look For

1. **Peak Locations**:
   - Shifted peaks indicate different typical feature magnitudes
   - FourierBN often shows more concentrated distributions

2. **Distribution Width**:
   - Narrower distributions = more consistent feature norms
   - Wider distributions = higher variance

3. **Multiple Peaks**:
   - May indicate different "modes" of activation
   - Can reveal bimodal behavior

4. **Tail Behavior**:
   - Long tails = occasional extreme values
   - Heavy tails may indicate numerical instability

5. **Layer-wise Changes**:
   - Compare early vs late layers
   - Look for progressive concentration or spreading

### Example Findings

**Fourier vs FourierBN**:
- FourierBN typically shows narrower, more concentrated distributions
- The inner batch normalization in FourierBN stabilizes feature magnitudes
- Early layers may show larger differences than late layers

**Norm-based vs Norm-BN**:
- Norm-BN variants often have more peaked distributions
- Better normalization leads to tighter distribution control

## Workflow: From Extraction to Comparison

### Step 1: Extract Distributions

```bash
python plots/extract_feature_norm_distributions.py \
  --models-dir /path/to/checkpoints \
  --output-dir plots/norm_distributions \
  --num-samples 256 \
  --num-bins 100 \
  --density
```

This creates `.npz` files like:
- `mnist_rot_fourier_relu_16_Normbn.npz`
- `mnist_rot_fourierbn_relu_16_Normbn.npz`

### Step 2: Compare Models

```bash
# Quick overview
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --plot-type overlay

# Detailed analysis of specific layer
python plots/compare_histograms.py \
  --model1 plots/norm_distributions/mnist_rot_fourier_relu_16_Normbn.npz \
  --model2 plots/norm_distributions/mnist_rot_fourierbn_relu_16_Normbn.npz \
  --layer 1 --freq 1 --plot-type both
```

### Step 3: Analyze Results

- Look at grid plots to identify interesting layer/frequency pairs
- Generate detailed plots for specific combinations
- Compare across different datasets (mnist_rot, eurosat, colorectal_hist)

## Tips

1. **Start with overlay grid plots** to get an overview of all differences
2. **Focus on middle layers** (layers 2-4) where differences are often most pronounced
3. **Compare frequency 0 (scalars)** vs higher frequencies (vectors) separately
4. **Use side-by-side plots** for publication-quality figures
5. **Generate comparisons for all datasets** to see if patterns are consistent

## Troubleshooting

### Missing Data

If you see warnings like "No data for Layer X, Frequency Y":
- Check that both `.npz` files have data for that combination
- Verify the extraction step completed successfully
- Check `metadata` in the `.npz` files

### Empty Plots

If plots appear empty:
- Verify `.npz` files are not corrupted: `python -c "import numpy as np; d=np.load('file.npz'); print(d.files)"`
- Check that `num_bins` in extraction was > 0
- Ensure models were trained and checkpoints exist

### Mismatched Scales

If one histogram is barely visible:
- This indicates a real difference in distribution scales
- Check the y-axis values to understand the magnitude difference
- Consider whether this indicates a problem with one model

## Related Scripts

- `extract_feature_norm_distributions.py`: Extract norm distributions from checkpoints
- `plot_saved_histograms.py`: Plot histograms for a single model
- `plot_kde_by_frequency.py`: Create KDE plots (smoothed distributions)

## Output Organization

```
plots/
├── norm_distributions/          # Extracted .npz files
│   ├── mnist_rot_fourier_relu_16_Normbn.npz
│   └── mnist_rot_fourierbn_relu_16_Normbn.npz
├── comparison_plots/            # Comparison visualizations
│   ├── mnist_rot_fourier_relu_16_vs_fourierbn_relu_16_grid_overlay.png
│   ├── mnist_rot_..._L3_F1_overlay.png
│   └── mnist_rot_..._L3_F1_sidebyside.png
├── histogram_plots/             # Single-model histograms
└── kde_plots/                   # Single-model KDE plots
```
