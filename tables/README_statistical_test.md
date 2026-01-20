# Statistical Test Script

This script performs statistical significance testing (paired or independent t-tests) to compare two model configurations across all datasets and augmentation settings.

## Key Changes

The script has been updated to:
- Remove `--dataset` and `--aug` parameters
- Automatically test across **all** dataset/augmentation combinations
- Support both **paired** and **independent** t-tests
- Generate comprehensive tables for all comparisons at once

## Usage

### Basic Usage (Paired t-test - default)

```bash
python tables/statistical_test.py \
  --model1 <activation1> <bn1> \
  --model2 <activation2> <bn2>
```

### Independent t-test

```bash
python tables/statistical_test.py \
  --model1 <activation1> <bn1> \
  --model2 <activation2> <bn2> \
  --test-type independent
```

### Additional Options

```bash
python tables/statistical_test.py \
  --model1 <activation1> <bn1> \
  --model2 <activation2> <bn2> \
  --test-type paired \
  --csv path/to/results.csv \
  --output-dir path/to/output \
  --no-save  # Don't save CSV/LaTeX tables
```

## Examples

### Example 1: Compare Fourier Activations (Paired)

```bash
python tables/statistical_test.py \
  --model1 fourier_relu_16 Normbn \
  --model2 fourier_elu_16 Normbn
```

Output:
```
========================================================================================================================
STATISTICAL COMPARISON: fourier_relu_16 + Normbn vs fourier_elu_16 + Normbn
Test Type: Paired t-test
========================================================================================================================

Dataset              Aug    Model 1 Mean   Model 2 Mean   Diff       p-value      Sig    N
------------------------------------------------------------------------------------------------------------------------
mnist_rot            No      98.88% ± 0.08   98.58% ± 0.04    +0.29%    0.000218   ***  5
mnist_rot            Yes     99.01% ± 0.04   98.89% ± 0.05    +0.12%    0.017621     *  5
========================================================================================================================

Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
```

### Example 2: Compare Gated Activations

```bash
python tables/statistical_test.py \
  --model1 gated_sigmoid Normbn \
  --model2 gated_shared_sigmoid Normbn
```

### Example 3: Compare Norm-based Activations (Independent)

```bash
python tables/statistical_test.py \
  --model1 norm_relu Normbn \
  --model2 normbn_relu Normbn \
  --test-type independent
```

## Output Files

The script automatically saves two files in `tables/tables/`:

1. **CSV file**: Full results with all statistics
   - Filename: `ttest_<model1>_vs_<model2>_<test_type>.csv`

2. **LaTeX table**: Formatted for inclusion in papers
   - Filename: `ttest_<model1>_vs_<model2>_<test_type>.tex`
   - Columns: Dataset, Augmentation, Mean Diff, p-value, Cohen's d, Significance

## Understanding the Output

### Console Output
- **Dataset**: Which dataset the comparison is on
- **Aug**: Whether augmentation was used (Yes/No)
- **Model 1/2 Mean**: Mean accuracy ± standard deviation
- **Diff**: Mean difference (Model 1 - Model 2) as percentage points
- **p-value**: Statistical significance (lower = more significant)
- **Sig**: Significance marker
  - `***` = p < 0.001 (highly significant)
  - `**` = p < 0.01 (very significant)
  - `*` = p < 0.05 (significant)
  - `ns` = not significant
- **N**: Number of samples (seeds) compared

### Paired vs Independent t-test

**Paired t-test** (default):
- Compares same seeds between models
- More powerful when samples are matched
- Use when models were trained with same random seeds

**Independent t-test**:
- Treats all samples as independent
- Use when samples are not paired or seeds differ

## Parameters

### Required
- `--model1 ACTIVATION BN`: First model configuration
- `--model2 ACTIVATION BN`: Second model configuration

### Optional
- `--test-type {paired,independent}`: Type of t-test (default: paired)
- `--csv PATH`: Path to results CSV (default: tables/csv_outputs/Results_with_invariance.csv)
- `--output-dir PATH`: Directory for output files (default: tables/tables)
- `--no-save`: Don't save CSV and LaTeX tables to disk

## Interpretation

### p-value
- **p < 0.001**: Highly significant (marked with ***)
- **p < 0.01**: Very significant (marked with **)
- **p < 0.05**: Significant (marked with *)
- **p ≥ 0.05**: Not significant (marked with ns)

### Cohen's d (Effect Size)
For paired t-tests, Cohen's d is calculated as:
- d = mean_difference / std_of_differences

Interpretation:
- **|d| < 0.2**: Negligible effect
- **0.2 ≤ |d| < 0.5**: Small effect
- **0.5 ≤ |d| < 0.8**: Medium effect
- **|d| ≥ 0.8**: Large effect

### t-statistic
- Positive values indicate Model 1 has higher accuracy
- Negative values indicate Model 2 has higher accuracy
- Magnitude indicates strength of the difference relative to variability

## Notes

- The script automatically iterates over all datasets and augmentation settings found in the data
- Results are only included if both models have data for that configuration
- Cohen's d provides effect size independent of sample size
- LaTeX tables use booktabs formatting for professional appearance
- For paired tests, only common seeds are compared
- Assumes approximately normal distribution of accuracies (reasonable for 5 seeds)
- Uses sample standard deviation (ddof=1)
