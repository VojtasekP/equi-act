# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Equi-act is a research project implementing equivariant neural networks for image classification using the `escnn` library. The project trains rotation-equivariant CNNs on datasets like rotated MNIST, EuroSAT, Resisc45, and colorectal histology images.

## Setup

**Python 3.10 is required** (pinned in pyproject.toml).

Recommended setup with `uv`:
```bash
uv sync --python 3.10
source .venv/bin/activate
```

Alternative with pip:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Training

Main entrypoint is `src/train.py`:

```bash
python src/train.py \
  --dataset <mnist_rot|resisc45|colorectal_hist|eurosat> \
  --activation_type <see activation types below> \
  --bn <IIDbn|Normbn|FieldNorm|GNormBatchNorm> \
  [--train_subset_fraction 0.5] \
  [--flip True] \
  [--precision 16-mixed]
```

Example commands:
```bash
# Rotated MNIST
python src/train.py --dataset mnist_rot --batch_size 64 --activation_type gated_sigmoid --bn Normbn

# EuroSAT with 25% train split
python src/train.py --dataset eurosat --train_subset_fraction 0.25 --activation_type fourier_relu_16 --bn Normbn
```

### Activation Types
- Gated: `gated_sigmoid`, `gated_shared_sigmoid`
- Norm-based: `norm_relu`, `norm_squash`
- Fourier: `fourier_relu_{4,8,16,32}`, `fourier_elu_{4,8,16,32}`, `fourierbn_relu_{16,32,64,128}`, `fourierbn_elu_{16,32,64,128}`
- Norm+BN: `normbn_relu`, `normbn_elu`, `normbn_sigmoid`, `normbnvec_relu`, `normbnvec_elu`, `normbnvec_sigmoid`
- Non-equivariant: `non_equi_relu`, `non_equi_bn`

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Test configuration is in `tests/conftest.py`.

## Wandb Sweeps

Sweep configurations are in `src/sweeps_configs/`. To run a sweep:

```bash
wandb sweep src/sweeps_configs/sweep_mnist_final.yaml
wandb agent <sweep_id>
```

Sweep files define hyperparameter grids for different datasets (mnist, eurosat, colorectal, resisc45).

## Architecture: RnNet.py (Critical)

**Location**: `src/nets/RnNet.py`

This is the core architecture file implementing equivariant neural networks using group theory and the `escnn` library. The code has been recently refactored using OOP patterns.

### Class Hierarchy

```
RnNet (Abstract Base)
  ├── R2NetBase (2D base, template methods)
  │     ├── R2Net (grid convolutions with R2Conv)
  │     └── R2PointNet (point cloud convolutions with R2PointConv)
  ├── R3Net (3D grid, not yet refactored)
  └── R3PointNet (3D point cloud, not yet refactored)
```

### Key Design Patterns

**R2NetBase**: Uses **Template Method Pattern** + **Strategy Pattern**
- Implements all block creation methods (`make_gated_block`, `make_norm_block`, `make_fourier_block`, etc.) as templates
- Subclasses implement only:
  - `_create_conv_layer()`: Returns appropriate conv layer (R2Conv vs R2PointConv)
  - `_get_conv_kwargs()`: Returns conv-specific parameters (kernel_size/padding for grid, empty dict for point cloud)
  - `_build_pooling()`: Grid uses spatial pooling, point cloud uses IdentityModule

**Benefits of this refactoring**:
- Eliminated ~400 lines of duplicated code between R2Net and R2PointNet
- Single source of truth for block creation logic
- Easy to add new activation types (only modify R2NetBase)
- R3Net and R3PointNet are NOT yet refactored (still inherit directly from RnNet)

### Important Implementation Details

1. **Activation Block Types**: The codebase supports 10+ different activation block types, each with specific field construction logic. These are implemented in R2NetBase as template methods.

2. **Field Types**: Uses `escnn.nn.FieldType` to represent equivariant feature spaces. Fields are split into:
   - **Scalar fields**: Trivial representations (invariant features)
   - **Vector fields**: Non-trivial irreducible representations (equivariant features)

3. **Batch Normalization**: Different BN types handle equivariance differently:
   - `IIDbn`: Independent BN per channel
   - `Normbn`: Norm-based BN preserving equivariance
   - `FieldNorm`: Normalization over entire field
   - `GNormBatchNorm`: Group norm + batch norm

4. **Custom Modules** in `src/nets/new_layers.py`:
   - `NormNonlinearityWithBN`: Combines norm-based non-linearity with batch normalization
   - `FourierPointwiseInnerBn`: Fourier activation with inner batch normalization

### When Modifying RnNet.py

- **R2 variants (R2Net, R2PointNet)**: Modifications to block logic go in `R2NetBase`, NOT the subclasses
- **R3 variants (R3Net, R3PointNet)**: Still have duplicated code - changes must be made in both classes
- **Adding new activation types**:
  1. Add mapping to `ACT_MAP` at top of file
  2. Add new `make_*_block()` method in R2NetBase (for R2) or in R3Net/R3PointNet individually (for R3)
  3. Update `_build_block()` in RnNet to call the new method

### Testing RnNet Changes

Validation script at `src/nets/validate_refactoring.py`:
```bash
python src/nets/validate_refactoring.py
```

Tests instantiation and forward passes for R2Net, R2PointNet, R3Net, R3PointNet with multiple activation types.

## Data Processing

**Location**: `src/datasets_utils/data_classes.py`

Implements PyTorch Lightning DataModules for:
- `MnistRotDataModule`: Rotated MNIST dataset
- `Resisc45DataModule`: Remote sensing image classification
- `ColorectalHistDataModule`: Colorectal histology images
- `EuroSATDataModule`: Satellite imagery classification

Each DataModule handles:
- Dataset downloading/preprocessing
- Train/val/test splits
- Data augmentation (rotation, flipping if `--flip True`)
- Normalization
- Batch loading

## Results and Analysis

**Location**: `tables/`

Scripts for generating LaTeX tables and analyzing results:
- `ds_acc_results.py`: Accuracy results aggregation
- `invariance_to_latex.py`: Convert invariance metrics to LaTeX tables
- `to_latex.py`: General results to LaTeX conversion
- `combine_invariance_and_results.py`: Merge invariance and accuracy results

Output directories:
- `tables/csv_outputs/`: CSV result files
- `tables/tex_outputs/`: LaTeX tables

## Equivariance Testing

**Location**: `src/nets/equivariance_metric.py`

Functions for testing rotational equivariance of trained models:
- Applies rotations to inputs
- Compares rotated predictions with predictions on rotated features
- Computes equivariance error metrics

Used during training with flags:
- `--invar_error_logging`: Enable invariance logging
- `--invar_check_every_n_epochs N`: Check every N epochs
- `--num_of_angles 32`: Number of rotation angles to test

## Important Notes

1. **GPU Usage**: Default is `cuda:1` (second GPU) if available, see `src/train.py:31`

2. **Channel Calculation**: `src/nets/calculate_channels.py` contains `adjust_channels()` which dynamically adjusts channel counts based on rotation order, activation type, and layer position. This is critical for maintaining consistent feature dimensions.

3. **Residual Connections**: Optional via `--residual True`. Implemented in `ResidualBlock` class with skip connections.

4. **escnn Library**: Custom fork from `https://github.com/QUVA-Lab/escnn.git` (see pyproject.toml). This library provides group-equivariant operations.

5. **Point Cloud Support**: R2PointNet and R3PointNet use point convolutions instead of grid convolutions. They:
   - Take (features, coords, edge_index) as input
   - Use `R2PointConv`/`R3PointConv` layers
   - Apply global max pooling instead of spatial pooling
   - Forward signature: `forward(input, coords, edge_index)`

6. **Lightning Integration**: Training uses PyTorch Lightning (`LitHnn` in train.py) with:
   - WandB logging
   - Early stopping
   - Model checkpointing
   - Learning rate monitoring
