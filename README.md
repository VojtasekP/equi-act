# Equi-Act: Equivariant Neural Networks for Image Classification

Implementation of rotation-equivariant CNNs using the `escnn` library. Explores various equivariant activation functions (gated, norm-based, Fourier) and batch normalization strategies.

## Installation

**Requires Python 3.10**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (if not installed)
uv sync --python 3.10                             # create venv and install dependencies
source .venv/bin/activate
```

## Project Structure

```
src/
├── train.py                     # Main training entrypoint
├── invariance_test.py           # Post-training invariance evaluation
├── nets/
│   ├── RnNet.py                 # Equivariant architectures (R2Net, R3Net, PointNet variants)
│   ├── baseline_resnet.py       # ResNet18 baseline
│   ├── new_layers.py            # Custom layers (NormBN, FourierBN)
│   └── equivariance_metric.py   # Equivariance error functions
├── datasets_utils/
│   └── data_classes.py          # DataModules (MNIST, EuroSAT, Colorectal, Resisc45)
└── sweeps_configs/              # WandB hyperparameter sweep configs

tables/
├── csv_outputs/                 # Experiment results
├── tex_outputs/                 # Generated LaTeX tables
├── ds_acc_results.py            # Fetch results from WandB
└── generate_latex_tables.py     # Create LaTeX tables

figures/                         # Thesis figures
```

## Training

```bash
python src/train.py --dataset mnist_rot --activation_type fourierbn_relu_16 --bn Normbn
```

### Datasets

`--dataset`: `mnist_rot` | `eurosat` | `colorectal_hist` | `resisc45`

### Activation Types

| Category | Options |
|----------|---------|
| Gated | `gated_sigmoid`, `gated_shared_sigmoid` |
| Norm-based | `norm_relu`, `norm_squash`, `normbn_relu`, `normbnvec_relu` |
| Fourier | `fourier_relu_{4,8,16,32}`, `fourierbn_relu_{8,16,32,64,128}` |
| Non-equivariant | `non_equi_relu`, `non_equi_bn` |

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | `equivariant`, `resnet18`, `pointcloud` | `equivariant` |
| `--bn` | `IIDbn`, `Normbn`, `FieldNorm`, `GNormBatchNorm` | `Normbn` |
| `--max_rot_order` | Rotation discretization order | `3` |
| `--flip` | Include reflections (O(2) vs SO(2)) | `False` |
| `--epochs` | Training epochs | `30` |
| `--batch_size` | Batch size | `4` |
| `--lr` | Learning rate | `1e-3` |
| `--train_subset_fraction` | Fraction of training data to use | `1.0` |
| `--precision` | `32-true`, `16-mixed` | `32-true` |

### Examples

```bash
# Fourier activation with inner batch norm
python src/train.py --dataset mnist_rot --activation_type fourierbn_relu_16 --bn Normbn --epochs 50

# Subset of training data
python src/train.py --dataset eurosat --activation_type gated_sigmoid --train_subset_fraction 0.25

# Baseline comparison
python src/train.py --dataset mnist_rot --model_type resnet18
```

## Invariance Evaluation

```bash
python src/invariance_test.py --checkpoint saved_models/model.ckpt --dataset mnist_rot
```

## WandB Sweeps

```bash
wandb sweep src/sweeps_configs/sweep_mnist_final.yaml
wandb agent <sweep_id>
```

## Results

```bash
python tables/ds_acc_results.py           # Fetch from WandB
python tables/generate_latex_tables.py    # Generate LaTeX
```

---

*This README was generated with [Claude Code](https://claude.ai/code).*
