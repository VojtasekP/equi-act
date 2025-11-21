# equi-act

## Setup
### Recomended way:

install uv if needed: https://github.com/astral-sh/uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
install the dependancies and activate env with uv
```bash
uv sync
source .venv/bin/activate
```

### Pip and conda
install with pip:
```bash
pip install -r requirements.txt
```
install with conda:
```bash
conda env create -f env.yml
conda activate equi-act
```
## Run
Main script in src/train.py
```bash
python src/train.py \
  --dataset <mnist_rot|resisc45|colorectal_hist|eurosat> \
  --activation_type <see below> \
  --bn <IIDbn|Normbn|FieldNorm|GNormBatchNorm> \
  [--train_subset_fraction 0.5] \
  [--flip True] \
  [--precision 16-mixed] \
  # and other flags
```

Quick samples to start:
```bash
# Rotated MNIST
python src/train.py --dataset mnist_rot --batch_size 64 --activation_type gated_sigmoid --bn Normbn

# EuroSAT with 25% train split
python src/train.py --dataset eurosat --train_subset_fraction 0.25 --activation_type fourier_relu_16 --bn Normbn

# Resisc45 with flips
python src/train.py --dataset resisc45 --flip True --activation_type gated_shared_sigmoid --bn IIDbn
```

## Choices
- `--dataset`: mnist_rot, resisc45, colorectal_hist, eurosat (`--train_subset_fraction` in (0,1] for smaller train set size)
- `--activation_type`: gated_sigmoid, gated_shared_sigmoid, norm_relu, norm_squash, fourier_relu_{4,8,16,32}, fourier_elu_{4,8,16,32}, non_equi_relu, non_equi_bn
- `--bn`: IIDbn, Normbn, FieldNorm, GNormBatchNorm

Other flags: invariance logging (`--invar_error_logging`, `--invar_check_every_n_epochs`, `--num_of_angles`, `--num_of_batches_to_use_for_invar_logging`), export paths (`--model_export_dir`, `--model_export_subdir`), W&B project name (`--project`).

## Project structure (key bits)
```
src/
  train.py                 # Lightning training script / CLI entrypoint
  nets/RnNet.py            # R2Net architecture and layers
  datasets_utils/
    data_classes.py        # LightningDataModules for mnist_rot, resisc45, colorectal_hist, eurosat
    mnist_download.py      # Helper to fetch rotated MNIST
```
