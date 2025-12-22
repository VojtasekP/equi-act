# MNIST Rotation Point Cloud Dataset

This directory contains tools to convert MNIST rotation images into 2D point cloud format and load them efficiently for training equivariant neural networks.

## Overview

The point cloud conversion treats each MNIST image as a 2D point cloud where:
- Each non-zero pixel becomes a point with (x, y) coordinates
- Pixel intensity becomes the point feature
- Coordinates are normalized to [-1, 1] range
- Graph connectivity is built using kNN or radius graph

This format is compatible with `R2PointNet` and `R3PointNet` architectures in [RnNet.py](../nets/RnNet.py).

## Quick Start

### Step 1: Convert MNIST to Point Clouds

```bash
# Using default directories
python src/datasets_utils/mnist_to_pointcloud.py

# With custom directories
python src/datasets_utils/mnist_to_pointcloud.py \
    --data_dir /path/to/mnist_rotation_new \
    --output_dir /path/to/output \
    --threshold 0.01 \
    --num_workers 8
```

**Parameters:**
- `--data_dir`: Directory containing `.amat` files (default: `src/datasets_utils/mnist_rotation_new`)
- `--output_dir`: Output directory for `.npz` files (default: `src/datasets_utils/mnist_rotation_pointcloud`)
- `--threshold`: Minimum pixel intensity to include (default: 0.01)
- `--num_workers`: Parallel workers for conversion (default: 8)

**Output:**
```
mnist_rotation_pointcloud/
├── train.npz          # Training point clouds
├── test.npz           # Test point clouds
└── metadata.txt       # Dataset metadata
```

**Conversion Statistics:**
- Training: 12,000 images → ~350-400 points per image on average
- Test: 50,000 images → ~350-400 points per image on average
- File size: ~100-150 MB per split (compressed)
- Processing time: ~2-3 minutes with 8 workers

### Step 2: Load and Use the Dataset

#### Option A: Use DataModule (Recommended)

```python
from datasets_utils.data_classes import MnistRotPointCloudDataModule

# Create datamodule
dm = MnistRotPointCloudDataModule(
    batch_size=64,
    augment_rotation=True,     # Random rotation augmentation
    k_neighbors=8,              # kNN graph with k=8
    num_workers=4
)

# Setup
dm.prepare_data()
dm.setup()

# Get dataloaders
train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

# Iterate
for batch in train_loader:
    # batch is a PyG Batch object
    x = batch.x            # Features (total_points, 1)
    pos = batch.pos        # Coordinates (total_points, 2)
    edge_index = batch.edge_index  # Edges (2, num_edges)
    y = batch.y            # Labels (batch_size,)
    batch_ids = batch.batch  # Graph assignment (total_points,)
```

#### Option B: Use Dataset Directly

```python
from datasets_utils.data_classes import MnistRotPointCloudDataset
from pathlib import Path

# Create dataset
dataset = MnistRotPointCloudDataset(
    npz_path=Path("src/datasets_utils/mnist_rotation_pointcloud/train.npz"),
    augment_rotation=True,
    k_neighbors=8
)

# Get sample
data = dataset[0]
print(f"Label: {data.y}")
print(f"Points: {data.pos.shape}")      # (N, 2)
print(f"Features: {data.x.shape}")      # (N, 1)
print(f"Edges: {data.edge_index.shape}")  # (2, num_edges)
```

### Step 3: Test the Implementation

```bash
# Run comprehensive tests
python src/datasets_utils/test_pointcloud_dataset.py
```

This will:
1. Test dataset loading and augmentation
2. Test datamodule and dataloaders
3. Compare kNN vs radius graph
4. Generate visualizations in `src/datasets_utils/pointcloud_visualizations/`

## API Reference

### MnistRotPointCloudDataset

PyTorch Dataset for loading pre-converted point clouds.

**Constructor:**
```python
MnistRotPointCloudDataset(
    npz_path: Path,                  # Path to .npz file
    transform=None,                   # Optional transform (not used currently)
    augment_rotation: bool = False,   # Random rotation augmentation
    k_neighbors: int = 8,             # kNN graph parameter
    radius: float = None              # Use radius graph if provided
)
```

**Returns:** PyG `Data` object with:
- `x`: Features (num_points, 1) - pixel intensities
- `pos`: Coordinates (num_points, 2) - normalized to [-1, 1]
- `edge_index`: Graph connectivity (2, num_edges)
- `y`: Label (scalar, 0-9)

### MnistRotPointCloudDataModule

Lightning DataModule for point cloud MNIST.

**Constructor:**
```python
MnistRotPointCloudDataModule(
    batch_size: int = 64,
    data_dir: Path | str = None,      # Directory with .npz files
    seed: int = 42,
    train_fraction: float = 1.0,      # Use subset of training data
    augment_rotation: bool = True,    # Augment training data
    k_neighbors: int = 8,             # kNN graph parameter
    radius: float = None,             # Use radius graph if provided
    num_workers: int = None           # Dataloader workers
)
```

**Methods:**
- `prepare_data()`: Check files exist
- `setup()`: Create train/val/test splits (80/20 train/val split)
- `train_dataloader()`: Returns PyG DataLoader
- `val_dataloader()`: Returns PyG DataLoader
- `test_dataloader()`: Returns PyG DataLoader

## Graph Construction

Two methods are supported:

### kNN Graph (Default)
```python
dataset = MnistRotPointCloudDataset(
    npz_path,
    k_neighbors=8  # Connect each point to 8 nearest neighbors
)
```
- **Pros**: Fixed connectivity, stable training
- **Cons**: May connect distant points if local density is low
- **Use when**: You want consistent graph structure

### Radius Graph
```python
dataset = MnistRotPointCloudDataset(
    npz_path,
    radius=0.3  # Connect points within radius 0.3
)
```
- **Pros**: Respects spatial locality
- **Cons**: Variable connectivity, some points may be isolated
- **Use when**: Spatial relationships are critical

## Data Augmentation

### Rotation Augmentation
When `augment_rotation=True`, each point cloud is randomly rotated by an angle in [0, 2π):

```python
# Rotation matrix in 2D
angle = random.uniform(0, 2π)
R = [[cos(θ), -sin(θ)],
     [sin(θ),  cos(θ)]]

coords_rotated = coords @ R^T
```

This is applied:
- **Training**: Yes (if `augment_rotation=True`)
- **Validation**: No
- **Test**: No

## Integration with R2PointNet

To use with the point cloud network:

```python
from nets.RnNet import R2PointNet
from datasets_utils.data_classes import MnistRotPointCloudDataModule

# Create model
model = R2PointNet(
    n_classes=10,
    max_rot_order=4,
    grey_scale=True,
    activation_type="gated_sigmoid",
    bn="IIDbn"
)

# Create datamodule
dm = MnistRotPointCloudDataModule(batch_size=32)
dm.prepare_data()
dm.setup()

# Forward pass
batch = next(iter(dm.train_dataloader()))

# R2PointNet expects (features, coords, edge_index)
output = model(
    input=batch.x,          # (total_points, 1)
    coords=batch.pos,       # (total_points, 2)
    edge_index=batch.edge_index  # (2, num_edges)
)
```

**Note:** R2PointNet's forward signature differs from R2Net:
- `R2Net.forward(input)` - takes batched images
- `R2PointNet.forward(input, coords, edge_index, batch=batch_indices)` - takes point cloud data with batch info

## Important Limitations

### Activation Types

Point cloud networks (R2PointNet, R3PointNet) **only support Fourier-based activation types**:
- `fourier_relu_4`, `fourier_relu_8`, `fourier_relu_16`, `fourier_relu_32`
- `fourier_elu_4`, `fourier_elu_8`, `fourier_elu_16`, `fourier_elu_32`

**Other activation types are NOT supported** and will raise an error during model initialization:
- ❌ `gated_sigmoid`, `gated_shared_sigmoid`
- ❌ `norm_relu`, `norm_sigmoid`, `norm_softplus`, `norm_squash`
- ❌ `normbn_relu`, `normbn_elu`, `normbn_sigmoid`
- ❌ `normbnvec_relu`, `normbnvec_elu`, `normbnvec_sigmoid`
- ❌ `fourierbn_relu`, `fourierbn_elu`
- ❌ `non_equi_relu`, `non_equi_bn`

**Reason:** These activation types use BatchNorm layers (InnerBatchNorm, IIDBatchNorm, etc.) that expect grid-structured data with spatial dimensions (4D/5D tensors). Point clouds have shape `(num_points, features)` which is a 2D tensor, causing dimension mismatches.

**Example:**
```python
# ✅ CORRECT - Using supported activation type
model = R2PointNet(
    n_classes=10,
    activation_type="fourier_relu_8",  # Works!
    grey_scale=True
)

# ❌ INCORRECT - Using unsupported activation type
model = R2PointNet(
    n_classes=10,
    activation_type="gated_sigmoid",  # Will raise ValueError!
    grey_scale=True
)
# ValueError: Point cloud networks only support fourier_relu_* and fourier_elu_* activation types.
```

## File Formats

### .npz Format (Output)
```python
{
    'coords': object array of (N_i, 2) float32 arrays,
    'features': object array of (N_i, 1) float32 arrays,
    'labels': (N,) int64 array,
    'num_points': (N,) int32 array
}
```

- `coords[i]`: Coordinates for i-th sample (variable length)
- `features[i]`: Features for i-th sample (variable length)
- `labels[i]`: Class label (0-9)
- `num_points[i]`: Number of points in i-th sample

### .amat Format (Input)
Original MNIST rotation format:
```
pixel_1 pixel_2 ... pixel_784 label
```
Each row is a flattened 28×28 image followed by its label.

## Performance Considerations

### Memory Usage
- **Images**: 12,000 × 28 × 28 × 4 bytes ≈ 38 MB
- **Point clouds**: Variable, typically ~100-150 MB compressed
- **In-memory**: Point clouds are loaded as numpy arrays (~500 MB)

### Loading Speed
- First epoch: Slow (loads .npz files)
- Subsequent epochs: Fast (data is cached in memory)
- Use `num_workers > 0` for faster batch loading

### Graph Construction
- kNN: ~0.5-1ms per sample (fast)
- Radius: ~1-2ms per sample (slightly slower)
- Built on-the-fly during `__getitem__` (not pre-computed)

### Recommendations
- Use `persistent_workers=True` to avoid reloading data
- Use `pin_memory=True` for GPU training
- Start with `k_neighbors=8`, tune if needed
- For large batches, consider pre-computing graphs offline

## Troubleshooting

### FileNotFoundError: Point cloud files not found
**Solution:** Run the conversion script first:
```bash
python src/datasets_utils/mnist_to_pointcloud.py
```

### RuntimeError: Not enough neighbors
**Cause:** Some point clouds have fewer points than `k_neighbors`
**Solution:** Reduce `k_neighbors` or use `radius` graph:
```python
dataset = MnistRotPointCloudDataset(npz_path, k_neighbors=4)  # Reduce k
# OR
dataset = MnistRotPointCloudDataset(npz_path, radius=0.3)     # Use radius
```

### Low accuracy with point clouds
**Possible causes:**
1. Graph connectivity too sparse/dense → Tune `k_neighbors` or `radius`
2. Threshold too high → More pixels filtered out → Less information
3. No rotation augmentation → Add `augment_rotation=True`
4. Wrong network architecture → Use `R2PointNet`, not `R2Net`

### Slow training
**Solutions:**
1. Increase `num_workers` in dataloader
2. Use smaller `k_neighbors` (fewer edges)
3. Enable `persistent_workers=True`
4. Use GPU for graph construction (if available)

## Advanced Usage

### Custom Threshold
Filter more/fewer pixels during conversion:
```bash
# More selective (fewer points, faster training)
python src/datasets_utils/mnist_to_pointcloud.py --threshold 0.1

# More inclusive (more points, richer features)
python src/datasets_utils/mnist_to_pointcloud.py --threshold 0.001
```

### Subset Training
Use only a fraction of training data:
```python
dm = MnistRotPointCloudDataModule(
    train_fraction=0.25  # Use 25% of training data
)
```

### Custom Data Directory
```python
dm = MnistRotPointCloudDataModule(
    data_dir="/custom/path/to/pointclouds"
)
```

### No Augmentation
```python
dm = MnistRotPointCloudDataModule(
    augment_rotation=False  # Disable rotation augmentation
)
```

## Comparison: Images vs Point Clouds

| Aspect | Images (R2Net) | Point Clouds (R2PointNet) |
|--------|----------------|---------------------------|
| Input | 28×28 grid | ~350-400 points |
| Convolution | R2Conv (grid) | R2PointConv (point) |
| Pooling | Spatial pooling | Global pooling |
| Edges | Implicit (grid) | Explicit (graph) |
| Memory | Fixed (28×28) | Variable (~350 points) |
| Training speed | Faster | Slower (graph construction) |
| Augmentation | GridAug | PointAug |

## Citation

If you use this point cloud implementation, please cite:
```bibtex
@misc{mnist_pointcloud,
  title={MNIST Rotation Point Cloud Dataset},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## Related Files

- [mnist_to_pointcloud.py](mnist_to_pointcloud.py): Conversion script
- [data_classes.py](data_classes.py): Dataset and DataModule implementations
- [test_pointcloud_dataset.py](test_pointcloud_dataset.py): Test suite
- [RnNet.py](../nets/RnNet.py): R2PointNet architecture
- [CLAUDE.md](../../CLAUDE.md): Project documentation
