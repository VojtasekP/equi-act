# Point Cloud Quick Reference

## 🚀 One-Line Switch

Change from grid to point cloud by changing just the dataset flag:

```bash
# Grid (Original)
python src/train.py --dataset mnist_rot

# Point Cloud (New)
python src/train.py --dataset mnist_rot_p
```

Everything else stays the same!

## 📋 Setup Checklist

- [ ] Convert MNIST to point clouds: `python src/datasets_utils/mnist_to_pointcloud.py`
- [ ] Verify setup: `python verify_pointcloud_setup.py`
- [ ] Test integration: `python test_train_pointcloud.py`
- [ ] Train model: `python src/train.py --dataset mnist_rot_p --epochs 50`

## 🎯 Common Commands

### Convert Data
```bash
python src/datasets_utils/mnist_to_pointcloud.py
```

### Test Dataset
```bash
python src/datasets_utils/test_pointcloud_dataset.py
```

### Train (Basic)
```bash
python src/train.py \
    --dataset mnist_rot_p \
    --epochs 50 \
    --batch_size 64
```

### Train (Full Options)
```bash
python src/train.py \
    --dataset mnist_rot_p \
    --epochs 50 \
    --batch_size 64 \
    --max_rot_order 4 \
    --activation_type gated_sigmoid \
    --k_neighbors 8 \
    --lr 0.001
```

### Train with Radius Graph
```bash
python src/train.py \
    --dataset mnist_rot_p \
    --radius 0.3
```

## 🔑 Key Arguments

| Argument | Grid (mnist_rot) | Point Cloud (mnist_rot_p) |
|----------|------------------|---------------------------|
| `--dataset` | `mnist_rot` | `mnist_rot_p` |
| `--k_neighbors` | N/A | `8` (default) |
| `--radius` | N/A | `None` (optional) |
| Everything else | ✅ Same | ✅ Same |

## 📊 Side-by-Side Comparison

```bash
# Grid version
python src/train.py \
    --dataset mnist_rot \
    --epochs 50 \
    --batch_size 64 \
    --max_rot_order 4 \
    --seed 42

# Point cloud version
python src/train.py \
    --dataset mnist_rot_p \
    --epochs 50 \
    --batch_size 64 \
    --max_rot_order 4 \
    --seed 42 \
    --k_neighbors 8
```

## 🐛 Quick Fixes

### Files not found?
```bash
python src/datasets_utils/mnist_to_pointcloud.py
```

### k >= num_points error?
```bash
# Reduce k or use radius
--k_neighbors 4
# OR
--radius 0.3
```

### Slow training?
```bash
--num_workers 8
--k_neighbors 6
```

### Out of memory?
```bash
--batch_size 32
--k_neighbors 6
```

## 📁 File Locations

| Purpose | Location |
|---------|----------|
| Training script | [src/train.py](src/train.py) |
| Conversion script | [src/datasets_utils/mnist_to_pointcloud.py](src/datasets_utils/mnist_to_pointcloud.py) |
| Dataset class | [src/datasets_utils/data_classes.py](src/datasets_utils/data_classes.py) |
| Point cloud data | `src/datasets_utils/mnist_rotation_pointcloud/` |
| Full guide | [TRAIN_POINTCLOUD_GUIDE.md](TRAIN_POINTCLOUD_GUIDE.md) |
| Setup guide | [MNIST_POINTCLOUD_SETUP.md](MNIST_POINTCLOUD_SETUP.md) |

## 💡 Tips

1. **Start simple**: Use `--k_neighbors 8` first
2. **Compare fairly**: Same hyperparameters for grid vs point cloud
3. **Monitor speed**: Point cloud is slower but more flexible
4. **Try radius**: `--radius 0.25` for spatial graphs
5. **Test fast**: Use `--epochs 1 --save False` for quick tests

## 🎓 What Gets Changed Automatically?

When you use `--dataset mnist_rot_p`:
- ✅ DataModule: `MnistRotPointCloudDataModule`
- ✅ Model: `R2PointNet` (via `LitHnnPointCloud`)
- ✅ Input: Point clouds with graphs
- ✅ Forward pass: `model(x, pos, edge_index)`

What stays the same:
- ✅ All hyperparameters
- ✅ Training loop
- ✅ Logging & checkpointing
- ✅ Sweep compatibility

---

**Need more info?** See [TRAIN_POINTCLOUD_GUIDE.md](TRAIN_POINTCLOUD_GUIDE.md)
