#!/usr/bin/env python3
"""
Quick verification script to check if point cloud implementation is working.

This script performs basic checks without running the full conversion or training.

Usage:
    python verify_pointcloud_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

print("=" * 70)
print("Point Cloud Setup Verification")
print("=" * 70)

# Check 1: Import dependencies
print("\n[1/5] Checking dependencies...")
try:
    import numpy as np
    import torch
    from torch_geometric.data import Data as PyGData
    from torch_geometric.nn import knn_graph, radius_graph
    print("✓ All dependencies available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("  Please install: pip install torch-geometric")
    sys.exit(1)

# Check 2: Import custom modules
print("\n[2/5] Checking custom modules...")
try:
    from datasets_utils.data_classes import (
        MnistRotPointCloudDataset,
        MnistRotPointCloudDataModule
    )
    print("✓ Point cloud dataset modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Check 3: Import R2PointNet
print("\n[3/5] Checking R2PointNet architecture...")
try:
    from nets.RnNet import R2PointNet
    print("✓ R2PointNet architecture available")
except ImportError as e:
    print(f"✗ Failed to import R2PointNet: {e}")
    sys.exit(1)

# Check 4: Verify conversion script exists
print("\n[4/5] Checking conversion script...")
conversion_script = Path("src/datasets_utils/mnist_to_pointcloud.py")
if conversion_script.exists():
    print(f"✓ Conversion script found: {conversion_script}")
else:
    print(f"✗ Conversion script not found: {conversion_script}")
    sys.exit(1)

# Check 5: Check if point cloud data exists
print("\n[5/5] Checking point cloud data...")
data_dir = Path("src/datasets_utils/mnist_rotation_pointcloud")
train_file = data_dir / "train.npz"
test_file = data_dir / "test.npz"

if train_file.exists() and test_file.exists():
    print(f"✓ Point cloud data found in {data_dir}/")

    # Quick test load
    try:
        dm = MnistRotPointCloudDataModule(batch_size=8, num_workers=0)
        dm.prepare_data()
        dm.setup()

        # Get one batch
        batch = next(iter(dm.train_dataloader()))

        print(f"  - Train samples: {len(dm.mnist_train)}")
        print(f"  - Val samples: {len(dm.mnist_val)}")
        print(f"  - Test samples: {len(dm.mnist_test)}")
        print(f"  - Sample batch shape: {batch.num_graphs} graphs, {batch.pos.shape[0]} points")
        print("✓ DataModule works correctly")

        # Test R2PointNet forward pass
        model = R2PointNet(
            n_classes=10,
            max_rot_order=2,
            channels_per_block=(8, 16),
            grey_scale=True,
            mnist=True
        )

        output = model(batch.x, batch.pos, batch.edge_index)
        print(f"✓ R2PointNet forward pass works (output shape: {output.shape})")

    except Exception as e:
        print(f"✗ Error testing datamodule: {e}")
        sys.exit(1)
else:
    print(f"✗ Point cloud data not found in {data_dir}/")
    print(f"\n  To generate point cloud data, run:")
    print(f"    python src/datasets_utils/mnist_to_pointcloud.py")
    print(f"\n  This will convert MNIST images to point clouds (~2-3 minutes)")

    # Check if original MNIST data exists
    mnist_dir = Path("src/datasets_utils/mnist_rotation_new")
    if mnist_dir.exists():
        print(f"\n  ✓ Original MNIST data found in {mnist_dir}/")
    else:
        print(f"\n  ✗ Original MNIST data not found in {mnist_dir}/")
        print(f"     It will be downloaded automatically when you run mnist_to_pointcloud.py")

# Summary
print("\n" + "=" * 70)
if train_file.exists() and test_file.exists():
    print("✓ Setup verification PASSED - You're ready to train!")
    print("\nNext steps:")
    print("  1. Test the implementation:")
    print("     python src/datasets_utils/test_pointcloud_dataset.py")
    print("\n  2. Train a model:")
    print("     python examples/train_mnist_pointcloud.py --max_epochs 10")
else:
    print("⚠ Setup verification INCOMPLETE - Point cloud data not generated")
    print("\nNext step:")
    print("  1. Generate point cloud data:")
    print("     python src/datasets_utils/mnist_to_pointcloud.py")
    print("\n  2. Re-run this verification:")
    print("     python verify_pointcloud_setup.py")

print("=" * 70)
