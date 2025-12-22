#!/usr/bin/env python3
"""
Quick test to verify train.py works with point cloud dataset.

This script does a minimal training run (1 epoch) to verify the integration.

Usage:
    # First ensure point cloud data exists
    python src/datasets_utils/mnist_to_pointcloud.py

    # Then test training
    python test_train_pointcloud.py
"""

import subprocess
import sys
from pathlib import Path

def check_pointcloud_data():
    """Check if point cloud data exists"""
    data_dir = Path("src/datasets_utils/mnist_rotation_pointcloud")
    train_file = data_dir / "train.npz"
    test_file = data_dir / "test.npz"

    if not train_file.exists() or not test_file.exists():
        print(f"ERROR: Point cloud data not found in {data_dir}/")
        print("\nPlease run the conversion script first:")
        print("  python src/datasets_utils/mnist_to_pointcloud.py")
        return False

    print(f"✓ Found point cloud data in {data_dir}/")
    return True


def test_grid_mnist():
    """Test with regular grid-based MNIST (baseline)"""
    print("\n" + "="*70)
    print("Test 1: Grid-based MNIST (mnist_rot)")
    print("="*70)

    cmd = [
        "python", "src/train.py",
        "--dataset", "mnist_rot",
        "--epochs", "1",
        "--batch_size", "32",
        "--channels_per_block", "8", "16",
        "--kernels_per_block", "3", "3",
        "--paddings_per_block", "1", "1",
        "--activation_type", "gated_sigmoid",
        "--save", "False",
        "--patience", "100",
        "--invar_error_logging", "False"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("✓ Grid-based MNIST test PASSED")
        return True
    else:
        print("✗ Grid-based MNIST test FAILED")
        return False


def test_pointcloud_mnist():
    """Test with point cloud MNIST (mnist_rot_p)"""
    print("\n" + "="*70)
    print("Test 2: Point Cloud MNIST (mnist_rot_p)")
    print("="*70)

    cmd = [
        "python", "src/train.py",
        "--dataset", "mnist_rot_p",
        "--epochs", "1",
        "--batch_size", "32",
        "--channels_per_block", "8", "16",
        "--kernels_per_block", "3", "3",
        "--paddings_per_block", "1", "1",
        "--activation_type", "gated_sigmoid",
        "--k_neighbors", "8",
        "--save", "False",
        "--patience", "100"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("✓ Point cloud MNIST test PASSED")
        return True
    else:
        print("✗ Point cloud MNIST test FAILED")
        return False


def test_pointcloud_radius_graph():
    """Test with point cloud MNIST using radius graph"""
    print("\n" + "="*70)
    print("Test 3: Point Cloud MNIST with Radius Graph")
    print("="*70)

    cmd = [
        "python", "src/train.py",
        "--dataset", "mnist_rot_p",
        "--epochs", "1",
        "--batch_size", "32",
        "--channels_per_block", "8", "16",
        "--kernels_per_block", "3", "3",
        "--paddings_per_block", "1", "1",
        "--activation_type", "gated_sigmoid",
        "--radius", "0.3",  # Use radius graph instead of kNN
        "--save", "False",
        "--patience", "100"
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("✓ Radius graph test PASSED")
        return True
    else:
        print("✗ Radius graph test FAILED")
        return False


def main():
    print("="*70)
    print("Testing train.py Point Cloud Integration")
    print("="*70)

    # Check data exists
    if not check_pointcloud_data():
        sys.exit(1)

    # Run tests
    results = []

    print("\nNote: Each test runs for 1 epoch only (for speed)")
    print("Full training would use more epochs (e.g., --epochs 50)")

    # Test 1: Grid-based (baseline)
    results.append(("Grid MNIST", test_grid_mnist()))

    # Test 2: Point cloud with kNN
    results.append(("Point Cloud kNN", test_pointcloud_mnist()))

    # Test 3: Point cloud with radius graph
    results.append(("Point Cloud Radius", test_pointcloud_radius_graph()))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25s} {status}")

    all_passed = all(passed for _, passed in results)

    print("="*70)
    if all_passed:
        print("✓ All tests PASSED!")
        print("\nYou can now train models with:")
        print("  Grid:        python src/train.py --dataset mnist_rot --epochs 50")
        print("  Point Cloud: python src/train.py --dataset mnist_rot_p --epochs 50")
    else:
        print("✗ Some tests FAILED")
        print("Please check the error messages above")
        sys.exit(1)


if __name__ == "__main__":
    main()
