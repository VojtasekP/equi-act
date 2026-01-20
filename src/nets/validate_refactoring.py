"""
Validation script for RnNet refactoring.

This script ensures that the refactored classes maintain the same functionality
as the original implementation by testing instantiation and forward passes.
"""

import torch
import sys
sys.path.insert(0, 'src')

from nets.RnNet import R2Net, R2PointNet, R3Net, R3PointNet


def validate_grid_2d():
    """Test R2Net instantiation and forward pass"""
    print("Testing R2Net...")
    model = R2Net(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16, 32),
        activation_type="gated_sigmoid",
        img_size=28
    )
    model.eval()  # Set to eval mode for batch_size=1
    # Test forward pass with dummy data
    x = torch.randn(1, 3, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    print("✓ R2Net validation passed")


def validate_point_2d():
    """Test R2PointNet instantiation and forward pass"""
    print("Testing R2PointNet...")
    # Use fourier_relu_8 which is compatible with point cloud data shapes
    model = R2PointNet(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16, 32),
        activation_type="fourier_relu_8"
    )
    model.eval()  # Set to eval mode for batch_size=1
    # Test forward pass with dummy point cloud
    features = torch.randn(100, 3)  # 100 points, 3 features
    coords = torch.randn(100, 2)    # 2D coordinates
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    output = model(features, coords, edge_index)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    print("✓ R2PointNet validation passed")


def validate_grid_3d():
    """Test R3Net instantiation and forward pass"""
    print("Testing R3Net...")
    model = R3Net(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16, 32),
        activation_type="gated_sigmoid",
        img_size=28
    )
    model.eval()  # Set to eval mode for batch_size=1
    # Test forward pass with dummy 3D data
    x = torch.randn(1, 1, 28, 28, 28)
    output = model(x)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    print("✓ R3Net validation passed")


def validate_point_3d():
    """Test R3PointNet instantiation and forward pass"""
    print("Testing R3PointNet...")
    # Use fourier_relu_8 which is compatible with point cloud data shapes
    model = R3PointNet(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16, 32),
        activation_type="fourier_relu_8"
    )
    model.eval()  # Set to eval mode for batch_size=1
    # Test forward pass with dummy 3D point cloud
    features = torch.randn(100, 1)  # 100 points, 1 feature
    coords = torch.randn(100, 3)    # 3D coordinates
    edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
    output = model(features, coords, edge_index)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"
    print("✓ R3PointNet validation passed")


def validate_all_activations():
    """Test different activation types work"""
    print("\nTesting different activation types...")
    activation_types = [
        "gated_sigmoid",
        "norm_relu",
        "normbn_relu"
    ]
    for act_type in activation_types:
        print(f"  Testing {act_type}...")
        model = R2Net(
            n_classes=10,
            max_rot_order=2,
            channels_per_block=(8, 16, 32),
            activation_type=act_type,
            img_size=28
        )
        model.eval()  # Set to eval mode for batch_size=1
        x = torch.randn(1, 3, 28, 28)
        output = model(x)
        assert output.shape == (1, 10), f"Expected (1, 10) for {act_type}, got {output.shape}"
        print(f"  ✓ R2Net with {act_type} validation passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running RnNet Validation Tests")
    print("=" * 60)
    print()

    try:
        validate_grid_2d()
        validate_point_2d()
        validate_grid_3d()
        validate_point_3d()
        validate_all_activations()

        print()
        print("=" * 60)
        print("✅ All validation tests passed!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Validation failed: {e}")
        print("=" * 60)
        raise
