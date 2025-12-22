"""
Test to verify that the edge_index propagation fix works.
This creates a minimal test that would have failed with the original error.
"""

import torch
import sys
sys.path.insert(0, 'src')

from nets.RnNet import R2PointNet

def test_edge_index_propagation():
    """
    Test that R2PointConv layers receive edge_index properly.
    The original error was:
    TypeError: _RdPointConv.forward() missing 1 required positional argument: 'edge_index'
    """
    print("Testing edge_index propagation through PointCloudSequentialModule...")

    # Create a simple R2PointNet with minimal layers
    # Using minimal architecture to isolate the fix
    model = R2PointNet(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16),
        kernels_per_block=(3, 3),
        paddings_per_block=(1, 1),
        activation_type="non_equi_relu",  # Avoids BatchNorm issues
        grey_scale=True
    )
    model.eval()

    # Create minimal point cloud data
    num_points = 50
    features = torch.randn(num_points, 1)  # Grey scale
    coords = torch.randn(num_points, 2)     # 2D coordinates
    edge_index = torch.randint(0, num_points, (2, 100))  # 100 edges

    try:
        # This would fail with the original code:
        # TypeError: _RdPointConv.forward() missing 1 required positional argument: 'edge_index'
        output = model(features, coords, edge_index)
        print(f"✅ Success! Output shape: {output.shape}")
        print("✅ edge_index is properly propagated to R2PointConv layers")
        return True
    except TypeError as e:
        if "edge_index" in str(e):
            print(f"❌ FAILED: {e}")
            print("❌ edge_index is NOT being propagated properly")
            return False
        raise

if __name__ == "__main__":
    success = test_edge_index_propagation()
    sys.exit(0 if success else 1)
