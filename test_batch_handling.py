"""Test that point cloud networks handle batched data correctly"""
import torch
import sys
sys.path.insert(0, 'src')

from nets.RnNet import R2PointNet
from torch_geometric.data import Data, Batch

def test_batch_pooling():
    """Verify that batched point clouds produce batch_size outputs"""
    print("Testing batch-aware pooling in R2PointNet...")

    model = R2PointNet(
        n_classes=10,
        max_rot_order=2,
        channels_per_block=(8, 16),
        activation_type="fourier_relu_8",
        grey_scale=True
    )
    model.eval()

    # Create 3 samples with different numbers of points
    samples = []
    for i in range(3):
        num_points = 30 + i * 10  # 30, 40, 50 points
        data = Data(
            x=torch.randn(num_points, 1),
            pos=torch.randn(num_points, 2),
            edge_index=torch.randint(0, num_points, (2, num_points * 3)),
            y=torch.tensor([i])
        )
        samples.append(data)

    # Batch them using PyG
    batch = Batch.from_data_list(samples)

    print(f"  Batch contains {batch.num_graphs} graphs")
    print(f"  Total points: {batch.x.shape[0]}")
    print(f"  batch.batch shape: {batch.batch.shape}")

    # Forward pass
    output = model(batch.x, batch.pos, batch.edge_index, batch=batch.batch)

    # Check output shape
    print(f"  Output shape: {output.shape}")
    assert output.shape == (3, 10), f"Expected (3, 10), got {output.shape}"
    print("✅ Batch pooling test passed!")

    # Also test single sample (batch=None) still works
    print("\nTesting single sample (no batch parameter)...")
    single_output = model(samples[0].x, samples[0].pos, samples[0].edge_index)
    print(f"  Single output shape: {single_output.shape}")
    assert single_output.shape == (1, 10), f"Expected (1, 10), got {single_output.shape}"
    print("✅ Single sample test passed!")

if __name__ == "__main__":
    test_batch_pooling()
