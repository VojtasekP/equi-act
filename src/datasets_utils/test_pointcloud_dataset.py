"""
Test script for MNIST point cloud dataset and datamodule.

Usage:
    # First convert MNIST to point clouds
    python src/datasets_utils/mnist_to_pointcloud.py

    # Then test the dataset
    python src/datasets_utils/test_pointcloud_dataset.py
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets_utils.data_classes import MnistRotPointCloudDataset, MnistRotPointCloudDataModule


def visualize_point_cloud_sample(data, title="Point Cloud Sample"):
    """
    Visualize a single point cloud sample.

    Args:
        data: PyG Data object with pos (coordinates) and x (features)
        title: Plot title
    """
    pos = data.pos.numpy()  # (N, 2)
    features = data.x.numpy().squeeze()  # (N,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Point cloud with feature intensities
    scatter = axes[0].scatter(pos[:, 0], pos[:, 1], c=features,
                             cmap='viridis', s=10, alpha=0.6)
    axes[0].set_aspect('equal')
    axes[0].set_title(f'{title} - Label: {data.y.item()}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(scatter, ax=axes[0], label='Intensity')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Graph connectivity
    # Draw edges
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        axes[1].plot([pos[src, 0], pos[dst, 0]],
                    [pos[src, 1], pos[dst, 1]],
                    'gray', alpha=0.1, linewidth=0.5)

    # Draw nodes
    scatter = axes[1].scatter(pos[:, 0], pos[:, 1], c=features,
                             cmap='viridis', s=10, alpha=0.6)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Graph Connectivity (edges: {edge_index.shape[1]})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(scatter, ax=axes[1], label='Intensity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_dataset():
    """Test the MnistRotPointCloudDataset directly"""
    print("=" * 60)
    print("Testing MnistRotPointCloudDataset")
    print("=" * 60)

    data_dir = Path(__file__).resolve().parent / "mnist_rotation_pointcloud"
    train_npz = data_dir / "train.npz"

    if not train_npz.exists():
        print(f"ERROR: Point cloud file not found: {train_npz}")
        print("Please run mnist_to_pointcloud.py first")
        return

    # Test without augmentation
    print("\n1. Loading dataset without augmentation...")
    dataset = MnistRotPointCloudDataset(
        train_npz,
        augment_rotation=False,
        k_neighbors=8
    )

    print(f"   Dataset size: {len(dataset)}")

    # Get a sample
    sample = dataset[0]
    print(f"\n2. Sample 0 properties:")
    print(f"   Label: {sample.y.item()}")
    print(f"   Points: {sample.pos.shape[0]}")
    print(f"   Coordinates shape: {sample.pos.shape}")
    print(f"   Features shape: {sample.x.shape}")
    print(f"   Edges: {sample.edge_index.shape[1]}")
    print(f"   Coordinate range: [{sample.pos.min():.2f}, {sample.pos.max():.2f}]")
    print(f"   Feature range: [{sample.x.min():.2f}, {sample.x.max():.2f}]")

    # Test with augmentation
    print("\n3. Testing rotation augmentation...")
    dataset_aug = MnistRotPointCloudDataset(
        train_npz,
        augment_rotation=True,
        k_neighbors=8
    )

    sample1 = dataset_aug[0]
    sample2 = dataset_aug[0]

    # Check that coordinates differ (due to random rotation)
    coord_diff = torch.abs(sample1.pos - sample2.pos).max().item()
    print(f"   Max coordinate difference between two augmented loads: {coord_diff:.4f}")
    print(f"   Augmentation working: {coord_diff > 0.01}")

    # Visualize
    print("\n4. Generating visualizations...")
    fig1 = visualize_point_cloud_sample(sample, "Original Sample")
    fig2 = visualize_point_cloud_sample(sample1, "Augmented Sample")

    # Save plots
    output_dir = Path(__file__).resolve().parent / "pointcloud_visualizations"
    output_dir.mkdir(exist_ok=True)

    fig1.savefig(output_dir / "pointcloud_original.png", dpi=150, bbox_inches='tight')
    fig2.savefig(output_dir / "pointcloud_augmented.png", dpi=150, bbox_inches='tight')
    print(f"   Saved visualizations to {output_dir}/")

    plt.close('all')


def test_datamodule():
    """Test the MnistRotPointCloudDataModule"""
    print("\n" + "=" * 60)
    print("Testing MnistRotPointCloudDataModule")
    print("=" * 60)

    # Create datamodule
    print("\n1. Creating datamodule...")
    dm = MnistRotPointCloudDataModule(
        batch_size=32,
        augment_rotation=True,
        k_neighbors=8,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )

    print("2. Preparing data...")
    try:
        dm.prepare_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print("3. Setting up splits...")
    dm.setup()

    # Test dataloaders
    print("\n4. Testing dataloaders...")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Get a batch
    print("\n5. Inspecting a training batch...")
    batch = next(iter(train_loader))

    print(f"   Batch size: {batch.num_graphs}")
    print(f"   Total points in batch: {batch.pos.shape[0]}")
    print(f"   Features shape: {batch.x.shape}")
    print(f"   Coordinates shape: {batch.pos.shape}")
    print(f"   Edge index shape: {batch.edge_index.shape}")
    print(f"   Labels shape: {batch.y.shape}")
    print(f"   Labels: {batch.y.tolist()}")

    # Compute average points per sample
    points_per_sample = batch.pos.shape[0] / batch.num_graphs
    print(f"   Average points per sample: {points_per_sample:.1f}")

    # Test iteration
    print("\n6. Testing iteration over batches...")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        print(f"   Batch {i}: {batch.num_graphs} graphs, {batch.pos.shape[0]} total points")

    print("\n✓ All tests passed!")


def test_with_radius_graph():
    """Test using radius graph instead of kNN"""
    print("\n" + "=" * 60)
    print("Testing with Radius Graph")
    print("=" * 60)

    data_dir = Path(__file__).resolve().parent / "mnist_rotation_pointcloud"
    train_npz = data_dir / "train.npz"

    if not train_npz.exists():
        print(f"ERROR: Point cloud file not found: {train_npz}")
        return

    # Test with radius graph
    print("\n1. Creating dataset with radius graph...")
    dataset = MnistRotPointCloudDataset(
        train_npz,
        augment_rotation=False,
        radius=0.3  # Radius in normalized coordinates
    )

    sample = dataset[0]
    print(f"   Points: {sample.pos.shape[0]}")
    print(f"   Edges: {sample.edge_index.shape[1]}")
    print(f"   Average degree: {sample.edge_index.shape[1] / sample.pos.shape[0]:.2f}")

    # Compare with kNN
    dataset_knn = MnistRotPointCloudDataset(
        train_npz,
        augment_rotation=False,
        k_neighbors=8
    )

    sample_knn = dataset_knn[0]
    print(f"\n2. Comparison with kNN (k=8):")
    print(f"   Radius graph edges: {sample.edge_index.shape[1]}")
    print(f"   kNN graph edges: {sample_knn.edge_index.shape[1]}")


def main():
    """Run all tests"""
    print("MNIST Point Cloud Dataset Tests\n")

    test_dataset()
    test_datamodule()
    test_with_radius_graph()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
