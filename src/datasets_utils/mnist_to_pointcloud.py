"""
Script to convert MNIST rotation dataset to point cloud format.

This script converts 28x28 grayscale MNIST images into 2D point clouds by:
1. Extracting non-zero pixel coordinates and intensities
2. Normalizing coordinates to [-1, 1] range
3. Saving as compressed .npz files for efficient loading

Output format:
- coords: (N, 2) array of (x, y) coordinates
- features: (N, 1) array of pixel intensities
- label: int label (0-9)

Usage:
    python src/datasets_utils/mnist_to_pointcloud.py --data_dir <path> --output_dir <path>
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def image_to_pointcloud(image: np.ndarray, threshold: float = 0.01, normalize: bool = True):
    """
    Convert a 2D image to a point cloud.

    Args:
        image: 2D numpy array (H, W) with pixel intensities
        threshold: Minimum intensity to include a pixel (filters near-zero pixels)
        normalize: Whether to normalize coordinates to [-1, 1]

    Returns:
        coords: (N, 2) array of (x, y) coordinates
        features: (N, 1) array of pixel intensities
    """
    # Get coordinates of non-zero pixels
    y_coords, x_coords = np.where(image > threshold)

    if len(x_coords) == 0:
        # Handle empty images - add a single point at center with zero intensity
        x_coords = np.array([image.shape[1] // 2])
        y_coords = np.array([image.shape[0] // 2])
        features = np.array([[0.0]])
    else:
        # Extract features (pixel intensities)
        features = image[y_coords, x_coords].reshape(-1, 1)

    # Stack coordinates
    coords = np.stack([x_coords, y_coords], axis=1).astype(np.float32)

    if normalize:
        # Normalize to [-1, 1] range
        # Assuming image is square (H == W)
        img_size = image.shape[0]
        coords = (coords - img_size / 2.0) / (img_size / 2.0)

    return coords.astype(np.float32), features.astype(np.float32)


def process_single_sample(args):
    """Process a single image and return point cloud data"""
    idx, image, label, threshold = args
    coords, features = image_to_pointcloud(image, threshold=threshold)
    return idx, coords, features, label


def convert_mnist_split(images: np.ndarray,
                        labels: np.ndarray,
                        output_dir: Path,
                        split_name: str,
                        threshold: float = 0.01,
                        num_workers: int = 8):
    """
    Convert a split of MNIST data to point clouds and save.

    Args:
        images: (N, H, W) array of images
        labels: (N,) array of labels
        output_dir: Directory to save point cloud files
        split_name: Name of the split (train/test)
        threshold: Intensity threshold for including pixels
        num_workers: Number of parallel workers
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = len(images)

    print(f"Converting {split_name} split ({n_samples} images) to point clouds...")

    # Prepare arguments for parallel processing
    args_list = [(i, images[i], labels[i], threshold) for i in range(n_samples)]

    # Process in parallel
    all_data = []
    with mp.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(process_single_sample, args_list),
                          total=n_samples,
                          desc=f"Processing {split_name}"):
            all_data.append(result)

    # Sort by index to maintain order
    all_data.sort(key=lambda x: x[0])

    # Save to a single compressed file
    output_file = output_dir / f"{split_name}.npz"

    # Extract data
    coords_list = [item[1] for item in all_data]
    features_list = [item[2] for item in all_data]
    labels_array = np.array([item[3] for item in all_data], dtype=np.int64)

    # Compute statistics
    num_points_per_sample = np.array([len(c) for c in coords_list])
    print(f"{split_name} point cloud statistics:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Points per sample: mean={num_points_per_sample.mean():.1f}, "
          f"std={num_points_per_sample.std():.1f}, "
          f"min={num_points_per_sample.min()}, max={num_points_per_sample.max()}")

    # Save as compressed npz with object arrays for variable-length data
    np.savez_compressed(
        output_file,
        coords=np.array(coords_list, dtype=object),
        features=np.array(features_list, dtype=object),
        labels=labels_array,
        num_points=num_points_per_sample
    )

    print(f"Saved {split_name} split to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert MNIST rotation dataset to point clouds")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing MNIST rotation .amat files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save point cloud files")
    parser.add_argument("--threshold", type=float, default=0.01,
                       help="Minimum pixel intensity to include in point cloud")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of parallel workers for conversion")

    args = parser.parse_args()

    # Set default directories
    if args.data_dir is None:
        args.data_dir = Path(__file__).resolve().parent / "mnist_rotation_new"
    else:
        args.data_dir = Path(args.data_dir)

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "mnist_rotation_pointcloud"
    else:
        args.output_dir = Path(args.output_dir)

    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Workers: {args.num_workers}")

    # Load and convert training data
    train_file = args.data_dir / "mnist_all_rotation_normalized_float_train_valid.amat"
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    print(f"\nLoading training data from {train_file}...")
    train_data = np.loadtxt(train_file, delimiter=' ')
    train_images = train_data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
    train_labels = train_data[:, -1].astype(np.int64)

    convert_mnist_split(train_images, train_labels, args.output_dir, "train",
                       threshold=args.threshold, num_workers=args.num_workers)

    # Load and convert test data
    test_file = args.data_dir / "mnist_all_rotation_normalized_float_test.amat"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    print(f"\nLoading test data from {test_file}...")
    test_data = np.loadtxt(test_file, delimiter=' ')
    test_images = test_data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
    test_labels = test_data[:, -1].astype(np.int64)

    convert_mnist_split(test_images, test_labels, args.output_dir, "test",
                       threshold=args.threshold, num_workers=args.num_workers)

    # Save metadata
    metadata = {
        "threshold": args.threshold,
        "image_size": 28,
        "normalized_coords": True,
        "coord_range": "[-1, 1]",
        "feature_dim": 1,
        "num_classes": 10,
    }

    metadata_file = args.output_dir / "metadata.txt"
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"\nConversion complete! Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
