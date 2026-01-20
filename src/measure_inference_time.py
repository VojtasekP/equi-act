#!/usr/bin/env python3
"""
Measure inference time for trained models across different datasets.

Usage:
    python src/measure_inference_time.py --checkpoint <path_to_checkpoint>

    # Or specify dataset and model type explicitly:
    python src/measure_inference_time.py --checkpoint saved_models/mnist_rot_final/equivariant_mnist_rot_seed0_aug_gated_sigmoid_Normbn.ckpt

    # For baseline ResNet18:
    python src/measure_inference_time.py --checkpoint saved_models/baseline_mnist/resnet18_mnist_rot_seed0_aug_resnet_scratch.ckpt --model_type resnet18

Outputs mean and std of inference time over 1000 samples from the train set.
"""

import argparse
import time
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import LitHnn
from nets.baseline_resnet import LitResNet18
from datasets_utils.data_classes import (
    MnistRotDataModule,
    ColorectalHistDataModule,
    EuroSATDataModule,
)


def infer_dataset_from_checkpoint(ckpt_path: str) -> str:
    """Infer dataset name from checkpoint filename."""
    name = Path(ckpt_path).name.lower()
    if "mnist" in name:
        return "mnist_rot"
    elif "colorectal" in name:
        return "colorectal_hist"
    elif "eurosat" in name:
        return "eurosat"
    else:
        raise ValueError(f"Cannot infer dataset from checkpoint name: {ckpt_path}")


def infer_model_type_from_checkpoint(ckpt_path: str) -> str:
    """Infer model type from checkpoint filename."""
    name = Path(ckpt_path).name.lower()
    if "resnet" in name:
        return "resnet18"
    else:
        return "equivariant"


def get_datamodule(dataset: str, batch_size: int = 1):
    """Create data module for the specified dataset."""
    if dataset == "mnist_rot":
        dm = MnistRotDataModule(
            batch_size=batch_size,
            data_dir="./src/datasets_utils/mnist_rotation_new",
            aug=False,
            normalize=True,
        )
    elif dataset == "colorectal_hist":
        dm = ColorectalHistDataModule(
            batch_size=batch_size,
            aug=False,
            normalize=True,
        )
    elif dataset == "eurosat":
        dm = EuroSATDataModule(
            batch_size=batch_size,
            seed=0,
            aug=False,
            normalize=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dm.prepare_data()
    dm.setup("fit")
    return dm


def load_model(ckpt_path: str, model_type: str, device: torch.device):
    """Load model from checkpoint."""
    if model_type == "resnet18":
        model = LitResNet18.load_from_checkpoint(ckpt_path, map_location=device)
    else:
        model = LitHnn.load_from_checkpoint(ckpt_path, map_location=device)

    model.to(device)
    model.eval()
    return model


def measure_inference_time(
    model,
    dataloader,
    device: torch.device,
    num_samples: int = 1000,
    warmup_samples: int = 50,
) -> tuple[float, float]:
    """
    Measure inference time for the model.

    Args:
        model: The model to measure
        dataloader: DataLoader providing samples
        device: Device to run on
        num_samples: Number of samples to measure (default: 1000)
        warmup_samples: Number of warmup samples before measuring (default: 50)

    Returns:
        Tuple of (mean_time_ms, std_time_ms) per sample
    """
    model.eval()
    times = []
    sample_count = 0
    warmup_count = 0

    # Handle model wrapping
    is_lit_model = hasattr(model, 'model')
    actual_model = model.model if is_lit_model else model

    # Create CUDA Events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print(device)

    with torch.no_grad():
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            batch_size = x.size(0)

            # Warmup phase
            if warmup_count < warmup_samples:
                _ = actual_model(x)
                warmup_count += batch_size
                continue

            if device.type == 'cuda':
                # Record the start event on GPU
                start_event.record()
                _ = actual_model(x)
                # Record the end event on GPU
                end_event.record()

                # Wait for GPU to finish ONLY to read the timer,
                # but the timer itself (end_event) was already "stamped" on the GPU.
                torch.cuda.synchronize()

                # Calculate time in milliseconds directly
                elapsed_time_ms = start_event.elapsed_time(end_event)
                time_per_sample = elapsed_time_ms / batch_size

            else:
                # Fallback for CPU (less precise but unavoidable)
                start = time.perf_counter()
                _ = actual_model(x)
                end = time.perf_counter()
                time_per_sample = ((end - start) * 1000) / batch_size

            # Store time per sample for this batch (one measurement per batch)
            times.append(time_per_sample)
            sample_count += batch_size

            if sample_count >= num_samples:
                break

    if len(times) == 0:
        raise RuntimeError("No samples were measured.")

    times = np.array(times)
    n_batches = len(times)
    mean_time = float(np.mean(times))
    # Standard error of the mean (std / sqrt(n))
    std_time = float(np.std(times) / np.sqrt(n_batches)) if n_batches > 1 else 0.0
    return mean_time, std_time


def main():
    parser = argparse.ArgumentParser(
        description="Measure inference time for trained models"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist_rot", "colorectal_hist", "eurosat"],
        default=None,
        help="Dataset to use (inferred from checkpoint if not specified)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["equivariant", "resnet18"],
        default=None,
        help="Model type (inferred from checkpoint if not specified)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to measure (default: 1000)",
    )
    parser.add_argument(
        "--warmup_samples",
        type=int,
        default=50,
        help="Number of warmup samples before measuring (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda:1 if available, else cpu)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV file to append results to",
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Infer dataset and model type if not specified
    dataset = args.dataset or infer_dataset_from_checkpoint(args.checkpoint)
    model_type = args.model_type or infer_model_type_from_checkpoint(args.checkpoint)

    # Set device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {dataset}")
    print(f"Model type: {model_type}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples}")
    print(f"Warmup samples: {args.warmup_samples}")
    print()

    # Load data
    print("Loading data...")
    dm = get_datamodule(dataset, batch_size=args.batch_size)
    train_loader = dm.train_dataloader()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, model_type, device)

    # Measure inference time
    print(f"Measuring inference time over {args.num_samples} samples...")
    mean_time, std_time = measure_inference_time(
        model,
        train_loader,
        device,
        num_samples=args.num_samples,
        warmup_samples=args.warmup_samples,
    )

    print()
    print("=" * 60)
    print(f"Results for: {Path(args.checkpoint).name}")
    print(f"  Mean inference time: {mean_time:.4f} ms")
    print(f"  Std inference time:  {std_time:.4f} ms")
    print("=" * 60)

    # Optionally write to CSV
    if args.output_csv:
        csv_path = Path(args.output_csv)
        write_header = not csv_path.exists()

        with open(csv_path, "a") as f:
            if write_header:
                f.write("checkpoint,dataset,model_type,batch_size,num_samples,mean_time_ms,std_time_ms\n")
            f.write(f"{Path(args.checkpoint).name},{dataset},{model_type},{args.batch_size},{args.num_samples},{mean_time:.6f},{std_time:.6f}\n")

        print(f"Results appended to: {args.output_csv}")

    return mean_time, std_time


if __name__ == "__main__":
    main()
