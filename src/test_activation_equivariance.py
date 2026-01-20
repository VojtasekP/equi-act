"""Test script to verify activation equivariance for a trained model.

Usage:
    python src/test_activation_equivariance.py \
        --checkpoint saved_models/equivariance/equivariant_mnist_rot_seed0_noaug_fourierbn_relu_16_Normbn.ckpt \
        --layer 3 \
        --num-angles 32
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets_utils.data_classes import MnistRotDataModule
from nets.equivariance_metric import check_activation_equivariance_r2, check_equivariance_batch_r2
from train import LitHnn


def main():
    parser = argparse.ArgumentParser(description="Test activation equivariance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--layer", type=int, default=3, help="Layer index (1-based)")
    parser.add_argument("--num-angles", type=int, default=32, help="Number of rotation angles")
    parser.add_argument("--num-batches", type=int, default=5, help="Number of test batches")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--output-plot", type=str, default=None, help="Optional: save plot to this path")
    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.checkpoint}")
    lit_model = LitHnn.load_from_checkpoint(args.checkpoint, map_location=device)
    lit_model.eval()
    lit_model.to(device)

    # Load dataset (assuming MNIST for this example)
    datamodule = MnistRotDataModule(batch_size=8, img_size=64, seed=42)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    # Collect equivariance errors
    activation_errors_per_batch = []
    full_layer_errors_per_batch = []
    theta_values = None

    print(f"\nTesting layer {args.layer} activation equivariance...")

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            x = x.to(device)

            # Test activation-only equivariance
            thetas_act, errs_act = check_activation_equivariance_r2(
                x, lit_model.model, num_samples=args.num_angles, layer=args.layer
            )
            activation_errors_per_batch.append(errs_act)

            # Test full layer equivariance for comparison
            thetas_full, errs_full = check_equivariance_batch_r2(
                x, lit_model.model, num_samples=args.num_angles, layer=args.layer
            )
            full_layer_errors_per_batch.append(errs_full)

            if theta_values is None:
                theta_values = thetas_act

            print(f"  Batch {batch_idx + 1}/{args.num_batches}: "
                  f"activation_err={errs_act.mean():.6f}, "
                  f"full_layer_err={errs_full.mean():.6f}")

    # Average over batches
    activation_errors = np.stack(activation_errors_per_batch).mean(axis=0)
    full_layer_errors = np.stack(full_layer_errors_per_batch).mean(axis=0)

    print(f"\n{'='*60}")
    print(f"Layer {args.layer} Equivariance Test Results:")
    print(f"{'='*60}")
    print(f"Activation-only equivariance error: {activation_errors.mean():.6f} ± {activation_errors.std():.6f}")
    print(f"Full layer equivariance error:      {full_layer_errors.mean():.6f} ± {full_layer_errors.std():.6f}")
    print(f"{'='*60}")

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Error curves
    angles_deg = np.degrees(theta_values)
    ax1.plot(angles_deg, activation_errors, 'o-', label='Activation only', linewidth=2)
    ax1.plot(angles_deg, full_layer_errors, 's-', label='Full layer (conv+bn+act)', linewidth=2)
    ax1.set_xlabel('Rotation Angle (degrees)')
    ax1.set_ylabel('Equivariance Error')
    ax1.set_title(f'Layer {args.layer} Equivariance Error vs Rotation Angle')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log scale comparison
    ax2.semilogy(angles_deg, activation_errors, 'o-', label='Activation only', linewidth=2)
    ax2.semilogy(angles_deg, full_layer_errors, 's-', label='Full layer (conv+bn+act)', linewidth=2)
    ax2.set_xlabel('Rotation Angle (degrees)')
    ax2.set_ylabel('Equivariance Error (log scale)')
    ax2.set_title(f'Layer {args.layer} Equivariance Error (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if args.output_plot:
        plt.savefig(args.output_plot, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {args.output_plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
