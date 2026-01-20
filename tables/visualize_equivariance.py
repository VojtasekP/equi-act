"""Visualize equivariance error curves for Fourier-BN activation functions.

This script loads trained models and plots equivariance error vs rotation angle
for activation functions at different layers (first, middle, last) for each dataset.
Creates 9 plots total: 3 datasets × 3 layers.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets_utils.data_classes import (
    ColorectalHistDataModule,
    EuroSATDataModule,
    MnistRotDataModule,
)
import nets.equivariance_metric as em
from train import LitHnn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize equivariance error curves for Fourier-BN models"
    )
    parser.add_argument(
        "--mnist-ckpt",
        type=str,
        required=True,
        help="Path to MNIST Fourier-BN checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--colorectal-ckpt",
        type=str,
        required=True,
        help="Path to Colorectal Fourier-BN checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--eurosat-ckpt",
        type=str,
        required=True,
        help="Path to EuroSAT Fourier-BN checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--num-angles",
        type=int,
        default=180,
        help="Number of rotation angles to test (default: 180)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run models on (default: auto)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index of sample to use from test set (default: 0)",
    )
    parser.add_argument(
        "--mnist-data-dir",
        type=str,
        default="./src/datasets_utils/mnist_rotation_new",
        help="Path to MNIST-rot dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4,
        help="Chunk size for processing rotations (default: 4)",
    )
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    """Select compute device based on user choice and availability."""
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device):
    """Load a Lightning checkpoint and return the model."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    lit_model = LitHnn.load_from_checkpoint(str(ckpt_path), map_location=device)
    lit_model.eval()
    lit_model.to(device)
    return lit_model


def get_sample(datamodule, sample_idx: int, device: torch.device):
    """Get a single sample from the test set."""
    if hasattr(datamodule, "prepare_data"):
        try:
            datamodule.prepare_data()
        except Exception:
            pass

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # Get first batch and extract the sample at sample_idx
    batch = next(iter(test_loader))
    x, y = batch

    if sample_idx >= x.shape[0]:
        raise ValueError(
            f"sample_idx {sample_idx} out of range for batch size {x.shape[0]}"
        )

    # Extract single sample and add batch dimension
    x_sample = x[sample_idx:sample_idx+1].to(device)
    y_sample = y[sample_idx]

    return x_sample, y_sample


def compute_single_layer_equivariance(
    model_obj,
    x_input: torch.Tensor,
    layer_idx: int,
    num_angles: int,
    chunk_size: int,
    device: torch.device
):
    """Compute equivariance for a single specific layer by manually handling transforms."""
    from escnn import nn as escnn_nn

    # Process input up to the layer we want to test
    # This gets us the intermediate features
    model_obj.init_nth_layer(layer_idx)

    # Get the input to the target layer by processing through previous layers
    # We need to manually handle this to avoid the mask issue
    with torch.no_grad():
        # Wrap input in GeometricTensor
        x_geo = escnn_nn.GeometricTensor(x_input, model_obj.input_type)

        # Apply mask
        x_masked = model_obj.mask(x_geo)

        # Process through layers up to (but not including) target layer
        x_intermediate = x_masked
        for i in range(layer_idx - 1):
            x_intermediate = model_obj.eq_layers[i](x_intermediate)

        # Now x_intermediate is the input to our target layer
        # Get the input and output types for the target layer
        target_layer = model_obj.eq_layers[layer_idx - 1]
        input_type = target_layer.in_type
        output_type = target_layer.out_type

        # Generate rotation angles
        r2_act = getattr(model_obj, "r2_act")
        thetas = np.linspace(0.0, 2*np.pi, num_angles, endpoint=False)
        elems = em.make_group_elements(r2_act, thetas)

        # Get reference output (apply target layer to intermediate features)
        y_ref = target_layer(x_intermediate)
        if not isinstance(y_ref, escnn_nn.GeometricTensor):
            y_ref = escnn_nn.GeometricTensor(y_ref, output_type)

        # Test equivariance: transform input, apply layer, vs apply layer, transform output
        errs = []
        B = x_input.shape[0]

        for start in range(0, len(elems), chunk_size):
            chunk_elems = elems[start:start + chunk_size]

            # Transform the intermediate features
            x_rot_list = [x_intermediate.transform(g) for g in chunk_elems]

            # Apply target layer to each transformed input
            y_rot_list = [target_layer(x_rot) for x_rot in x_rot_list]

            # Also transform the reference output
            y_ref_rot_list = [y_ref.transform(g) for g in chunk_elems]

            # Compute relative errors
            for y_rot, y_ref_rot in zip(y_rot_list, y_ref_rot_list):
                y_rot_tensor = y_rot.tensor if isinstance(y_rot, escnn_nn.GeometricTensor) else y_rot
                y_ref_rot_tensor = y_ref_rot.tensor if isinstance(y_ref_rot, escnn_nn.GeometricTensor) else y_ref_rot

                e = em.rel_err(y_rot_tensor, y_ref_rot_tensor).mean()
                errs.append(e)

        errs = torch.stack(errs).detach().cpu().numpy()
        return thetas, errs


def compute_equivariance_curves_per_layer(
    model,
    x: torch.Tensor,
    num_angles: int,
    chunk_size: int,
    device: torch.device
):
    """Compute equivariance error curves for first, middle, and last layers."""
    # Determine total number of layers
    total_layers = len(getattr(model.model, "eq_layers", []))

    if total_layers == 0:
        raise ValueError("Model has no equivariant layers (eq_layers attribute is empty)")

    print(f"Model has {total_layers} equivariant layers")

    # Select layer indices: first, middle, last
    first_layer = 1
    middle_layer = max(1, total_layers // 2)
    last_layer = total_layers

    print(f"Testing layers: {first_layer} (first), {middle_layer} (middle), {last_layer} (last)")

    results = {}

    for layer_name, layer_idx in [("first", first_layer), ("middle", middle_layer), ("last", last_layer)]:
        print(f"  Computing equivariance for layer {layer_idx} ({layer_name})...")
        with torch.no_grad():
            thetas, errors = compute_single_layer_equivariance(
                model.model,
                x,
                layer_idx,
                num_angles,
                chunk_size,
                device
            )

        results[layer_name] = {
            "thetas": thetas,
            "errors": errors,
            "layer_idx": layer_idx
        }

    return results


def create_plot(results: dict, output_dir: Path, num_angles: int):
    """Create a 3×3 grid plot showing equivariance curves for all datasets and layers.

    Rows: MNIST, Colorectal, EuroSAT
    Columns: First layer, Middle layer, Last layer
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    datasets = ["MNIST", "Colorectal", "EuroSAT"]
    layers = ["first", "middle", "last"]
    layer_titles = ["First Layer", "Middle Layer", "Last Layer"]

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (layer, layer_title) in enumerate(zip(layers, layer_titles)):
            ax = axes[row_idx, col_idx]

            if dataset in results and layer in results[dataset]:
                thetas = results[dataset][layer]["thetas"]
                errors = results[dataset][layer]["errors"]
                layer_num = results[dataset][layer]["layer_idx"]

                # Convert radians to degrees for plotting
                angles_deg = np.degrees(thetas)

                ax.plot(angles_deg, errors, linewidth=2, color=f"C{row_idx}")
                ax.set_xlabel("Rotation Angle (degrees)", fontsize=11)
                ax.set_ylabel("Equivariance Error", fontsize=11)

                # Title with dataset and layer info
                if row_idx == 0:
                    ax.set_title(f"{dataset} - {layer_title}\n(Layer {layer_num})",
                                fontsize=12, fontweight="bold")
                else:
                    ax.set_title(f"{dataset} - {layer_title}\n(Layer {layer_num})",
                                fontsize=12, fontweight="bold")

                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 360)

                # Add statistics
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                ax.text(
                    0.05, 0.95,
                    f"Mean: {mean_error:.6f}\nMax: {max_error:.6f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    fontsize=9,
                )
            else:
                ax.text(
                    0.5, 0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_title(f"{dataset} - {layer_title}", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f"equivariance_layers_{num_angles}angles.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    output_path_png = output_dir / f"equivariance_layers_{num_angles}angles.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path_png}")

    plt.close()


def main():
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Number of angles: {args.num_angles}")
    print(f"Sample index: {args.sample_idx}")
    print(f"Chunk size: {args.chunk_size}\n")

    results = {}

    # Process MNIST
    print("=" * 60)
    print("Processing MNIST")
    print("=" * 60)
    mnist_model = load_model(args.mnist_ckpt, device)
    mnist_dm = MnistRotDataModule(
        batch_size=32,
        data_dir=args.mnist_data_dir,
        img_size=64,
        seed=42,
    )
    x_mnist, y_mnist = get_sample(mnist_dm, args.sample_idx, device)
    print(f"Sample shape: {x_mnist.shape}, label: {y_mnist}")

    mnist_layer_results = compute_equivariance_curves_per_layer(
        mnist_model, x_mnist, args.num_angles, args.chunk_size, device
    )
    results["MNIST"] = mnist_layer_results

    # Print summary
    for layer_name in ["first", "middle", "last"]:
        mean_err = np.mean(mnist_layer_results[layer_name]["errors"])
        max_err = np.max(mnist_layer_results[layer_name]["errors"])
        print(f"  {layer_name.capitalize()} layer - Mean: {mean_err:.6f}, Max: {max_err:.6f}")
    print()

    # Process Colorectal
    print("=" * 60)
    print("Processing Colorectal")
    print("=" * 60)
    colorectal_model = load_model(args.colorectal_ckpt, device)
    colorectal_dm = ColorectalHistDataModule(
        batch_size=32,
        img_size=64,
        seed=42,
        train_fraction=1.0,
    )
    x_colorectal, y_colorectal = get_sample(colorectal_dm, args.sample_idx, device)
    print(f"Sample shape: {x_colorectal.shape}, label: {y_colorectal}")

    colorectal_layer_results = compute_equivariance_curves_per_layer(
        colorectal_model, x_colorectal, args.num_angles, args.chunk_size, device
    )
    results["Colorectal"] = colorectal_layer_results

    # Print summary
    for layer_name in ["first", "middle", "last"]:
        mean_err = np.mean(colorectal_layer_results[layer_name]["errors"])
        max_err = np.max(colorectal_layer_results[layer_name]["errors"])
        print(f"  {layer_name.capitalize()} layer - Mean: {mean_err:.6f}, Max: {max_err:.6f}")
    print()

    # Process EuroSAT
    print("=" * 60)
    print("Processing EuroSAT")
    print("=" * 60)
    eurosat_model = load_model(args.eurosat_ckpt, device)
    eurosat_dm = EuroSATDataModule(
        batch_size=32,
        img_size=64,
        seed=42,
        train_fraction=1.0,
    )
    x_eurosat, y_eurosat = get_sample(eurosat_dm, args.sample_idx, device)
    print(f"Sample shape: {x_eurosat.shape}, label: {y_eurosat}")

    eurosat_layer_results = compute_equivariance_curves_per_layer(
        eurosat_model, x_eurosat, args.num_angles, args.chunk_size, device
    )
    results["EuroSAT"] = eurosat_layer_results

    # Print summary
    for layer_name in ["first", "middle", "last"]:
        mean_err = np.mean(eurosat_layer_results[layer_name]["errors"])
        max_err = np.max(eurosat_layer_results[layer_name]["errors"])
        print(f"  {layer_name.capitalize()} layer - Mean: {mean_err:.6f}, Max: {max_err:.6f}")
    print()

    # Create combined plot
    print("=" * 60)
    print("Creating plot")
    print("=" * 60)
    create_plot(results, output_dir, args.num_angles)

    print("\nDone!")


if __name__ == "__main__":
    main()
