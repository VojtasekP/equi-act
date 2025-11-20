import numpy as np
import matplotlib.pyplot as plt


def create_c8_basis_filter(k_size, k, radial_profile):
    """
    Creates a filter from the steerable basis for the C8 group.

    Args:
        k_size (int): The size of the filter kernel.
        k (int): The frequency of the irreducible representation (from 0 to 7).
        radial_profile (list or np.array): A list of weights, one for each
                                           unique radius in the filter grid.

    Returns:
        np.array: A complex-valued (k_size x k_size) basis filter.

    """
    if k_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # 1. Create the polar coordinate grid
    center = (k_size - 1) / 2
    x, y = np.meshgrid(
        np.linspace(-center, center, k_size),
        np.linspace(-center, center, k_size)
    )
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    # 2. Discretize the radial profile
    unique_radii = np.unique(r)
    assert len(radial_profile) == len(unique_radii), \
        f"Expected {len(unique_radii)} radial weights, but got {len(radial_profile)}."
    radius_to_weight = {radius: weight for radius, weight in zip(unique_radii, radial_profile)}
    R_grid = np.zeros_like(r)
    for radius, weight in radius_to_weight.items():
        R_grid[r == radius] = weight

    # 3. Apply the C8 transformation rule
    # The basis function is R(r) * e^(i*k*phi). This is a circular harmonic,
    # which is the basis for SO(2). When sampled on the C8 grid points,
    # it provides a basis for C8 steerable filters.
    W = R_grid * np.exp(1j * k * phi)

    # Set the center to zero for non-trivial representations
    if k != 0:
        W[r == 0] = 0

    return W


# --- Main execution ---
if __name__ == '__main__':
    KERNEL_SIZE = 15

    # Define a simple decaying radial profile
    # For KERNEL_SIZE=11, there are 19 unique radii
    num_radii = 34
    RADIAL_PROFILE = np.linspace(1.0, 0.2, num=num_radii)

    # --- Create a plot for all 8 basis filters (k=0 to 7) ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Steerable Basis Filters for the $C_8$ Group (Real Part)', fontsize=18)

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for k in range(8):
        # Create the basis filter for the k-th representation
        basis_filter = create_c8_basis_filter(
            k_size=KERNEL_SIZE,
            k=k,
            radial_profile=RADIAL_PROFILE
        )

        # Plot the real part of the filter
        ax = axes[k]
        im = ax.imshow(basis_filter.real, cmap='viridis')
        ax.set_title(f'Basis for k={k}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()