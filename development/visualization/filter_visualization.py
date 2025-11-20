import numpy as np
import matplotlib.pyplot as plt


def get_unique_radii_count(k_size):
    """Calculates the number of unique radii for a given kernel size."""
    if k_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    center = (k_size - 1) / 2
    x, y = np.meshgrid(
        np.linspace(-center, center, k_size),
        np.linspace(-center, center, k_size)
    )
    r = np.sqrt(x ** 2 + y ** 2)
    return len(np.unique(r))


def create_harmonic_filter(k_size, m, radial_profile, beta):
    """
    Creates a complex-valued circular harmonic filter on a Cartesian grid.

    Args:
        k_size (int): The size of the filter kernel (e.g., 7 for a 7x7 grid).
        m (int): The rotation order of the harmonic.
        radial_profile (list or np.array): A list of learnable weights, one for each
                                           unique radius in the filter grid.
        beta (float): The learnable phase offset in radians.

    Returns:
        np.array: A complex-valued (k_size x k_size) filter.
    """
    if k_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # 1. Create the Cartesian coordinate grid
    center = (k_size - 1) / 2
    x, y = np.meshgrid(
        np.linspace(-center, center, k_size),
        np.linspace(-center, center, k_size)
    )

    # 2. Convert grid to polar coordinates
    r = np.sqrt(x ** 2 + y ** 2)  # Radius
    phi = np.arctan2(y, x)  # Angle (phi)

    # 3. Discretize the radial profile
    unique_radii = np.unique(r)

    assert len(radial_profile) == len(unique_radii), \
        f"For k_size={k_size}, expected {len(unique_radii)} radial weights, but got {len(radial_profile)}."

    radius_to_weight = {radius: weight for radius, weight in zip(unique_radii, radial_profile)}

    R_grid = np.zeros_like(r)
    for radius, weight in radius_to_weight.items():
        R_grid[r == radius] = weight

    # 4. Calculate the harmonic filter using the formula W = R(r) * e^(i*(m*phi + beta))
    W = R_grid * np.exp(1j * (m * phi + beta))

    # 5. The value at the center for m != 0 is always zero
    if m != 0:
        W[r == 0] = 0

    return W


# --- Main execution ---
if __name__ == '__main__':
    # --- Parameters for our example filter ---
    # YOU CAN CHANGE THE KERNEL_SIZE HERE
    KERNEL_SIZE = 201
    ROTATION_ORDER_M = 11
    PHASE_OFFSET_BETA = np.pi / 1  # 90 degrees

    # --- Dynamically create the radial profile ---
    # This automatically generates a decaying profile with the correct length for any kernel size.
    # In a real network, these values would be learned.
    num_radii = get_unique_radii_count(KERNEL_SIZE)
    RADIAL_PROFILE = np.linspace(1.0, 0.1, num=num_radii)

    print(f"Kernel Size: {KERNEL_SIZE}x{KERNEL_SIZE}")
    print(f"Required number of radial profile weights: {num_radii}")

    # --- Create the filter ---
    harmonic_filter = create_harmonic_filter(
        k_size=KERNEL_SIZE,
        m=ROTATION_ORDER_M,
        radial_profile=RADIAL_PROFILE,
        beta=PHASE_OFFSET_BETA
    )

    # --- Visualize the filter ---
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f'Circular Harmonic Filter (m={ROTATION_ORDER_M}, size={KERNEL_SIZE}x{KERNEL_SIZE})', fontsize=16)

    # 1. Cartesian Coordinate Visualization
    ax1 = fig.add_subplot(1, 2, 1)
    # We visualize the real part of the filter, as done in the paper
    cartesian_plot = ax1.imshow(harmonic_filter.real, cmap='viridis')
    ax1.set_title('Cartesian Visualization (Real Part)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(cartesian_plot, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Polar Coordinate Visualization
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')

    center = (KERNEL_SIZE - 1) / 2
    x_coords, y_coords = np.meshgrid(range(KERNEL_SIZE), range(KERNEL_SIZE))
    r_coords = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)
    phi_coords = np.arctan2(y_coords - center, x_coords - center)

    polar_plot = ax2.scatter(
        phi_coords.flatten(),
        r_coords.flatten(),
        c=harmonic_filter.real.flatten(),
        cmap='viridis'
    )
    ax2.set_title('Polar Visualization (Real Part)')
    fig.colorbar(polar_plot, ax=ax2, fraction=0.046, pad=0.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()