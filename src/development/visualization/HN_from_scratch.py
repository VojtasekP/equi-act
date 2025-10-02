import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===============================================================
#  STEP 1: The Harmonic Convolutional Layer
# ===============================================================
class HarmonicConvLayer(nn.Module):
    """
    A custom layer for a single Harmonic Network convolution.
    It learns filters as a basis of circular harmonics.
    """

    def __init__(self, in_channels, out_channels, k_size, max_order):
        super(HarmonicConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.rotation_orders = np.arange(0, max_order, dtype=np.int64)

        # --- TODO 1: Define Learnable Parameters ---
        # The learnable parameters are the radial profiles (R(r)) and phase offsets (beta).
        # We need one set for each filter connecting an input channel to an output channel
        # for each rotation order.

        self.num_radii = self._get_unique_radii_count(k_size)
        self.num_orders = len(self.rotation_orders)

        # Shape: (out_channels, in_channels, num_orders, num_radii)
        self.radial_profiles = nn.Parameter(torch.randn(out_channels, in_channels, self.num_orders, self.num_radii))

        # Shape: (out_channels, in_channels, num_orders)
        self.phase_offsets = nn.Parameter(torch.zeros(out_channels, in_channels, self.num_orders))
        # --- TODO 2: Pre-calculate non-learnable filter components ---
        # These are fixed grids that don't change during training.
        # Using register_buffer makes them part of the module's state without being parameters.
        self.register_buffer('r_grid', self._create_r_grid(k_size))
        self.register_buffer('phi_grid', self._create_phi_grid(k_size))
        self.register_buffer('radius_to_index', self._create_radius_to_index_map(k_size))

    def forward(self, x_complex):
        # --- TODO 4: Implement the forward pass ---
        # 1. Assemble the complex-valued filters on the fly.
        # This will result in a tensor of shape:
        # (out_channels, in_channels, num_orders, k_size, k_size)
        filters_complex = self._assemble_filters()

        # We will convolve each rotation order stream separately and sum the results.
        total_output = 0

        # 2. Iterate through the rotation order streams and apply convolutions.
        for i, m in enumerate(self.rotation_orders):
            # Get the filters for the current rotation order 'm'.
            # Shape: (out_channels, in_channels, k_size, k_size)
            w_m = filters_complex[:,:,m,:,:]
            w_real, w_imag = w_m.real, w_m.complex
            # out_real = (x_real * w_real) - (x_imag * w_imag)
            # out_imag = (x_real * w_imag) + (x_imag * w_real)
            x_real, x_imag = x_complex.real, x_complex.imag

            out_real = F.conv2d(x_real, w_real, padding='same') - F.conv2d(x_imag, w_imag, padding='same')
            out_imag = F.conv2d(x_real, w_imag, padding='same') + F.conv2d(x_imag, w_real, padding='same')

            total_output = total_output + torch.complex(out_real, out_imag)
        return total_output

    def _get_unique_radii_count(self, k_size):
        if k_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        center = (k_size - 1) / 2
        x_grid = torch.linspace(-center, center, k_size)
        y_gird = x_grid
        x, y = torch.meshgrid(x_grid, y_gird)
        r = torch.sqrt(x ** 2 + y ** 2)
        return len(np.unique(r))

    def _create_r_grid(self, k_size):
        if k_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        center = (k_size - 1) / 2
        x_grid = torch.linspace(-center, center, k_size)
        y_gird = x_grid
        x, y = torch.meshgrid(x_grid, y_gird)
        r = torch.sqrt(x ** 2 + y ** 2)
        return r

    def _create_phi_grid(self, k_size):
        if k_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        center = (k_size - 1) / 2
        x_grid = torch.linspace(-center, center, k_size)
        y_gird = x_grid
        x, y = torch.meshgrid(x_grid, y_gird)
        theta = torch.arctan2(y,x)
        return theta

    def _create_radius_to_index_map(self, k_size):
        if k_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        r_grid = self._create_r_grid(k_size)
        unique_radii = torch.unique(r_grid)

        radius_to_index = {radius.item(): i for i, radius in enumerate(unique_radii)}
        index_grid = torch.zeros_like(r_grid, dtype=torch.long)
        for radius, index in radius_to_index.items():
            index_grid[r_grid == radius] = index

        return index_grid
    def _assemble_filters(self):
        # --- TODO 3: Implement the filter assembly logic ---
        # This is where you build the complex filter tensor from the learnable parameters.
        # Shape of fitler = (out_channels, in_channels, rotation_order +1, ksize, ksize)
        # 1. Expand radial_profiles and phase_offsets to match the grid shape.
        rotation_orders = self.rotation_orders.reshape(1, 1, -1, 1, 1)

        # Reshape phi_grid to enable proper broadcasting
        phi_grid = self.phi_grid.reshape(1, 1, 1, *self.phi_grid.shape)

        # Reshape phase_offsets for broadcasting
        phase_offsets = self.phase_offsets.reshape(*self.phase_offsets.shape, 1, 1)

        # Get radial profiles with proper indexing
        R = self.radial_profiles[:, :, :, self.radius_to_index]
        # Calculate phase with properly shaped tensors
        phase = rotation_orders * phi_grid + phase_offsets
        print(phase.shape, rotation_orders.shape, phi_grid.shape, phase_offsets.shape)

        # Calculate final filters
        filters = R * torch.exp(1j * phase)
        return filters


# ===============================================================
#  STEP 2: Helper for the H-Net Style ReLU
# ===============================================================
class NormReLU(nn.Module):
    """
    Applies ReLU to the magnitude of complex-valued features,
    preserving the phase.
    """

    def forward(self, x_complex):
        # Correctly calculate magnitude (abs value)
        magnitude = x_complex.abs()

        new_magnitude = F.relu(magnitude)
        unit_phase = x_complex / (magnitude + 1e-8)

        return new_magnitude.unsqueeze(-1) * unit_phase


# ===============================================================
#  STEP 3: The Full H-Net Model
# ===============================================================
class HNet(nn.Module):
    def __init__(self, n_classes=10):
        super(HNet, self).__init__()
        # --- TODO 6: Define the network architecture ---
        self.conv1 = HarmonicConvLayer(in_channels=1, out_channels=16, k_size=7, rotation_orders=[0, 1])
        self.norm_relu1 = NormReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = HarmonicConvLayer(in_channels=16, out_channels=32, k_size=5, rotation_orders=[0, 1])
        self.norm_relu2 = NormReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        # Final layers for classification
        self.flatten = nn.Flatten()
        # TODO: Calculate the correct input features for the linear layer
        # based on the output size of the final conv/pool layer.
        self.fc = nn.Linear(in_features=..., out_features=n_classes)

    def forward(self, x):
        # --- TODO 7: Implement the full forward pass ---
        # 1. Convert the real-valued input `x` to a complex tensor.
        x_complex = torch.complex(x, torch.zeros_like(x))

        # 2. Pass data through the network.
        x = self.conv1(x_complex)
        x = self.norm_relu1(x)
        x = self.pool1(x.abs())  # AvgPool works on real tensors, so we use the magnitude.

        x = self.conv2(torch.complex(x, torch.zeros_like(x)))  # Re-complexify
        x = self.norm_relu2(x)
        x = self.pool2(x.abs())

        # 3. Flatten and pass to the classifier.
        x = self.flatten(x)
        logits = self.fc(x)
        return logits


# ===============================================================
#  STEP 4: Training and Evaluation Loop
# ===============================================================
if __name__ == '__main__':
    # --- TODO 8: Set up your training loop ---
    # 1. Instantiate the model: model = HNet()
    # 2. Create your data loaders (e.g., for rotated MNIST).
    # 3. Define loss function (e.g., nn.CrossEntropyLoss).
    # 4. Define optimizer (e.g., optim.Adam).
    # 5. Write the training and evaluation loops.

    print("H-Net From Scratch Template: Fill in the TODO sections to begin.")   