import torch
from escnn import gspaces, nn


def make_gated_block_3d(act, in_type: nn.FieldType, irrep_repr, channels: int,
                        layers_num: int, kernel_size: int = 3, pad: int = 1,
                        use_bn: bool = True):
    """
    Gated non-linearity block for 3D SO(3)-equivariant features.
    Mirrors your 2D version but uses R3Conv and IIDBatchNorm3d.
    """
    layers = []
    cur_type = in_type

    for _ in range(layers_num):
        feature_repr = irrep_repr * channels  # list of irreps repeated "channels" times
        feature_type = nn.FieldType(act, feature_repr)

        # build gated type: one trivial gate per feature field
        gate_repr = [act.trivial_repr] * len(feature_repr)
        full_type = nn.FieldType(act, gate_repr + feature_repr)

        layers.append(nn.R3Conv(cur_type, full_type, kernel_size=kernel_size, padding=pad, bias=False))
        layers.append(nn.GatedNonLinearity1(full_type, drop_gates=True))

        if use_bn:
            layers.append(nn.IIDBatchNorm3d(feature_type))

        cur_type = feature_type

    return nn.SequentialModule(*layers), cur_type


def make_norm_block_3d(act, in_type: nn.FieldType, irrep_repr, channels: int,
                       layers_num: int, kernel_size: int = 3, pad: int = 1,
                       use_bn: bool = True, non_linearity: str = 'n_relu'):
    """
    Norm-based non-linearity block for 3D SO(3)-equivariant features.
    Supported non_linearity values: 'n_relu', 'n_sigmoid', 'n_softplus', 'squash'.
    """
    if non_linearity not in ['n_relu', 'n_sigmoid', 'n_softplus', 'squash']:
        raise ValueError(f"Unsupported non-linearity: {non_linearity}. "
                         f"Supported: 'n_relu', 'n_sigmoid', 'n_softplus', 'squash'.")

    layers = []
    cur_type = in_type

    for _ in range(layers_num):
        feature_repr = irrep_repr * channels
        feature_type = nn.FieldType(act, feature_repr)

        layers.append(nn.R3Conv(cur_type, feature_type, kernel_size=kernel_size, padding=pad, bias=False))

        if use_bn:
            layers.append(nn.IIDBatchNorm3d(feature_type))

        layers.append(nn.NormNonLinearity(in_type=feature_type, function=non_linearity))
        cur_type = feature_type

    return nn.SequentialModule(*layers), cur_type


class SONet(torch.nn.Module):
    """
    SO(3)-equivariant 3D network (R^3 signals) analogous to your HNet, for volumetric inputs.
    Input tensor shape: [B, C_in, D, H, W].
    """

    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),
                 layers_per_block=2,
                 gated=True,
                 kernel_size=3,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 use_bn=True,
                 non_linearity='n_relu',
                 gray_scale=True):
        super().__init__()

        # SO(3) acting on R^3 (3D volumes)
        self.r3_act = gspaces.rot3dOnR3(maximum_frequency=max_rot_order)

        # Irreps l = 0..L (dims 1, 3, 5, ...)
        self.irreps = [self.r3_act.irrep(l) for l in range(max_rot_order + 1)]

        # input type: trivial scalars per input channel (e.g., density/intensity)
        in_channels = 1 if gray_scale else 3
        self.input_type = nn.FieldType(self.r3_act, in_channels * [self.r3_act.trivial_repr])

        cur_type = self.input_type
        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        # choose padding consistent with kernel size
        pad = (kernel_size - 1) // 2


        for channels in channels_per_block:
            if tensor_poly:
                block, cur_type = make_tensorpoly_block_3d(
                    act=self.r3_act,
                    in_type=cur_type,
                    tensor_irrep_l=tensor_irrep_l,
                    channels=channels,
                    layers_num=layers_per_block,
                    kernel_size=kernel_size,
                    pad=pad,
                    use_bn=use_bn
                )
            elif gated:
                block, cur_type = make_gated_block_3d(
                    act=self.r3_act,
                    in_type=cur_type,
                    irrep_repr=self.irreps,
                    channels=channels,
                    layers_num=layers_per_block,
                    kernel_size=kernel_size,
                    pad=pad,
                    use_bn=use_bn
                )

            else:
                block, cur_type = make_norm_block_3d(
                    act=self.r3_act,
                    in_type=cur_type,
                    irrep_repr=self.irreps,
                    channels=channels,
                    layers_num=layers_per_block,
                    kernel_size=kernel_size,
                    pad=pad,
                    use_bn=use_bn,
                    non_linearity=non_linearity
                )

            self.blocks.append(block)

            # Pointwise pooling is allowed because it acts only on spatial dims;
            # escnn provides antialiased avg pooling that works with geometric tensors.
            self.pools.append(nn.PointwiseAvgPoolAntialiased(cur_type, sigma=pool_sigma, stride=pool_stride))

            # Invariant readout: map to purely trivial (rotation-invariant) fields, then global average
        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r3_act, c * [self.r3_act.trivial_repr])
        self.invariant_map = nn.R3Conv(cur_type, self.out_inv_type, kernel_size=1, bias=False)

        self.avg = torch.nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, n_classes),
        )


def forward(self, input: torch.Tensor):
    # input: [B, C_in, D, H, W]
    x = nn.GeometricTensor(input, self.input_type)

    for block, pool in zip(self.blocks, self.pools):
        x = block(x)
        x = pool(x)

    x = self.invariant_map(x)  # -> invariant fiber type
    x = self.avg(x.tensor).flatten(1)  # [B, c]
    x = self.head(x)
    return x


def make_tensorpoly_block_3d(act, in_type: nn.FieldType, tensor_irrep_l: int,
                             channels: int, layers_num: int, kernel_size: int = 3, pad: int = 1,
                             use_bn: bool = True):
    """
    Tensor-polynomial (quadratic) non-linearity block using TensorProductModule.

    Note:
      - TensorProductModule requires UNIFORM field types (same rep repeated).
      - We therefore map to a uniform type of irrep(l) repeated `channels` times,
        apply the tensor product module (quadratic nonlinearity), and stay in that uniform type.
      - Depth > 1 implicitly raises the polynomial degree (quadratic per layer).

    Args:
      act: escnn gspace (rot3dOnR3)
      in_type: incoming FieldType
      tensor_irrep_l: which SO(3) irrep (l) to use for the uniform block (e.g., 0, 1, 2)
      channels: number of fields (copies) of that irrep
      layers_num: how many (Conv -> BN? -> TensorProductModule) layers to stack
      kernel_size, pad, use_bn: usual conv hyperparams

    Returns:
      (block, out_type)
    """
    ir = act.irrep(tensor_irrep_l)
    layers = []
    cur_type = in_type

    for _ in range(layers_num):
        uniform_type = nn.FieldType(act, [ir] * channels)

        # Linear equivariant lift into the uniform representation
        layers.append(nn.R3Conv(cur_type, uniform_type, kernel_size=kernel_size, padding=pad, bias=False))
        if use_bn:
            layers.append(nn.IIDBatchNorm3d(uniform_type))

        # Quadratic tensor-product nonlinearity with learnable projection
        layers.append(nn.TensorProductModule(uniform_type, uniform_type))

        cur_type = uniform_type

    return nn.SequentialModule(*layers), cur_type
