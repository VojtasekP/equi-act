import torch
from escnn import gspaces, nn

def make_gated_block(act, in_type: nn.FieldType, irrep_repr, channels: int,
                    layers_num: int, kernel_size: int = 5, pad: int = 2,
                    use_bn: bool = True):
    layers = []
    cur_type = in_type

    for _ in range(layers_num):
        feature_repr = irrep_repr * channels
        feature_type = nn.FieldType(act, feature_repr)


        gate_repr = [act.trivial_repr] * len(feature_repr)
        full_type = nn.FieldType(act, gate_repr + feature_repr)

        layers.append(nn.R2Conv(cur_type, full_type, kernel_size=kernel_size, padding=pad, bias=False))


        layers.append(nn.GatedNonLinearity1(full_type, drop_gates=True))

        if use_bn:
            layers.append(nn.IIDBatchNorm2d(feature_type))


        cur_type = feature_type

    return nn.SequentialModule(*layers), cur_type

def make_norm_block(act, in_type: nn.FieldType, irrep_repr, channels: int,
                     layers_num: int, kernel_size: int = 5, pad: int = 2,
                     use_bn: bool = True, non_linearity: str = 'n_relu'):
    layers = []
    cur_type = in_type
    if non_linearity not in ['n_relu', 'n_sigmoid', 'n_softplus', 'squash']:
        raise ValueError(f"Unsupported non-linearity: {non_linearity}. "
                         f"Supported options are: 'n_relu', 'n_sigmoid', 'n_softplus', 'squash'.")
    for _ in range(layers_num):
        feature_repr = irrep_repr * channels
        feature_type = nn.FieldType(act, feature_repr)

        layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=kernel_size, padding=pad, bias=False))

        if use_bn:
            layers.append(nn.IIDBatchNorm2d(feature_type))

        layers.append(nn.NormNonLinearity(in_type=feature_type, function=non_linearity))
        cur_type = feature_type

    return nn.SequentialModule(*layers), cur_type

def make_fourier_block(act, in_type: nn.FieldType, irrep_repr, channels: int,
                     layers_num: int, kernel_size: int = 5, pad: int = 2,
                     use_bn: bool = True):
    layers = []
    cur_type = in_type
    for _ in range(layers_num):
        feature_repr = irrep_repr * channels
        feature_type = nn.FieldType(act, feature_repr)

        layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=kernel_size, padding=pad, bias=False))

        if use_bn:
            layers.append(nn.IIDBatchNorm2d(feature_type))

        layers.append(nn.QuotientFourierPointwise(feature_type))
        cur_type = feature_type



class HNet(torch.nn.Module):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),          # number of channels per block
                 layers_per_block=2,             # depth per block
                 gated=True,                     # switch between gated vs norm blocks
                 kernel_size=5,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 use_bn=True,
                 non_linearity='n_relu',
                 grey_scale=False):
        super().__init__()


        self.r2_act = gspaces.rot2dOnR2(-1, maximum_frequency=max_rot_order)
        self.irreps = [self.r2_act.irrep(m) for m in range(max_rot_order + 1)]
        if grey_scale:
            self.input_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        else:
            self.input_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])
        cur_type = self.input_type
        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i, channels in enumerate(channels_per_block):
            if gated:
                block, cur_type = make_gated_block(act=self.r2_act,
                                                   in_type=cur_type,
                                                   irrep_repr=self.irreps,
                                                   channels=channels,
                                                   layers_num=layers_per_block,
                                                   kernel_size=kernel_size,
                                                   pad=(kernel_size-1) // 2,
                                                   use_bn=use_bn)
            else:

                block, cur_type = make_norm_block(act=self.r2_act,
                                                  in_type=cur_type,
                                                  irrep_repr=self.irreps,
                                                  channels=channels,26
                                                  layers_num=layers_per_block,
                                                  kernel_size=kernel_size,
                                                  pad=(kernel_size-1) // 2,
                                                  use_bn=use_bn,
                                                  non_linearity=non_linearity)

            self.blocks.append(block)

            pool = nn.PointwiseAvgPoolAntialiased(cur_type, sigma=pool_sigma, stride=pool_stride)
            self.pools.append(pool)

        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(cur_type, self.out_inv_type, kernel_size=1, bias=False)

        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, c//2),

            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//2, c//4),

            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//4, n_classes),
        )

    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input, self.input_type)

        # run blocks + pools
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)

        # invariant readout
        x = self.invariant_map(x)
        x = self.avg(x.tensor).squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x