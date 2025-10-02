import torch
from escnn import gspaces, nn
import layers

class HNet(torch.nn.Module):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),
                 layers_per_block=2,
                 activation_type="gated",
                 kernel_size=5,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 bn="IIDbn",
                 non_linearity='relu',
                 grey_scale=False):
        super().__init__()
        assert activation_type in ["gated", "norm_relu", "norm_squash", "fourier"]
        assert non_linearity in ['relu', 'sigmoid', 'softplus', 'squash']
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        self.r2_act = gspaces.rot2dOnR2(-1, maximum_frequency=max_rot_order)
        if bn == "Normbn":
            assert max_rot_order > 0, "NormBatchNorm is not defined for trivial representations. Please set max_rot_order > 0."
            self.irreps = [self.r2_act.irrep(m) for m in range(1, max_rot_order + 1)]
        else:
            self.irreps = [self.r2_act.irrep(m) for m in range(max_rot_order + 1)]
        if grey_scale:
            self.input_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        else:
            self.input_type = nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        self.batch_norm = self._create_bn(bn)
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.non_linearity = non_linearity
        cur_type = self.input_type
        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i, channels in enumerate(channels_per_block):
            if activation_type=="gated":
                block, cur_type = layers.make_gated_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        layers_num=layers_per_block)
                
            elif activation_type in ["norm_relu", "norm_squash"]:

                block, cur_type = layers.make_norm_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        layers_num=layers_per_block)
                
            else:  # fourier
                block, cur_type = layers.make_fourier_block(
                                        in_type=cur_type
                                        channels=channels,
                                        layers_num=layers_per_block)

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

            torch.nn.BatchNorm1d(c//2),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//2, c//4),

            torch.nn.BatchNorm1d(c//2),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//4, n_classes),
        )

    def _create_bn(bn_type):
        if bn_type == "IIDbn":
            return nn.IIDBatchNorm2d
        elif bn_type == "Normbn":
            return nn.NormBatchNorm
        elif bn_type == "FieldNorm":
            return nn.FieldNorm
        elif bn_type == "GNormBatchNorm":
            return nn.GNormBatchNorm
        else:
            raise ValueError(f"Unsupported batch norm type: {bn_type}. Supported types are: 'IIDbn', 'Normbn', 'FieldNorm', 'GNormBatchNorm'.")
        
    def make_gated_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type

        for _ in range(layers_num):

            feature_repr = self.irrep_repr * channels
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            gate_repr = [self.r2_act.trivial_repr] * len(feature_repr)
            full_type = nn.FieldType(self.r2_act, gate_repr + feature_repr)

            layers.append(nn.R2Conv(cur_type, full_type, kernel_size=self.kernel_size, padding=self.pad, bias=False))


            layers.append(nn.GatedNonLinearity1(full_type, drop_gates=True))

            layers.append(self.batch_norm(feature_type))


            cur_type = feature_type

        return nn.SequentialModule(*layers), cur_type

    def make_norm_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        assert self.non_linearity not in ['relu', 'sigmoid', 'softplus', 'squash']

        if self.non_linearity in ['relu', 'sigmoid', 'softplus']: 
            non_linearity = 'n_' + non_linearity
        for _ in range(layers_num):
            feature_repr = self.irrep_repr * channels
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=self.kernel_size, padding=self.pad, bias=False))

            layers.append(nn.NormNonLinearity(in_type=feature_type, function=non_linearity))

            layers.append(self.batch_norm(feature_type))

            cur_type = feature_type

        return nn.SequentialModule(*layers), cur_type
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        G = in_type.fibergroup
        assert self.non_linearity in ['relu', 'elu', 'sigmoid', 'tanh']
        non_linearity = 'p_' + non_linearity
        for _ in range(layers_num):
            feature_repr = self.irrep_repr * channels
            activation = nn.FourierPointwise(feature_type, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=non_linearity, N=16)
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=self.kernel_size, padding=self.pad, bias=False))

            layers.append(activation)
            feature_type = activation.out_type

            layers.append(self.batch_norm(feature_type))


            cur_type = feature_type
        return nn.SequentialModule(*layers), cur_type
    
    def make_adaptive_sampling_fourier_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        G = in_type.fibergroup
        assert self.non_linearity in ['relu', 'elu', 'sigmoid', 'tanh']
        non_linearity = 'p_' + non_linearity
        for _ in range(layers_num):
            feature_repr = self.irrep_repr * channels
            activation = nn.FourierPointwise(feature_type, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=non_linearity, N=16, adaptive_sampling=True)
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=self.kernel_size, padding=self.pad, bias=False))

            layers.append(activation)
            feature_type = activation.out_type

            layers.append(self.batch_norm(feature_type))


            cur_type = feature_type
        return nn.SequentialModule(*layers), cur_type

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