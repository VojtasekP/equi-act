import torch
from escnn import gspaces, nn


ACT_MAP = {
    "norm_relu": "n_relu",
    "norm_sigmoid": "n_sigmoid",
    "norm_softplus": "n_softplus",
    "norm_squash": "squash",
    "pointwise_relu": "p_relu",
    "pointwise_elu": "p_elu",
    "fourier_relu": "p_relu",
    "fourier_elu": "p_elu",
    "gated_relu": "relu",
}

BN_MAP = {
    "IIDbn": nn.IIDBatchNorm2d,
    "Normbn": nn.NormBatchNorm,
    "FieldNorm": nn.FieldNorm,
    "GNormBatchNorm": nn.GNormBatchNorm
}
class HNet(torch.nn.Module):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 channels_per_block=(8, 16, 128),
                 layers_per_block=2,
                 activation_type="gated_relu",
                 kernel_size=5,
                 pool_stride=2,
                 pool_sigma=0.66,
                 invariant_channels=64,
                 bn="IIDbn",
                 grey_scale=False):
        super().__init__()
        assert activation_type in ["gated_relu", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu", "pointwise_relu"]
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        self.r2_act = gspaces.rot2dOnR2(-1, maximum_frequency=max_rot_order)
        self.max_rot_order = max_rot_order

        self._create_irreps(bn)
        self._create_input_type(grey_scale)


        self.batch_norm = self._create_bn(bn) # create batch norm object
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.non_linearity = self._create_fund_non_linearity(activation_type) # determines the non-linearity used in norm and fourier blocks, gated is fixed on relu
        cur_type = self.input_type
        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i, channels in enumerate(channels_per_block):
            if activation_type=="gated_relu":
                block, cur_type = self.make_gated_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        layers_num=layers_per_block)
                
            elif activation_type in ["norm_relu", "norm_squash", "norm_sigmoid", "norm_softplus"]:

                block, cur_type = self.make_norm_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        layers_num=layers_per_block)
                
            elif activation_type in ["fourier_relu", "fourier_elu"]:  # fourier
                block, cur_type = self.make_fourier_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        layers_num=layers_per_block)

            self.blocks.append(block)

            pool = nn.PointwiseAvgPoolAntialiased2D(cur_type, sigma=pool_sigma, stride=pool_stride)
            self.pools.append(pool)

        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        self.invariant_map = nn.R2Conv(cur_type, self.out_inv_type, kernel_size=3, padding=1, bias=False)

        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        print(c)
        self.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c, c//2),

            torch.nn.BatchNorm1d(c//2),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//2, c//4),

            torch.nn.BatchNorm1d(c//4),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(c//4, n_classes),
        )
    def _create_irreps(self, bn_type):
        if bn_type == "Normbn": # for normbn we cannot have trivial representations
            assert self.max_rot_order > 0, "NormBatchNorm is not defined for trivial representations. Please set max_rot_order > 0."
            self.irreps = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)]
        else:
            self.irreps = [self.r2_act.irrep(m) for m in range(self.max_rot_order + 1)]

    def _create_input_type(self, grey_scale):
        if grey_scale:
            self.input_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        else:
            self.input_type = nn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

    def _create_bn(self, bn_type):
        try:
            return BN_MAP[bn_type]
        except KeyError:
            raise ValueError(f"Unsupported batch norm type: {bn_type}")

    def _create_fund_non_linearity(self, activation_type):
        try:
            return ACT_MAP[activation_type]
        except KeyError:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        

    def make_gated_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type

        for _ in range(layers_num):

            feature_repr = self.irreps * channels
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            gate_repr = [self.r2_act.trivial_repr] * len(feature_repr)
            full_type = nn.FieldType(self.r2_act, gate_repr + feature_repr)

            layers.append(nn.R2Conv(cur_type, full_type, kernel_size=self.kernel_size, padding=self.pad))


            layers.append(nn.GatedNonLinearity1(full_type, drop_gates=True))

            layers.append(self.batch_norm(feature_type))


            cur_type = feature_type

        return nn.SequentialModule(*layers), cur_type

    def make_norm_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False
        for _ in range(layers_num):
            feature_repr = self.irreps * channels
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=self.kernel_size, padding=self.pad))
            layers.append(nn.NormNonLinearity(in_type=feature_type, function=self.non_linearity, bias=bias))

            layers.append(self.batch_norm(feature_type))

            cur_type = feature_type

        return nn.SequentialModule(*layers), cur_type
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type

        G = in_type.fibergroup
        for _ in range(layers_num):
            feature_repr = self.irreps * channels
            feature_type = nn.FieldType(self.r2_act, feature_repr)

            activation = nn.FourierPointwise(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=16)
            feature_type = activation.out_type

            layers.append(nn.R2Conv(cur_type, feature_type, kernel_size=self.kernel_size, padding=self.pad))

            layers.append(activation)


            layers.append(self.batch_norm(feature_type))

            cur_type = feature_type
        return nn.SequentialModule(*layers), cur_type
    
    def make_adaptive_sampling_fourier_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        G = in_type.fibergroup
        for _ in range(layers_num):
            feature_repr = self.irreps * channels
            activation = AdaptiveFourierPointwise(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=16, adaptive_sampling=True)
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

    def forward_features(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
            
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        x = self.invariant_map(x)
        x = self.avg(x.tensor).squeeze(-1).squeeze(-1)
        return x
