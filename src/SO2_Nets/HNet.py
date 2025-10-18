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
    "gated_sigmoid": "sigmoid",
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
                 kernels_per_block=(3, 3, 3),
                 paddings_per_block=(1, 1, 1),
                 pool_after_every_n_blocks=2,
                 activation_type="gated_sigmoid",
                 pool_stride=2,
                 pool_sigma=0.66,
                 invar_type='norm',
                 pool_type='max',
                 invariant_channels=64,
                 bn="IIDbn",
                 img_size=29,
                 grey_scale=False):
        super().__init__()
        assert activation_type in ["gated_sigmoid", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu", "pointwise_relu"]
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]
        
        assert len(channels_per_block) == len(kernels_per_block) == len(paddings_per_block), "channels_per_block, kernels_per_block and padding_per_block must have the same length"
        self.r2_act = gspaces.rot2dOnR2(-1, maximum_frequency=max_rot_order)
        self.max_rot_order = max_rot_order

        self._create_irreps(bn)
        self._create_input_type(grey_scale)


        self.img_size = img_size
        self.mask = nn.MaskModule(self.input_type, img_size, margin=1)

        self.batch_norm = self._create_bn(bn) # create batch norm object
        self.bn_type = bn
        self.non_linearity = self._create_fund_non_linearity(activation_type) # determines the non-linearity used in norm and fourier blocks, gated is fixed on relu
        cur_type = self.input_type
        self.blocks = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i, (channels, kernel, padding) in enumerate(zip(channels_per_block, kernels_per_block, paddings_per_block)):
            if activation_type=="gated_sigmoid":
                block, cur_type = self.make_gated_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        kernel_size=kernel,
                                        pad=padding)
                
            elif activation_type in ["norm_relu", "norm_squash", "norm_sigmoid", "norm_softplus"]:

                block, cur_type = self.make_norm_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        kernel_size=kernel,
                                        pad=padding)
                
            elif activation_type in ["fourier_relu", "fourier_elu"]:  # fourier
                block, cur_type = self.make_fourier_block(
                                        in_type=cur_type,
                                        channels=channels,
                                        kernel_size=kernel,
                                        pad=padding)

            self.blocks.append(block)


            if ((i+1) % pool_after_every_n_blocks) == 0 and (i != len(channels_per_block)-1):
                if pool_type == 'avg':
                    pool = nn.PointwiseAvgPoolAntialiased2D(cur_type, sigma=pool_sigma, stride=pool_stride)
                elif pool_type == 'max':
                    pool = nn.NormMaxPool(cur_type, kernel_size=pool_stride)
                else:
                    raise ValueError(f"Unsupported pool type: {pool_type}")
                cur_type = pool.out_type
                self.blocks.append(pool)

        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        if invar_type == 'conv2triv':
            self.invariant_map = nn.R2Conv(cur_type, self.out_inv_type, kernel_size=1, padding=0, bias=False)
        elif invar_type == 'norm':
            self.invariant_map = nn.NormPool(in_type=cur_type)
        else:
            raise ValueError(f"Unsupported invariant map type: {invar_type}")
        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        invariant_size = len(self.invariant_map.out_type)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(invariant_size, c),
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.3),

            torch.nn.Linear(c, c),
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(p=0.3),

            torch.nn.Linear(c, n_classes),
        )
    def _create_irreps(self, bn_type):
        
        self.type_1_irrep = [self.r2_act.irrep(0)]        
        self.type_2_irreps = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)]

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
        

    def make_gated_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        cur_type = in_type

        vector_rep = self.type_2_irreps * channels
        scalar_rep = self.type_1_irrep * channels

        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        
        feat_type_out = scalar_field + vector_field

        gate_repr = [self.r2_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r2_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field
        layers.append(nn.R2Conv(cur_type, full_field, kernel_size=kernel_size, padding=pad))

        non_linearity = nn.MultipleModule(in_type=full_field, 
                                        labels=['elu']*len(scalar_rep) + ['gate']*len(gate_field),
                                        modules=[(nn.ELU(scalar_field), 'elu'), (nn.GatedNonLinearity1(gate_field, drop_gates=True), 'gate')]
                                        )
        layers.append(non_linearity)

        if self.bn_type == "Normbn":
            identity = nn.IdentityModule(scalar_field)
            batch_norm = nn.MultipleModule(in_type=feat_type_out,
                                        labels=['none']*len(scalar_rep) + ['norm']*len(vector_rep),
                                        modules=[(identity, 'none'), (self.batch_norm(vector_field), 'norm')]
                                        )
            layers.append(batch_norm)
        else:               
            layers.append(self.batch_norm(feat_type_out))


        cur_type = feat_type_out

        return nn.SequentialModule(*layers), cur_type

    def make_norm_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        cur_type = in_type
        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False
        vector_rep = self.type_2_irreps * channels
        scalar_rep = self.type_1_irrep * channels
        
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)

        feat_type_out = scalar_field + vector_field

        layers.append(nn.R2Conv(cur_type, feat_type_out, kernel_size=kernel_size, padding=pad))

        elu = nn.ELU(scalar_field)
        norm_nonlin = nn.NormNonLinearity(in_type=vector_field, function=self.non_linearity, bias=bias)

        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['elu']*len(scalar_rep) + ['norm']*len(vector_rep), 
                                            modules=[(elu,'elu'), (norm_nonlin, 'norm')]
                                            )

        layers.append(non_linearity)
        if self.bn_type == "Normbn":
            identity = nn.IdentityModule(scalar_field)
            batch_norm = nn.MultipleModule(in_type=feat_type_out,
                                        labels=['none']*len(scalar_rep) + ['norm']*len(vector_rep),
                                        modules=[(identity, 'none'), (self.batch_norm(vector_field), 'norm')]
                                        )
            layers.append(batch_norm)
        else:               
            layers.append(self.batch_norm(feat_type_out))

        cur_type = feat_type_out

        return nn.SequentialModule(*layers), cur_type
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        cur_type = in_type
        scalar_field = nn.FieldType(self.r2_act, self.type_1_irrep * channels)
        G = in_type.fibergroup

        activation = nn.FourierPointwise(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=16)
        feat_type_out = activation.out_type

        layers.append(nn.R2Conv(cur_type, feat_type_out, kernel_size=kernel_size, padding=pad))

        layers.append(activation)

        layers.append(self.batch_norm(feat_type_out))

        cur_type = feat_type_out
        return nn.SequentialModule(*layers), cur_type
    
    def make_adaptive_sampling_fourier_block(self, in_type: nn.FieldType, channels: int, layers_num: int):
        layers = []
        cur_type = in_type
        G = in_type.fibergroup
        for _ in range(layers_num):
            feature_repr = self.type_2_irreps * channels
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
        x = self.mask(x)
        # run blocks + pools
        for block in self.blocks: 
            x = block(x)

        # invariant readout
        x = self.invariant_map(x)
        x = x.tensor
        x = self.avg(x)
        x = self.head(x.view(x.shape[0], -1))
        return x

    def forward_features(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
            
        for block in self.blocks:
            x = block(x)
        x = self.invariant_map(x)
        x = x.tensor
        x = self.avg(x)
        return x
