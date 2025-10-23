import torch
from escnn import gspaces, nn
from typing import List, Tuple, Optional
from escnn.group import directsum
from SO2_Nets.calculate_channels import adjust_channels
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
    "gated_shared_sigmoid": "sigmoid"
}

BN_MAP = {
    "IIDbn": nn.IIDBatchNorm2d,
    "Normbn": nn.NormBatchNorm,
    "FieldNorm": nn.FieldNorm,
    "GNormBatchNorm": nn.GNormBatchNorm
}

def _split_scalar_vector(ft: nn.FieldType) -> Tuple[List[int], List[int]]:
    # For SO(2)/O(2): 1D = scalar/pseudoscalar, 2D = vector (k>0 real irreps)
    scalar_idx = [i for i, rep in enumerate(ft.representations) if rep.size == 1]
    vector_idx = [i for i, rep in enumerate(ft.representations) if rep.size == 2]
    return scalar_idx, vector_idx

def _labels_for_fields(n_fields: int, scalar_idx: List[int]) -> List[str]:
    scalar_set = set(scalar_idx)
    return ['scalar' if i in scalar_set else 'vector' for i in range(n_fields)]

class RetypeModule(nn.EquivariantModule):
    def __init__(self, in_type:nn.FieldType, out_type: nn.FieldType):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type

    def forward(self, input: nn.GeometricTensor) -> nn.GeometricTensor:
        return nn.GeometricTensor(input.tensor, self.out_type)
    
    def evaluate_output_shape(self, input_shape):
        # identity on the data shape
        return input_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-6):
        # You can optionally implement a quick stochastic check here,
        # but for a pure retype this is usually skipped or delegated.
        return True
class R2Net(torch.nn.Module):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 flip=False,
                 channels_per_block=(8, 16, 128),
                 kernels_per_block=(3, 3, 3),
                 paddings_per_block=(1, 1, 1),
                 conv_sigma=0.6,
                 pool_after_every_n_blocks=2,
                 activation_type="gated_sigmoid",
                 pool_size=2,
                 pool_sigma=0.66,
                 invar_type='norm',
                 pool_type='max',
                 invariant_channels=64,
                 bn="IIDbn",
                 img_size=29,
                 grey_scale=False):
        super().__init__()
        assert activation_type in ["gated_sigmoid", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu", "pointwise_relu", "gated_shared_sigmoid"]
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]
        
        assert len(channels_per_block) == len(kernels_per_block) == len(paddings_per_block), "channels_per_block, kernels_per_block and padding_per_block must have the same length"
        if flip:
            self.r2_act = gspaces.flipRot2dOnR2(maximum_frequency=max_rot_order)
        else:
            self.r2_act = gspaces.rot2dOnR2(maximum_frequency=max_rot_order)

        self.max_rot_order = max_rot_order

        input_channels = 1 if grey_scale else 3
        self.input_type = nn.FieldType(self.r2_act, input_channels* [self.r2_act.trivial_repr])

        self.img_size = img_size
        self.mask = nn.MaskModule(self.input_type, img_size, margin=1) 

        self.bn_type = bn
        self.batch_norm = self._create_bn() # create batch norm object
        self.non_linearity = self._create_fund_non_linearity(activation_type) # determines the non-linearity used in norm and fourier blocks, gated is fixed on relu


        self.activation_type = activation_type


        if activation_type == "gated_sigmoid":
            build_layer = self.make_gated_block
        elif activation_type in ["norm_relu", "norm_squash", "norm_sigmoid"]:
            build_layer = self.make_norm_block
        elif activation_type in ["fourier_relu", "fourier_elu"]:
            build_layer = self.make_fourier_block
        elif activation_type in ["gated_shared_sigmoid"]:
            build_layer = self.make_gated_block_shared

        eq_layers = []

        self.conv_sigma = conv_sigma
        self.pool_size = pool_size
        self.invar_type = invar_type
        assert pool_type == 'max', "Only max pooling is currently supported"
        self.pool_type = pool_type
        self.pool_sigma = pool_sigma

        self.LAYER = 0
        self.LAYERS_NUM = len(channels_per_block)
        cur_type = self.input_type
        
        for i, (channels, kernel, padding) in enumerate(zip(channels_per_block, kernels_per_block, paddings_per_block)):

            self.LAYER += 1
            eq_layers += build_layer(cur_type, channels, kernel, padding)
            cur_type = eq_layers[-1].out_type

            if (i + 1) % pool_after_every_n_blocks == 0 and (i + 1) != self.LAYERS_NUM:
                eq_layers += self._build_pooling(eq_layers[-1].out_type)

        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        
        eq_layers += self._build_invariant_map(eq_layers[-1].out_type)

        self.pool = nn.PointwiseAdaptiveMaxPool(eq_layers[-1].out_type, (1, 1))
        invariant_size = len(eq_layers[-1].out_type)

        self.eq_layers = nn.SequentialModule(*eq_layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(invariant_size, c),
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),

            torch.nn.Linear(c, n_classes),
        )

    def _create_bn(self):
        try:
            return BN_MAP[self.bn_type]
        except KeyError:
            raise ValueError(f"Unsupported batch norm type: {self.bn_type}")

    def _create_fund_non_linearity(self, activation_type):
        try:
            return ACT_MAP[activation_type]
        except KeyError:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
    def _build_batch_norm(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return nn.IIDBatchNorm2d(in_type)
        if self.bn_type == "Normbn":
            return nn.MultipleModule(in_type=in_type,
                                     labels=labels,
                                     modules=[(nn.InnerBatchNorm(trivials), 'trivial'),
                                              (self.batch_norm(others), 'others')]
                                              )
        else:
            return self.batch_norm(in_type)
        
    def _build_pooling(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return [nn.PointwiseMaxPool2D(in_type, kernel_size=self.pool_stride)]
        modules = [
            (nn.PointwiseMaxPool2D(trivials, kernel_size=self.pool_size), 'trivial'),
            (nn.NormMaxPool(others, kernel_size=self.pool_size), 'others')
        ]
        return [nn.MultipleModule(
            in_type=in_type,
            labels=labels,
            modules=modules
        )]
    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if self.invar_type == 'conv2triv':
            return [nn.R2Conv(in_type, self.out_inv_type, kernel_size=1, padding=0, bias=False)]
        elif self.invar_type == 'norm':
            modules = [
                (nn.IdentityModule(trivials), 'trivial'),
                (nn.NormPool(others), 'others')
            ]
            return [nn.MultipleModule(
                in_type=in_type,
                labels=labels,
                modules=modules
            )]
        else:
            raise ValueError(f"Unsupported invariant map type: {self.invar_type}")

    def make_gated_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order)
        vector_rep = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)] * channels
        scalar_rep = [self.r2_act.irrep(0)] * channels

        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)

        gate_repr = [self.r2_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r2_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field
        non_linearity = nn.MultipleModule(in_type=full_field, 
                            labels=['scalar']*len(scalar_rep) + ['gated']*len(gate_field),
                            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')]
                            )

        layers.append(
            nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )
        
        layers.append(non_linearity)

        layers.append(self._build_batch_norm(layers[-1].out_type))

        return layers
    def make_gated_block_shared(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order)
        vector_rep = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)] # without * channels, after directsum we will multiply
        scalar_rep = [self.r2_act.irrep(0)] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)

        vec_rep_dirsum= directsum(vector_rep, name="vec")
        vector_field = nn.FieldType(self.r2_act, [vec_rep_dirsum] * channels)

        gates = nn.FieldType(self.r2_act, scalar_rep)

        gate_field = (gates + vector_field).sorted()

        full_field = (scalar_field + gate_field).sorted()
        layers.append(
            nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )

        # nelinearity: ELU na scalars, GatedNL na (gates ⊕ vector), gates po NL dropneme
        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=["scalar"] * len(scalar_field) + ["gate"] * len(gate_field),
            modules=[
                (nn.ELU(scalar_field), "scalar"),
                (nn.GatedNonLinearity1(gate_field), "gate"),
            ],
        )
        layers.append(non_linearity)

        layers.append(self._build_batch_norm(layers[-1].out_type))

        return layers
    def make_norm_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order)
        vector_rep = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)] * channels
        scalar_rep = [self.r2_act.irrep(0)] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field
        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False

        layers.append(nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad))

        elu = nn.ELU(scalar_field)
        norm_nonlin = nn.NormNonLinearity(in_type=vector_field, function=self.non_linearity, bias=bias)

        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep), 
                                            modules=[(elu,'scalar'), (norm_nonlin, 'vector')]
                                            )

        layers.append(non_linearity)

        layers.append(self._build_batch_norm(layers[-1].out_type))

        return layers
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order)
        reps = [self.r2_act.irrep(m) for m in range(0, self.max_rot_order + 1)]
        full_rep = reps * channels
        base_field = nn.FieldType(self.r2_act, full_rep)
        G = in_type.fibergroup

        activation = nn.FourierPointwise(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=16)
        feat_type_out = activation.in_type

        layers.append(nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))

        layers.append(activation)
        layers.append(RetypeModule(feat_type_out,base_field))

        layers.append(self._build_batch_norm(layers[-1].out_type))
        return layers
    

    def forward(self, input: torch.Tensor):

        x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
        for layer in self.eq_layers:
            x = layer(x)

        x = self.pool(x)
        x = x.tensor
        x = self.head(x.view(x.shape[0], -1))
        return x

    def forward_features(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
        for layer in self.eq_layers:
            x = layer(x)
        x = self.pool(x)
        x = x.tensor
        return x
