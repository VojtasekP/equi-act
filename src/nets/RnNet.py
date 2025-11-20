import torch
from escnn import gspaces, nn
from typing import List, Tuple, Optional
from escnn.group import directsum
from nets.calculate_channels import adjust_channels
from abc import ABC, abstractmethod
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
    "gated_shared_sigmoid": "sigmoid",
    "non_equi_relu": "p_relu",
    "non_equi_bn": "sigmoid"
}

BN_MAP_2d = {
    "IIDbn": nn.IIDBatchNorm2d,
    "Normbn": nn.NormBatchNorm,
    "FieldNorm": nn.FieldNorm,
    "GNormBatchNorm": nn.GNormBatchNorm
}

BN_MAP_3d = {
    "IIDbn": nn.IIDBatchNorm3d,
    "Normbn": nn.NormBatchNorm,
    "FieldNorm": nn.FieldNorm,
    "GNormBatchNorm": nn.GNormBatchNorm
}

def _split_scalar_vector(ft: nn.FieldType) -> Tuple[List[int], List[int]]:
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
        return input_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-6):

        return True
    
class NonEquivariantTorchOp(nn.EquivariantModule):
    """
    Wraps a regular torch.nn module to operate on GeometricTensor data.
    This deliberately breaks equivariance because the wrapped module is unaware
    of the field structure and only sees the raw tensor.
    """
    def __init__(self, in_type: nn.FieldType, torch_module: torch.nn.Module):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.torch_module = torch_module

    def forward(self, input: nn.GeometricTensor):
        x = self.torch_module(input.tensor)
        return nn.GeometricTensor(x, self.out_type)
    
    def evaluate_output_shape(self, input_shape):
        return input_shape

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-6):
        return False
    
class RnNet(ABC, torch.nn.Module):
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
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
                 grid_size=29,
                 mnist=False):
        super().__init__()

        assert hasattr(self, "r2_act"), "Subclass must set self.r2_act before calling RnNet.__init__"
        assert hasattr(self, "input_type"), "Subclass must set self.input_type before calling RnNet.__init__"

        assert len(channels_per_block) == len(kernels_per_block) == len(paddings_per_block), "channels_per_block, kernels_per_block and padding_per_block must have the same length"

        self.max_rot_order = max_rot_order
        self.bn_type = bn
        self.batch_norm = self._create_bn() # create batch norm object

        self.mnist = mnist

        if activation_type.split("_")[0] in ["fourier"]:
            #remove the last _n
            self.N=int(activation_type.split("_")[-1])
            activation_type = "_".join(activation_type.split("_")[:-1])
        self.activation_type = activation_type
        # here might confusion arise as we added non_equi_bn to activation types, even that it is gated sigmoid with classical nonequi batch norm. This decision was made purly out of implementational ease
        
        assert activation_type in ["gated_sigmoid", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu", "pointwise_relu", "gated_shared_sigmoid", "non_equi_relu", "non_equi_bn"]
        if activation_type == "gated_sigmoid":
            build_layer = self.make_gated_block
        elif activation_type in ["norm_relu", "norm_squash", "norm_sigmoid"]:
            build_layer = self.make_norm_block
        elif activation_type in ["fourier_relu", "fourier_elu"]:
            build_layer = self.make_fourier_block
        elif activation_type in ["gated_shared_sigmoid"]:
            build_layer = self.make_gated_block_shared
        elif activation_type == "non_equi_relu":
            build_layer = self.make_non_equi_layer_nonlin
        elif activation_type == "non_equi_bn":
            build_layer = self.make_non_equi_layer_bn
        self.non_linearity = self._create_fund_non_linearity(activation_type) # determines the non-linearity used in norm and fourier blocks, gated is fixed on relu

        self.mask = nn.MaskModule(self.input_type, grid_size, margin=1)

        eq_layers = []

        self.conv_sigma = conv_sigma
        self.pool_size = pool_size
        self.invar_type = invar_type
        assert pool_type == 'max', "Only max pooling is currently supported"
        self.pool_type = pool_type
        self.pool_sigma = pool_sigma
        self.pool_stride = self.pool_size

        self.LAYER = 0
        self.LAYERS_NUM = len(channels_per_block)
        cur_type = self.input_type
        
        for i, (channels, kernel, padding) in enumerate(zip(channels_per_block, kernels_per_block, paddings_per_block)):

            self.LAYER += 1
            eq_layers += build_layer(cur_type, channels, kernel, padding)
            cur_type = eq_layers[-1].out_type

            if (i + 1) % pool_after_every_n_blocks == 0 and (i + 1) != self.LAYERS_NUM:
                eq_layers += self._build_pooling(eq_layers[-1].out_type)
                cur_type = eq_layers[-1].out_type
        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        
        self.invar_map = self._build_invariant_map(eq_layers[-1].out_type)

        self.pool = nn.PointwiseAdaptiveMaxPool(self.invar_map.out_type, (1, 1))
        invariant_size = len(self.invar_map.out_type)

        self.eq_layers = nn.SequentialModule(*eq_layers)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(invariant_size, c),
            torch.nn.BatchNorm1d(c),
            torch.nn.ELU(inplace=True),

            torch.nn.Linear(c, n_classes),
        )
    @abstractmethod
    def _create_bn(self):
        pass

    def _create_fund_non_linearity(self, activation_type):
        try:
            return ACT_MAP[activation_type]
        except KeyError:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        
    @abstractmethod
    def _build_batch_norm(self, in_type: nn.FieldType):
        pass
    
    @abstractmethod
    def _build_pooling(self, in_type: nn.FieldType):
        pass

    @abstractmethod
    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        pass

    @abstractmethod
    def make_gated_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass

    @abstractmethod
    def make_gated_block_shared(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass
    
    @abstractmethod
    def make_norm_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass

    @abstractmethod
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass
    def forward(self, input: torch.Tensor):

        x = nn.GeometricTensor(input, self.input_type)
    
        x = self.mask(x)
        for layer in self.eq_layers:
            x = layer(x)
        x = self.invar_map(x)
        x = self.pool(x)
        x = x.tensor
        x = self.head(x.view(x.shape[0], -1))
        return x

    def forward_invar_features(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
        for layer in self.eq_layers:
            x = layer(x)
        x = self.invar_map(x)
        return x
    
    def forward_features(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
        for layer in self.eq_layers:
            x = layer(x)
        return x
    
    def init_nth_layer(self, n: int):
        self.target_layer = n - 1

    def forward_upto_nth_layer(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        x = self.mask(x)
        if self.target_layer == 0:
            return x
        for i, layer in enumerate(self.eq_layers):
            x = layer(x)
            if i == self.target_layer - 1:
                return x
        raise ValueError(f"Layer index {self.target_layer} out of range")
    def forward_nth_layer(self, input: torch.Tensor):
        if isinstance(input, nn.GeometricTensor):
            x = input
        else:
            x = nn.GeometricTensor(input, self.input_type)
        for i, layer in enumerate(self.eq_layers):
            if i == self.target_layer:
                x = layer(x)
                return x
        raise ValueError(f"Layer index {self.target_layer} out of range")
    
class R2Net(RnNet):
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
                 grey_scale=False,
                 mnist=False):
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        if flip:
            r2_act = gspaces.flipRot2dOnR2(maximum_frequency=max_rot_order)
        else:
            r2_act = gspaces.rot2dOnR2(maximum_frequency=max_rot_order)
        print(f"Using group: {r2_act.name}, {r2_act._sg_id}")

        input_channels = 1 if grey_scale else 3
        input_type = nn.FieldType(r2_act, input_channels * [r2_act.trivial_repr])

        self.r2_act = r2_act
        self.input_type = input_type
        self.img_size = img_size

        super().__init__(n_classes=n_classes,
                         max_rot_order=max_rot_order,
                         channels_per_block=channels_per_block,
                         kernels_per_block=kernels_per_block,
                         paddings_per_block=paddings_per_block,
                         conv_sigma=conv_sigma,
                         pool_after_every_n_blocks=pool_after_every_n_blocks,
                         activation_type=activation_type,
                         pool_size=pool_size,
                         pool_sigma=pool_sigma,
                         invar_type=invar_type,
                         pool_type=pool_type,
                         invariant_channels=invariant_channels,
                         bn=bn,
                         grid_size=img_size,
                         mnist=mnist)

    def _create_bn(self):
        try:
            return BN_MAP_2d[self.bn_type]
        except KeyError:
            raise ValueError(f"Unsupported batch norm type: {self.bn_type}")
        
    def _build_batch_norm(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return nn.InnerBatchNorm(in_type)
        return nn.MultipleModule(in_type=in_type,
                                    labels=labels,
                                    modules=[(nn.InnerBatchNorm(trivials), 'trivial'),
                                            (self.batch_norm(others), 'others')],
                                    reshuffle=0
                                            )
        
    def _build_pooling(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        print(f"Building pooling with {len(trivials)} trivial and {len(others)} other fields")
        if len(others) == 0:
            return [nn.PointwiseMaxPool2D(in_type, kernel_size=self.pool_stride)]
        modules = [
            (nn.PointwiseMaxPool2D(trivials, kernel_size=self.pool_size), 'trivial'),
            (nn.NormMaxPool(others, kernel_size=self.pool_size), 'others')
        ]
        return [nn.MultipleModule(
            in_type=in_type,
            labels=labels,
            modules=modules,
            reshuffle=0
        )]
    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if self.invar_type == 'conv2triv':
            return nn.R2Conv(in_type, self.out_inv_type, kernel_size=1, padding=0, bias=False)
        elif self.invar_type == 'norm':
            modules = [
                (nn.IdentityModule(trivials), 'trivial'),
                (nn.NormPool(others), 'others')
            ]
            return nn.MultipleModule(
                in_type=in_type,
                labels=labels,
                modules=modules,
                reshuffle=0
            )
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
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        # vector_rep = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)] * channels
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)

        gate_repr = [self.r2_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r2_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field
        non_linearity = nn.MultipleModule(in_type=full_field, 
                            labels=['scalar']*len(scalar_rep) + ['gated']*len(gate_field),
                            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
                            reshuffle=0
                            )

        layers.append(
            nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))
        
        layers.append(non_linearity)


        return layers
    
    def make_non_equi_layer_nonlin(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        Conv -> equivariant batch-norm -> plain torch ReLU (breaks equivariance).
        """
        layers = []
        channels = adjust_channels(channels,
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order,
                                   mnist=self.mnist)
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        layers.append(nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))
        layers.append(self._build_batch_norm(layers[-1].out_type))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.ReLU(inplace=True)))
        return layers

    def make_non_equi_layer_bn(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        Conv -> non-equivariant torch BatchNorm2d -> gated nonlinearity (non-shared).
        """
        layers = []
        channels = adjust_channels(channels,
                                   self.r2_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order,
                                   mnist=self.mnist)
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)

        gate_repr = [self.r2_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r2_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field

        layers.append(nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.BatchNorm2d(layers[-1].out_type.size)))

        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=['scalar'] * len(scalar_rep) + ['gated'] * len(gate_field),
            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
            reshuffle=0
        )
        layers.append(non_linearity)
        return layers
    
    def make_gated_block_shared(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r2_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult)


        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)

        vec_rep_dirsum= directsum(vector_rep, name="vec")
        vector_field = nn.FieldType(self.r2_act, [vec_rep_dirsum] * channels)

        gates = nn.FieldType(self.r2_act, scalar_rep)

        gate_field = (gates + vector_field)

        full_field = (scalar_field + gate_field)
        layers.append(
            nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))
        
        # nelinearity: ELU na scalars, GatedNL na (gates ⊕ vector), gates po NL dropneme
        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=["scalar"] * len(scalar_field) + ["gate"] * len(gate_field),
            modules=[
                (nn.ELU(scalar_field), "scalar"),
                (nn.GatedNonLinearity1(gate_field), "gate"),
            ],
            reshuffle=0
        )
        layers.append(non_linearity)


        return layers
    def make_norm_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r2_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        vector_rep = []
        for irr in self.r2_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult* channels)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field
        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False

        layers.append(nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))

        elu = nn.ELU(scalar_field)
        norm_nonlin = nn.NormNonLinearity(in_type=vector_field, function=self.non_linearity, bias=bias)

        layers.append(self._build_batch_norm(layers[-1].out_type))

        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep), 
                                            modules=[(elu,'scalar'), (norm_nonlin, 'vector')]
                                            )

        layers.append(non_linearity)



        return layers
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r2_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        reps = []
        for irr in self.r2_act.irreps:

            mult = int(irr.size // irr.sum_of_squares_constituents)
            reps.extend([irr] * mult)
        full_rep = reps * channels
        base_field = nn.FieldType(self.r2_act, full_rep)
        G = in_type.fibergroup
        activation = nn.FourierPointwise(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=self.N)
        feat_type_out = activation.in_type

        conv = nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        layers.append(conv)

        layers.append(RetypeModule(conv.out_type, base_field))

        bn_module = self._build_batch_norm(base_field)
        layers.append(bn_module)

        trivial_indices = [i for i, rep in enumerate(base_field) if rep.is_trivial()]
        other_indices = [i for i, rep in enumerate(base_field) if not rep.is_trivial()]
        bn_order = trivial_indices + other_indices
        restore_positions = {idx: pos for pos, idx in enumerate(bn_order)}
        restore_perm = [restore_positions[i] for i in range(len(base_field))]
        if restore_perm != list(range(len(base_field))):
            layers.append(nn.ReshuffleModule(bn_module.out_type, restore_perm))

        layers.append(RetypeModule(base_field, feat_type_out))

        layers.append(activation)

        layers.append(RetypeModule(feat_type_out, base_field))
        return layers
    
class R3Net(RnNet):
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
                 mnist=False):
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        if flip:
            r3_act = gspaces.flipRot3dOnR3(maximum_frequency=max_rot_order)
        else:
            r3_act = gspaces.rot3dOnR3(maximum_frequency=max_rot_order)
        print(f"Using group: {r3_act.name}")

        input_type = nn.FieldType(r3_act, [r3_act.trivial_repr])

        self.r3_act = r3_act
        self.input_type = input_type
        self.img_size = img_size

        super().__init__(n_classes=n_classes,
                         max_rot_order=max_rot_order,
                         channels_per_block=channels_per_block,
                         kernels_per_block=kernels_per_block,
                         paddings_per_block=paddings_per_block,
                         conv_sigma=conv_sigma,
                         pool_after_every_n_blocks=pool_after_every_n_blocks,
                         activation_type=activation_type,
                         pool_size=pool_size,
                         pool_sigma=pool_sigma,
                         invar_type=invar_type,
                         pool_type=pool_type,
                         invariant_channels=invariant_channels,
                         bn=bn,
                         grid_size=img_size,
                         mnist=mnist)

    def _create_bn(self):
        try:
            return BN_MAP_3d[self.bn_type]
        except KeyError:
            raise ValueError(f"Unsupported batch norm type: {self.bn_type}")
        
    def _build_batch_norm(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return nn.InnerBatchNorm(in_type)
        return nn.MultipleModule(in_type=in_type,
                                    labels=labels,
                                    modules=[(nn.InnerBatchNorm(trivials), 'trivial'),
                                            (self.batch_norm(others), 'others')],
                                    reshuffle=0
                                            )
        
    def _build_pooling(self, in_type: nn.FieldType):
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if len(others) == 0:
            return [nn.PointwiseMaxPool3D(in_type, kernel_size=self.pool_stride)]
        modules = [
            (nn.PointwiseMaxPool3D(trivials, kernel_size=self.pool_size), 'trivial'),
            (nn.NormMaxPool(others, kernel_size=self.pool_size), 'others')
        ]
        return [nn.MultipleModule(
            in_type=in_type,
            labels=labels,
            modules=modules,
            reshuffle=0
        )]
    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if self.invar_type == 'conv2triv':
            return nn.R3Conv(in_type, self.out_inv_type, kernel_size=1, padding=0, bias=False)
        elif self.invar_type == 'norm':
            modules = [
                (nn.IdentityModule(trivials), 'trivial'),
                (nn.NormPool(others), 'others')
            ]
            return nn.MultipleModule(
                in_type=in_type,
                labels=labels,
                modules=modules,
                reshuffle=0
            )
        else:
            raise ValueError(f"Unsupported invariant map type: {self.invar_type}")

    def make_gated_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r3_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        # vector_rep = [self.r2_act.irrep(m) for m in range(1, self.max_rot_order + 1)] * channels
        vector_rep = []
        for irr in self.r3_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult* channels)
        scalar_rep = [self.r3_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r3_act, scalar_rep)
        vector_field = nn.FieldType(self.r3_act, vector_rep)

        gate_repr = [self.r3_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r3_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field
        non_linearity = nn.MultipleModule(in_type=full_field, 
                            labels=['scalar']*len(scalar_rep) + ['gated']*len(gate_field),
                            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
                            reshuffle=0
                            )

        layers.append(
            nn.R3Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))
        
        layers.append(non_linearity)


        return layers
    def make_gated_block_shared(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r3_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        vector_rep = []
        for irr in self.r3_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult* channels)
        scalar_rep = [self.r3_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r3_act, scalar_rep)

        vec_rep_dirsum= directsum(vector_rep, name="vec")
        vector_field = nn.FieldType(self.r3_act, [vec_rep_dirsum] * channels)

        gates = nn.FieldType(self.r3_act, scalar_rep)

        gate_field = (gates + vector_field)

        full_field = (scalar_field + gate_field)
        layers.append(
            nn.R2Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))
        
        # nelinearity: ELU na scalars, GatedNL na (gates ⊕ vector), gates po NL dropneme
        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=["scalar"] * len(scalar_field) + ["gate"] * len(gate_field),
            modules=[
                (nn.ELU(scalar_field), "scalar"),
                (nn.GatedNonLinearity1(gate_field), "gate"),
            ],
            reshuffle=0
        )

        layers.append(non_linearity)


        return layers
    def make_norm_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r3_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        vector_rep = []
        for irr in self.r3_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult* channels)
        scalar_rep = [self.r3_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r3_act, scalar_rep)
        vector_field = nn.FieldType(self.r3_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False

        layers.append(nn.R3Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))

        elu = nn.ELU(scalar_field)
        norm_nonlin = nn.NormNonLinearity(in_type=vector_field, function=self.non_linearity, bias=bias)

        layers.append(self._build_batch_norm(layers[-1].out_type))

        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep), 
                                            modules=[(elu,'scalar'), (norm_nonlin, 'vector')]
                                            )

        layers.append(non_linearity)



        return layers
    
    def make_fourier_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        layers = []
        channels = adjust_channels(channels, 
                                self.r3_act,
                                kernel_size,
                                activation_type=self.activation_type,
                                layer_index=self.LAYER,
                                last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                max_frequency=self.max_rot_order,
                                mnist=self.mnist)
        reps = []
        for irr in self.r3_act.irreps:

            mult = int(irr.size // irr.sum_of_squares_constituents)
            reps.extend([irr] * mult)
        full_rep = reps * channels
        base_field = nn.FieldType(self.r3_act, full_rep)
        G = in_type.fibergroup
        activation = nn.FourierPointwise(self.r3_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=self.N)
        feat_type_out = activation.in_type

        conv = nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        layers.append(conv)

        layers.append(RetypeModule(conv.out_type, base_field))

        bn_module = self._build_batch_norm(base_field)
        layers.append(bn_module)

        trivial_indices = [i for i, rep in enumerate(base_field) if rep.is_trivial()]
        other_indices = [i for i, rep in enumerate(base_field) if not rep.is_trivial()]
        bn_order = trivial_indices + other_indices
        restore_positions = {idx: pos for pos, idx in enumerate(bn_order)}
        restore_perm = [restore_positions[i] for i in range(len(base_field))]
        if restore_perm != list(range(len(base_field))):
            layers.append(nn.ReshuffleModule(bn_module.out_type, restore_perm))

        layers.append(RetypeModule(base_field, feat_type_out))

        layers.append(activation)

        layers.append(RetypeModule(feat_type_out, base_field))
        return layers
    
    def make_non_equi_layer_nonlin(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        Conv -> equivariant batch-norm -> plain torch ReLU (breaks equivariance).
        """
        layers = []
        channels = adjust_channels(channels,
                                   self.r3_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order,
                                   mnist=self.mnist)
        vector_rep = []
        for irr in self.r3_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)
        scalar_rep = [self.r3_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r3_act, scalar_rep)
        vector_field = nn.FieldType(self.r3_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        layers.append(nn.R3Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))
        layers.append(self._build_batch_norm(layers[-1].out_type))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.ReLU(inplace=True)))
        return layers

    def make_non_equi_layer_bn(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        Conv -> non-equivariant torch BatchNorm3d -> gated nonlinearity (non-shared).
        """
        layers = []
        channels = adjust_channels(channels,
                                   self.r3_act,
                                   kernel_size,
                                   activation_type=self.activation_type,
                                   layer_index=self.LAYER,
                                   last_layer=True if self.LAYER == self.LAYERS_NUM else False,
                                   max_frequency=self.max_rot_order,
                                   mnist=self.mnist)
        vector_rep = []
        for irr in self.r3_act.irreps:
            if irr.is_trivial():
                continue
            mult = int(irr.size // irr.sum_of_squares_constituents)
            vector_rep.extend([irr] * mult * channels)
        scalar_rep = [self.r3_act.trivial_repr] * channels
        scalar_field = nn.FieldType(self.r3_act, scalar_rep)
        vector_field = nn.FieldType(self.r3_act, vector_rep)

        gate_repr = [self.r3_act.trivial_repr] * len(vector_rep)
        gate_field = nn.FieldType(self.r3_act, gate_repr) + vector_field
        full_field = scalar_field + gate_field

        layers.append(nn.R3Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.BatchNorm3d(layers[-1].out_type.size)))

        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=['scalar'] * len(scalar_rep) + ['gated'] * len(gate_field),
            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
            reshuffle=0
        )
        layers.append(non_linearity)
        return layers
