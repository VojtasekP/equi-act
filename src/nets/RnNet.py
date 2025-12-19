import torch
from escnn import gspaces, nn
from typing import List, Tuple, Optional
from escnn.group import directsum
from nets.calculate_channels import adjust_channels
from abc import ABC, abstractmethod
from nets.new_layers import NormNonlinearityWithBN, FourierPointwiseInnerBn
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
    "non_equi_bn": "sigmoid",
    "normbn_relu": "relu",
    "normbn_elu": "elu",
    "normbn_sigmoid": "sigmoid",
    "normbnvec_relu": "relu",
    "normbnvec_elu": "elu",
    "normbnvec_sigmoid": "sigmoid",
    "fourierbn_relu":"p_relu",
    "fourierbn_elu":"p_elu"
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


class ResidualBlock(nn.EquivariantModule):
    """
    Wrap a main equivariant branch with an optional skip mapping (1x1 conv) to enable residual connections.
    """
    def __init__(self, main: nn.SequentialModule, skip: nn.EquivariantModule | None = None):
        super().__init__()
        self.main = main
        self.skip = skip
        self.in_type = main.in_type
        self.out_type = main.out_type

    def forward(self, input: nn.GeometricTensor) -> nn.GeometricTensor:
        y = self.main(input)
        shortcut = input if self.skip is None else self.skip(input)
        assert shortcut.type == y.type, "Residual shortcut and main branch must produce the same type"
        return nn.GeometricTensor(shortcut.tensor + y.tensor, y.type)

    def evaluate_output_shape(self, input_shape):
        return self.main.evaluate_output_shape(input_shape)

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-6):
        return self.main.check_equivariance(atol, rtol)
    
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
                 mnist=False,
                 residual: bool = False):
        super().__init__()

        assert hasattr(self, "r2_act"), "Subclass must set self.r2_act before calling RnNet.__init__"
        assert hasattr(self, "input_type"), "Subclass must set self.input_type before calling RnNet.__init__"

        assert len(channels_per_block) == len(kernels_per_block) == len(paddings_per_block), "channels_per_block, kernels_per_block and padding_per_block must have the same length"

        self.max_rot_order = max_rot_order
        self.bn_type = bn
        self.batch_norm = self._create_bn() # create batch norm object

        self.mnist = mnist

        if activation_type.split("_")[0] in ["fourier", "fourierbn"]:
            #remove the last _n
            self.N=int(activation_type.split("_")[-1])
            activation_type = "_".join(activation_type.split("_")[:-1])
        self.activation_type = activation_type
        # here might confusion arise as we added non_equi_bn to activation types, even that it is gated sigmoid with classical nonequi batch norm. This decision was made purly out of implementational ease
        
        assert activation_type in ["gated_sigmoid", "norm_relu", "norm_squash", "fourier_relu", "fourier_elu", "pointwise_relu", "gated_shared_sigmoid", "non_equi_relu", "non_equi_bn", "normbn_relu", "normbn_elu", "normbn_sigmoid", "normbnvec_relu", "normbnvec_elu", "normbnvec_sigmoid",  "fourierbn_elu", "fourierbn_relu"]
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
        elif activation_type in ["normbn_relu", "normbn_elu", "normbn_sigmoid"]:
            build_layer = self.make_normbn_relu_block
        elif activation_type in ["fourierbn_elu", "fourierbn_relu"]:
            build_layer = self.make_fourierbn_block
        elif activation_type in ["normbnvec_relu", "normbnvec_elu", "normbnvec_sigmoid"]:
            build_layer = self.make_normbn_relu_only_on_vec_block
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
        self.use_residual = residual

        self.LAYER = 0
        self.LAYERS_NUM = len(channels_per_block)
        cur_type = self.input_type
        
        for i, (channels, kernel, padding) in enumerate(zip(channels_per_block, kernels_per_block, paddings_per_block)):

            self.LAYER += 1
            block_modules = build_layer(cur_type, channels, kernel, padding)
            block_seq = nn.SequentialModule(*block_modules)
            if self.use_residual and (i + 1) % 2 == 0:
                skip = None
                if block_seq.in_type != block_seq.out_type:
                    skip = nn.R2Conv(block_seq.in_type, block_seq.out_type, kernel_size=1, padding=0, sigma=self.conv_sigma, bias=False)
                block = ResidualBlock(block_seq, skip)
            else:
                block = block_seq
            eq_layers.append(block)
            cur_type = block.out_type

            if (i + 1) % pool_after_every_n_blocks == 0 and (i + 1) != self.LAYERS_NUM:
                eq_layers += self._build_pooling(eq_layers[-1].out_type)
                cur_type = eq_layers[-1].out_type
        c = invariant_channels
        self.out_inv_type = nn.FieldType(self.r2_act, c * [self.r2_act.trivial_repr])
        
        self.invar_map = self._build_invariant_map(eq_layers[-1].out_type)

        self.pool = nn.PointwiseAdaptiveMaxPool(self.invar_map.out_type, (1, 1))
        # (B, )
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

    @abstractmethod
    def make_normbn_relu_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass

    @abstractmethod
    def make_normbn_relu_only_on_vec_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        pass
    
    @abstractmethod
    def make_fourierbn_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        x = self.pool(x)
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
                 mnist=False,
                 residual: bool = False):
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
                         mnist=mnist,
                         residual=residual)

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
            vector_rep.extend([irr] * mult)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        vector_rep = vector_rep*channels
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
    
    def make_fourierbn_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        activation = FourierPointwiseInnerBn(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=self.N)
        feat_type_out = activation.in_type

        conv = nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
        layers.append(conv)

        layers.append(activation)
        layers.append(RetypeModule(feat_type_out, base_field))
        return layers
    
    def make_normbn_relu_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        vector_rep=vector_rep*channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        conv = nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma, bias=True)
        layers.append(conv)

        non_linearity = NormNonlinearityWithBN(in_type=feat_type_out, function=self.non_linearity)


        layers.append(non_linearity)
        return layers

    def make_normbn_relu_only_on_vec_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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

        conv = nn.R2Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma, bias=True)
        layers.append(conv)

        innerbn_elu = nn.SequentialModule(nn.InnerBatchNorm(scalar_field), nn.ELU(scalar_field))
        norm_nonlinbn = NormNonlinearityWithBN(in_type=vector_field, function=self.non_linearity)


        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep), 
                                            modules=[(innerbn_elu,'scalar'), (norm_nonlinbn, 'vector')]
                                            )

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

# ============================================================================
# Point Cloud Variants
# ============================================================================

class R2PointNet(RnNet):
    """
    Point cloud variant of R2Net using R2PointConv instead of R2Conv.
    Operates on 2D point clouds with (x, y) coordinates.
    """
    def __init__(self,
                 n_classes=10,
                 max_rot_order=2,
                 flip=False,
                 channels_per_block=(8, 16, 128),
                 kernels_per_block=(3, 3, 3),  # Used for determining channel adjustments
                 paddings_per_block=(1, 1, 1),  # Not used in point conv but kept for compatibility
                 conv_sigma=0.6,
                 pool_after_every_n_blocks=2,
                 activation_type="gated_sigmoid",
                 pool_size=2,
                 pool_sigma=0.66,
                 invar_type='norm',
                 pool_type='max',
                 invariant_channels=64,
                 bn="IIDbn",
                 grey_scale=False,
                 mnist=False,
                 residual: bool = False,
                 point_conv_n_rings: int = 3,
                 point_conv_frequencies_cutoff: float = 3.0):
        """
        Args:
            point_conv_n_rings: Number of concentric rings for point convolution bases
            point_conv_frequencies_cutoff: Maximum circular harmonic frequency
        """
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        if flip:
            r2_act = gspaces.flipRot2dOnR2(maximum_frequency=max_rot_order)
        else:
            r2_act = gspaces.rot2dOnR2(maximum_frequency=max_rot_order)
        print(f"Using group: {r2_act.name}, {r2_act._sg_id} [Point Cloud Mode]")

        input_channels = 1 if grey_scale else 3
        input_type = nn.FieldType(r2_act, input_channels * [r2_act.trivial_repr])

        self.r2_act = r2_act
        self.input_type = input_type
        self.point_conv_n_rings = point_conv_n_rings
        self.point_conv_frequencies_cutoff = point_conv_frequencies_cutoff

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
                         grid_size=28,  # Not used for point clouds
                         mnist=mnist,
                         residual=residual)

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
        # For point clouds, we can't use spatial pooling directly
        # Instead, return an identity - pooling will be handled differently
        return [nn.IdentityModule(in_type)]

    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if self.invar_type == 'conv2triv':
            # Use 1x1 point convolution
            return nn.R2PointConv(in_type, self.out_inv_type, n_rings=1,
                                 sigma=self.conv_sigma, bias=False)
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
            nn.R2PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                          sigma=self.conv_sigma,
                          frequencies_cutoff=self.point_conv_frequencies_cutoff)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))

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
            nn.R2PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                          sigma=self.conv_sigma,
                          frequencies_cutoff=self.point_conv_frequencies_cutoff)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))

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
            vector_rep.extend([irr] * mult)
        scalar_rep = [self.r2_act.trivial_repr] * channels
        vector_rep = vector_rep*channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        bias = True if self.non_linearity in ['n_relu', 'n_softplus'] else False

        layers.append(nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                                    sigma=self.conv_sigma,
                                    frequencies_cutoff=self.point_conv_frequencies_cutoff))

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

        conv = nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff)
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

    def make_fourierbn_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        activation = FourierPointwiseInnerBn(self.r2_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=self.N)
        feat_type_out = activation.in_type

        conv = nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff)
        layers.append(conv)

        layers.append(activation)
        layers.append(RetypeModule(feat_type_out, base_field))
        return layers

    def make_normbn_relu_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        vector_rep=vector_rep*channels
        scalar_field = nn.FieldType(self.r2_act, scalar_rep)
        vector_field = nn.FieldType(self.r2_act, vector_rep)
        feat_type_out = scalar_field + vector_field

        conv = nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff, bias=True)
        layers.append(conv)

        non_linearity = NormNonlinearityWithBN(in_type=feat_type_out, function=self.non_linearity)

        layers.append(non_linearity)
        return layers

    def make_normbn_relu_only_on_vec_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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

        conv = nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff, bias=True)
        layers.append(conv)

        innerbn_elu = nn.SequentialModule(nn.InnerBatchNorm(scalar_field), nn.ELU(scalar_field))
        norm_nonlinbn = NormNonlinearityWithBN(in_type=vector_field, function=self.non_linearity)

        non_linearity = nn.MultipleModule(in_type=feat_type_out,
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep),
                                            modules=[(innerbn_elu,'scalar'), (norm_nonlinbn, 'vector')]
                                            )

        layers.append(non_linearity)

        return layers

    def make_non_equi_layer_nonlin(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        PointConv -> equivariant batch-norm -> plain torch ReLU (breaks equivariance).
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

        layers.append(nn.R2PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                                     sigma=self.conv_sigma,
                                     frequencies_cutoff=self.point_conv_frequencies_cutoff))
        layers.append(self._build_batch_norm(layers[-1].out_type))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.ReLU(inplace=True)))
        return layers

    def make_non_equi_layer_bn(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        PointConv -> non-equivariant torch BatchNorm1d -> gated nonlinearity (non-shared).
        Note: For point clouds, we use BatchNorm1d instead of BatchNorm2d
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

        layers.append(nn.R2PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                                     sigma=self.conv_sigma,
                                     frequencies_cutoff=self.point_conv_frequencies_cutoff))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.BatchNorm1d(layers[-1].out_type.size)))

        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=['scalar'] * len(scalar_rep) + ['gated'] * len(gate_field),
            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
            reshuffle=0
        )
        layers.append(non_linearity)
        return layers

    def forward(self, input: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass for point cloud data.

        Args:
            input: Feature tensor of shape (num_points, num_features)
            coords: Coordinate tensor of shape (num_points, 2)
            edge_index: Edge connectivity of shape (2, num_edges)
        """
        x = nn.GeometricTensor(input, self.input_type)
        x.coords = coords
        x.edge_index = edge_index

        # Note: MaskModule not applicable for point clouds
        for layer in self.eq_layers:
            x = layer(x)
            # Preserve coords and edge_index through layers
            if hasattr(x, 'coords'):
                pass  # Already set
            else:
                x.coords = coords
                x.edge_index = edge_index

        x = self.invar_map(x)

        # Global pooling for point clouds - aggregate over all points
        x_tensor = x.tensor
        if len(x_tensor.shape) == 2:  # (num_points, features)
            x_pooled = x_tensor.max(dim=0, keepdim=True)[0]  # (1, features)
        else:
            x_pooled = x_tensor

        x_out = self.head(x_pooled.view(1, -1))
        return x_out


class R3PointNet(RnNet):
    """
    Point cloud variant of R3Net using R3PointConv instead of R3Conv.
    Operates on 3D point clouds with (x, y, z) coordinates.
    """
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
                 mnist=False,
                 residual: bool = False,
                 point_conv_n_rings: int = 3,
                 point_conv_frequencies_cutoff: float = 3.0):
        """
        Args:
            point_conv_n_rings: Number of concentric rings for point convolution bases
            point_conv_frequencies_cutoff: Maximum frequency for spherical harmonics
        """
        assert bn in ["IIDbn", "Normbn", "FieldNorm", "GNormBatchNorm"]

        if flip:
            r3_act = gspaces.flipRot3dOnR3(maximum_frequency=max_rot_order)
        else:
            r3_act = gspaces.rot3dOnR3(maximum_frequency=max_rot_order)
        print(f"Using group: {r3_act.name} [Point Cloud Mode]")

        input_type = nn.FieldType(r3_act, [r3_act.trivial_repr])

        self.r3_act = r3_act
        self.input_type = input_type
        self.point_conv_n_rings = point_conv_n_rings
        self.point_conv_frequencies_cutoff = point_conv_frequencies_cutoff

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
                         grid_size=28,  # Not used for point clouds
                         mnist=mnist,
                         residual=residual)

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
        # For point clouds, spatial pooling doesn't apply
        return [nn.IdentityModule(in_type)]

    def _build_invariant_map(self, in_type: nn.FieldType) -> list:
        labels = ["trivial" if r.is_trivial() else "others" for r in in_type]
        cur_type_labeled = in_type.group_by_labels(labels)
        trivials = cur_type_labeled["trivial"]
        others = cur_type_labeled["others"]
        if self.invar_type == 'conv2triv':
            return nn.R3PointConv(in_type, self.out_inv_type, n_rings=1,
                                 sigma=self.conv_sigma, bias=False)
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
            nn.R3PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                          sigma=self.conv_sigma,
                          frequencies_cutoff=self.point_conv_frequencies_cutoff)
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
            nn.R3PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                          sigma=self.conv_sigma,
                          frequencies_cutoff=self.point_conv_frequencies_cutoff)
        )

        layers.append(self._build_batch_norm(layers[-1].out_type))

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

        layers.append(nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                                    sigma=self.conv_sigma,
                                    frequencies_cutoff=self.point_conv_frequencies_cutoff))

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

        conv = nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff)
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

    def make_fourierbn_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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
        activation = FourierPointwiseInnerBn(self.r3_act, channels=channels, irreps = G.bl_irreps(self.max_rot_order), function=self.non_linearity, N=self.N)
        feat_type_out = activation.in_type

        conv = nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff)
        layers.append(conv)

        layers.append(activation)
        layers.append(RetypeModule(feat_type_out, base_field))
        return layers

    def make_normbn_relu_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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

        conv = nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff, bias=True)
        layers.append(conv)

        non_linearity = NormNonlinearityWithBN(in_type=feat_type_out, function=self.non_linearity)

        layers.append(non_linearity)
        return layers

    def make_normbn_relu_only_on_vec_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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

        conv = nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                             sigma=self.conv_sigma,
                             frequencies_cutoff=self.point_conv_frequencies_cutoff, bias=True)
        layers.append(conv)

        innerbn_elu = nn.SequentialModule(nn.InnerBatchNorm(scalar_field), nn.ELU(scalar_field))
        norm_nonlinbn = NormNonlinearityWithBN(in_type=vector_field, function=self.non_linearity)

        non_linearity = nn.MultipleModule(in_type=feat_type_out,
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep),
                                            modules=[(innerbn_elu,'scalar'), (norm_nonlinbn, 'vector')]
                                            )

        layers.append(non_linearity)

        return layers

    def make_non_equi_layer_nonlin(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        PointConv -> equivariant batch-norm -> plain torch ReLU (breaks equivariance).
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

        layers.append(nn.R3PointConv(in_type, feat_type_out, n_rings=self.point_conv_n_rings,
                                     sigma=self.conv_sigma,
                                     frequencies_cutoff=self.point_conv_frequencies_cutoff))
        layers.append(self._build_batch_norm(layers[-1].out_type))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.ReLU(inplace=True)))
        return layers

    def make_non_equi_layer_bn(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
        """
        PointConv -> non-equivariant torch BatchNorm1d -> gated nonlinearity (non-shared).
        Note: For point clouds, we use BatchNorm1d instead of BatchNorm3d
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

        layers.append(nn.R3PointConv(in_type, full_field, n_rings=self.point_conv_n_rings,
                                     sigma=self.conv_sigma,
                                     frequencies_cutoff=self.point_conv_frequencies_cutoff))
        layers.append(NonEquivariantTorchOp(layers[-1].out_type, torch.nn.BatchNorm1d(layers[-1].out_type.size)))

        non_linearity = nn.MultipleModule(
            in_type=full_field,
            labels=['scalar'] * len(scalar_rep) + ['gated'] * len(gate_field),
            modules=[(nn.ELU(scalar_field), 'scalar'), (nn.GatedNonLinearity1(gate_field), 'gated')],
            reshuffle=0
        )
        layers.append(non_linearity)
        return layers

    def forward(self, input: torch.Tensor, coords: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass for point cloud data.

        Args:
            input: Feature tensor of shape (num_points, num_features)
            coords: Coordinate tensor of shape (num_points, 3)
            edge_index: Edge connectivity of shape (2, num_edges)
        """
        x = nn.GeometricTensor(input, self.input_type)
        x.coords = coords
        x.edge_index = edge_index

        # Note: MaskModule not applicable for point clouds
        for layer in self.eq_layers:
            x = layer(x)
            # Preserve coords and edge_index through layers
            if hasattr(x, 'coords'):
                pass  # Already set
            else:
                x.coords = coords
                x.edge_index = edge_index

        x = self.invar_map(x)

        # Global pooling for point clouds - aggregate over all points
        x_tensor = x.tensor
        if len(x_tensor.shape) == 2:  # (num_points, features)
            x_pooled = x_tensor.max(dim=0, keepdim=True)[0]  # (1, features)
        else:
            x_pooled = x_tensor

        x_out = self.head(x_pooled.view(1, -1))
        return x_out

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
            nn.R3Conv(in_type, full_field, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
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

        conv = nn.R3Conv(in_type, feat_type_out, kernel_size=kernel_size, padding=pad, sigma=self.conv_sigma)
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
    def make_normbn_relu_block(self, in_type: nn.FieldType, channels: int, kernel_size: int, pad: int):
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

        innerbn_elu = nn.SequentialModule(nn.InnerBatchNorm(scalar_field), nn.ELU(scalar_field))
        norm_nonlin = nn.NormNonLinearity(in_type=vector_field, function=self.non_linearity, bias=bias)


        non_linearity = nn.MultipleModule(in_type=feat_type_out, 
                                            labels=['scalar']*len(scalar_rep) + ['vector']*len(vector_rep), 
                                            modules=[(innerbn_elu,'scalar'), (norm_nonlin, 'vector')]
                                            )

        layers.append(non_linearity)



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
